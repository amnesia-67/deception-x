"""Process isolation/IPC regressions with tiny stdlib fake worker scripts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from sunhacks.local_digest import DigestRow,combine_ensemble_scores,load_digest_schema
from sunhacks.local_stack_process import ProcessDigestStack,DigestWorkerError,worker_environment

FAKE_WORKER=r'''
import sys,json,os,time
mode=MODE
protocol="sunhacks-xgb-worker-v1"
for line in sys.stdin.buffer:
    r=json.loads(line)
    if r['action']=='prepare':
        if mode=='startup_exit':sys.exit(9)
        if mode=='startup_timeout':time.sleep(2)
        if mode=='oversize':sys.stdout.write('x'*70000+'\n');sys.stdout.flush();continue
        reply={'status':'ready','protocol':protocol,'pid':os.getpid(),'feature_count':89,
               'model_sha256':r['model_sha256'],'schema_sha256':r['schema_sha256'],
               'feature_names_sha256':r['feature_names_sha256'],'torch_imported':False,
               'backend':'xgboost_cpu_process','request_id':r['request_id'],'xgboost_version':'fake'}
        if mode=='wrong_hash':reply['model_sha256']='0'*64
        if mode=='env':
            if any(key in os.environ for key in ('DEEPSEEK_API_KEY','SUNHACKS_SOCKET_TOKEN','PYTHONPATH','KMP_DUPLICATE_LIB_OK')):
                reply['torch_imported']=True
    else:
        if mode=='predict_exit':sys.exit(139)
        if mode=='predict_timeout':time.sleep(2)
        if mode=='bad_json':sys.stdout.write('not-json\n');sys.stdout.flush();continue
        if mode=='nan_json':sys.stdout.write('{"raw_score":NaN}\n');sys.stdout.flush();continue
        reply={'status':'success','protocol':protocol,'request_id':r['request_id'],
               'source_window_id':r['source_window_id'],'raw_score':.4}
        if mode=='wrong_request':reply['request_id']='0'*32
        if mode=='bool_window':reply['source_window_id']=True
        if mode=='nan_preserved' and r['values'][10] is not None:reply['raw_score']=False
    sys.stdout.write(json.dumps(reply)+'\n');sys.stdout.flush()
'''


def artifacts(tmp_path,mode='normal',timeout=.6):
    manifest={'format':'sunhacks-ensemble-v1','branches':['tuned_dasheng','xgb_digest'],
              'platt_A':[1.2,1.4],'platt_B':[.2,.4],'weights':[.6,.4]}
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    bp=tmp_path/'model.json';bp.write_text('{"test":true}')
    sp=tmp_path/'schema.json';sp.write_bytes((Path(__file__).resolve().parents[1]/'sunhacks/vendor/friend_digest_schema.json').read_bytes())
    wp=tmp_path/'worker.py';wp.write_text(FAKE_WORKER.replace('MODE',repr(mode)))
    stack=ProcessDigestStack(mp,bp,schema_path=sp,_worker_path=wp,timeout_seconds=timeout,
        expected_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
        expected_model_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),
        expected_schema_sha256=hashlib.sha256(sp.read_bytes()).hexdigest())
    return stack,manifest,bp


def row():
    names=tuple(load_digest_schema()['feature_names']);values=np.ones((1,89),np.float32);values[0,10]=np.nan
    return DigestRow(names,values,(names[10],))


def test_persistent_child_keeps_scores_local_and_preserves_nan(tmp_path):
    stack,manifest,_=artifacts(tmp_path,'nan_preserved')
    ready=stack.prepare();process=stack._process
    assert ready['backend']=='xgboost_cpu_process' and ready['parent_imports_xgboost'] is False
    assert stack.prepare()['worker_pid']==process.pid and stack.prepare()['cache_hit'] is True
    result=stack.infer(row(),.7,source_window_id=4)
    assert result['stack']['raw_score']==pytest.approx(combine_ensemble_scores(.7,.4,manifest))
    assert result['stack']['not_used_modalities']==['vision']
    assert result['tabular']['missing_features']==list(row().missing_features)
    assert result['tabular']['source_window_id']==4
    stack.close();assert process.poll() is not None
    stack.close()
    with pytest.raises(DigestWorkerError,match='closed'):stack.prepare()


def test_worker_environment_drops_provider_keys_and_global_omp_override(tmp_path,monkeypatch):
    for key in ('DEEPSEEK_API_KEY','SUNHACKS_SOCKET_TOKEN','PYTHONPATH','KMP_DUPLICATE_LIB_OK'):
        monkeypatch.setenv(key,'test-private-value')
    env=worker_environment()
    assert all(key not in env for key in ('DEEPSEEK_API_KEY','SUNHACKS_SOCKET_TOKEN','PYTHONPATH','KMP_DUPLICATE_LIB_OK'))
    stack,_,_=artifacts(tmp_path,'env')
    assert stack.prepare()['status']=='ready'
    stack.close()


@pytest.mark.parametrize('mode',['startup_exit','startup_timeout','oversize','wrong_hash'])
def test_bad_or_hung_startup_is_bounded_and_reaped(tmp_path,mode):
    stack,_,_=artifacts(tmp_path,mode,timeout=.1)
    started=time.monotonic()
    with pytest.raises(DigestWorkerError):stack.prepare()
    assert time.monotonic()-started<1.5
    assert stack._process is None
    stack.close()


@pytest.mark.parametrize('mode',['predict_exit','predict_timeout','bad_json','nan_json','wrong_request','bool_window'])
def test_crash_timeout_or_bad_prediction_invalidates_child(tmp_path,mode):
    stack,_,_=artifacts(tmp_path,mode,timeout=.15)
    stack.prepare();process=stack._process
    with pytest.raises(DigestWorkerError):stack.infer(row(),.7,source_window_id=1)
    assert stack._process is None and process.poll() is not None
    with pytest.raises(DigestWorkerError,match='not prepared'):stack.infer(row(),.7)
    stack.close()


def test_artifact_tamper_is_rejected_before_any_child_is_created(tmp_path):
    stack,_,bp=artifacts(tmp_path)
    bp.write_bytes(b'altered')
    with pytest.raises(ValueError,match='hash'):stack.prepare()
    assert stack._process is None
    stack.close()


def test_close_interrupts_an_active_worker_rpc_without_waiting_for_rpc_lock(tmp_path):
    stack,_,_=artifacts(tmp_path,'predict_timeout',timeout=5)
    stack.prepare();process=stack._process;errors=[]
    thread=threading.Thread(target=lambda:run())
    def run():
        try:stack.infer(row(),.7)
        except Exception as exc:errors.append(exc)
    thread.start();time.sleep(.04)
    started=time.monotonic();stack.close();thread.join(1)
    assert time.monotonic()-started<1.5 and not thread.is_alive()
    assert errors and isinstance(errors[0],DigestWorkerError)
    assert process.poll() is not None


def test_parent_module_never_imports_xgboost_and_worker_never_imports_torch():
    folder=Path(__file__).resolve().parents[1]/'sunhacks'
    parent=ast.parse((folder/'local_stack_process.py').read_text())
    worker=ast.parse((folder/'_xgb_worker.py').read_text())
    def imports(tree):
        out=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):out.extend(item.name for item in node.names)
            elif isinstance(node,ast.ImportFrom):out.append(node.module or '')
        return out
    assert 'xgboost' not in imports(parent)
    assert 'torch' not in imports(worker) and not any(name.startswith('sunhacks') for name in imports(worker))


def test_worker_numeric_contract_preserves_nan_without_loading_a_real_model():
    path=Path(__file__).resolve().parents[1]/'sunhacks'/'_xgb_worker.py'
    spec=importlib.util.spec_from_file_location('isolated_xgb_worker_test',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    names=load_digest_schema()['feature_names'];seen=[]
    class Model:
        def predict(self,matrix):seen.append(matrix);return np.array([.4])
    state={'model':Model(),'numpy':np,'names':names,'xgb':SimpleNamespace(DMatrix=lambda values,feature_names:values)}
    values=[1.0]*89;values[10]=None
    request={'protocol':worker.PROTOCOL,'values':values,'source_window_id':7,'feature_names_sha256':worker.names_digest(names)}
    result=worker.predict(state,request)
    assert result['raw_score']==.4 and np.isnan(seen[0][0,10])
    request['values'][2]=True
    with pytest.raises(ValueError,match='invalid row'):worker.predict(state,request)


def test_waiting_for_rpc_lock_shares_the_request_deadline(tmp_path):
    stack,_,_=artifacts(tmp_path,timeout=.05)
    stack._rpc_lock.acquire()
    started=time.monotonic()
    try:
        with pytest.raises(DigestWorkerError,match="busy"):
            stack.prepare()
    finally:stack._rpc_lock.release()
    assert .03 < time.monotonic()-started < .5
    assert stack._process is None
    stack.close()
