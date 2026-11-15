"""Dual-schema IPC contracts; no real model or accelerator is loaded."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sunhacks.local_digest import DigestRow,combine_ensemble_scores,digest_contract
from sunhacks.local_stack_process import ProcessDigestStack,DigestWorkerError

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dual_schema_worker',ROOT/'sunhacks/_xgb_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
OLD='v2-no-within-z';NEW='v3-fixed-lag-motion'


@pytest.mark.parametrize('version,count',[(OLD,89),(NEW,88)])
def test_worker_contract_binds_known_version_hash_and_count(version,count):
    digest=worker.SCHEMA_CONTRACTS[version][0]
    assert worker.schema_contract({'feature_version':version,'schema_sha256':digest,'feature_count':count})==(version,count)


@pytest.mark.parametrize('change',[{'feature_version':'unreviewed'}, {'schema_sha256':'0'*64}, {'feature_count':89}, {'feature_count':True}])
def test_new_worker_schema_rejects_cross_version_or_unpinned_contract(change):
    request={'feature_version':NEW,'schema_sha256':worker.SCHEMA_CONTRACTS[NEW][0],'feature_count':88}
    request.update(change)
    with pytest.raises(ValueError):worker.schema_contract(request)


def test_new_worker_version_requires_explicit_count_but_old_request_can_be_legacy():
    with pytest.raises(ValueError):worker.schema_contract({'feature_version':NEW,'schema_sha256':worker.SCHEMA_CONTRACTS[NEW][0]})
    assert worker.schema_contract({'schema_sha256':worker.SCHEMA_CONTRACTS[OLD][0]})==(OLD,89)


def test_new_worker_predict_uses88_fields_and_preserves_missing_values():
    names=[f'f_{i}' for i in range(88)];seen=[]
    state={'model':SimpleNamespace(predict=lambda matrix:(seen.append(matrix) or np.array([.4]))),
           'numpy':np,'xgb':SimpleNamespace(DMatrix=lambda values,feature_names:values),
           'names':names,'feature_version':NEW,'feature_count':88}
    values=[1.0]*88;values[3]=None
    request={'protocol':worker.PROTOCOL,'feature_version':NEW,'feature_count':88,'values':values,
             'feature_names_sha256':worker.names_digest(names),'source_window_id':9}
    reply=worker.predict(state,request)
    assert reply['feature_count']==88 and reply['feature_version']==NEW
    assert reply['raw_score']==.4 and np.isnan(seen[0][0,3])
    request['values']=[1.0]*89
    with pytest.raises(ValueError,match='row width'):worker.predict(state,request)
    request['values']=[1.0]*88;request.pop('feature_version')
    with pytest.raises(ValueError,match='identity'):worker.predict(state,request)


@pytest.mark.parametrize('version,count,reply,expected',[
    (OLD,89,{'feature_count':89},True),
    (OLD,89,{},True),
    (NEW,88,{'feature_count':88},False),
    (NEW,88,{'feature_version':NEW,'feature_count':88},True),
    (NEW,88,{'feature_version':OLD,'feature_count':89},False),
    (NEW,88,{'feature_version':NEW,'feature_count':True},False),
])
def test_parent_accepts_legacy_replies_only_for_legacy_schema(version,count,reply,expected):
    seam=SimpleNamespace(feature_version=version,feature_count=count)
    assert ProcessDigestStack._response_has_schema(seam,reply) is expected


FAKE=r'''
import json,os,sys
for line in sys.stdin.buffer:
 r=json.loads(line)
 reply={'protocol':r['protocol'],'request_id':r['request_id'],'feature_count':r['feature_count']}
 if MODE!='missing_version':reply['feature_version']=r['feature_version']
 if r['action']=='prepare':
  reply.update(status='ready',pid=os.getpid(),model_sha256=r['model_sha256'],schema_sha256=r['schema_sha256'],feature_names_sha256=r['feature_names_sha256'],torch_imported=False,backend='xgboost_cpu_process')
 else:reply.update(status='success',source_window_id=r['source_window_id'],raw_score=.4)
 print(json.dumps(reply),flush=True)
'''


def fixture(tmp_path,mode='normal'):
    # Self-contained technical manifest: no checkout-specific review directory,
    # downloaded artifact or trained model is required by these IPC tests.
    manifest={'format':'sunhacks-ensemble-v2','feature_version':NEW,
        'branches':['tuned_dasheng','xgb_digest'],'platt_A':[1.,1.],
        'platt_B':[0.,0.],'weights':[.6,.4],'threshold':.4238521751969923,
        'threshold_provenance':'Generated dual-schema IPC fixture.',
        'digest_schema_sha256':digest_contract(NEW)['schema_sha256']}
    bp=tmp_path/'model.json';bp.write_text('{"fake":true}')
    manifest['digest_model_sha256']=hashlib.sha256(bp.read_bytes()).hexdigest()
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    schema_path=ROOT/'sunhacks/vendor/friend_digest_schema_v3.json'
    script=tmp_path/'worker.py';script.write_text(FAKE.replace('MODE',repr(mode)))
    obj=ProcessDigestStack(mp,bp,feature_version=NEW,schema_path=schema_path,_worker_path=script,
        expected_manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
        expected_model_sha256=manifest['digest_model_sha256'])
    return obj,manifest


def test_new_parent_runs88_field_process_and_propagates_export_policy(tmp_path):
    obj,manifest=fixture(tmp_path)
    try:
        ready=obj.prepare()
        assert ready['feature_count']==88 and ready['feature_version']==NEW
        names=tuple(obj.schema['feature_names']);values=np.ones((1,88),np.float32)
        result=obj.infer(DigestRow(names,values,(),feature_version=NEW),.7,source_window_id=3)
        assert result['tabular']['model_id']=='teammate/xgb_digest_88'
        assert result['tabular']['feature_count']==88
        assert result['stack']['feature_version']==NEW
        assert result['stack']['threshold']==manifest['threshold']
        assert result['stack']['threshold_provenance']==manifest['threshold_provenance']
        assert result['stack']['raw_score']==pytest.approx(combine_ensemble_scores(.7,.4,manifest))
    finally:obj.close()


def test_new_parent_rejects_missing_version_handshake_and_reaps_worker(tmp_path):
    obj,_=fixture(tmp_path,'missing_version')
    try:
        with pytest.raises(DigestWorkerError,match='handshake'):obj.prepare()
        assert obj._process is None
    finally:obj.close()


def test_worker_schema_pins_match_the_parent_reviewed_contracts():
    for version,(digest,count) in worker.SCHEMA_CONTRACTS.items():
        contract=digest_contract(version)
        assert contract['schema_sha256']==digest
        assert contract['feature_count']==count
