import copy,hashlib,json,threading
from pathlib import Path
import numpy as np
import pytest,torch
from sunhacks import pilot_audio as pilot
from sunhacks.local_dasheng import tensor_schema
from test_local_dasheng import TinyFold
from test_split_runtime import setup,inputs
from test_showcase import real_latest
from sunhacks.showcase import ShowcaseSession


def export(tmp_path,monkeypatch):
    config={'backbone':'dasheng','lora_r':8,'lora_blocks':4,'lora_dropout':.1,'head_dropout':.3}
    model=TinyFold()
    with torch.no_grad():model.head.out.bias.fill_(1.2)
    state={'lora':model.backbone.state_dict(),'head':model.head.state_dict()}
    path=tmp_path/'fold0.pt';torch.save({'config':config,'state':state},path);data=path.read_bytes()
    digest=hashlib.sha256(data).hexdigest();monkeypatch.setattr(pilot,'FOLD_SHA256',digest)
    meta={'format':'sunhacks-pilot-webcam-v1','profile':'webcam_pilot','backbone':'dasheng','model':pilot.MODEL,
          'folds':{'fold0.pt':{'size':len(data),'sha256':digest,'config':config,'state_schema':tensor_schema(state)}},
          'calibration':{'A':1.9653717801525192,'B':-.11875779621198643},'threshold':.5436123431421811,
          'threshold_provenance':'Exported validation cutoff','metrics':{},'caveats':[]}
    raw=json.dumps(meta).encode();manifest=tmp_path/'deploy_meta.json';manifest.write_bytes(raw)
    monkeypatch.setattr(pilot,'MANIFEST_SHA256',hashlib.sha256(raw).hexdigest())
    return manifest,meta


def test_one_model_loads_strict_state_and_uses_own_calibration_and_threshold(tmp_path,monkeypatch):
    path,meta=export(tmp_path,monkeypatch);m=pilot.PilotDasheng(path,tmp_path,model_factory=lambda _:TinyFold())
    out=m.infer(np.ones(16000,np.float32)*.01,source_window_id=17)
    assert len(out['per_fold_scores'])==1 and out['profile']=='webcam_pilot' and out['source_window_id']==17
    decision=out['_pilot_decision']
    assert decision['threshold']==meta['threshold'] and decision['source_window_id']==17
    assert decision['raw_score']==pilot.calibrated_probability(out['raw_score'],meta['calibration'])
    assert decision['used_modalities']==['tuned_dasheng_webcam_pilot']
    assert m.prepare()['classifier']==pilot.MODEL and m.prepare()['folds']==1
    m.close()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,-.1,1.1])
def test_invalid_probabilities_rejected(value):
    with pytest.raises(ValueError):pilot.calibrated_probability(value,{'A':1,'B':0})


def test_published_calibration_matches_expected_probability():
    p=pilot.calibrated_probability(.7721172972209245,{'A':1.9653717801525192,'B':-.11875779621198643})
    assert p==pytest.approx(.9071730386843133)


@pytest.mark.parametrize('change',['manifest','checkpoint','extra_fold'])
def test_model_provenance_mismatches_rejected_before_factory(tmp_path,monkeypatch,change):
    manifest,_=export(tmp_path,monkeypatch)
    if change=='manifest':manifest.write_text(manifest.read_text()+' ')
    if change=='checkpoint':(tmp_path/'fold0.pt').write_bytes(b'changed')
    if change=='extra_fold':(tmp_path/'fold1.pt').write_bytes(b'extra')
    with pytest.raises(ValueError):pilot.PilotDasheng(manifest,tmp_path,model_factory=lambda _:pytest.fail('Invalid weights reached model construction'))


def test_runtime_uses_pilot_decision_even_if_old_stack_disagrees(monkeypatch):
    runtime,video,dash,*_=setup(monkeypatch);runtime.classifier_profile='webcam_pilot'
    original=dash.infer
    def audio(*a,**kw):
        out=original(*a,**kw);out['_pilot_decision']={'status':'trained_classifier','profile':'webcam_pilot','raw_score':.50,
            'threshold':.5436123431421811,'threshold_provenance':'pilot export','source_window_id':kw['source_window_id']}
        return out
    dash.infer=audio
    clip,wave,text,ts=inputs();out=runtime.infer(clip,wave,text,7,timestamps=ts)
    assert out['branches']['stack']['raw_score']==.5 and out['branches']['stack']['threshold']==.5436123431421811
    assert out['branches']['stack']['profile']=='webcam_pilot';runtime.close()


def test_failed_selected_pilot_does_not_reuse_old_classifier(monkeypatch):
    runtime,video,dash,*_=setup(monkeypatch);runtime.classifier_profile='webcam_pilot'
    dash.infer=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Pilot failure'))
    clip,wave,text,ts=inputs();out=runtime.infer(clip,wave,text,7,timestamps=ts)
    assert out['branches']['stack']['status']=='unavailable' and out['branches']['stack']['raw_score'] is None
    runtime.close()


def test_showcase_pilot_threshold_and_contradiction_only_rule():
    latest=real_latest(.5);latest['remote_video_enabled']=False
    latest['branches']['stack'].update(profile='webcam_pilot',threshold=.5436123431421811)
    for name in ('text','emotion','tabular'):latest['branches'][name]={'status':'error','source_window_id':1}
    latest['claim_check_enabled']=True;latest['claim_check']={'status':'no_contradiction','kind':'multimodal','source_window_id':1,'reason':'Fixture','evidence':[]}
    state={'session_id':'s','latest':latest,'answer':{}}
    value=ShowcaseSession._result(state,'s');assert value['label']=='truth' and value['classifier_profile']=='webcam_pilot'
    latest['branches']['stack']['raw_score']=.6
    assert ShowcaseSession._result(state,'s')['label']=='lie'
    latest['branches']['stack']['raw_score']=.5;latest['claim_check']['status']='contradiction'
    assert ShowcaseSession._result(state,'s')['label']=='lie'
