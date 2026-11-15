import threading
from types import SimpleNamespace
import numpy as np
import pytest
from sunhacks.pipeline import Pipeline
from sunhacks.showcase import ShowcaseSession
from test_showcase import real_latest
from test_split_runtime import setup,inputs


def test_disabled_video_runs_local_stack_without_calling_remote(monkeypatch):
    runtime,video,*_=setup(monkeypatch)
    runtime.config['remote_vision']={'enabled':False}
    video.predict=lambda *a,**k:pytest.fail('Disabled video was sent')
    clip,audio,tr,ts=inputs();out=runtime.infer(clip,audio,tr,1,timestamps=ts)
    assert out['branches']['stack']['status']=='trained_classifier'
    assert out['branches']['vision']['status']=='disabled'
    assert out['branches']['vision']['raw_score'] is None
    assert out['remote_video_enabled'] is False
    assert 'vision' not in out['timings']['jobs']
    runtime.close()


@pytest.mark.parametrize('enabled',[False,True])
def test_local_result_requires_video_only_when_enabled(enabled):
    latest=real_latest();latest['remote_video_enabled']=enabled
    latest['branches']['vision']={'status':'disabled','source_window_id':1,'raw_score':None}
    result=ShowcaseSession._result({'session_id':'s','latest':latest},'s')
    assert result['model_result_valid'] is (not enabled)
    assert result['label']==('inconclusive' if enabled else 'lie')
    assert result['branch_scores']['video'] is None


def test_disabled_video_does_not_bypass_required_local_model_or_face_gate():
    latest=real_latest();latest['remote_video_enabled']=False
    latest['branches']['vision']={'status':'disabled'}
    latest['digest']['face']['face_detect_rate']=0
    assert not ShowcaseSession._result({'session_id':'s','latest':latest},'s')['model_result_valid']
    latest['digest']['face']['face_detect_rate']=1
    latest['branches']['dasheng']['status']='unavailable'
    assert not ShowcaseSession._result({'session_id':'s','latest':latest},'s')['model_result_valid']


def test_disabling_socket_releases_client_and_enabling_requires_fresh_check():
    closed=[]
    p=Pipeline({'runtime_mode':'split_local_video','remote_vision':{'enabled':True}})
    p.runtime_preparation={'status':'ready','local_weights_loaded':True}
    p.video_client=SimpleNamespace(close=lambda:closed.append(True))
    p.set_remote_video(False)
    assert closed==[True] and p.video_client is None
    assert p._split_runtime_state()['live_ready']
    p.set_remote_video(True)
    assert not p._split_runtime_state()['live_ready']


def test_disabled_prepare_constructs_no_video_client_or_probe(monkeypatch):
    import sunhacks.remote_jepa_vitl as remote
    import sunhacks.split_runtime as split
    monkeypatch.setattr(remote,'RemoteJepa21LClassifier',lambda *a,**k:pytest.fail('GPU client constructed'))
    class Runtime:
        def __init__(self,*a,**kw):self.config=a[0];self.video_client=None
        def prepare(self):return {'status':'ready','local_weights_loaded':True}
        def snapshot(self):return {'active_jobs':[]}
    monkeypatch.setattr(split,'SplitRuntime',Runtime)
    p=Pipeline({'runtime_mode':'split_local_video','remote_vision':{'enabled':False}})
    p.prepare_runtime();p.prepare_thread.join(2)
    assert p.video_client is None and p._split_runtime_state()['live_ready']
    assert p.runtime_preparation['video_check']['status']=='disabled'


@pytest.mark.parametrize('value',[None,'false',0,1,[],{}])
def test_toggle_rejects_coercible_boolean_values(value):
    p=Pipeline({'runtime_mode':'split_local_video'})
    with pytest.raises(ValueError):p.set_remote_video(value)


@pytest.mark.parametrize('status',['capturing','finishing','processing'])
def test_active_answer_cannot_change_video_policy(status):
    p=Pipeline({'runtime_mode':'split_local_video','remote_vision':{'enabled':True}})
    p.answer={'status':status}
    with pytest.raises(RuntimeError):p.set_remote_video(False)
    assert p.config['remote_vision']['enabled']


def test_recorded_disabled_policy_survives_later_current_setting_change():
    latest=real_latest();latest['remote_video_enabled']=False
    latest['branches']['vision']={'status':'disabled'}
    result=ShowcaseSession._result({'session_id':'s','latest':latest,'runtime':{'remote_video_enabled':True}},'s')
    assert result['model_result_valid'] and result['remote_video_enabled'] is False
