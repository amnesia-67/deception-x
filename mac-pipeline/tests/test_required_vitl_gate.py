from types import SimpleNamespace
import pytest
from sunhacks.pipeline import Pipeline
from sunhacks.showcase import ShowcaseSession


def test_model_blocker_prevents_any_b_client_or_model_preparation(monkeypatch):
    import sunhacks.remote_jepa_vitl as remote
    monkeypatch.setattr(remote,'RemoteJepa21LClassifier',lambda *a,**k:pytest.fail('B client constructed despite L requirement'))
    p=Pipeline({'runtime_mode':'split_local_video','deployment':{'runtime_blocker':'V-JEPA 2.1 ViT-L required; B rejected'}})
    state=p.prepare_runtime()
    assert state['status']=='blocked' and state['live_ready'] is False
    assert p.prepare_thread is None and p.fast_runtime is None and p.video_client is None
    with pytest.raises(RuntimeError):p.start('live',auto_stop_seconds=7)
    assert p.capture is None
    view=ShowcaseSession(p).snapshot()
    assert view['runtime_blocked'] and not view['model_ready'] and not view['can_prepare']


def test_l_requirement_invalidates_cached_b_ready_flags():
    p=Pipeline({'runtime_mode':'split_local_video','deployment':{'runtime_blocker':'L required'}})
    p.runtime_preparation={'status':'ready','local_weights_loaded':True,'components':{}}
    p.video_client=SimpleNamespace(snapshot=lambda:{'classifier_ready':True,'classifier_response_validated':True,'participant_media_authorized':True})
    view=p._split_runtime_state()
    assert not view['live_ready'] and view['status']=='blocked' and view['reason']=='L required'
