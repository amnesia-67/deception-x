from types import SimpleNamespace
import threading
import sys
import pytest
from sunhacks import capture


def fake_audio(monkeypatch,*,fail=False):
    calls=[]
    sd=SimpleNamespace(_initialized=1)
    def terminate():calls.append('terminate');sd._initialized-=1
    def initialize():calls.append('initialize');sd._initialized+=1
    def query(device,kind):calls.append(('query',device));return {'name':'Blue Snowball ','default_samplerate':48000.}
    class Stream:
        def __init__(self,**kwargs):
            calls.append(('open',kwargs['device']));self.callback=kwargs['callback']
            if fail:raise RuntimeError('Device disconnected')
        def __enter__(self):
            for _ in range(3):self.callback(None,4800,None,False)
            return self
        def __exit__(self,*args):calls.append('closed')
    sd._terminate=terminate;sd._initialize=initialize;sd.query_devices=query;sd.InputStream=Stream
    monkeypatch.setitem(sys.modules,'sounddevice',sd)
    monkeypatch.setattr(capture,'media_authorization_status',lambda:{'audio':'authorized'})
    return sd,calls


def test_idle_reconnect_reinitializes_and_checks_only_selected_microphone(monkeypatch):
    sd,calls=fake_audio(monkeypatch)
    result=capture.reconnect_microphone('Blue Snowball ')
    assert result['status']=='ready' and result['audio_saved'] is False
    assert result['samples_received']==14400 and result['sample_rate']==48000
    assert calls[:2]==['terminate','initialize']
    assert ('open','Blue Snowball ') in calls and 'closed' in calls
    assert sd._initialized==1


def test_reconnect_cannot_reset_active_audio_backend(monkeypatch):
    sd,calls=fake_audio(monkeypatch);capture._AUDIO_BACKEND_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError,match='active microphone'):capture.reconnect_microphone('Blue Snowball ')
        assert calls==[]
    finally:capture._AUDIO_BACKEND_LOCK.release()


def test_live_capture_cannot_start_during_backend_reconnect(monkeypatch):
    sd,calls=fake_audio(monkeypatch);capture._AUDIO_BACKEND_LOCK.acquire()
    try:
        c=capture.LiveCapture();c._audio_worker()
        assert c.snapshot()['audio']['status']=='missing' and not calls
    finally:capture._AUDIO_BACKEND_LOCK.release()


def test_reconnect_never_prompts_for_new_microphone_permission(monkeypatch):
    sd,calls=fake_audio(monkeypatch)
    monkeypatch.setattr(capture,'media_authorization_status',lambda:{'audio':'not_determined'})
    with pytest.raises(RuntimeError,match='Authorize'):capture.reconnect_microphone()
    assert calls==[]


def test_failed_microphone_releases_backend_lease_without_switching_device(monkeypatch):
    sd,calls=fake_audio(monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='disconnected'):capture.reconnect_microphone('Blue Snowball ')
    assert capture._AUDIO_BACKEND_LOCK.acquire(blocking=False)
    capture._AUDIO_BACKEND_LOCK.release()
    assert [x for x in calls if isinstance(x,tuple) and x[0]=='open']==[('open','Blue Snowball ')]


def test_incompatible_backend_refcount_requires_restart_without_terminating(monkeypatch):
    sd,calls=fake_audio(monkeypatch);sd._initialized=2
    with pytest.raises(RuntimeError,match='Restart'):capture.reconnect_microphone()
    assert calls==[]


def test_reconnect_route_is_idle_only_and_leaves_models_in_place(monkeypatch,tmp_path):
    import httpx
    from sunhacks.pipeline import Pipeline
    from sunhacks.server import make_server
    p=Pipeline({'audio_device':'Blue Snowball ','dataset_root':str(tmp_path)})
    calls=[]
    def reconnect(device):calls.append(device);return {'status':'ready','audio_saved':False}
    monkeypatch.setattr(capture,'reconnect_microphone',reconnect)
    server=make_server(p,0);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',trust_env=False) as client:
            assert client.post('/api/audio/reconnect',json={}).json()['status']=='ready'
            assert calls==['Blue Snowball '] and p.fast_runtime is None
            p.answer={'status':'capturing'}
            assert client.post('/api/audio/reconnect',json={}).status_code==400
            assert calls==['Blue Snowball ']
            p.answer={}
            assert client.post('/api/audio/reconnect',json={},headers={'Origin':'https://other.example'}).status_code==403
    finally:server.shutdown();server.server_close();worker.join(1);p.close()
