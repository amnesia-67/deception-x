import threading
import httpx
import pytest
from sunhacks.pipeline import Pipeline
from sunhacks.server import make_server


def test_showcase_pages_and_shared_capture_ownership():
    pipeline=Pipeline()
    class Controller:
        active=False
        prepares=0
        def is_active(self):return self.active
        def snapshot(self):return {'stage':'warmup' if self.active else 'idle','model_ready':False}
        def start_warmup(self):self.active=True;return self.snapshot()
        def cancel(self):self.active=False;return self.snapshot()
        def prepare_models(self):self.prepares+=1;return self.snapshot()
        def configure_video(self,enabled):
            if type(enabled) is not bool:raise ValueError('enabled must be a boolean')
            self.video=enabled;return self.snapshot()
        claim_check=True
        answers=[]
        def claim_check_enabled(self):return self.claim_check
        def configure_claim_check(self,enabled):
            if type(enabled) is not bool:raise ValueError('enabled must be a boolean')
            self.claim_check=enabled;return self.snapshot()
        def start_answer(self,*,visual_claim_check):self.answers.append(visual_claim_check);return self.snapshot()
    controller=Controller();pipeline.showcase=controller
    server=make_server(pipeline,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',trust_env=False,timeout=2) as client:
            assert 'THE SEVEN-SECOND TEST' in client.get('/').text
            assert 'Multimodal workbench' in client.get('/dashboard').text
            assert client.get('/showcase').status_code==200
            assert client.get('/api/showcase').json()['stage']=='idle'
            assert client.post('/api/showcase/video',json={'enabled':False}).status_code==200
            assert controller.video is False
            assert client.post('/api/showcase/video',json={'enabled':'false'}).status_code==400
            # The showcase page no longer chooses the claim check; the server-side setting does.
            assert client.post('/api/showcase/answer',json={'visual_claim_check':False}).status_code==200
            assert controller.answers==[True]
            assert client.post('/api/showcase/claim-check',json={'enabled':False}).status_code==200
            assert client.post('/api/showcase/claim-check',json={'enabled':'true'}).status_code==400
            assert client.post('/api/showcase/answer',json={'visual_claim_check':True}).status_code==200
            assert controller.answers==[True,False]
            assert client.post('/api/showcase/startwarmup',headers={'Origin':'https://evil.example'},json={}).status_code==403
            assert not controller.active
            assert client.post('/api/showcase/startwarmup',json={}).status_code==200
            assert controller.active
            for route,body in [('/api/session',{'mode':'live'}),('/api/prepare',{}),('/api/answer/finish',{'stop_now':True}),('/api/transcript',{'text':'unrelated'})]:
                response=client.post(route,json=body)
                assert response.status_code==400
                assert 'showcase is active' in response.json()['error']
            assert pipeline.capture is None
            assert client.post('/api/stop',json={}).status_code==200
            assert not controller.active
            assert client.post('/api/showcase/prepare',json={}).status_code==200
            assert controller.prepares==1
            assert client.post('/api/transcript',json={'text':'permitted after cancel'}).status_code==200
    finally:
        server.shutdown();server.server_close();thread.join(1);pipeline.stop()


def test_new_participant_forgets_data_while_retaining_prepared_runtime():
    from types import SimpleNamespace
    pipeline=Pipeline()
    runtime=SimpleNamespace(snapshot=lambda:{'active_jobs':[],'active_prepares':0})
    pipeline.fast_runtime=runtime
    pipeline.session_id='prior';pipeline.latest={'transcript':{'text':'private previous answer'}}
    pipeline.transcript={'text':'previous transcript'}
    pipeline.calibration_values=[.9];pipeline.calibration={'status':'complete','data':'prior'}
    pipeline.clear_session_data()
    assert pipeline.fast_runtime is runtime
    assert pipeline.latest is None and pipeline.session_id is None
    assert pipeline.transcript['text']==''
    assert pipeline.calibration['status']=='not_started' and not pipeline.calibration_values


def test_new_participant_cannot_clear_an_active_capture():
    from types import SimpleNamespace
    pipeline=Pipeline();pipeline.session_id='active';pipeline.latest={'keep':'until stopped'}
    pipeline.capture=SimpleNamespace(snapshot=lambda:{'camera':{'status':'running'},'audio':{'status':'running'}})
    with pytest.raises(RuntimeError,match='Stop capture'):pipeline.clear_session_data()
    assert pipeline.latest=={'keep':'until stopped'} and pipeline.session_id=='active'
