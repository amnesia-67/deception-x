"""Exact-input inspection without extra calls, stale evidence or public photos."""
import json
import threading

import pytest

from sunhacks.claim_check import input_fingerprint
from sunhacks.claim_runtime import AnswerClaimTask, ClaimRuntime
from test_claim_check import checker, frames
from test_claim_pipeline_integration import run_answer


def completed(monkeypatch):
    p,_,claims=run_answer(monkeypatch,visual=True)
    p._infer_answer()
    p.fast_runtime.snapshot=lambda: {}
    images=claims.begins[0]['images'];text=p.latest['transcript']['text']
    p.latest['claim_check'].update(status='no_contradiction',kind='multimodal',call_made=True,
        sent_keyframes=2,sent_transcript=True,transmission_status='response_received',
        input_sha256=input_fingerprint(text,images))
    return p,images,text


def test_fingerprint_matches_actual_transport_observations(monkeypatch):
    c,calls=checker(monkeypatch);text='Two plus two equals four. ';images=frames()
    result=c.check(text,images,visual_authorized=True,source_window_id=1)
    body=json.loads(calls[0].content)
    sent_text=json.loads(body['state'][0])['recognized_transcript']
    sent_images=[item['image_url']['url'].removeprefix('data:image/jpeg;base64,') for item in body['state'][1:]]
    assert result['input_sha256']==input_fingerprint(sent_text,sent_images)
    assert result['input_sha256']!=input_fingerprint(text.strip(),sent_images)
    assert len(calls)==1


def test_preview_exact_images_and_text_but_polled_state_has_no_images(monkeypatch):
    p,images,text=completed(monkeypatch)
    value=p.claim_input_preview()
    assert value['status']=='available' and value['images']==list(images) and value['transcript']==text
    assert value['selected_frame_indices']==[42,85]
    assert value['decision_effect']=='contradiction_only'
    assert all(image not in json.dumps(p.state()) for image in images)
    assert p.claim_input_preview()==value
    p.stop()
    assert p._claim_input_preview is None and p.claim_input_preview()['status']=='unavailable'


@pytest.mark.parametrize('change', ['text','photos','session','window','claim_source','consent','enabled','not_sent','failed','active','cancelled','hash'])
def test_preview_rejects_stale_unconsented_or_unconfirmed_inputs(monkeypatch,change):
    p,_,_=completed(monkeypatch)
    if change=='text':p.latest['transcript']['text']+=' changed'
    if change=='photos':p._claim_input_preview['images']=frames()
    if change=='session':p.session_id='other'
    if change=='window':p.latest['window']['id']=2
    if change=='claim_source':p.latest['claim_check']['source_window_id']=True
    if change=='consent':p.latest['visual_claim_check']=False
    if change=='enabled':p.latest['claim_check_enabled']=False
    if change=='not_sent':p.latest['claim_check']['sent_transcript']=False
    if change=='failed':p.latest['claim_check']['transmission_status']='attempted_outcome_unknown'
    if change=='active':p.answer['status']='processing'
    if change=='cancelled':p.stop_event.set()
    if change=='hash':p.latest['claim_check']['input_sha256']='0'*64
    assert p.claim_input_preview()['status']=='unavailable'


def test_no_preview_without_photo_authorization(monkeypatch):
    p,_,_=run_answer(monkeypatch,visual=False);p._infer_answer()
    assert p._claim_input_preview is None and p.claim_input_preview()['status']=='unavailable'


@pytest.mark.parametrize('digest',['abc','G'*64,'a'*65,False,None])
def test_runtime_rejects_invalid_fingerprint(digest):
    from test_claim_runtime import result
    runtime=ClaimRuntime({'deadline_seconds':.1})
    task=AnswerClaimTask(runtime,7,None)
    try:
        with pytest.raises(ValueError):task._validated(result(input_sha256=digest))
        assert task._validated(result(input_sha256='a'*64))['input_sha256']=='a'*64
    finally:runtime.close()


def test_preview_route_is_explicit_same_origin_and_not_cacheable():
    import httpx
    from sunhacks.pipeline import Pipeline
    from sunhacks.server import make_server
    p=Pipeline();p.claim_input_preview=lambda: {'status':'available','images':['fixture-a','fixture-b']}
    server=make_server(p,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',trust_env=False) as client:
            response=client.get('/api/claim-inputs')
            assert response.json()['images']==['fixture-a','fixture-b']
            assert response.headers['cache-control']=='no-store'
            assert 'fixture-a' not in client.get('/api/state').text
            assert client.get('/api/claim-inputs',headers={'Origin':'https://other.example'}).status_code==403
            assert client.get('/api/claim-inputs',headers={'Host':'other.example'}).status_code==403
    finally:
        server.shutdown();server.server_close();thread.join(1);p.close()


def test_participant_reset_forgets_preview(monkeypatch):
    p,_,_=completed(monkeypatch)
    assert p._claim_input_preview is not None
    p.clear_session_data()
    assert p._claim_input_preview is None and p.claim_input_preview()['status']=='unavailable'
