import base64
import io
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from sunhacks import remote_jepa_classifier as remote
from sunhacks.pipeline import Pipeline
from test_remote_jepa_classifier import FakeSocket


def ready_client(monkeypatch,edge=960):
    c=remote.RemoteJepa21Classifier('192.0.2.1',reviewed_source_revision=remote.SOURCE_REVISION,
            participant_media_authorized=True,max_frame_edge=edge)
    sock=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *args:sock)
    c.probe();return c


def test_960_transport_preserves_960_camera_geometry_and_reports_jpeg_contract(monkeypatch):
    c=ready_client(monkeypatch)
    clip=np.zeros((16,540,960,3),np.uint8);clip[...,0]=150
    sock=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *args:sock)
    r=c.predict(clip,window_id=1,timestamps=np.arange(16)/30)
    first=Image.open(io.BytesIO(base64.b64decode(sock.request['frames'][0])))
    assert first.size==(960,540)
    assert r['source_frame_shape']==r['transport_frame_shape']==[540,960,3]
    assert r['transport_resized'] is False
    assert r['transport_jpeg_quality']==90 and r['transport_jpeg_subsampling']=='4:4:4'
    assert r['input_rgb_bytes']==clip.nbytes
    assert set(sock.request)=={'action','frames','face_crop'}


def test_legacy640_transport_is_explicitly_reported(monkeypatch):
    c=ready_client(monkeypatch,edge=640)
    clip=np.zeros((16,540,960,3),np.uint8)
    sock=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *args:sock)
    r=c.predict(clip,window_id=1,timestamps=np.arange(16)/30)
    first=Image.open(io.BytesIO(base64.b64decode(sock.request['frames'][0])))
    assert first.size==(640,360) and r['transport_resized'] is True
    assert r['source_frame_shape']==[540,960,3] and r['transport_frame_shape']==[360,640,3]


def test_transport_byte_limit_rejects_instead_of_weakening_jpeg(monkeypatch):
    c=ready_client(monkeypatch)
    monkeypatch.setattr(remote,'MAX_REQUEST_BYTES',100)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Oversize request must not connect'))
    with pytest.raises(remote.RemoteClassifierError) as exc:
        c.predict(np.zeros((16,540,960,3),np.uint8),window_id=1,timestamps=np.arange(16)/30)
    assert exc.value.code=='request_too_large'


@pytest.mark.parametrize('edge',[True,63,961,'960'])
def test_unbounded_or_noninteger_transport_resolution_rejected(edge):
    with pytest.raises(ValueError):remote.RemoteJepa21Classifier('192.0.2.1',max_frame_edge=edge)


@pytest.mark.parametrize('configured,expected',[(None,640),(960,960)])
def test_answer_capture_receives_explicit_profile_edge(monkeypatch,configured,expected):
    import sunhacks.capture as cap
    calls=[]
    class FakeCapture:
        def __init__(self,**kwargs):calls.append(kwargs)
        def start(self):pass
        def stop(self):pass
        def drain(self):return [],[]
        def snapshot(self):return {'camera':{'status':'stopped'},'audio':{'status':'stopped'}}
    monkeypatch.setattr(cap,'LiveCapture',FakeCapture)
    config={'runtime_mode':'split_local_video'}
    if configured is not None:config['capture_max_frame_edge']=configured
    p=Pipeline(config);p.fast_runtime=SimpleNamespace(snapshot=lambda:{'active_jobs':[]})
    monkeypatch.setattr(p,'_split_runtime_state',lambda:{'live_ready':True})
    monkeypatch.setattr(p,'_ingest_answer',lambda:None);monkeypatch.setattr(p,'_infer_answer',lambda:None)
    p.start('live',auto_stop_seconds=7)
    assert calls[0]['max_frame_edge']==expected
    assert p.answer_frames.maxlen==1000
    p.stop()


def test_jpeg_huffman_optimization_preserves_decoded_pixels(monkeypatch):
    c=ready_client(monkeypatch)
    rng=np.random.default_rng(4)
    clip=rng.integers(0,256,size=(16,48,64,3),dtype=np.uint8)
    sock=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *args:sock)
    result=c.predict(clip,window_id=1,timestamps=np.arange(16)/30)
    actual=np.array(Image.open(io.BytesIO(base64.b64decode(sock.request['frames'][0]))))
    reference=io.BytesIO();Image.fromarray(clip[0]).save(reference,format='JPEG',quality=90,subsampling=0,optimize=False)
    expected=np.array(Image.open(io.BytesIO(reference.getvalue())))
    assert np.array_equal(actual,expected)
    assert result['transport_jpeg_huffman_optimized'] is True
