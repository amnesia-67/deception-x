import importlib.util
import io
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from sunhacks import remote_jepa_classifier as remote


def health():
    return {'status':'ok','jepa_loaded':True,'jepa':{
        'model':remote.MODEL_LABEL,'model_id':remote.MODEL_ID,'folds':5,'device':'cuda',
        'manifest_sha256':remote.MANIFEST_SHA256,'contract':dict(remote.CONTRACT)}}


def answer(count=64):
    return {'status':'success','model':remote.MODEL_LABEL,'model_id':remote.MODEL_ID,
        'manifest_sha256':remote.MANIFEST_SHA256,'contract':dict(remote.CONTRACT),
        'classifier_only':True,'calibrated':False,'device':'cuda','raw_score':.52,
        'per_fold_probability':[.3,.4,.5,.6,.7],'temporal_attention_32':[.03125]*32,
        'prep_ms':2.2,'inference_ms':120.,'n_frames_received':count,
        'verdict':'must-not-expose','transcript':'must-not-expose'}

class FakeSocket:
    def __init__(self,raw=None,mutate=None,chunk_size=16384,on_recv=None,error=None):
        self.raw,self.mutate,self.chunk_size,self.on_recv,self.error=raw,mutate,chunk_size,on_recv,error
        self.requests=[]
        self.timeouts=[]
        self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def settimeout(self,value):self.timeouts.append(value)
    def connect(self,endpoint):
        self.endpoint=endpoint
        if self.error:raise self.error
    def sendall(self,data):
        self.requests.append(data)
        self.request=json.loads(data)
        response=health() if self.request['action']=='ping' else answer(len(self.request['frames']))
        if self.mutate:self.mutate(response)
        self.output=self.raw if self.raw is not None else json.dumps(response,separators=(',',':')).encode()+b'\n'
    def recv(self,maximum):
        if self.on_recv:self.on_recv()
        n=min(maximum,self.chunk_size)
        result,self.output=self.output[:n],self.output[n:]
        return result


def client(monkeypatch,authorized=True,reviewed=True):
    instance=remote.RemoteJepa21Classifier('192.0.2.1',reviewed_source_revision=remote.SOURCE_REVISION if reviewed else None,
                                         participant_media_authorized=authorized)
    probe_socket=FakeSocket()
    monkeypatch.setattr(remote.socket,'socket',lambda *args:probe_socket)
    assert instance.probe()['classifier_ready'] is True
    return instance


def clip():
    value=np.zeros((64,4,8,3),dtype=np.uint8)
    value[...,0]=255
    return value


def predict(monkeypatch,instance=None,**socket_kwargs):
    instance=instance if instance is not None else client(monkeypatch)
    stream=FakeSocket(**socket_kwargs)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    result=instance.predict(clip(),window_id=7,timestamps=np.arange(64)/24,face_crop=True)
    return result,stream


def test_only_video_request_preserves_classifier_identity_and_local_timestamps(monkeypatch):
    branch,stream=predict(monkeypatch)
    assert set(stream.request)=={'action','frames','face_crop'}
    assert stream.request['action']=='jepa_video'
    assert stream.request['face_crop'] is True
    assert len(stream.requests)==1 and stream.requests[0].endswith(b'\n')
    assert branch['status']=='trained_classifier'
    assert branch['model_id']==remote.MODEL_ID
    assert branch['classifier_only'] is True and branch['calibrated'] is False
    assert branch['source_window_id']==7
    assert branch['_source_frame_timestamps']==tuple(np.arange(64)/24)
    assert branch['server_echoes_source_identity'] is False
    assert branch['encoder_unique_source_frames']==64
    assert '_embedding' not in branch and '_sequence_embedding' not in branch
    assert 'must-not-expose' not in repr(branch)
    assert branch['server_timings']=={'prep_ms':2.2,'inference_ms':120.}
    assert stream.closed
    import base64
    image=np.asarray(Image.open(io.BytesIO(base64.b64decode(stream.request['frames'][0]))))
    assert image[0,0,0]>250 and image[0,0,2]<5


def test_nonperson_fixture_method_works_without_participant_permission(monkeypatch):
    instance=client(monkeypatch,authorized=False)
    stream=FakeSocket()
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    branch=instance.benchmark_synthetic(variant=1,window_id=99)
    assert branch['synthetic_fixture'] is True
    assert branch['source_frames_sent']==128 and branch['encoder_unique_source_frames']==64
    assert branch['source_window_id']==99
    assert stream.request['face_crop'] is False
    assert not hasattr(instance,'analyze')


def test_participant_gate_cannot_be_bypassed_by_fixture_flag(monkeypatch):
    instance=client(monkeypatch,authorized=False)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No participant socket allowed'))
    with pytest.raises(remote.RemoteClassifierError) as caught:
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    assert caught.value.code=='media_authorization_required'


def test_review_and_probe_gates_no_automatic_network(monkeypatch):
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No automatic connection'))
    instance=remote.RemoteJepa21Classifier('192.0.2.1',participant_media_authorized=True)
    with pytest.raises(remote.RemoteClassifierError,match='reviewed'):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    instance._reviewed_source=remote.SOURCE_REVISION
    with pytest.raises(remote.RemoteClassifierError,match='probe'):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))


@pytest.mark.parametrize('mutation',[
    lambda item:item['jepa'].__setitem__('model','vjepa2_vitl_lora_5fold'),
    lambda item:item['jepa'].__setitem__('model_id','facebook/vjepa2-vitl-fpc64-256'),
    lambda item:item['jepa'].__setitem__('manifest_sha256','0'*64),
    lambda item:item['jepa'].__setitem__('folds',4),
    lambda item:item['jepa']['contract'].__setitem__('classifier_only',1),
    lambda item:item.__setitem__('jepa_loaded',1),
])
def test_mismatched_health_cannot_enable_classifier(monkeypatch,mutation):
    instance=remote.RemoteJepa21Classifier('192.0.2.1')
    stream=FakeSocket(mutate=mutation)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    assert instance.probe()['classifier_ready'] is False


@pytest.mark.parametrize('field,value',[
    ('model','vjepa2_vitl_lora_5fold'),('model_id','facebook/vjepa2-vitl-fpc64-256'),
    ('manifest_sha256','0'*64),('classifier_only',1),('calibrated',0),('device','cpu'),
    ('raw_score',True),('raw_score',float('nan')),('raw_score',1.1),
    ('per_fold_probability',[.5]*4),('per_fold_probability',[.5,True,.5,.5,.5]),
    ('temporal_attention_32',[.1]*32),('n_frames_received',63),('n_frames_received',True),
    ('inference_ms',0),('prep_ms',float('inf')),
])
def test_response_validation_rejects_wrong_identity_shapes_numbers(monkeypatch,field,value):
    instance=client(monkeypatch)
    stream=FakeSocket(mutate=lambda item:item.__setitem__(field,value))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteClassifierError):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    assert stream.closed
    assert instance.snapshot()['classifier_ready'] is False


@pytest.mark.parametrize('payload',[b'{}',b'not-json\n',b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{}\n{}\n',b'\xff\n'])
def test_strict_bounded_ndjson(monkeypatch,payload):
    instance=client(monkeypatch)
    stream=FakeSocket(raw=payload)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteClassifierError):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    assert stream.closed


def test_response_byte_limit(monkeypatch):
    instance=client(monkeypatch)
    stream=FakeSocket(raw=b'x'*(remote.MAX_RESPONSE_BYTES+1))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteClassifierError,match='byte limit'):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))


def test_request_byte_limit_before_connection(monkeypatch):
    instance=client(monkeypatch)
    monkeypatch.setattr(remote,'MAX_REQUEST_BYTES',100)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Oversized payload must not connect'))
    with pytest.raises(remote.RemoteClassifierError,match='16MiB'):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))


@pytest.mark.parametrize('timestamps',[np.arange(63),np.zeros(64),np.ones(64,dtype=bool),np.arange(64,dtype=np.complex64),np.full(64,float('nan'))])
def test_invalid_timestamps_do_not_send(monkeypatch,timestamps):
    instance=client(monkeypatch)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Invalid timestamps must not send'))
    with pytest.raises(remote.RemoteClassifierError):instance.predict(clip(),window_id=1,timestamps=timestamps)
    assert instance.snapshot()['busy'] is False


def test_one_inflight_request_and_local_metadata_cannot_be_reused(monkeypatch):
    instance=client(monkeypatch)
    entered,release=threading.Event(),threading.Event()
    def pause():
        entered.set()
        assert release.wait(2)
    stream=FakeSocket(on_recv=pause)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    outputs=[]
    thread=threading.Thread(target=lambda:outputs.append(instance.predict(clip(),window_id=10,timestamps=np.arange(64)+100)))
    thread.start()
    assert entered.wait(2)
    assert instance.snapshot()['busy'] is True
    with pytest.raises(remote.RemoteClassifierError) as caught:
        instance.predict(clip(),window_id=11,timestamps=np.arange(64)+200)
    assert caught.value.code=='busy'
    assert instance.probe()['status']=='busy'
    release.set()
    thread.join(2)
    assert outputs[0]['source_window_id']==10
    assert outputs[0]['_source_frame_timestamps'][0]==100
    assert instance.snapshot()['busy'] is False


def test_total_deadline_not_reset_by_response_chunks(monkeypatch):
    instance=client(monkeypatch)
    now=[0.]
    monkeypatch.setattr(remote.time,'monotonic',lambda:now[0])
    stream=FakeSocket(chunk_size=10,on_recv=lambda:now.__setitem__(0,now[0]+6))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteClassifierError,match='deadline'):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    assert stream.timeouts[-3:]==[15.,9.,3.]
    assert stream.closed


def test_close_discards_result_and_disables_new_calls(monkeypatch):
    instance=client(monkeypatch)
    stream=FakeSocket(on_recv=instance.close)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteClassifierError,match='closed'):
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    assert instance.snapshot()['classifier_ready'] is False
    assert instance.probe()['status']=='closed'


def test_duplicate_selected_timestamps_remain_explicit(monkeypatch):
    instance=client(monkeypatch)
    stream=FakeSocket()
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    stamps=np.repeat(np.arange(32,dtype=float),2)
    branch=instance.predict(clip(),window_id=1,timestamps=stamps)
    assert branch['_source_frame_timestamps']==tuple(stamps)
    assert branch['source_unique_timestamps']==32
    assert branch['encoder_unique_source_frames']==32


def test_close_during_packing_prevents_video_socket(monkeypatch):
    instance=client(monkeypatch)
    original=Image.Image.save
    def save(image,*args,**kwargs):
        original(image,*args,**kwargs)
        instance.close()
    monkeypatch.setattr(Image.Image,'save',save)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Close during packing must not connect'))
    with pytest.raises(remote.RemoteClassifierError) as caught:
        instance.predict(clip(),window_id=1,timestamps=np.arange(64))
    assert caught.value.code=='closed'
    assert instance.snapshot()['status']=='closed'


def test_close_during_probe_cannot_restore_ready(monkeypatch):
    instance=remote.RemoteJepa21Classifier('192.0.2.1')
    stream=FakeSocket(on_recv=instance.close)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    assert instance.probe()['status']=='closed'
    snapshot=instance.snapshot()
    assert snapshot['status']=='closed'
    assert snapshot['classifier_ready'] is False
    assert snapshot['classifier_response_validated'] is False


def test_invalid_window_cannot_enter_public_snapshot(monkeypatch):
    instance=client(monkeypatch)
    secret=np.zeros((10,10,3))
    with pytest.raises(remote.RemoteClassifierError):instance.predict(clip(),window_id=secret,timestamps=np.arange(64))
    assert 'source_window_id' not in instance.snapshot()
