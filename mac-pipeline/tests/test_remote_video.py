import base64
import copy
import io
import json

import numpy as np
from PIL import Image
import pytest

from sunhacks import remote_client as remote
MODEL = 'facebookresearch/vjepa2:vjepa2_1_vit_large_384'
REVISION = 'reviewed-source-and-checkpoint-revision'
PPHASH = 'a'*64


def preprocessing(model=MODEL):
    return {'id':remote.VISION_MODEL_SPECS[model]['preprocessing_id'],
            'input_stage':'transport_rgb_uint8','applied_once':True,
            'spatial_size':remote.VISION_MODEL_SPECS[model]['spatial_size'],
            'config_sha256':PPHASH}


def health(model=MODEL):
    return {'status':'ok','message':'pong','protocols':[remote.VISION_PROTOCOL],
            'encoder':{'loaded':True,'real_encoder':True,'synthetic':False,
                       'model_id':model,'revision':REVISION,'device':'cuda:0',
                       'input_frames':64,'output_sequence_shape':[32,1024],
                       'pooling':remote.VISION_POOLING,'preprocessing':preprocessing(model)}}


def response(request):
    return {'status':'ok','protocol':remote.VISION_PROTOCOL,'action':'encode_video',
            'request_id':request['request_id'],'window_id':request['window_id'],
            'model_id':request['model_id'],'revision':request['revision'],
            'device':'cuda:0','real_encoder':True,'synthetic':False,
            'sequence_shape':[32,1024], 'pooling':remote.VISION_POOLING,
            'timestamp_basis':request['timestamp_basis'], 'sample_fps':request['sample_fps'],
            'frame_timestamps':[frame['timestamp'] for frame in request['frames']],
            'preprocessing':preprocessing(request['model_id']),
            'sequence':np.linspace(0,1,32*1024).reshape(32,1024).tolist(),
            'timings':{'decode_ms':2.,'preprocess_ms':1.,'encoder_ms':100.,'pooling_ms':1.,'total_ms':105.}}


class VideoSocket:
    def __init__(self, mutate=None, raw=None, split=16384, error=None, on_recv=None):
        self.mutate, self.raw, self.split, self.error = mutate, raw, split, error
        self.on_recv = on_recv
        self.request = None
        self.timeouts = []
        self.closed = False
        self.sent = []
    def __enter__(self): return self
    def __exit__(self,*args): self.closed = True
    def settimeout(self, value): self.timeouts.append(value)
    def connect(self, destination):
        self.destination = destination
        if self.error: raise self.error
    def sendall(self, packed):
        self.sent.append(packed)
        self.request = json.loads(packed)
        answer = response(self.request)
        if self.mutate: self.mutate(answer)
        self.output = self.raw if self.raw is not None else json.dumps(answer,separators=(',',':')).encode()+b'\n'
    def recv(self, maximum):
        if self.on_recv: self.on_recv()
        size = min(maximum,self.split)
        chunk,self.output = self.output[:size],self.output[size:]
        return chunk


def client(monkeypatch, *, model=MODEL, reviewed=True, probe_health=None, **kwargs):
    obj = remote.RemoteVisionClient('192.0.2.1', expected_model_id=model,
                                   expected_revision=REVISION, expected_preprocessing_sha256=PPHASH,
                                   protocol_reviewed=reviewed, **kwargs)
    declared = health(model) if probe_health is None else probe_health
    obj.health_client.probe = lambda: {'status':'encoder_protocol_unverified','health':declared,'encoder_verified':False}
    obj.probe()
    return obj


def clip():
    arr = np.zeros((64,4,8,3),dtype=np.uint8)
    arr[...,0] = 255
    return arr


def encode(monkeypatch, obj=None, **socket_kwargs):
    stream = VideoSocket(**socket_kwargs)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    obj = obj if obj is not None else client(monkeypatch)
    return obj.encode(clip(), window_id=7, sample_fps=24), stream


def test_valid_real_response_preserves_private_features_and_color(monkeypatch):
    branch, stream = encode(monkeypatch)
    assert branch['status'] == 'base'
    assert branch['model_id'] == MODEL
    assert branch['revision'] == REVISION
    assert branch['_sequence_embedding'].shape == (32,1024)
    assert branch['_sequence_embedding'].dtype == np.float32
    np.testing.assert_allclose(branch['_embedding'],branch['_sequence_embedding'].mean(0),atol=2e-7)
    assert branch['remote_timings']['encoder_ms'] == 100
    assert branch['window_id'] == 7
    assert branch['input_contract'] == remote.VISION_PROTOCOL
    assert stream.closed
    assert len(stream.sent) == 1 and stream.sent[0].endswith(b'\n')
    assert stream.request['action'] == 'encode_video'
    assert 'f_v' not in stream.request and 'auth_token' not in stream.request
    assert len(stream.request['frames']) == 64
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(stream.request['frames'][0]['jpeg_base64']))))
    assert decoded.shape == (4,8,3)
    assert decoded[0,0,0] >= 250 and decoded[0,0,2] < 5
    assert stream.request['timestamp_basis'] == 'relative_clip_seconds'
    assert stream.request['frames'][-1]['timestamp'] == 63/24


def test_aspect_ratio_capture_cap_without_model_crop(monkeypatch):
    stream = VideoSocket()
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    arr = np.zeros((64,100,1000,3),dtype=np.uint8)
    client(monkeypatch).encode(arr,window_id=1,sample_fps=16)
    assert stream.request['transport']['size'] == [640,64]
    decoded = Image.open(io.BytesIO(base64.b64decode(stream.request['frames'][0]['jpeg_base64'])))
    assert decoded.size == (640,64)


def test_vjepa2_is_preserved_and_not_relabelled_21(monkeypatch):
    model = 'facebook/vjepa2-vitl-fpc64-256'
    branch, _ = encode(monkeypatch,client(monkeypatch,model=model))
    assert branch['model_id'] == model
    assert branch['preprocessing']['spatial_size'] == [256,256]


def test_supplied_capture_timestamps_echo(monkeypatch):
    obj = client(monkeypatch)
    stream = VideoSocket(split=37)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    stamps = [100 + index/24 for index in range(64)]
    branch = obj.encode(clip(),window_id=3,timestamps=stamps,sample_fps=24)
    assert branch['window_id'] == 3
    assert stream.request['timestamp_basis'] == 'client_monotonic_seconds'
    assert [frame['timestamp'] for frame in stream.request['frames']] == stamps


def test_health_is_declaration_not_verification(monkeypatch):
    obj = client(monkeypatch)
    result = obj.probe()
    assert result['status'] == 'encoder_protocol_unverified'
    assert result['encoder_contract_declared'] is True
    assert result['encoder_verified'] is False


@pytest.mark.parametrize('mutation',[
    lambda x:x.pop('protocols'),
    lambda x:x['encoder'].pop('loaded'),
    lambda x:x['encoder'].__setitem__('loaded',1),
    lambda x:x['encoder'].__setitem__('real_encoder',False),
    lambda x:x['encoder'].__setitem__('synthetic',True),
    lambda x:x['encoder'].__setitem__('model_id','another-model'),
    lambda x:x['encoder'].__setitem__('output_sequence_shape',[1024]),
    lambda x:x['encoder'].__setitem__('input_frames',16),
    lambda x:x['encoder'].__setitem__('revision','changed'),
    lambda x:x['encoder'].__setitem__('device','cpu'),
    lambda x:x['encoder']['preprocessing'].__setitem__('applied_once',1),
])
def test_incomplete_or_mismatched_capability_blocks_media(monkeypatch,mutation):
    value = health()
    mutation(value)
    obj = client(monkeypatch,probe_health=value)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No video socket permitted'))
    with pytest.raises(remote.RemoteVisionError): obj.encode(clip(),window_id=0,sample_fps=24)


def test_review_gate_and_missing_health_open_no_socket(monkeypatch):
    obj = client(monkeypatch,reviewed=False)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No socket permitted'))
    with pytest.raises(remote.RemoteVisionError,match='not been reviewed'): obj.encode(clip(),window_id=0,sample_fps=24)
    obj.protocol_reviewed = True
    obj._declared_encoder = None
    with pytest.raises(remote.RemoteVisionError,match='probe'): obj.encode(clip(),window_id=0,sample_fps=24)


def test_missing_review_pins_open_no_socket(monkeypatch):
    obj = remote.RemoteVisionClient('192.0.2.1',expected_model_id=MODEL,protocol_reviewed=True)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No socket permitted'))
    with pytest.raises(remote.RemoteVisionError,match='Pin'): obj.encode(clip(),window_id=0,sample_fps=24)


@pytest.mark.parametrize('field,value',[
    ('protocol','legacy'),('action','jepa'),('request_id','wrong'),('window_id',9),
    ('window_id',True),('model_id','facebook/vjepa2-vitl-fpc64-256'),('revision','unknown'),
    ('device','mps'),('real_encoder',1),('real_encoder',False),('synthetic',0),('synthetic',True),
    ('pooling','mean_all_tokens'),('sequence_shape',[1024]),('sequence',[[0]*1024]),
    ('sequence',[[True]*1024]*32),('sample_fps',8),('frame_timestamps',[0]*64),
    ('timestamp_basis','relative_other'),('timings',{}),('preprocessing',{}),
])
def test_strict_response_contract(monkeypatch,field,value):
    stream = VideoSocket(mutate=lambda answer:answer.__setitem__(field,value))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)
    assert stream.closed


@pytest.mark.parametrize('missing', ['revision','real_encoder','synthetic','sequence','preprocessing','timings'])
def test_omitted_provenance_rejected(monkeypatch,missing):
    stream = VideoSocket(mutate=lambda answer:answer.pop(missing))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)


@pytest.mark.parametrize('raw',[b'{}',b'not-json\n',b'{"status":"ok","status":"ok"}\n',b'{"x":NaN}\n',b'{}\n{}\n',b'\xff\n'])
def test_strict_ndjson_framing(monkeypatch,raw):
    stream = VideoSocket(raw=raw)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)
    assert stream.closed


def test_response_byte_cap(monkeypatch):
    stream = VideoSocket(raw=b'x'*(remote.MAX_VIDEO_RESPONSE_BYTES+1))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError,match='1 MiB'): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)
    assert stream.closed


def test_request_byte_cap_before_connect(monkeypatch):
    monkeypatch.setattr(remote,'MAX_VIDEO_REQUEST_BYTES',100)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No oversized request socket'))
    with pytest.raises(remote.RemoteVisionError,match='16 MiB'): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)


def test_float32_overflow_and_nonfinite_rejected(monkeypatch):
    for invalid in (float('nan'),float('inf'),1e100):
        stream = VideoSocket(mutate=lambda answer:answer['sequence'][0].__setitem__(0,invalid))
        monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
        with pytest.raises(remote.RemoteVisionError): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)
        assert stream.closed


@pytest.mark.parametrize('timings',[
    {'decode_ms':0,'preprocess_ms':0,'encoder_ms':0,'pooling_ms':0,'total_ms':0},
    {'decode_ms':0,'preprocess_ms':0,'encoder_ms':100,'pooling_ms':0,'total_ms':99},
    {'decode_ms':-1,'preprocess_ms':0,'encoder_ms':100,'pooling_ms':0,'total_ms':100},
])
def test_impossible_or_incomplete_timings_rejected(monkeypatch,timings):
    stream = VideoSocket(mutate=lambda answer:answer.__setitem__('timings',timings))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)


@pytest.mark.parametrize('kwargs',[
    {'clip':np.zeros((16,4,8,3),dtype=np.uint8)}, {'clip':np.zeros((64,4,8,3),dtype=np.float32)},
    {'window_id':True},{'window_id':-1},{'window_id':2**53}, {'sample_fps':0},{'sample_fps':True},
    {'timestamps':[0]*64},{'timestamps':[0]*63},{'timestamps':[float('nan')]*64},
])
def test_input_errors_do_not_open_video_socket(monkeypatch,kwargs):
    obj = client(monkeypatch)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('No invalid input socket'))
    values = {'clip':clip(),'window_id':7,'sample_fps':24}
    values.update(kwargs)
    with pytest.raises(remote.RemoteVisionError): obj.encode(**values)


def test_separate_auth_token_only(monkeypatch):
    monkeypatch.setenv('SUNHACKS_SOCKET_TOKEN','socket-only-secret')
    monkeypatch.setenv('DEEPSEEK_API_KEY','deepseek-never-sent')
    branch, stream = encode(monkeypatch,client(monkeypatch,shared_token_env='SUNHACKS_SOCKET_TOKEN'))
    assert stream.request['auth_token'] == 'socket-only-secret'
    assert b'deepseek-never-sent' not in stream.sent[0]
    assert 'socket-only-secret' not in repr(branch)
    with pytest.raises(ValueError): remote.RemoteVisionClient('192.0.2.1',expected_model_id=MODEL,shared_token_env='DEEPSEEK_API_KEY')


def test_total_deadline_includes_jpeg_packing(monkeypatch):
    now = [0]
    def clock():
        now[0] += 0.25
        return now[0]
    obj = client(monkeypatch)
    monkeypatch.setattr(remote.time,'monotonic',clock)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Packing exceeded total deadline before connect'))
    with pytest.raises(remote.RemoteVisionError,match='deadline'): obj.encode(clip(),window_id=7,sample_fps=24)


def test_deadline_decreases_across_response_chunks(monkeypatch):
    now = [0.]
    monkeypatch.setattr(remote.time,'monotonic',lambda:now[0])
    stream = VideoSocket(split=100,on_recv=lambda:now.__setitem__(0,now[0]+3))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError,match='deadline'): client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)
    assert stream.timeouts[-3:] == [8.,5.,2.]
    assert stream.closed


def test_offline_adapter_remains_unavailable_no_features(monkeypatch):
    obj = client(monkeypatch,reviewed=False)
    runner = remote.make_vision_runner(obj,sample_fps=24)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Offline adapter must not connect'))
    branch = runner(clip(),2)
    assert branch['status'] == 'unavailable'
    assert branch['model_id'] == MODEL
    assert branch['remote_error_code'] == 'review_required'
    assert '_embedding' not in branch and '_sequence_embedding' not in branch


def test_adapter_passes_window_timestamps(monkeypatch):
    obj = client(monkeypatch)
    stream = VideoSocket()
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    stamps = [index/24+200 for index in range(64)]
    runner = remote.make_vision_runner(obj,sample_fps=24,timestamps_provider=lambda window_id:stamps if window_id==2 else None)
    result = runner(clip(),2)
    assert result['status'] == 'base'
    assert stream.request['frames'][0]['timestamp'] == 200


@pytest.mark.parametrize('field,value',[
    ('host','192.0.2.2'),('port',5003),('expected_revision','new-revision'),
    ('expected_preprocessing_sha256','b'*64),('expected_model_id','facebook/vjepa2-vitl-fpc64-256'),
])
def test_changed_endpoint_or_model_pins_require_fresh_probe(monkeypatch,field,value):
    obj = client(monkeypatch)
    setattr(obj,field,value)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Changed identity must not send frames'))
    with pytest.raises(remote.RemoteVisionError,match='changed'): obj.encode(clip(),window_id=1,sample_fps=24)
    assert obj._declared_encoder is None


def test_mutated_cached_capability_is_invalidated(monkeypatch):
    obj = client(monkeypatch)
    obj._declared_encoder['model_id'] = 'another-model'
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Changed declaration must not send frames'))
    with pytest.raises(remote.RemoteVisionError,match='changed'): obj.encode(clip(),window_id=1,sample_fps=24)


def test_public_preprocessing_allowlist_hides_untrusted_extras(monkeypatch):
    def mutate(answer): answer['preprocessing'].update(auth_token='never-expose',frames=['private'])
    branch,_ = encode(monkeypatch,mutate=mutate)
    assert set(branch['preprocessing']) == {'id','input_stage','applied_once','spatial_size','config_sha256'}
    assert 'never-expose' not in repr(branch)


@pytest.mark.parametrize('mutation',[
    lambda answer:answer['frame_timestamps'].__setitem__(0,False),
    lambda answer:answer.__setitem__('sample_fps',True),
    lambda answer:answer.__setitem__('sequence_shape',[32.,1024.]),
])
def test_boolean_and_float_shape_echoes_rejected(monkeypatch,mutation):
    obj = client(monkeypatch)
    stream = VideoSocket(mutate=mutation)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError): obj.encode(clip(),window_id=7,sample_fps=1)


@pytest.mark.parametrize('stamps',[np.arange(64,dtype=np.complex128),np.array(0.)])
def test_bad_numpy_timestamp_kinds_are_controlled(monkeypatch,stamps):
    obj = client(monkeypatch)
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Bad timestamps must not send frames'))
    with pytest.raises(remote.RemoteVisionError): obj.encode(clip(),window_id=1,sample_fps=24,timestamps=stamps)


def test_huge_json_feature_integer_is_controlled_protocol_error(monkeypatch):
    stream = VideoSocket(mutate=lambda answer:answer['sequence'][0].__setitem__(0,10**400))
    monkeypatch.setattr(remote.socket,'socket',lambda *args:stream)
    with pytest.raises(remote.RemoteVisionError) as caught: client(monkeypatch).encode(clip(),window_id=7,sample_fps=24)
    assert caught.value.code == 'protocol_error'
    assert stream.closed


def test_required_auth_without_config_blocks_video(monkeypatch):
    declared = health()
    declared['auth_required'] = True
    obj = client(monkeypatch,probe_health=declared)
    assert obj.probe()['status'] == 'auth_configuration'
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Missing auth must not send frames'))
    with pytest.raises(remote.RemoteVisionError): obj.encode(clip(),window_id=1,sample_fps=24)


def test_cannot_reprobe_old_host_and_send_to_mutated_video_host(monkeypatch):
    obj = client(monkeypatch)
    obj.host = '192.0.2.2'
    obj.health_client.probe = lambda:pytest.fail('Mismatched endpoint must not even probe')
    assert obj.probe()['status'] == 'configuration_error'
    monkeypatch.setattr(remote.socket,'socket',lambda *args:pytest.fail('Unprobed endpoint must not receive video'))
    with pytest.raises(remote.RemoteVisionError): obj.encode(clip(),window_id=1,sample_fps=24)
