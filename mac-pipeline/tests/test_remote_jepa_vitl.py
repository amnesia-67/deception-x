"""Generated transport fixtures only; no GPU, sensors or external networking."""
import json
import numpy as np
import pytest
from sunhacks import remote_jepa_vitl as remote
from test_remote_jepa_classifier import FakeSocket as LegacySocket
from sunhacks.local_digest import select_timestamp_frames


class FakeSocket(LegacySocket):
    def sendall(self,data):
        self.requests.append(data);self.request=json.loads(data)
        response={'status':'ok','jepa_loaded':True,'jepa':{
            'model':remote.MODEL_LABEL,'model_id':remote.MODEL_ID,'folds':5,'device':'cuda',
            'manifest_sha256':remote.MANIFEST_SHA256,'contract':remote.CONTRACT}}
        if self.request['action']=='jepa_video':
            idx,_=select_timestamp_frames(self.request['timestamps'])
            response={'status':'success','model':remote.MODEL_LABEL,'model_id':remote.MODEL_ID,
                'manifest_sha256':remote.MANIFEST_SHA256,'contract':remote.CONTRACT,
                'classifier_only':True,'calibrated':False,'device':'cuda','raw_score':.52,
                'per_fold_probability':[.3,.4,.5,.6,.7],'temporal_attention_32':[1/32]*32,
                'prep_ms':2.2,'inference_ms':120.,'n_frames_received':len(self.request['frames']),
                'face':{'face_detect_rate':1.,'centre_crop_fallback':False,'unique_frames_used':len(set(idx))},
                'server_revision':{'code':{'commit':remote.SOURCE_REVISION,'dirty':False,
                    'source_tree_sha256':remote.SOURCE_TREE_SHA256},
                    'artifacts_sha256':{'models/vjepa2_1_l_lora/deploy_meta.json':remote.MANIFEST_SHA256}},
                **{k:self.request[k] for k in ('request_id','source_id','client_revision')}}
        if self.mutate:self.mutate(response)
        self.output=json.dumps(response,separators=(',',':')).encode()+b'\n'


def client(monkeypatch,authorized=True):
    c=remote.RemoteJepa21LClassifier('192.0.2.1',reviewed_source_revision=remote.SOURCE_REVISION,
        participant_media_authorized=authorized,max_frame_edge=960)
    monkeypatch.setattr(remote.socket,'socket',lambda *a:FakeSocket())
    assert c.probe()['classifier_ready']
    return c


def test_l_input_uses_raw_timestamps_once_and_echoed_source(monkeypatch):
    c=client(monkeypatch);stream=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    stamps=100+np.arange(211)/30
    branch=c.predict(np.zeros((211,4,8,3),np.uint8),window_id=17,timestamps=stamps)
    assert set(stream.request)=={'action','frames','timestamps','request_id','source_id','client_revision'}
    idx,_=select_timestamp_frames(stamps-stamps[0])
    kept=np.unique(np.r_[0,idx,210])
    assert np.array_equal(stream.request['timestamps'],(stamps-stamps[0])[kept])
    selected,_=select_timestamp_frames(stream.request['timestamps'])
    assert np.array_equal(kept[selected],idx)
    assert branch['model_id']==remote.MODEL_ID and remote.CONTRACT['hidden_width']==1024
    assert remote.CONTRACT['frames_per_view']==64 and remote.CONTRACT['frame_stride']==1
    assert branch['encoder_unique_source_frames']==128 and branch['source_frames_sent']==len(kept)
    assert branch['raw_source_frames']==211
    assert branch['server_echoes_source_identity'] and branch['source_window_id']==17
    assert branch['request_id']==stream.request['request_id']
    assert branch['remote_source_revision_verified'] is False


@pytest.mark.parametrize('field,value',[('source_id',True),('source_id',18),('source_id','17'),
    ('request_id','stale'),('client_revision','old'),('model_id','facebookresearch/vjepa2:vjepa2_1_vit_base_384'),
    ('manifest_sha256','0'*64),('contract',{}),('server_revision',{}),('raw_score',None)])
def test_wrong_l_identity_or_stale_response_is_rejected(monkeypatch,field,value):
    c=client(monkeypatch);stream=FakeSocket(mutate=lambda r:r.update({field:value}))
    monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    with pytest.raises(remote.RemoteClassifierError):
        c.predict(np.zeros((128,4,8,3),np.uint8),window_id=17,timestamps=np.arange(128)/24)
    assert not c.snapshot()['classifier_ready']


def test_sparse_generated_capture_reports_real_unique_frames(monkeypatch):
    c=client(monkeypatch);stream=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    out=c.predict(np.zeros((35,4,8,3),np.uint8),window_id=1,timestamps=np.arange(35)/5)
    assert out['encoder_unique_source_frames']==35


@pytest.mark.parametrize('stamps',[np.zeros(128),np.arange(128)[::-1],np.arange(128,dtype=float)*np.nan])
def test_raw_timestamps_must_be_strict_finite_capture(monkeypatch,stamps):
    c=client(monkeypatch);monkeypatch.setattr(remote.socket,'socket',lambda *a:pytest.fail('invalid input sent'))
    with pytest.raises(remote.RemoteClassifierError,match='timestamp'):
        c.predict(np.zeros((128,4,8,3),np.uint8),window_id=1,timestamps=stamps)


def test_constructor_and_blocked_participant_do_not_connect(monkeypatch):
    monkeypatch.setattr(remote.socket,'socket',lambda *a:pytest.fail('unauthorized socket opened'))
    c=remote.RemoteJepa21LClassifier('192.0.2.1')
    with pytest.raises(remote.RemoteClassifierError,match='disabled'):
        c.predict(None,window_id=1,timestamps=None)


def test_server_sampling_mismatch_rejected(monkeypatch):
    c=client(monkeypatch)
    stream=FakeSocket(mutate=lambda r:r['face'].update(unique_frames_used=64))
    monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    with pytest.raises(remote.RemoteClassifierError,match='sampling'):
        c.predict(np.zeros((128,4,8,3),np.uint8),window_id=1,timestamps=np.arange(128)/24)


@pytest.mark.parametrize('fingerprint',sorted(remote.REVIEWED_SOURCE_TREES))
def test_both_reconstructed_source_trees_are_accepted_even_with_windows_dirty_flag(monkeypatch,fingerprint):
    c=client(monkeypatch)
    def mutate(r):r['server_revision']['code'].update(source_tree_sha256=fingerprint,dirty=True)
    stream=FakeSocket(mutate=mutate);monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    out=c.predict(np.zeros((128,4,8,3),np.uint8),window_id=1,timestamps=np.arange(128)/24)
    assert out['server_declared_source_tree_sha256']==fingerprint
    assert out['remote_source_revision_verified'] is False


@pytest.mark.parametrize('dirty',[False,True])
def test_matching_commit_cannot_authorize_unknown_source_bytes(monkeypatch,dirty):
    c=client(monkeypatch)
    def mutate(r):r['server_revision']['code'].update(source_tree_sha256='0'*64,dirty=dirty)
    stream=FakeSocket(mutate=mutate);monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    with pytest.raises(remote.RemoteClassifierError,match='reviewed source bytes'):
        c.predict(np.zeros((128,4,8,3),np.uint8),window_id=1,timestamps=np.arange(128)/24)


def test_known_eval_draft_tree_does_not_bypass_manifest_check(monkeypatch):
    c=client(monkeypatch)
    def mutate(r):
        r['server_revision']['code']['source_tree_sha256']=remote.SOURCE_TREE_SHA256_WITH_EVAL_DRAFT
        r['server_revision']['artifacts_sha256']['models/vjepa2_1_l_lora/deploy_meta.json']='0'*64
    stream=FakeSocket(mutate=mutate);monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    with pytest.raises(remote.RemoteClassifierError,match='L manifest'):
        c.predict(np.zeros((128,4,8,3),np.uint8),window_id=1,timestamps=np.arange(128)/24)


def test_detailed_camera_frames_fit_full_wire_budget_without_temporal_loss(monkeypatch):
    import base64,io
    from PIL import Image
    c=client(monkeypatch);stream=FakeSocket();monkeypatch.setattr(remote.socket,'socket',lambda *a:stream)
    # Force the real-camera failure with a smaller budget, using generated
    # detailed RGB frames. Low-entropy black startup fixtures miss this case.
    monkeypatch.setattr(remote,'MAX_REQUEST_BYTES',500_000)
    clip=np.random.default_rng(42).integers(0,256,(32,192,256,3),dtype=np.uint8)
    stamps=200+np.arange(32)/30
    out=c.predict(clip,window_id=9,timestamps=stamps)
    assert len(stream.requests[-1])<=remote.MAX_REQUEST_BYTES
    assert out['request_bytes']<=remote.MAX_REQUEST_BYTES
    assert out['transport_budget_adapted']
    assert out['raw_source_frames']==out['source_frames_sent']==out['encoder_unique_source_frames']==32
    assert np.allclose(stream.request['timestamps'],stamps-stamps[0])
    sizes={Image.open(io.BytesIO(base64.b64decode(frame))).size for frame in stream.request['frames']}
    assert sizes=={(out['transport_frame_shape'][1],out['transport_frame_shape'][0])}
