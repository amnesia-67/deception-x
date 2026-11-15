import hashlib
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pytest

from sunhacks.dataset_store import DatasetStore,write_json
from sunhacks.dataset_recorder import DatasetRecorder
from sunhacks.pipeline import Pipeline
from sunhacks.temporal import AudioChunk,VideoFrame


def media(start=10.):
    image=np.zeros((64,96,3),np.uint8);image[:,:,1]=150
    return tuple(VideoFrame(start+i/30,image,i) for i in range(210)),(.1*np.sin(np.arange(112000)*.1)).astype(np.float32)


def pending(store):
    store.initialize();path=store.root/'.pending'/'fixture';path.mkdir()
    (path/'video.mp4').write_bytes(b'fixture-video');(path/'audio.wav').write_bytes(b'fixture-audio')
    return {'take_id':'fixture','directory':path,'metadata':{'schema':'sunhacks-labeled-av-v1'}}


def test_split_stays_with_participant_across_labels_and_restart(tmp_path):
    store=DatasetStore(tmp_path/'dataset');a=store.save(pending(store),'person1','truth','My statement')
    store=DatasetStore(store.root);b=store.save(pending(store),'person1','lie')
    assert a['split']==b['split']
    assert (Path(a['path'])/'metadata.json').exists()
    meta=json.loads((Path(a['path'])/'metadata.json').read_text())
    assert meta['label']=='truth' and meta['statement']=='My statement' and meta['label_source']=='participant_self_report'
    assert store.counts()[a['split']]=={'truth':1,'lie':1}
    assert not any((store.root/'.pending').iterdir())


def test_random_group_assignment_targets_75_15_10_without_label_or_order_dependence(tmp_path):
    store=DatasetStore(tmp_path);manifest=store.initialize();manifest['seed']='fixed-test-seed'
    people=[f'person_{i}' for i in range(3000)];splits=[store.split_for(p,manifest) for p in people]
    for split,ratio in [('train',.75),('validation',.15),('test',.1)]:assert abs(splits.count(split)/len(splits)-ratio)<.03
    assert [store.split_for(p,manifest) for p in people[::-1]]==splits[::-1]


@pytest.mark.parametrize('person',['../outside','/tmp/name','',None,'a'*65,'has spaces'])
def test_invalid_participant_cannot_be_saved(tmp_path,person):
    store=DatasetStore(tmp_path);take=pending(store)
    with pytest.raises(ValueError):store.save(take,person,'truth')
    assert take['directory'].exists() and sum(sum(v.values()) for v in store.counts().values())==0


@pytest.mark.parametrize('label',['Truth','unknown',None])
def test_only_explicit_labels_are_saved(tmp_path,label):
    store=DatasetStore(tmp_path);take=pending(store)
    with pytest.raises(ValueError):store.save(take,'person1',label)
    assert take['directory'].exists()


@pytest.mark.skipif(not shutil.which('ffmpeg'),reason='ffmpeg unavailable')
def test_real_seven_second_mp4_wav_and_timestamp_sidecar(tmp_path):
    import subprocess
    from scipy.io import wavfile
    store=DatasetStore(tmp_path);frames,audio=media()
    take=store.encode(frames,audio,10.,{},cancelled=threading.Event())
    assert take['metadata']['source_frame_count']==210 and take['metadata']['audio']['samples']==112000
    assert len(take['metadata']['frame_timestamps_seconds'])==210
    rate,read=wavfile.read(take['directory']/'audio.wav');assert rate==16000 and np.array_equal(read,audio)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(take['directory']/'video.mp4')]))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert video['codec_name']=='h264' and int(video['nb_frames'])==210
    assert abs(float(probe['format']['duration'])-7.)<.05
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    assert not list(take['directory'].glob('frames-*'))
    for name,digest in take['metadata']['files_sha256'].items():assert hashlib.sha256((take['directory']/name).read_bytes()).hexdigest()==digest


class Capture:
    def __init__(self,**kwargs):self.kwargs=kwargs;self.status='stopped';self.sent=False
    def start(self):self.status='running'
    def stop(self):self.status='stopped'
    def snapshot(self):return {k:{'status':self.status} for k in ('camera','audio')}
    def drain(self):
        if self.sent:return [],[]
        self.sent=True;frames,audio=media(time.monotonic()-7.1)
        return list(frames),[AudioChunk(frames[0].timestamp,16000,audio)]


class FastStore(DatasetStore):
    def encode(self,frames,audio,start,devices,*,cancelled):
        assert len(frames)==210 and len(audio)==112000
        return pending(self)


def recorder(tmp_path,cls=Capture):
    pipeline=Pipeline({'audio_device':'Blue Snowball','dataset_root':str(tmp_path)})
    store=FastStore(tmp_path);r=DatasetRecorder(pipeline,capture_factory=cls,store=store);pipeline.dataset_recorder=r
    return pipeline,r


def wait(r,status):
    end=time.monotonic()+2
    while r.snapshot()['status']!=status and time.monotonic()<end:time.sleep(.005)
    assert r.snapshot()['status']==status,r.snapshot()


def test_recording_does_not_require_models_and_needs_explicit_label(tmp_path):
    p,r=recorder(tmp_path);r.start('p_one');wait(r,'awaiting_label')
    assert r.is_active() and p.fast_runtime is None and p.video_client is None
    assert r._capture.kwargs['audio_device']=='Blue Snowball'
    assert r.preview('fixture')==b'fixture-video'
    assert p.state()['dataset']['status']=='awaiting_label'
    assert sum(sum(v.values()) for v in r.store.counts().values())==0
    with pytest.raises(ValueError):r.save('stale','truth')
    saved=r.save('fixture','truth');assert saved['saved']['label']=='truth'
    with pytest.raises(RuntimeError):r.save('fixture','lie')
    assert not r.is_active();p.close()


def test_discard_deletes_unlabeled_take_and_invalidates_preview(tmp_path):
    p,r=recorder(tmp_path);r.start('p_one');wait(r,'awaiting_label');r.cancel()
    if r._thread:r._thread.join(1)
    assert not r.is_active() and not list((tmp_path/'.pending').iterdir())
    with pytest.raises(ValueError):r.preview('fixture')
    p.close()


def test_cancel_stops_capture_and_never_encodes_incomplete_take(tmp_path):
    class Empty(Capture):
        def drain(self):return [],[]
    p,r=recorder(tmp_path,Empty);r.start('p_one');r.cancel();r._thread.join(1)
    assert not r.is_active() and r.snapshot()['capture']['camera']['status']=='stopped'
    assert not list((tmp_path/'.pending').iterdir());p.close()


def test_collector_refuses_other_capture_and_pending_take(tmp_path):
    p,r=recorder(tmp_path);p.answer={'status':'capturing'}
    with pytest.raises(RuntimeError,match='Stop'):r.start('p_one')
    p.answer={};r.start('p_one');wait(r,'awaiting_label')
    with pytest.raises(RuntimeError,match='Label or discard'):r.start('p_two')
    p.close()


def test_dataset_routes_share_capture_ownership_and_do_not_publish_media_in_state(tmp_path):
    import httpx
    from sunhacks.server import make_server
    p,r=recorder(tmp_path)
    server=make_server(p,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}',trust_env=False,timeout=2) as client:
            assert client.post('/api/dataset/start',json={'participant_id':'p_one'}).status_code==200
            wait(r,'awaiting_label')
            for path,body in [('/api/session',{'mode':'demo'}),('/api/prepare',{}),('/api/showcase/answer',{})]:
                assert client.post(path,json=body).status_code==400
            preview=client.get('/api/dataset/preview?take_id=fixture')
            assert preview.content==b'fixture-video' and preview.headers['cache-control']=='no-store'
            assert client.get('/api/dataset/preview?take_id=wrong').status_code==404
            assert client.get('/api/dataset/preview?take_id=fixture',headers={'Origin':'https://other.example'}).status_code==403
            state=client.get('/api/state');assert 'fixture-video' not in state.text and 'fixture-audio' not in state.text
            assert client.post('/api/dataset/save',json={'take_id':'stale','label':'lie'}).status_code==400
            response=client.post('/api/dataset/save',json={'take_id':'fixture','label':'lie'})
            assert response.status_code==200 and response.json()['saved']['label']=='lie'
            assert client.get('/api/dataset/preview?take_id=fixture').status_code==404
            assert 'Record labeled data' in client.get('/dashboard').text
            r.start('p_one');wait(r,'awaiting_label')
            assert client.post('/api/stop',json={}).status_code==200
            assert not r.is_active()
    finally:server.shutdown();server.server_close();thread.join(1);p.close()


@pytest.mark.parametrize('problem',['audio_gap','video_gap','dropped','device_error'])
def test_bad_capture_never_enters_labelled_dataset(tmp_path,problem):
    class Broken(Capture):
        def snapshot(self):
            value=super().snapshot()
            if problem=='dropped':value['audio']['overflow_callbacks']=1
            if problem=='device_error' and self.status=='running':value['camera'].update(status='missing',error='Camera unavailable')
            return value
        def drain(self):
            frames,audio=media(time.monotonic()-8)
            if problem=='audio_gap':audio=audio[:100000]
            if problem=='video_gap':frames=frames[:50]+frames[80:]
            return list(frames),[AudioChunk(frames[0].timestamp,16000,audio)]
    p,r=recorder(tmp_path,Broken);r.start('p_one');wait(r,'error')
    assert not r.is_active() and not list((tmp_path/'.pending').iterdir())
    assert sum(sum(v.values()) for v in r.store.counts().values())==0;p.close()


def test_real_wall_clock_timer_stops_a_generated_stream_after_seven_seconds(tmp_path):
    class Ticker(Capture):
        def start(self):
            super().start();self.begin=time.monotonic();self.next_frame=0;self.next_sample=0
        def drain(self):
            now=time.monotonic();count=int((now-self.begin)*30);end=int((now-self.begin)*16000)
            image=np.zeros((64,96,3),np.uint8)
            frames=[VideoFrame(self.begin+i/30,image,i) for i in range(self.next_frame,count)]
            chunks=[AudioChunk(self.begin+self.next_sample/16000,16000,np.zeros(end-self.next_sample,np.float32))] if end>self.next_sample else []
            self.next_frame=count;self.next_sample=end
            return frames,chunks
    p,r=recorder(tmp_path,Ticker);begin=time.monotonic();r.start('p_timer')
    end=begin+9
    while r.snapshot()['status'] not in ('awaiting_label','error') and time.monotonic()<end:time.sleep(.01)
    assert r.snapshot()['status']=='awaiting_label',r.snapshot()
    assert 7<=time.monotonic()-begin<8
    assert r.snapshot()['elapsed_seconds']==7. and r.snapshot()['capture']['camera']['status']=='stopped'
    p.close()
