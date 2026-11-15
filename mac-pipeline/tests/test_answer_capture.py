import time
import threading
from collections import deque
import numpy as np
import pytest
from sunhacks.answer_capture import select_answer
from sunhacks.temporal import VideoFrame,AudioChunk,AudioBuffer
from sunhacks.pipeline import Pipeline


def frames(duration=6.,fps=30):
    start=time.monotonic()-duration
    image=np.zeros((4,6,3),np.uint8)
    return [VideoFrame(start+i/fps,image,i) for i in range(round(duration*fps)+1)]


def test_answer_selection_retains_raw_sources_without_claiming_video_completed():
    source=frames();window=select_answer(source);meta=window.metadata()
    assert len(window.frames)==128 and len({f.sequence for f in window.frames})==128
    assert meta['native_video_unique_frames'] is None
    assert len(window._source_frames)==len(source)
    assert meta['local_cache_frames']==128
    assert meta['effective_video_input_fps'] is None
    assert window.start==source[0].timestamp and window.end==source[-1].timestamp


@pytest.mark.parametrize('duration',[3.,31.])
def test_answer_duration_bounds(duration):
    with pytest.raises(ValueError):select_answer(frames(duration))


def test_capture_gap_is_not_padded_or_hidden():
    source=frames();source=[f for f in source if not source[0].timestamp+2<f.timestamp<source[0].timestamp+2.8]
    with pytest.raises(ValueError):select_answer(source)


def test_unprepared_split_profile_does_not_open_capture(monkeypatch):
    from sunhacks.capture import LiveCapture
    monkeypatch.setattr(LiveCapture,'start',lambda self:pytest.fail('Unprepared capture opened'))
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    with pytest.raises(RuntimeError,match='Prepare'):pipeline.start('live')
    assert pipeline.capture is None


def queued_answer():
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.mode='live';pipeline.session_id='answer-session'
    pipeline.answer={'status':'finishing','elapsed_seconds':6.,'source_frames':181}
    pipeline.answer_finish_event=threading.Event();pipeline.answer_finish_event.set()
    pipeline.answer_frames=deque(maxlen=1000);pipeline.audio=AudioBuffer(retention_seconds=35)
    source=frames();pipeline.created=source[0].timestamp
    pipeline.answer_stop_at=source[-1].timestamp
    audio=np.full(96000,.01,np.float32)
    class Capture:
        stopped=False
        def drain(self):
            if self.stopped:return [],[]
            self.stopped_data=True
            return source,[AudioChunk(source[0].timestamp,16000,audio)]
        def snapshot(self):return {'camera':{'status':'stopped' if self.stopped else 'running'},'audio':{'status':'stopped' if self.stopped else 'running'}}
        def stop(self):self.stopped=True
    pipeline.capture=Capture();pipeline._ingest_answer()
    assert pipeline.capture.stopped
    assert pipeline.answer['status']=='processing'
    assert not pipeline.answer_frames and not pipeline.audio.chunks
    assert pipeline.work.qsize()==1
    return pipeline


def test_finish_stops_sensors_before_the_inference_job_is_queued():queued_answer()


@pytest.mark.parametrize('cancel',['event','session'])
def test_late_answer_result_cannot_publish_into_stopped_or_new_session(cancel):
    pipeline=queued_answer()
    class Runtime:
        def infer(self,clip,audio,transcript,window_id,**kwargs):
            if cancel=='event':kwargs['stop_event'].set()
            else:pipeline.session_id='new-session'
            return {'branches':{'stack':{'status':'trained_classifier','raw_score':.4}}}
    pipeline.fast_runtime=Runtime();pipeline._infer_answer()
    assert pipeline.latest is None


def test_finish_rejects_too_short_capture_without_triggering_inference():
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.answer={'status':'capturing','elapsed_seconds':2,'min_seconds':5}
    pipeline.answer_finish_event=threading.Event()
    with pytest.raises(RuntimeError,match='at least5'):pipeline.finish_answer()
    assert not pipeline.answer_finish_event.is_set()


def test_stop_releases_retained_raw_answer_frames_and_audio():
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.answer={'status':'capturing'};pipeline.answer_frames=deque(frames())
    pipeline.audio=AudioBuffer();pipeline.audio.add(AudioChunk(0,16000,np.ones(16000,np.float32)))
    pipeline.stop()
    assert len(pipeline.answer_frames)==0 and len(pipeline.audio.chunks)==0


def test_failed_video_never_publishes_fabricated_processed_fps(monkeypatch):
    pipeline=queued_answer()
    class Runtime:
        def infer(self,*args,**kwargs):
            return {"branches":{"vision":{"status":"unavailable","elapsed_ms":0}},"transcript":{}}
    pipeline.fast_runtime=Runtime()
    pipeline._infer_answer()
    assert pipeline.latest is not None
    assert pipeline.metrics['video_frames_processed']==0
    assert pipeline.metrics['unique_frames_per_second']==0
    assert pipeline.metrics['video_service_unique_fps'] is None


def test_finish_checks_camera_coverage_before_stopping_streams():
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.answer={'status':'capturing','elapsed_seconds':5.,'min_seconds':5}
    pipeline.answer_frames=deque(frames(duration=5.,fps=24))
    pipeline.answer_finish_event=threading.Event()
    with pytest.raises(RuntimeError,match='128 captured'):pipeline.finish_answer()
    assert not pipeline.answer_finish_event.is_set()
    assert pipeline.answer['status']=='capturing'


@pytest.mark.parametrize('worker_ready',[True,False])
def test_live_readiness_tracks_digest_worker_after_a_crash(worker_ready):
    from types import SimpleNamespace
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.runtime_preparation={'status':'ready','local_weights_loaded':True,'components':{}}
    pipeline.video_client=SimpleNamespace(snapshot=lambda:{'classifier_ready':True,
        'classifier_response_validated':True,'participant_media_authorized':True})
    digest=SimpleNamespace(snapshot=lambda:{'ready':worker_ready,'status':'ready' if worker_ready else 'unprepared'})
    pipeline.fast_runtime=SimpleNamespace(digest_stack=digest,snapshot=lambda:{'active_jobs':[]})
    state=pipeline._split_runtime_state()
    assert state['live_ready'] is worker_ready
    assert state['local_weights_loaded'] is worker_ready
    if not worker_ready:
        assert state['status']=='partial'
        assert 'Prepare local models again' in state['reason']


def test_readiness_gate_preserves_runtime_test_seams_without_worker_snapshot():
    from types import SimpleNamespace
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.runtime_preparation={'status':'ready','local_weights_loaded':True,'components':{}}
    pipeline.video_client=SimpleNamespace(snapshot=lambda:{'classifier_ready':True,
        'classifier_response_validated':True,'participant_media_authorized':True})
    pipeline.fast_runtime=SimpleNamespace(digest_stack=object(),snapshot=lambda:{'active_jobs':[]})
    assert pipeline._split_runtime_state()['live_ready'] is True


def test_stop_now_always_stops_short_recording_and_reports_insufficient_input():
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    source=frames(duration=2.,fps=30)
    pipeline.mode='live';pipeline.session_id='short'
    pipeline.created=source[0].timestamp
    pipeline.answer={'status':'capturing','elapsed_seconds':2.,'min_seconds':5}
    pipeline.answer_frames=deque(source);pipeline.audio=AudioBuffer()
    pipeline.answer_finish_event=threading.Event();pipeline.answer_stop_at=None
    class Capture:
        stopped=False
        def drain(self):return [],[]
        def snapshot(self):return {'camera':{'status':'stopped' if self.stopped else 'running'},'audio':{'status':'stopped' if self.stopped else 'running'}}
        def stop(self):self.stopped=True
    pipeline.capture=Capture()
    pipeline.finish_answer(stop_now=True)
    assert pipeline.answer_finish_event.is_set()
    pipeline._ingest_answer()
    assert pipeline.capture.stopped
    assert pipeline.answer['status']=='error'
    assert pipeline.work.empty()
    assert not pipeline.answer_frames
