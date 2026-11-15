import threading
import time
from collections import deque
import numpy as np
import pytest
from sunhacks.pipeline import Pipeline
from sunhacks.temporal import AudioBuffer,AudioChunk,VideoFrame


def timed_capture(monkeypatch,*,startup_seconds=0,source_seconds=8,elapsed_now=8,audio_ready=True):
    import sunhacks.answer_pipeline as module
    base=100.
    monkeypatch.setattr(module.time,'monotonic',lambda:base+elapsed_now)
    pipeline=Pipeline({'runtime_mode':'split_local_video'})
    pipeline.mode='live';pipeline.session_id='timed';pipeline.created=base-startup_seconds
    pipeline.answer={'status':'capturing','auto_stop_seconds':7.,'timer_started_at':None,'deadline_at':None}
    pipeline.answer_frames=deque(maxlen=1000);pipeline.audio=AudioBuffer(retention_seconds=35)
    pipeline.answer_finish_event=threading.Event();pipeline.answer_stop_at=None
    source=[VideoFrame(base+i/30,np.zeros((4,6,3),np.uint8),i) for i in range(round(source_seconds*30)+1)]
    class Capture:
        stopped=False
        drains=0
        def drain(self):
            self.drains+=1
            if self.drains>1:
                pipeline.stop_event.set();return [],[]
            return source,[AudioChunk(base if audio_ready else base+1,16000,np.full(round(source_seconds*16000),.01,np.float32))]
        def snapshot(self):return {'camera':{'status':'stopped' if self.stopped else 'running'},'audio':{'status':'stopped' if self.stopped else 'running'}}
        def stop(self):self.stopped=True
    pipeline.capture=Capture()
    pipeline._ingest_answer()
    return pipeline


def test_server_timer_trims_overrun_and_stops_before_inference(monkeypatch):
    pipeline=timed_capture(monkeypatch)
    assert pipeline.capture.stopped
    assert pipeline.answer['status']=='processing'
    window,audio,meta,_=pipeline.work.get_nowait()
    assert window.end-window.start==pytest.approx(7.)
    assert max(f.timestamp for f in window.frames)<=107.
    assert audio is not None and meta['status']=='available'
    assert window.metadata()['effective_video_input_fps'] is None
    assert len({f.sequence for f in window.frames})/(window.end-window.start)>8
    assert not pipeline.answer_frames and not pipeline.audio.chunks


def test_startup_delay_is_not_subtracted_from_seven_second_answer(monkeypatch):
    pipeline=timed_capture(monkeypatch,startup_seconds=6,source_seconds=6,elapsed_now=6)
    assert not pipeline.capture.stopped
    assert pipeline.work.empty()
    assert pipeline.answer['timer_started_at']==100.
    assert pipeline.answer['deadline_at']==107.


@pytest.mark.parametrize('invalid',[True,4,31,float('nan'),'7'])
def test_invalid_timer_never_opens_capture(invalid,monkeypatch):
    from sunhacks.capture import LiveCapture
    monkeypatch.setattr(LiveCapture,'start',lambda self:pytest.fail('Invalid timer opened capture'))
    with pytest.raises(ValueError):Pipeline({'runtime_mode':'split_local_video'}).start('live',auto_stop_seconds=invalid)
