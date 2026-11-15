"""Bounded whole-answer capture; timestamps are selected once for both branches."""
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class AnswerWindow:
    id: int
    frames: tuple
    start: float
    end: float
    source_frame_count: int
    source_span_seconds: float
    _source_frames: tuple = field(default=(), repr=False, compare=False)
    @property
    def sample_fps(self):return 127/max(1e-9,self.frames[-1].timestamp-self.frames[0].timestamp)
    def metadata(self):
        duration=self.end-self.start
        return {"id":self.id,"start":self.start,"end":self.end,"duration_seconds":duration,
                "capture_mode":"whole_answer","transport_frames":None,"local_cache_frames":len(self.frames),
                "unique_frames":len({f.sequence for f in self.frames}),
                "source_frames_captured":self.source_frame_count,
                "timestamps":[f.timestamp for f in self.frames],
                "native_video_unique_frames":None,
                "effective_video_input_fps":None,
                "actual_sample_fps":self.sample_fps,"sample_fps":self.sample_fps,
                "max_frame_gap_seconds":float(np.max(np.diff([f.timestamp for f in self.frames]))),
                "coverage":1.,"temporal_recipe":"128 nearest-time local cache frames; remote L uses original timestamps, two64-frame stride1 views; completion unverified",
                "quality_limit":"Short webcam answers differ from courtroom training; score quality is unverified."}


def select_answer(frames, *, identifier=1, min_seconds=5., max_seconds=30.):
    """Same time-nearest128 recipe as reviewed training/service. Never pad gaps."""
    frames=tuple(frames)
    if len(frames)<128:raise ValueError("Need at least128 captured frames; capture a longer answer")
    ts=np.array([f.timestamp for f in frames],np.float64)
    if not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):raise ValueError("Capture timestamps must increase")
    if len({f.sequence for f in frames})!=len(frames):raise ValueError("Repeated capture source IDs")
    duration=float(ts[-1]-ts[0])
    if not min_seconds<=duration<=max_seconds+.2:raise ValueError(f"Answer must span{min_seconds:g}..{max_seconds:g} seconds")
    target=ts[0]+(np.arange(128)+.5)*duration/128
    right=np.searchsorted(ts,target).clip(0,len(ts)-1);left=(right-1).clip(0,len(ts)-1)
    indices=np.where(abs(ts[left]-target)<abs(ts[right]-target),left,right)
    selected=tuple(frames[int(i)] for i in indices)
    if len({f.sequence for f in selected})<128:
        raise ValueError("Camera timing is too sparse for128 distinct cache frames; try a longer answer")
    if max(np.diff(ts))>.25:raise ValueError("Camera has a gap over250ms; answer was not padded")
    return AnswerWindow(identifier,selected,float(ts[0]),float(ts[-1]),len(frames),duration,frames)
