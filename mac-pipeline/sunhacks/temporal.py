"""Timestamp-first bounded buffers; capture speed is independent of inference speed."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class VideoFrame:
    timestamp: float
    rgb: np.ndarray
    sequence: int

@dataclass(frozen=True)
class AudioChunk:
    start: float
    sample_rate: int
    samples: np.ndarray

@dataclass(frozen=True)
class VideoWindow:
    id: int
    frames: tuple[VideoFrame, ...]
    sample_fps: float
    target_end: float

    @property
    def start(self): return self.frames[0].timestamp
    @property
    def end(self): return self.frames[-1].timestamp
    def metadata(self):
        stamps = [f.timestamp for f in self.frames]
        return {"id": self.id, "start": self.start, "end": self.end,
                "sample_fps": self.sample_fps, "unique_frames": len(set(f.sequence for f in self.frames)),
                "coverage": 1.0, "timestamps": stamps,
                "actual_sample_fps": (len(stamps)-1)/(stamps[-1]-stamps[0]),
                "max_frame_gap_seconds": max(np.diff(stamps)),
                "target_end": self.target_end}

class WindowBuffer:
    def __init__(self, sample_fps=24.0, frames_per_window=64, hop_seconds=32/24,
                 retention_seconds=8.0, max_frames=300):
        if not 8 < sample_fps <= 60 or frames_per_window != 64:
            raise ValueError("V-JEPA contract requires 64 frames and sampling rate in (8,60]")
        if hop_seconds <= 0 or retention_seconds < (frames_per_window-1)/sample_fps:
            raise ValueError("Invalid window retention/hop")
        self.sample_fps, self.count, self.hop = sample_fps, frames_per_window, hop_seconds
        self.retention = retention_seconds
        self.frames = deque(maxlen=max_frames)
        self.next_end = None
        self.emitted = self.rejected = 0
        self.last_timestamp = -math.inf

    def add(self, frame):
        if not math.isfinite(frame.timestamp) or frame.timestamp <= self.last_timestamp:
            return False
        self.frames.append(frame)
        self.last_timestamp = frame.timestamp
        if self.next_end is None:
            self.next_end = frame.timestamp + (self.count-1)/self.sample_fps
        while self.frames and self.frames[0].timestamp < frame.timestamp-self.retention:
            self.frames.popleft()
        return True

    def ready(self):
        if not self.frames or self.next_end is None or self.frames[-1].timestamp < self.next_end:
            return []
        # Never generate an unbounded catch-up backlog following a pause.
        skipped = max(0, int((self.frames[-1].timestamp-self.next_end)//self.hop))
        self.rejected += skipped
        self.next_end += skipped*self.hop
        target_end = self.next_end
        self.next_end += self.hop
        targets = target_end - np.arange(self.count-1, -1, -1)/self.sample_fps
        candidates = list(self.frames)
        stamps = np.array([f.timestamp for f in candidates])
        right = np.searchsorted(stamps, targets).clip(0, len(stamps)-1)
        left = (right-1).clip(0, len(stamps)-1)
        indices = np.where(abs(stamps[left]-targets) <= abs(stamps[right]-targets), left, right)
        selected = tuple(candidates[int(i)] for i in indices)
        if len(set(f.sequence for f in selected)) != self.count or np.max(abs(stamps[indices]-targets)) > .75/self.sample_fps:
            self.rejected += 1
            return []
        self.emitted += 1
        return [VideoWindow(self.emitted, selected, self.sample_fps, target_end)]

class AudioBuffer:
    def __init__(self, rate=16000, retention_seconds=8):
        self.rate, self.retention = rate, retention_seconds
        self.chunks = deque(maxlen=500)
        self.last_start = -math.inf
    def add(self, chunk):
        if chunk.sample_rate != self.rate or chunk.samples.ndim != 1 or not np.isfinite(chunk.samples).all():
            raise ValueError("Audio must be finite float mono at 16kHz")
        if chunk.start < self.last_start: return False
        self.last_start = chunk.start
        self.chunks.append(chunk)
        while self.chunks and self.chunks[0].start+len(self.chunks[0].samples)/self.rate < chunk.start-self.retention:
            self.chunks.popleft()
        return True
    def slice(self, start, end):
        # Missing positions are tracked separately; zeros never count as observed audio.
        count = max(0, round((end-start)*self.rate))
        samples = np.zeros(count, np.float32)
        valid = np.zeros(count, bool)
        for chunk in self.chunks:
            offset = round((chunk.start-start)*self.rate)
            a, b = max(0, offset), min(count, offset+len(chunk.samples))
            if b > a:
                samples[a:b] = chunk.samples[a-offset:b-offset]
                valid[a:b] = True
        coverage = float(valid.mean()) if count else 0.0
        # Partial waveforms do not pass to encoders: no fabricated silence or hidden concatenation.
        metadata = {"start": start, "end": end, "sample_rate": self.rate,
                    "coverage": coverage, "missing_samples": int((~valid).sum()),
                    "status": "available" if count and valid.all() else "missing"}
        return (samples if count and valid.all() else None), metadata
