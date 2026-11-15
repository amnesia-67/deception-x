"""Independent integration and regression tests; no devices, weights, or remote APIs."""
import asyncio
import json
import queue
import threading
import time

import httpx
import numpy as np
import pytest

from sunhacks.pipeline import Pipeline
from sunhacks.temporal import AudioBuffer, AudioChunk, VideoFrame, VideoWindow, WindowBuffer


def make_window(identifier=1, end=None):
    end = time.monotonic() if end is None else end
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    frames = tuple(VideoFrame(end - (63-index)/24, image, index) for index in range(64))
    return VideoWindow(identifier, frames, 24, end)


class TwoIngestSteps:
    def __init__(self): self.steps = 0
    def is_set(self): return self.steps >= 2
    def wait(self, timeout): self.steps += 1; return self.is_set()
    def set(self): self.steps = 2


def test_live_capture_constructor_matches_implementation(monkeypatch):
    from sunhacks.capture import LiveCapture
    monkeypatch.setattr(LiveCapture, "start", lambda self: None)
    monkeypatch.setattr(LiveCapture, "stop", lambda self: None)
    monkeypatch.setattr(Pipeline, "_ingest", lambda self: None)
    monkeypatch.setattr(Pipeline, "_infer", lambda self: None)
    pipeline = Pipeline()
    try:
        state = pipeline.start("live")
        assert state["mode"] == "live"
        assert pipeline.capture.camera_index == 0
        assert pipeline.capture.camera_fps == 30
    finally:
        pipeline.stop()


def test_inference_worker_integrates_diagnostics_and_publishes(monkeypatch):
    from sunhacks.models import ModelRegistry
    pipeline = Pipeline()
    pipeline.mode = "demo"
    pipeline.metrics = {"completed_windows": 0}
    window = make_window()
    pipeline.created = window.start
    audio = np.zeros(round((window.end-window.start)*16000), dtype=np.float32)
    pipeline.work.put((window, audio, {"status": "available", "coverage": 1}, time.monotonic()))
    monkeypatch.setattr(ModelRegistry, "run", lambda *a, **kw: {"vision": {"status": "mock"}})
    original_get=pipeline.work.get
    calls=[]
    def one_result(*args,**kwargs):
        if calls:
            pipeline.stop_event.set()
            raise queue.Empty
        calls.append(True)
        return original_get(*args,**kwargs)
    monkeypatch.setattr(pipeline.work, "get", one_result)
    pipeline._infer()
    assert pipeline.latest is not None
    assert pipeline.latest["features"]["audio"]["status"] == "ok"
    assert pipeline.latest["features"]["video"]["frames"] == 64
    assert pipeline.metrics["completed_windows"] == 1


def test_video_window_waits_for_audio_tail_arriving_next_drain():
    """A video-ready tick must not permanently discard a slightly delayed mic tail."""
    pipeline = Pipeline()
    pipeline.mode = "live"
    pipeline.video = WindowBuffer()
    pipeline.audio = AudioBuffer()
    pipeline.stop_event = TwoIngestSteps()
    window = make_window()
    early_count = round((window.end-window.start-.025)*16000)
    audio = np.zeros(early_count, dtype=np.float32)
    tail = np.zeros(1600, dtype=np.float32)
    class Capture:
        count = 0
        def drain(self):
            self.count += 1
            if self.count == 1:
                return list(window.frames), [AudioChunk(window.start, 16000, audio)]
            return [], [AudioChunk(window.start+early_count/16000, 16000, tail)]
    pipeline.capture = Capture()
    pipeline._ingest()
    queued = pipeline.work.get_nowait()
    assert queued[1] is not None, "Complete mic tail arrived but queued window permanently retained missing audio"
    assert queued[2]["status"] == "available"


def test_partial_audio_is_missing_but_observed_silence_is_available():
    audio = AudioBuffer()
    audio.add(AudioChunk(10, 16000, np.zeros(1600, np.float32)))
    values, meta = audio.slice(10, 10.2)
    assert values is None
    assert meta["coverage"] == pytest.approx(.5)
    assert meta["missing_samples"] == 1600
    audio.add(AudioChunk(10.1, 16000, np.zeros(1600, np.float32)))
    values, meta = audio.slice(10, 10.2)
    assert meta["status"] == "available"
    assert len(values) == 3200
    assert np.count_nonzero(values) == 0


def test_sparse_camera_never_fabricates_requested_sampling_rate():
    video = WindowBuffer(sample_fps=24)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for index in range(60):
        video.add(VideoFrame(index/10, image, index))
    assert video.ready() == []
    assert video.rejected > 0






def test_loopback_server_blocks_foreign_origin_and_bad_payload():
    from sunhacks.server import make_server
    pipeline = Pipeline()
    server = make_server(pipeline, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(base_url=base, timeout=2, trust_env=False) as client:
            assert client.get("/api/state").status_code == 200
            response = client.post("/api/session", headers={"Origin": "https://evil.example"}, json={"mode": "demo"})
            assert response.status_code == 403
            assert pipeline.mode == "stopped"
            assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 403
            assert client.post("/api/transcript", json=[]).status_code == 400
            assert client.post("/api/transcript", json={"text": "x"*16001}).status_code == 400
            assert client.post("/api/session", json={"mode": "unknown"}).status_code == 400
            assert client.post("/api/transcript", content="text/plain").status_code == 415
            assert client.post("/api/transcript", content="{" ,headers={"Content-Type":"application/json"}).status_code == 400
            assert pipeline.mode == "stopped"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)
        pipeline.stop()


def test_latest_window_queue_is_bounded_and_counts_replacement():
    pipeline = Pipeline()
    pipeline.mode = "live"
    pipeline.metrics = {"dropped_windows": 0}
    pipeline.video = WindowBuffer()
    pipeline.audio = AudioBuffer()
    pipeline.stop_event = TwoIngestSteps()
    image = np.zeros((4, 6, 3), np.uint8)
    base = time.monotonic()-4
    class Capture:
        count = 0
        def drain(self):
            self.count += 1
            indices = range(64) if self.count == 1 else range(64, 97)
            frames = [VideoFrame(base+i/24, image, i) for i in indices]
            start = base if self.count == 1 else base+2.75
            return frames, [AudioChunk(start, 16000, np.zeros(44000, np.float32))]
    pipeline.capture = Capture()
    pipeline._ingest()
    assert pipeline.work.qsize() == 1
    window, audio, metadata, _ = pipeline.work.get_nowait()
    assert window.id == 2
    assert audio is not None and metadata["status"] == "available"
    assert pipeline.metrics["dropped_windows"] == 1


def test_inference_completing_after_stop_does_not_publish(monkeypatch):
    from sunhacks.models import ModelRegistry
    pipeline = Pipeline()
    pipeline.mode = "demo"
    window = make_window()
    pipeline.work.put((window, None, {"status": "missing"}, time.monotonic()))
    def finish_after_stop(*args, **kwargs):
        pipeline.stop_event.set()
        return {"vision": {"status": "mock"}}
    monkeypatch.setattr(ModelRegistry, "run", finish_after_stop)
    pipeline._infer()
    assert pipeline.latest is None
    assert not hasattr(pipeline,"images")


def test_restart_refuses_to_spawn_over_previous_live_worker():
    pipeline = Pipeline()
    class StillRunning:
        def join(self, timeout=None): pass
        def is_alive(self): return True
    pipeline.threads = [StillRunning()]
    with pytest.raises(RuntimeError, match="still stopping"):
        pipeline.start("demo")
    assert pipeline.mode == "stopped"
    assert pipeline.session_id is None
