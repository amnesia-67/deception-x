import queue
"""Live-path regressions using bounded fixtures; no weights, sensors, keys or APIs."""
import copy
import threading
import time

import numpy as np
import pytest

from sunhacks.models import MODEL_IDS, ModelRegistry
from sunhacks.pipeline import Pipeline
from sunhacks.temporal import VideoFrame, VideoWindow
from sunhacks import teammate_fusion


def _branches(text, audio, *, demo=False):
    status = "mock" if demo else "base"
    value = {name: {"status": status, "model_id": MODEL_IDS[name]}
             for name in ("vision", "dasheng", "text")}
    value["vision"]["_sequence_embedding"] = np.ones((32, 1024), np.float32)
    value["vision"]["_embedding"] = np.ones(1024, np.float32)
    value["dasheng"]["_sequence_embedding"] = np.ones((12, 768), np.float32)
    value["dasheng"]["_embedding"] = np.ones(768, np.float32)
    value["text"]["_embedding"] = np.ones(1024, np.float32)
    if not text:
        value["text"] = {"status": "unavailable", "model_id": MODEL_IDS["text"]}
    if audio is None:
        value["dasheng"] = {"status": "unavailable", "model_id": MODEL_IDS["dasheng"]}
    return value


def _pipeline(monkeypatch, *, mode="live", audio=True):
    config = {"models": {
        "asr": {"path": "/fixture-only/asr", "backend": "transformers"},
        "fusion": {"architecture": teammate_fusion.ARCHITECTURE},
    }}
    pipeline = Pipeline(config)
    pipeline.mode = mode
    pipeline.session_id = "fixture-session"
    pipeline.metrics = {"completed_windows": 0}
    end = time.monotonic()
    image = np.ones((2, 3, 3), np.uint8)
    frames = tuple(VideoFrame(end - (63-index)/24, image, index) for index in range(64))
    window = VideoWindow(1, frames, 24, end)
    pipeline.created = window.start
    waveform = np.full(round((window.end-window.start)*16000), .01, np.float32) if audio else None
    metadata = {"status": "available" if audio else "missing", "coverage": 1 if audio else 0}
    pipeline.work.put((window, waveform, metadata, time.monotonic()))
    # Publish at most one result without creating live worker threads.
    original_get=pipeline.work.get
    calls=[]
    def one_result(*args,**kwargs):
        if calls:
            pipeline.stop_event.set()
            raise queue.Empty
        calls.append(True)
        return original_get(*args,**kwargs)
    monkeypatch.setattr(pipeline.work, "get", one_result)
    monkeypatch.setattr(teammate_fusion, "_load", lambda cfg: pytest.fail("No head weights should load in this regression"))
    return pipeline


def test_synthetic_demo_never_runs_configured_local_asr(monkeypatch):
    pipeline = _pipeline(monkeypatch, mode="demo")
    monkeypatch.setattr(ModelRegistry, "transcribe", lambda *args: pytest.fail("Synthetic demo invoked ASR"))
    monkeypatch.setattr(ModelRegistry, "run", lambda self, clip, audio, text, demo=False: _branches(text, audio, demo=demo))
    pipeline._infer()
    assert pipeline.latest["mock"] is True
    assert pipeline.latest["branches"]["fusion"]["status"] == "unavailable"
    assert pipeline.latest["transcript"]["status"] == "missing"


@pytest.mark.parametrize("status", ["partial", "error", "missing"])
def test_incomplete_asr_is_visible_but_never_sent_to_qwen_or_head(monkeypatch, status):
    pipeline = _pipeline(monkeypatch)
    texts = []
    monkeypatch.setattr(ModelRegistry, "transcribe", lambda *args: {"status": status, "text": "Incomplete fixture speech"})
    def run(self, clip, audio, text, demo=False):
        texts.append(text)
        return _branches(text, audio)
    monkeypatch.setattr(ModelRegistry, "run", run)
    pipeline._infer()
    assert texts == [""]
    assert pipeline.latest["transcript"]["status"] == status
    assert pipeline.latest["branches"]["text"]["status"] == "unavailable"
    assert pipeline.latest["branches"]["fusion"]["status"] == "unavailable"
    assert pipeline.latest["branches"]["fusion"]["raw_score"] is None


def test_missing_audio_never_invokes_asr_or_fabricates_head_input(monkeypatch):
    pipeline = _pipeline(monkeypatch, audio=False)
    pipeline.transcript = {"status": "available", "text": "Manual fixture context", "scope": "session_context"}
    monkeypatch.setattr(ModelRegistry, "transcribe", lambda *args: pytest.fail("ASR invoked for missing audio"))
    monkeypatch.setattr(ModelRegistry, "run", lambda self, clip, audio, text, demo=False: _branches(text, audio))
    pipeline._infer()
    assert pipeline.latest["features"]["audio"]["status"] == "missing"
    assert pipeline.latest["branches"]["fusion"]["status"] == "unavailable"


def test_stop_during_asr_prevents_subsequent_encoder_work(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    calls = []
    def transcribe(*args):
        pipeline.stop_event.set()
        return {"status": "base", "text": "Late automatic fixture transcript"}
    def run(*args, **kwargs):
        calls.append("encoder")
        return {}
    monkeypatch.setattr(ModelRegistry, "transcribe", transcribe)
    monkeypatch.setattr(ModelRegistry, "run", run)
    pipeline._infer()
    assert calls == [], "Stopping during ASR still started expensive downstream encoders"
    assert pipeline.latest is None


def test_stop_during_encoders_prevents_subsequent_head_work(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    heads = []
    monkeypatch.setattr(ModelRegistry, "transcribe", lambda *args: {"status": "base", "text": "Fixture transcript"})
    def run(self, clip, audio, text, demo=False):
        pipeline.stop_event.set()
        return _branches(text, audio)
    def head(*args):
        heads.append(True)
        return {"status": "unavailable"}
    monkeypatch.setattr(ModelRegistry, "run", run)
    monkeypatch.setattr(teammate_fusion, "run_teammate_fusion", head)
    pipeline._infer()
    assert heads == [], "Stopped work continued to the head after the encoder returned"
    assert pipeline.latest is None


def test_private_sequences_never_enter_public_state(monkeypatch):
    pipeline = _pipeline(monkeypatch)
    monkeypatch.setattr(ModelRegistry, "transcribe", lambda *args: {"status": "base", "text": "Recognized fixture transcript"})
    monkeypatch.setattr(ModelRegistry, "run", lambda self, clip, audio, text, demo=False: _branches(text, audio))
    monkeypatch.setattr(teammate_fusion, "run_teammate_fusion", lambda *args: {"status": "synthetic_trained", "raw_score": .4})
    pipeline._infer()
    public = pipeline.state()
    assert "_sequence_embedding" not in str(public)
    assert "_embedding" not in str(public)
    assert public["latest"]["transcript"]["scope"] == "video_window"


def test_missing_window_cannot_reuse_prior_private_sequence(monkeypatch):
    registry = ModelRegistry({})
    registry._sequences["vision"] = np.ones((32, 1024), np.float32)
    registry._sequences["dasheng"] = np.ones((32, 768), np.float32)
    monkeypatch.setattr(registry, "_release", lambda branch: None)
    result = registry.run(None, None, "", demo=False)
    assert all("_sequence_embedding" not in item for item in result.values())
    assert registry._sequences == {}
