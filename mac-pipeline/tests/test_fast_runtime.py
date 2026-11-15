"""Bounded orchestration tests using events and fake encoders only."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from sunhacks import fast_runtime as runtime
from sunhacks.models import DIMS, MODEL_IDS


def inputs():
    return np.zeros((64, 1, 1, 3), np.uint8), np.ones(16, np.float32)


def transcript(text="Manual words."):
    return {"status": "available", "text": text, "source": "manual", "scope": "session_context"}


def branch_output(branch):
    return {"status": "base", "model_id": MODEL_IDS[branch], "elapsed_ms": 1.0,
            "embedding_shape": [DIMS[branch]], "_embedding": np.ones(DIMS[branch], np.float32),
            "reason": "Fake encoder for scheduling tests only."}


@pytest.fixture
def fake(monkeypatch):
    state = {"loads": [], "runs": [], "asr": [], "clears": 0,
             "asr_prepares": [], "run_hooks": {}, "transcribe_hook": None}
    class Registry:
        def __init__(self, config):
            self.config = config
            self._loaded = {}
            self.closed = 0
        def _load(self, branch):
            state["loads"].append(branch)
            self._loaded[branch] = object()
        def run(self, clip, audio, text, *, branches, should_stop):
            branch = branches[0]
            state["runs"].append((branch, text))
            hook = state["run_hooks"].get(branch)
            if hook:
                hook()
            if should_stop():
                return {branch: {"status": "cancelled", "reason": "Fixture stopped."}}
            if branch == "text" and not text:
                return {branch: {"status": "unavailable", "reason": "Missing transcript."}}
            return {branch: branch_output(branch)}
        def transcribe(self, audio):
            state["asr"].append(audio)
            if state["transcribe_hook"]:
                return state["transcribe_hook"]()
            return {"status": "base", "text": "Recognized words.", "source": "local_asr"}
        def close(self):
            self.closed += 1
    def prepare(cfg):
        state["asr_prepares"].append(dict(cfg))
        return {"status": "ready", "kernels_warmed": False, "cache_hit": False}
    def clear():
        state["clears"] += 1
        if state.get("cleared_event"):
            state["cleared_event"].set()
    monkeypatch.setattr(runtime, "ModelRegistry", Registry)
    monkeypatch.setattr(runtime, "prepare_asr", prepare)
    monkeypatch.setattr(runtime, "clear_asr_cache", clear)
    return state


def config(**options):
    return {"models": {"asr": {"path": "/fixture", "device": "mps"},
                       "dasheng": {"device": "cpu"}, "text": {"device": "mps"},
                       "emotion": {"device": "cpu"}},
            "fast_runtime": {"emotion_enabled": False, **options}}


def test_prepare_is_serial_resident_and_does_not_claim_kernel_warmup(fake):
    original = config(emotion_enabled=True, prepare_emotion=True)
    unit = runtime.FastRuntime(original, vision_runner=lambda *args: branch_output("vision"))
    try:
        report = unit.prepare()
        assert report["status"] == "ready" and report["local_weights_loaded"] is True
        assert report["kernels_warmed"] is False
        assert fake["loads"] == ["dasheng", "text", "emotion"]
        assert fake["runs"] == fake["asr"] == []
        assert fake["asr_prepares"][0]["keep_loaded"] is True
        assert all(unit.config["models"][name]["keep_loaded"] for name in ("dasheng", "text", "emotion", "asr"))
        assert "keep_loaded" not in original["models"]["text"]
    finally:
        unit.close()


def test_required_paths_overlap_and_asr_precedes_text(fake):
    gate = threading.Event()
    started = {name: threading.Event() for name in ("vision", "dasheng", "semantic")}
    def vision(clip, window_id):
        started["vision"].set()
        assert gate.wait(2)
        return branch_output("vision")
    def dasheng():
        started["dasheng"].set()
        assert gate.wait(2)
    def recognize():
        started["semantic"].set()
        assert gate.wait(2)
        assert not any(name == "text" for name, text in fake["runs"])
        return {"status": "base", "text": "Real recognized fixture."}
    fake["run_hooks"]["dasheng"] = dasheng
    fake["transcribe_hook"] = recognize
    unit = runtime.FastRuntime(config(), vision_runner=vision)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(unit.infer, *inputs(), transcript(), 17)
            try:
                assert all(event.wait(1) for event in started.values())
            finally:
                gate.set()
            result = pending.result(2)
        assert all(result["branches"][name]["status"] == "base" for name in ("vision", "dasheng", "text"))
        assert ("text", "Real recognized fixture.") in fake["runs"]
        assert result["transcript"]["source_window_id"] == 17
        assert all(result["branches"][name]["source_window_id"] == 17 for name in ("vision", "dasheng", "text"))
    finally:
        gate.set()
        unit.close()


def test_late_emotion_does_not_block_required_outputs(fake):
    gate = threading.Event()
    entered = threading.Event()
    def emotion():
        entered.set()
        assert gate.wait(2)
    fake["run_hooks"]["emotion"] = emotion
    unit = runtime.FastRuntime(config(emotion_enabled=True), vision_runner=lambda *args: branch_output("vision"))
    try:
        result = unit.infer(*inputs(), transcript(), 7)
        assert entered.wait(1)
        pending = result["_emotion_future"]
        assert not pending.done()
        assert result["branches"]["text"]["status"] == "base"
        assert result["branches"]["emotion"]["background_pending"] is True
        gate.set()
        assert pending.result(1)["source_window_id"] == 7
    finally:
        gate.set()
        unit.close()


def test_timeout_retains_one_slot_and_never_reuses_older_vision(fake):
    gate = threading.Event()
    started = threading.Event()
    finished = threading.Event()
    calls = []
    def vision(clip, window_id):
        calls.append(window_id)
        started.set()
        assert gate.wait(2)
        finished.set()
        return branch_output("vision")
    unit = runtime.FastRuntime(config(deadline_seconds=.05), vision_runner=vision)
    try:
        first = unit.infer(*inputs(), transcript(), 1)
        assert started.is_set() and first["timings"]["jobs"]["vision"]["status"] == "timeout"
        second = unit.infer(*inputs(), transcript(), 2)
        assert calls == [1]
        assert second["timings"]["jobs"]["vision"]["status"] == "busy"
        assert second["branches"]["vision"]["source_window_id"] == 2
        assert "_embedding" not in second["branches"]["vision"]
        assert len(unit._inflight) <= 4
        gate.set()
        assert finished.wait(1)
    finally:
        gate.set()
        unit.close()


def test_close_waits_to_release_models_until_running_job_finishes(fake):
    gate = threading.Event()
    fake["cleared_event"] = threading.Event()
    def vision(*args):
        assert gate.wait(2)
        return branch_output("vision")
    unit = runtime.FastRuntime(config(deadline_seconds=.03), vision_runner=vision)
    try:
        unit.infer(*inputs(), transcript(), 1)
        closing = unit.close()
        assert closing["status"] == "closing"
        assert unit.registry.closed == fake["clears"] == 0
        gate.set()
        assert fake["cleared_event"].wait(1)
        assert unit.registry.closed == fake["clears"] == 1
        assert unit.close()["status"] == "closed"
        assert unit.registry.closed == fake["clears"] == 1
    finally:
        gate.set()
        unit.close()


def test_no_remote_runner_never_invokes_local_vision(fake):
    unit = runtime.FastRuntime(config())
    try:
        prepared = unit.prepare()
        result = unit.infer(*inputs(), transcript(), 9)
        assert prepared["status"] == "partial"
        assert result["branches"]["vision"]["status"] == "unavailable"
        assert "no local vision fallback" in result["branches"]["vision"]["reason"]
        assert "vision" not in fake["loads"]
        assert not any(branch == "vision" for branch, text in fake["runs"])
    finally:
        unit.close()


def test_remote_exception_is_explicit_and_does_not_fall_back(fake):
    def fail(*args): raise ConnectionError("Fixture peer unavailable")
    unit = runtime.FastRuntime(config(), vision_runner=fail)
    try:
        result = unit.infer(*inputs(), transcript(), 2)
        assert result["branches"]["vision"]["status"] == "error"
        assert "ConnectionError" in result["branches"]["vision"]["reason"]
        assert not any(branch == "vision" for branch, text in fake["runs"])
    finally:
        unit.close()


def test_partial_asr_does_not_supply_invented_complete_text(fake):
    fake["transcribe_hook"] = lambda: {"status": "partial", "text": "Incomplete..."}
    unit = runtime.FastRuntime(config(), vision_runner=lambda *args: branch_output("vision"))
    try:
        result = unit.infer(*inputs(), transcript(), 3)
        assert result["transcript"]["status"] == "partial"
        assert ("text", "") in fake["runs"]
        assert result["branches"]["text"]["status"] == "unavailable"
    finally:
        unit.close()


def test_stopped_session_submits_no_model_jobs(fake):
    stopped = threading.Event()
    stopped.set()
    unit = runtime.FastRuntime(config(), vision_runner=lambda *args: pytest.fail("Unexpected remote call"))
    try:
        result = unit.infer(*inputs(), transcript(), 4, stop_event=stopped)
        assert fake["runs"] == fake["asr"] == []
        assert all(result["branches"][branch]["status"] == "cancelled" for branch in ("vision", "dasheng", "text"))
    finally:
        unit.close()


def test_progress_callback_failure_does_not_break_inference(fake):
    def fail(event): raise RuntimeError("Fixture UI failed")
    unit = runtime.FastRuntime(config(), vision_runner=lambda *args: branch_output("vision"), on_progress=fail)
    try:
        assert unit.prepare()["status"] == "ready"
        assert unit.infer(*inputs(), transcript(), 4)["branches"]["text"]["status"] == "base"
    finally:
        unit.close()


def test_preparation_excludes_inference_and_defers_close_until_load_finishes(fake, monkeypatch):
    gate, entered = threading.Event(), threading.Event()
    fake["cleared_event"] = threading.Event()
    def prepare(cfg):
        entered.set()
        assert gate.wait(2)
        return {"status": "ready", "kernels_warmed": False}
    monkeypatch.setattr(runtime, "prepare_asr", prepare)
    unit = runtime.FastRuntime(config(), vision_runner=lambda *args: pytest.fail("Unexpected remote call"))
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(unit.prepare)
            try:
                assert entered.wait(1)
                result = unit.infer(*inputs(), transcript(), 1)
                assert all(result["timings"]["jobs"][key]["status"] == "preparing"
                           for key in ("vision", "dasheng", "semantic"))
                assert fake["runs"] == []
                assert unit.close()["status"] == "closing"
                assert unit.registry.closed == fake["clears"] == 0
            finally:
                gate.set()
            assert pending.result(1)["status"] == "stopped"
        assert fake["cleared_event"].wait(1)
        assert unit.registry.closed == fake["clears"] == 1
        assert fake["loads"] == []
    finally:
        gate.set()
        unit.close()


def test_busy_semantics_do_not_reuse_old_transcript_or_start_late_text(fake):
    gate = threading.Event()
    def recognize():
        assert gate.wait(2)
        return {"status": "base", "text": "Previous source words."}
    fake["transcribe_hook"] = recognize
    unit = runtime.FastRuntime(config(deadline_seconds=.03), vision_runner=lambda *args: branch_output("vision"))
    try:
        first = unit.infer(*inputs(), transcript("First manual context."), 1)
        assert first["timings"]["jobs"]["semantic"]["status"] == "timeout"
        pending = unit._inflight["semantic"]
        second = unit.infer(*inputs(), transcript("Second manual context."), 2)
        assert second["timings"]["jobs"]["semantic"]["status"] == "busy"
        assert len(fake["asr"]) == 1
        assert second["transcript"]["text"] == "Second manual context."
        assert second["branches"]["text"]["source_window_id"] == 2
        assert "_embedding" not in second["branches"]["text"]
        gate.set()
        late = pending.result(1)
        assert late["text"]["status"] == "cancelled"
        assert late["text"]["source_window_id"] == 1
        assert "_embedding" not in late["text"]
    finally:
        gate.set()
        unit.close()


def test_remote_identity_and_private_sequence_are_preserved(fake):
    remote = {**branch_output("vision"), "model_id": "verified/example-different-model",
              "revision": "verified-revision", "_sequence_embedding": np.ones((32, 1024))}
    unit = runtime.FastRuntime(config(), vision_runner=lambda *args: remote)
    try:
        result = unit.infer(*inputs(), transcript(), 5)
        actual = result["branches"]["vision"]
        assert actual["model_id"] == "verified/example-different-model"
        assert actual["revision"] == "verified-revision"
        assert actual["_sequence_embedding"] is remote["_sequence_embedding"]
        assert "source_window_id" not in remote
    finally:
        unit.close()


def test_close_reports_closing_until_both_cleanup_steps_finish(fake, monkeypatch):
    registry_entered, registry_release = threading.Event(), threading.Event()
    cache_entered, cache_release = threading.Event(), threading.Event()
    unit = runtime.FastRuntime(config())
    original_registry_close = unit.registry.close
    def registry_close():
        registry_entered.set()
        assert registry_release.wait(2)
        original_registry_close()
    def cache_clear():
        cache_entered.set()
        assert cache_release.wait(2)
        fake["clears"] += 1
    monkeypatch.setattr(unit.registry, "close", registry_close)
    monkeypatch.setattr(runtime, "clear_asr_cache", cache_clear)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            closing = caller.submit(unit.close)
            try:
                assert registry_entered.wait(1)
                during_registry = unit.close()
                assert during_registry["status"] == "closing"
                assert during_registry["cleanup_in_progress"] is True
                assert not unit._cleanup_done and not unit._cleanup_finished.is_set()
                registry_release.set()
                assert cache_entered.wait(1)
                during_cache = unit.close()
                assert during_cache["status"] == "closing"
                assert during_cache["cleanup_in_progress"] is True
                assert fake["clears"] == 0
                assert not unit._cleanup_done and not unit._cleanup_finished.is_set()
            finally:
                registry_release.set()
                cache_release.set()
            assert closing.result(1)["status"] == "closed"
        assert unit._cleanup_finished.is_set()
        assert unit.close()["cleanup_in_progress"] is False
        assert unit.registry.closed == fake["clears"] == 1
    finally:
        registry_release.set()
        cache_release.set()
        unit.close()


def test_emotion_finishing_after_stop_discards_features(fake, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    stop = threading.Event()
    unit = runtime.FastRuntime(config(emotion_enabled=True), vision_runner=lambda *args: branch_output("vision"))
    original_run = unit.registry.run
    def native_like_run(clip, audio, text, *, branches, should_stop):
        if branches != ("emotion",):
            return original_run(clip, audio, text, branches=branches, should_stop=should_stop)
        # The native registry checks cancellation before a forward, not while
        # its accelerator/CPU kernel is already executing.
        assert not should_stop()
        entered.set()
        assert release.wait(2)
        return {"emotion": branch_output("emotion")}
    monkeypatch.setattr(unit.registry, "run", native_like_run)
    try:
        result = unit.infer(*inputs(), transcript(), 19, stop_event=stop)
        assert entered.wait(1)
        future = result["_emotion_future"]
        stop.set()
        release.set()
        late = future.result(1)
        assert late["status"] == "cancelled" and late["source_window_id"] == 19
        assert "_embedding" not in late
    finally:
        release.set()
        unit.close()
