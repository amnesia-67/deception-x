"""Display-only rPPG heart rate: estimator accuracy, monitor states and wiring."""
from pathlib import Path
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sunhacks import rppg
from sunhacks.capture import LiveCapture, add_frame_tap, remove_frame_tap

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT.parent / "fixtures" / "golden_v1"


def _pulse_trace(bpm=72.0, seconds=11.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(1 / 34, 1 / 26, int(seconds * 30)))  # jittery ~30 fps
    pulse = np.sin(2 * np.pi * bpm / 60 * t) + 0.3 * np.sin(4 * np.pi * bpm / 60 * t + .4)
    illumination = 1 + .02 * np.sin(2 * np.pi * .08 * t)              # slow lighting drift
    rgb = np.stack([150 + .25 * pulse, 105 + .6 * pulse, 85 + .2 * pulse], 1) * illumination[:, None]
    return t, rgb + rng.normal(0, .35, rgb.shape)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pos_estimator_recovers_synthetic_72_bpm_within_3(seed):
    t, rgb = _pulse_trace(72, seed=seed)
    grid, uniform = rppg.resample_uniform(t, rgb, rppg.RESAMPLE_HZ)
    result = rppg.spectral_bpm(rppg.pos_bvp(uniform, rppg.RESAMPLE_HZ), rppg.RESAMPLE_HZ)
    assert abs(result["bpm"] - 72) <= 3
    assert result["quality"] >= .6 and rppg.quality_label(result["quality"]) == "good"


@pytest.mark.parametrize("bpm", [48, 72, 96, 150, 200])
def test_spectral_estimator_covers_the_0_7_to_3_5_hz_band(bpm):
    fs = 30.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * bpm / 60 * t) + np.random.default_rng(bpm).normal(0, .3, len(t))
    assert abs(rppg.spectral_bpm(signal, fs)["bpm"] - bpm) <= 3


def test_noise_only_trace_reports_low_quality():
    noise = np.random.default_rng(5).normal(0, 1, 300)
    assert rppg.spectral_bpm(noise, 30.0)["quality"] < .35


class _FixedFace:
    kind = "test_fixed_box"

    def __init__(self, box=(20.0, 10.0, 100.0, 100.0)):
        self.box = box

    def detect(self, rgb):
        return self.box

    def close(self):
        pass


def _skin_frames(bpm=72.0, seconds=11.0, seed=3):
    t, rgb = _pulse_trace(bpm, seconds, seed)
    base = np.random.default_rng(seed).normal(0, 6, (120, 160, 3))  # static skin texture
    return [np.clip(base + c, 0, 255).astype(np.uint8) for c in rgb], t


def _monitor(clock_box, **config):
    monitor = rppg.HeartRateMonitor({"method": "pos", "auto_download": False, **config},
                                    start_thread=False, clock=lambda: clock_box[0])
    monitor._tracker, monitor._tracker_ready = _FixedFace(), True
    return monitor


def test_monitor_end_to_end_states_and_72_bpm_display():
    now = [0.0]
    monitor = _monitor(now)
    assert monitor.snapshot()["status"] == "idle"
    frames, stamps = _skin_frames()
    seen_acquiring = False
    for frame, stamp in zip(frames, stamps):
        now[0] = stamp
        monitor.offer_frame(frame, stamp)       # capture-thread API: bounded append only
        for ts, rgb in list(monitor._pending):  # drain synchronously (no worker in tests)
            monitor.process_frame(rgb, ts)
        monitor._pending.clear()
        monitor.maybe_compute(stamp)
        state = monitor.snapshot()
        if state["status"] == "acquiring":
            seen_acquiring = True
            assert state["bpm"] is None and state["warming_up"] and 0 <= state["progress"] <= 1
    assert seen_acquiring
    state = monitor.snapshot()
    assert state["status"] == "tracking" and state["display_only"] is True
    assert abs(state["bpm"] - 72) <= 3
    assert state["method"] == "pos" and state["quality_label"] in ("good", "fair")
    assert len(state["pulse"]) == rppg.PULSE_POINTS and max(map(abs, state["pulse"])) <= 1
    assert state["band_hz"] == [0.7, 3.5] and state["window_seconds"] <= rppg.WINDOW_SECONDS + .1
    monitor.capture_stopped()
    assert monitor.snapshot()["status"] == "idle" and monitor.snapshot()["bpm"] is None
    monitor.close()


def test_monitor_goes_idle_without_frames_and_reports_missing_face():
    now = [0.0]
    monitor = _monitor(now)
    monitor._tracker = _FixedFace(None)
    frame = np.full((120, 160, 3), 120, np.uint8)
    for i in range(30):
        now[0] = i / 30
        monitor.offer_frame(frame, now[0])
        monitor.process_frame(frame, now[0])
    assert monitor.snapshot()["status"] == "no_face"
    now[0] += rppg.IDLE_SECONDS + .1
    assert monitor.snapshot()["status"] == "idle"
    monitor.close()


def test_offer_frame_is_bounded_and_never_blocks():
    monitor = rppg.HeartRateMonitor({"method": "pos", "auto_download": False}, start_thread=False)
    frame = np.zeros((4, 4, 3), np.uint8)
    for i in range(rppg.MAX_PENDING + 25):
        monitor.offer_frame(frame, float(i))
    assert len(monitor._pending) == rppg.MAX_PENDING and monitor._dropped == 25
    monitor.close()


def test_live_capture_feeds_taps_without_a_second_camera_and_survives_tap_errors():
    capture = LiveCapture(max_video_frames=10)
    opened = []

    class Camera:
        reads = 0
        def __init__(self, *args):
            opened.append(args)
        def isOpened(self):
            return True
        def set(self, *args):
            return True
        def get(self, prop):
            return 30.0
        def read(self):
            Camera.reads += 1
            if Camera.reads > 5:
                capture._stop.set()
            return True, np.full((48, 64, 3), Camera.reads, np.uint8)
        def release(self):
            pass

    class Recorder:
        def __init__(self):
            self.frames, self.stopped = [], 0
        def offer_frame(self, rgb, timestamp):
            self.frames.append((timestamp, rgb))
        def capture_stopped(self):
            self.stopped += 1

    class Broken:
        def offer_frame(self, rgb, timestamp):
            raise RuntimeError("observer bug")
        def capture_stopped(self):
            raise RuntimeError("observer bug")

    fake = SimpleNamespace(CAP_AVFOUNDATION=1, CAP_ANY=0, VideoCapture=Camera, CAP_PROP_FRAME_WIDTH=3,
                           CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38, COLOR_BGR2RGB=4,
                           INTER_AREA=3, cvtColor=lambda image, code: image[..., ::-1])
    recorder, broken = Recorder(), Broken()
    add_frame_tap(broken); add_frame_tap(recorder)
    try:
        with patch("sunhacks.capture.media_authorization_status", return_value={"camera": "authorized", "audio": "authorized"}), \
             patch.dict("sys.modules", {"cv2": fake}):
            capture._camera_worker()
    finally:
        remove_frame_tap(broken); remove_frame_tap(recorder)
    video, _ = capture.drain()
    assert len(opened) == 1                                   # the app's only camera open
    assert len(recorder.frames) == len(video) == 5            # same frames as the pipeline queue
    assert all(a[1] is b.rgb for a, b in zip(recorder.frames, video))
    assert recorder.stopped == 1


def test_heart_rate_is_display_only_and_not_read_by_scoring_code():
    scoring = ["heads.py", "teammate_fusion.py", "local_stack.py", "local_stack_process.py", "split_runtime.py",
               "fast_runtime.py", "showcase.py", "answer_pipeline.py", "claim_runtime.py", "local_digest.py",
               "features.py", "models.py", "pilot_audio.py"]
    for name in scoring:
        text = (ROOT / "sunhacks" / name).read_text()
        assert not re.search(r"heart_rate|rppg|\bbpm\b", text, re.I), name


def test_pipeline_state_exposes_heart_rate_beside_capture():
    from sunhacks.pipeline import Pipeline
    pipeline = Pipeline({})
    assert pipeline.state()["heart_rate"] is None
    pipeline.heart_rate = rppg.HeartRateMonitor({"method": "pos", "auto_download": False}, start_thread=False)
    value = pipeline.state()["heart_rate"]
    assert value["status"] == "idle" and value["display_only"] is True and value["bpm"] is None
    pipeline.close()


def test_manifest_pins_hash_verified_weights():
    info = rppg.manifest()
    assert info["parameters"] < 5_000_000 and "Privacy Protection Addendum" in info["license"]
    for entry in info["assets"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert entry["url"].startswith("https://") and entry["bytes"] <= entry["max_bytes"]
    assert re.search(r"/[0-9a-f]{40}/", info["assets"]["facephys_onnx"]["url"])  # pinned commit


def test_fetch_rejects_hash_mismatch_and_keeps_cache_clean(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"not the model")
    entry = {"filename": "x.onnx", "url": source.as_uri(), "sha256": "0" * 64, "max_bytes": 1024}
    with pytest.raises(ValueError, match="sha256 mismatch"):
        rppg.fetch_asset(entry, directory=tmp_path / "cache")
    assert not any((tmp_path / "cache").iterdir())
    import hashlib
    entry["sha256"] = hashlib.sha256(b"not the model").hexdigest()
    assert rppg.fetch_asset(entry, directory=tmp_path / "cache").read_bytes() == b"not the model"
    with pytest.raises(FileNotFoundError):
        rppg.fetch_asset({**entry, "filename": "missing.onnx"}, directory=tmp_path / "cache", allow_download=False)


def _cached(name):
    entry = rppg.manifest()["assets"][name]
    path = rppg.cache_dir() / entry["filename"]
    return path if path.is_file() and rppg.sha256_file(path) == entry["sha256"] else None


def test_facephys_streaming_model_when_cached():
    pytest.importorskip("onnxruntime")
    model_path, state_path = _cached("facephys_onnx"), _cached("facephys_state")
    if model_path is None or state_path is None:
        pytest.skip("FacePhys weights not cached; run `python -m sunhacks fetch-rppg`")
    model = rppg.FacePhysStream(model_path, state_path)
    assert len(model.state_names) == 46
    frame = np.full((36, 36, 3), 128, np.uint8)
    first = [model.step(frame, 1 / 30) for _ in range(5)]
    model.reset()
    assert [model.step(frame, 1 / 30) for _ in range(5)] == pytest.approx(first)
    assert all(np.isfinite(first))


def test_golden_fixture_frames_run_through_the_monitor():
    """No ground-truth pulse exists for the golden frames; this checks the path runs."""
    from PIL import Image
    import json
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((GOLDEN / "frames").glob("*.jpg"))]
    stamps = json.loads((GOLDEN / "timestamps.json").read_text())
    result = rppg.analyze_frames(frames, stamps, config={"method": "pos", "auto_download": False})
    assert result["status"] in ("tracking", "acquiring", "no_face")
    assert result["display_only"] is True
    if result["bpm"] is not None:
        assert 42 <= result["bpm"] <= 210


def test_missing_weights_fall_back_to_pos(tmp_path):
    now = [0.0]
    monitor = _monitor(now, method="facephys", cache_dir=str(tmp_path))  # empty cache, no download
    monitor._ensure_model_loading()
    monitor._model_thread.join(timeout=30)
    assert monitor.snapshot()["model"]["status"] == "unavailable"
    frames, stamps = _skin_frames(seconds=9)
    for frame, stamp in zip(frames, stamps):
        now[0] = stamp
        monitor.offer_frame(frame, stamp)
        monitor.process_frame(frame, stamp)
        monitor.maybe_compute(stamp)
    state = monitor.snapshot()
    assert state["status"] == "tracking" and state["method"] == "pos" and abs(state["bpm"] - 72) <= 3
    monitor.close()
