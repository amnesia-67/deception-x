from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sunhacks.capture import LiveCapture, StreamingResampler


def test_resampler_keeps_contiguous_sample_clock():
    rate = 48000
    resampler = StreamingResampler(rate)
    chunks = []
    for index in range(10):
        positions = np.arange(index * 4800, (index + 1) * 4800)
        chunk = resampler.push(np.sin(2 * np.pi * 440 * positions / rate).astype(np.float32), 100 + index * .1)
        if chunk is not None:
            chunks.append(chunk)
    assert sum(len(chunk.samples) for chunk in chunks) >= 15900
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start == pytest.approx(previous.start + len(previous.samples) / 16000, abs=1e-8)
    values = np.concatenate([chunk.samples for chunk in chunks])
    reference = np.sin(2 * np.pi * 440 * np.arange(len(values)) / 16000)
    assert np.mean(np.abs(values[100:] - reference[100:])) < .003
    assert len(resampler.buffer) < 400


def test_resampler_antialiases_high_frequency():
    rate = 48000
    samples = np.sin(2 * np.pi * 12000 * np.arange(rate) / rate).astype(np.float32)
    chunk = StreamingResampler(rate).push(samples, 0)
    assert chunk.sample_rate == 16000
    assert np.sqrt(np.mean(chunk.samples[100:-100] ** 2)) < .01


def test_resampler_resets_on_capture_gap():
    resampler = StreamingResampler(16000)
    a = resampler.push(np.zeros(1600), 10)
    b = resampler.push(np.zeros(1600), 11)
    assert a.start == 10
    assert b.start == 11
    assert resampler.discontinuities == 1


def test_resampler_44100_and_empty():
    resampler = StreamingResampler(44100)
    assert resampler.push(np.empty(0), 0) is None
    chunks = [resampler.push(np.ones(4410, dtype=np.float32), i * .1) for i in range(4)]
    assert all(chunk.sample_rate == 16000 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start == pytest.approx(previous.start + len(previous.samples) / 16000)
    assert np.mean(np.concatenate([chunk.samples for chunk in chunks])[100:]) == pytest.approx(1, abs=.002)


def test_callback_queue_is_bounded_and_drain_consumes():
    capture = LiveCapture(max_audio_chunks=2)
    capture._resampler = StreamingResampler(16000)
    stamps = SimpleNamespace(currentTime=5, inputBufferAdcTime=4.9)
    with patch("sunhacks.capture.time.monotonic", return_value=10):
        for _ in range(3):
            capture._audio_callback(np.ones((160, 1), dtype=np.float32), 160, stamps, None)
    status = capture.snapshot()["audio"]
    assert status["queued_chunks"] == 2
    assert status["queue_dropped"] == 1
    assert status["clipped_samples"] == 480
    frames, audio = capture.drain()
    assert frames == []
    assert audio[0].start == pytest.approx(9.9)
    assert capture.snapshot()["audio"]["queued_chunks"] == 0
    assert capture.drain() == ([], [])


def test_missing_devices_remain_missing():
    capture = LiveCapture()
    with patch.dict("sys.modules", {"cv2": None}):
        capture._camera_worker()
    assert capture.snapshot()["camera"]["status"] == "missing"
    assert capture.drain() == ([], [])
    assert capture.snapshot()["camera"]["measured_capture_fps"] is None


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        LiveCapture(max_video_frames=0)
    with pytest.raises(ValueError):
        StreamingResampler(0)


def test_mac_permission_preflight_does_not_open_unauthorized_device():
    capture = LiveCapture()
    with patch("sunhacks.capture.media_authorization_status", return_value={"camera":"not_determined", "audio":"denied"}), \
         patch.dict("sys.modules", {"cv2": None, "sounddevice": None}):
        capture._camera_worker()
        capture._audio_worker()
    state = capture.snapshot()
    assert state["camera"]["status"] == "missing"
    assert "authorize-camera" in state["camera"]["error"]
    assert "System Settings" in state["audio"]["error"]


def test_camera_open_completing_after_stop_cannot_start_or_enqueue():
    capture = LiveCapture()
    class LateCamera:
        released = False
        def __init__(self, *args):
            capture._stop.set()
        def release(self):
            self.released = True
        def isOpened(self):
            raise AssertionError("Late camera must not be activated")
    fake = SimpleNamespace(CAP_AVFOUNDATION=1, CAP_ANY=0, VideoCapture=LateCamera)
    with patch("sunhacks.capture.media_authorization_status", return_value={"camera":"authorized", "audio":"authorized"}), \
         patch.dict("sys.modules", {"cv2": fake}):
        capture._camera_worker()
    assert capture.snapshot()["camera"]["status"] == "stopped"
    assert capture.drain() == ([], [])


def test_callback_after_stop_is_ignored():
    capture = LiveCapture()
    capture._stop.set()
    capture._audio_callback(np.ones((160, 1), dtype=np.float32),160,SimpleNamespace(currentTime=1,inputBufferAdcTime=.9),None)
    assert capture.snapshot()["audio"]["native_samples_received"] == 0
    assert capture.drain() == ([], [])


def test_starting_status_and_fps_alias():
    capture = LiveCapture()
    with patch("sunhacks.capture.threading.Thread") as thread:
        thread.return_value.is_alive.return_value = False
        capture.start()
        assert capture.snapshot()["camera"]["status"] == "starting"
        assert capture.snapshot()["audio"]["status"] == "starting"
        capture._video_times.extend([10.0,10.04,10.08])
        assert capture.snapshot()["camera"]["observed_fps"] == pytest.approx(25)
        capture.stop()
    assert capture.snapshot()["camera"]["status"] == "stopped"


def test_restart_rejected_when_previous_worker_is_still_opening():
    capture = LiveCapture()
    worker = SimpleNamespace(is_alive=lambda: True)
    capture._threads = [worker]
    with pytest.raises(RuntimeError, match="still stopping"):
        capture.start()


def test_fixed_adc_mapping_ignores_callback_scheduling_jitter():
    capture = LiveCapture()
    capture._resampler = StreamingResampler(48000)
    jitter = [0, .03, -.01, .09, .002, .05, -.02, .15]
    outputs = []
    for index, delay in enumerate(jitter):
        current = 5 + index * .1
        stamp = SimpleNamespace(currentTime=current, inputBufferAdcTime=current - .1)
        with patch("sunhacks.capture.time.monotonic", return_value=10 + index * .1 + delay):
            capture._audio_callback(np.zeros((4800,1),dtype=np.float32),4800,stamp,None)
        _, chunks = capture.drain()
        outputs.extend(chunks)
    assert capture._resampler.discontinuities == 0
    for previous, current in zip(outputs, outputs[1:]):
        assert current.start == pytest.approx(previous.start + len(previous.samples) / 16000,abs=1e-8)
    state = capture.snapshot()["audio"]
    assert state["adc_timeline_gap_callbacks"] == 0
    assert state["fixed_portaudio_to_monotonic_offset_seconds"] == 5
    assert state["maximum_callback_scheduling_delta_seconds"] == pytest.approx(.15)
    assert state["timestamp_fallback_callbacks"] == 0


def test_fixed_adc_mapping_preserves_real_adc_gap():
    capture = LiveCapture()
    capture._resampler = StreamingResampler(16000)
    outputs = []
    for index, adc in enumerate([4.9,5.0,5.2,5.3]):
        with patch("sunhacks.capture.time.monotonic",return_value=10+index*.1):
            capture._audio_callback(np.zeros((1600,1),dtype=np.float32),1600,
                                    SimpleNamespace(currentTime=adc+.1,inputBufferAdcTime=adc),None)
        outputs.extend(capture.drain()[1])
    assert capture._resampler.discontinuities == 1
    assert capture.snapshot()["audio"]["adc_timeline_gap_callbacks"] == 1
    assert outputs[2].start == pytest.approx(10.2)
    assert outputs[2].start-outputs[1].start == pytest.approx(.2)


def test_overflow_explicitly_breaks_resampling_continuity():
    capture = LiveCapture()
    capture._resampler = StreamingResampler(16000)
    for index in range(2):
        with patch("sunhacks.capture.time.monotonic",return_value=10+index*.1):
            capture._audio_callback(np.zeros((1600,1),dtype=np.float32),1600,
                                    SimpleNamespace(currentTime=5+index*.1,inputBufferAdcTime=4.9+index*.1),
                                    SimpleNamespace(input_overflow=index==1))
        capture.drain()
    assert capture._resampler.discontinuities == 1
    assert capture.snapshot()["audio"]["overflow_callbacks"] == 1


def test_invalid_adc_reports_estimated_timestamp_fallback():
    capture = LiveCapture()
    capture._resampler = StreamingResampler(16000)
    with patch("sunhacks.capture.time.monotonic", return_value=10):
        capture._audio_callback(np.zeros((1600,1),dtype=np.float32),1600,
                                SimpleNamespace(currentTime=0,inputBufferAdcTime=0),None)
    state = capture.snapshot()["audio"]
    assert state["timestamp_fallback_callbacks"] == 1
    assert "estimated" in state["last_timestamp_quality"]
    assert capture.drain()[1][0].start == pytest.approx(9.9)
