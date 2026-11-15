"""Local capture with bounded queues and a shared monotonic host clock.

Camera timestamps are host receipt timestamps, not exposure timestamps. Audio ADC
clock is mapped using one fixed host offset per stream; uncertainty is reported.
Nothing in this module writes media to disk or replaces missing media with mocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import gcd
import platform
import threading
import time
from typing import Any

import numpy as np

from .temporal import AudioChunk, VideoFrame

# Display-only frame observers (the rPPG heart-rate monitor). They receive the
# same read-only RGB frames already queued for the pipeline; no second camera is
# opened. Observers must return immediately (bounded append) and never raise.
_FRAME_TAPS: list[Any] = []
_FRAME_TAPS_LOCK = threading.Lock()


def add_frame_tap(tap: Any) -> None:
    """Register an object with ``offer_frame(rgb, timestamp)`` and ``capture_stopped()``."""
    with _FRAME_TAPS_LOCK:
        if tap not in _FRAME_TAPS:
            _FRAME_TAPS.append(tap)


def remove_frame_tap(tap: Any) -> None:
    with _FRAME_TAPS_LOCK:
        if tap in _FRAME_TAPS:
            _FRAME_TAPS.remove(tap)


def _frame_taps() -> tuple[Any, ...]:
    with _FRAME_TAPS_LOCK:
        return tuple(_FRAME_TAPS)


# PortAudio caches CoreAudio device handles for the lifetime of the backend.
# All owned microphone streams share this lease with the explicit idle reset.
_AUDIO_BACKEND_LOCK = threading.Lock()


def reconnect_microphone(audio_device=None):
    """Refresh idle PortAudio and count a brief local check; retain no audio.

    sounddevice 0.5.6 (pinned) has no public hotplug refresh. Its paired backend
    helpers are used only with a known initialization count and no owned stream.
    No macOS permission prompt or alternate-device fallback is attempted.
    """
    if media_authorization_status()["audio"] not in ("authorized","not_applicable"):
        raise RuntimeError("Authorize microphone access before reconnecting it")
    if not _AUDIO_BACKEND_LOCK.acquire(blocking=False):
        raise RuntimeError("Stop the active microphone stream before reconnecting")
    try:
        import sounddevice as sd
        count=getattr(sd,"_initialized",None)
        if type(count) is not int or count not in (0,1) or not callable(getattr(sd,"_initialize",None)) or not callable(getattr(sd,"_terminate",None)):
            raise RuntimeError("Restart the app to refresh this audio backend version")
        selected=audio_device
        if type(selected) is int:selected=sd.query_devices(selected,kind="input")["name"]
        if count:sd._terminate()
        sd._initialize()
        device=sd.query_devices(selected,kind="input")
        rate=int(round(device["default_samplerate"]))
        if rate<8000:raise RuntimeError("The microphone has no valid native sample rate")
        ready=threading.Event();observed={"samples":0,"overflow":False}
        def callback(data,frames,times,status):
            observed["samples"]+=frames
            observed["overflow"]|=bool(status)
            if observed["samples"]>=rate//4:ready.set()
        with sd.InputStream(device=selected,samplerate=rate,channels=1,dtype="float32",
                            blocksize=max(1,rate//10),callback=callback):
            if not ready.wait(1.5):raise RuntimeError("Microphone opened but delivered no complete check; reconnect its USB cable and try again")
        if observed["overflow"]:raise RuntimeError("Microphone check overflowed; stop other audio work and retry")
        return {"status":"ready","device_name":device["name"],"sample_rate":rate,
                "samples_received":observed["samples"],"audio_saved":False,
                "message":"Microphone reconnected and checked locally. No audio was saved. Try Add new data again."}
    finally:_AUDIO_BACKEND_LOCK.release()


@dataclass(frozen=True)
class _NativeAudio:
    start: float
    rate: int
    samples: np.ndarray
    discontinuity: bool = False


class StreamingResampler:
    """Polyphase resampling with overlap and deferred edge samples.

    Keeping the source offset divisible by the rational denominator preserves
    resampling phase between drains. A small tail waits for the next block so
    zeros are not repeatedly inserted at callback boundaries.
    """
    def __init__(self, source_rate: int, target_rate: int = 16000):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive")
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        factor = gcd(self.source_rate, self.target_rate)
        self.up, self.down = self.target_rate // factor, self.source_rate // factor
        self.guard = ((32 * max(self.up, self.down) // self.up + self.down - 1) // self.down) * self.down
        self.buffer = np.empty(0, dtype=np.float32)
        self.input_offset = 0
        self.emitted = 0
        self.total_input = 0
        self.start: float | None = None
        self.discontinuities = 0

    def push(self, samples: np.ndarray, start: float, *, discontinuity: bool = False) -> AudioChunk | None:
        from scipy.signal import resample_poly
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return None
        if not np.isfinite(values).all() or not np.isfinite(start):
            raise ValueError("Audio samples and timestamps must be finite")
        expected = None if self.start is None else self.start + self.total_input / self.source_rate
        # With a fixed clock mapping, this threshold detects ADC timeline gaps,
        # not Python callback scheduling delay. Overflow explicitly breaks FIR
        # continuity even when a driver does not advance its ADC timestamp.
        if discontinuity or (expected is not None and abs(start - expected) > 0.005):
            self.buffer = np.empty(0, dtype=np.float32)
            self.input_offset = self.emitted = self.total_input = 0
            self.start = None
            self.discontinuities += 1
        if self.start is None:
            self.start = float(start)
        self.buffer = np.concatenate((self.buffer, values))
        self.total_input += values.size
        if self.source_rate == self.target_rate:
            output = self.buffer.copy()
            chunk_start = self.start + self.emitted / self.target_rate
            self.emitted += output.size
            self.input_offset = self.total_input
            self.buffer = np.empty(0, dtype=np.float32)
            return AudioChunk(chunk_start, self.target_rate, output)
        safe_end = max(self.emitted, (self.total_input - self.guard) * self.up // self.down)
        if safe_end <= self.emitted:
            return None
        filtered = resample_poly(self.buffer, self.up, self.down).astype(np.float32)
        output_offset = self.input_offset * self.up // self.down
        begin, end = self.emitted - output_offset, safe_end - output_offset
        output = filtered[begin:end].copy()
        chunk_start = self.start + self.emitted / self.target_rate
        self.emitted += output.size
        # Retain enough pre-history for the next output's FIR support.
        keep_source = max(self.input_offset, self.emitted * self.down // self.up - self.guard)
        keep_source = keep_source // self.down * self.down
        discard = keep_source - self.input_offset
        self.buffer = self.buffer[discard:].copy()
        self.input_offset = keep_source
        return AudioChunk(chunk_start, self.target_rate, output)


def media_authorization_status() -> dict[str, str]:
    """Read macOS AVFoundation authorization without requesting or changing it.

    The returned permission is for this process/launching app. A sandboxed probe
    can differ from a terminal launch; neither probe grants capture access.
    """
    if platform.system() != "Darwin":
        return {"camera": "not_applicable", "audio": "not_applicable"}
    try:
        import ctypes
        ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        string = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
            ("objc_msgSend", objc))
        authorization = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc))
        mapping = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}
        result = {}
        for key, media in (("camera", b"vide"), ("audio", b"soun")):
            media_type = string(objc.objc_getClass(b"NSString"),
                                objc.sel_registerName(b"stringWithUTF8String:"), media)
            value = authorization(objc.objc_getClass(b"AVCaptureDevice"),
                                  objc.sel_registerName(b"authorizationStatusForMediaType:"), media_type)
            result[key] = mapping.get(value, "unknown")
        return result
    except Exception:
        # Other platforms or unavailable APIs should still reach device checks.
        return {"camera": "unknown", "audio": "unknown"}


def _permission_message(modality: str, permission: str) -> str:
    settings = "Camera" if modality == "camera" else "Microphone"
    if permission == "not_determined":
        return (f"macOS {settings} authorization has not been requested. Run the explicit "
                f"authorize-{modality} command from a terminal on its main thread before starting live capture.")
    return (f"macOS {settings} authorization is {permission} for this process/launching app. "
            f"Check System Settings > Privacy & Security > {settings}, enable the launching app, "
            "then restart it. Capture does not change permissions.")


def camera_authorization_probe(camera_index: int = 0) -> dict[str, Any]:
    """Explicit main-thread camera open/read; a pending OS prompt belongs to user.

    This one-shot probe never stores frames. It does not bypass denied/restricted
    permissions. Opening an undetermined device may wait for the user to respond
    to macOS; do not call it from the HTTP server or a worker thread.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Camera authorization probe must run on the process main thread")
    permission = media_authorization_status()["camera"]
    if permission in {"denied", "restricted"}:
        return {"status": "missing", "authorization": permission,
                "error": _permission_message("camera", permission)}
    import cv2
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        return {"status": "available" if ok else "missing",
                "authorization": media_authorization_status()["camera"],
                "frame_received": bool(ok), "shape": list(frame.shape) if ok else None,
                "note": "One frame read into memory and discarded; no recording."}
    finally:
        cap.release()


def microphone_authorization_probe(audio_device: int | str | None = None) -> dict[str, Any]:
    """Explicit main-thread one-shot audio input check; no recording or storage."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Microphone authorization probe must run on the process main thread")
    permission = media_authorization_status()["audio"]
    if permission in {"denied", "restricted"}:
        return {"status": "missing", "authorization": permission,
                "error": _permission_message("audio", permission)}
    import sounddevice as sd
    device = sd.query_devices(audio_device, kind="input")
    rate = int(round(device["default_samplerate"]))
    with sd.InputStream(device=audio_device, samplerate=rate, channels=1, dtype="float32") as stream:
        samples, overflowed = stream.read(max(1, rate // 4))
    return {"status": "available", "authorization": media_authorization_status()["audio"],
            "device_name": device["name"], "sample_rate": rate, "samples_received": int(samples.size),
            "overflowed": bool(overflowed), "note": "Quarter-second input read into memory and discarded; no recording."}


def probe_devices() -> dict[str, Any]:
    """Query audio metadata without opening camera or microphone streams."""
    result: dict[str, Any] = {
        "camera": {"status": "not_opened", "note": "Camera enumeration requires opening an AVFoundation device; use live start to test."},
        "audio": {"status": "unavailable", "devices": []},
        "authorization": media_authorization_status(),
    }
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        result["audio"] = {"status": "available" if any(d["max_input_channels"] > 0 for d in devices) else "missing",
                           "devices": [{"index": i, "name": d["name"], "input_channels": d["max_input_channels"],
                                        "default_sample_rate": d["default_samplerate"]}
                                       for i, d in enumerate(devices) if d["max_input_channels"] > 0]}
    except Exception as exc:
        result["audio"]["error"] = f"{type(exc).__name__}: {exc}"
    return result


class LiveCapture:
    def __init__(self, camera_index: int = 0, audio_device: int | str | None = None,
                 camera_fps: float = 30.0, width: int = 1280, height: int = 720,
                 max_frame_edge: int = 640, max_video_frames: int = 180,
                 max_audio_chunks: int = 120, audio_sample_rate: int = 16000):
        if camera_fps <= 0 or width <= 0 or height <= 0 or max_frame_edge <= 0:
            raise ValueError("Camera dimensions and FPS must be positive")
        if max_video_frames < 1 or max_audio_chunks < 1 or audio_sample_rate != 16000:
            raise ValueError("Queues must be bounded and output audio must be 16kHz")
        self.camera_index, self.audio_device = camera_index, audio_device
        self.camera_fps, self.width, self.height = camera_fps, width, height
        self.max_frame_edge, self.audio_sample_rate = max_frame_edge, audio_sample_rate
        self._video: deque[VideoFrame] = deque(maxlen=max_video_frames)
        self._audio: deque[_NativeAudio] = deque(maxlen=max_audio_chunks)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stream: Any = None
        self._cap: Any = None
        self._resampler: StreamingResampler | None = None
        self._audio_clock_offset: float | None = None
        self._audio_expected_adc: float | None = None
        self._running = False
        self._sequence = 0
        self._video_times: deque[float] = deque(maxlen=120)
        self._status: dict[str, Any] = {
            "camera": {"status": "stopped", "requested_fps": camera_fps,
                       "timestamp_source": "host_monotonic_after_read", "exposure_timestamp_available": False,
                       "frames_received": 0, "queue_dropped": 0},
            "audio": {"status": "stopped", "output_sample_rate": audio_sample_rate,
                      "timestamp_source": "PortAudio_inputBufferAdcTime_plus_fixed_host_monotonic_offset",
                      "clock_mapping_method": "offset = host_monotonic_at_first_valid_callback - PortAudio_currentTime; retained for stream lifetime",
                      "clock_mapping_uncertainty": "Fixed initial callback scheduling bias, host API clock precision and possible clock drift; not hardware AV synchronization",
                      "timestamp_fallback_callbacks": 0, "adc_timeline_gap_callbacks": 0,
                      "maximum_callback_scheduling_delta_seconds": 0.0,
                      "native_samples_received": 0, "queue_dropped": 0, "overflow_callbacks": 0,
                      "clipped_samples": 0},
        }

    def start(self) -> None:
        if self._running:
            return
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Previous capture device initialization is still stopping; restart the process after authorization")
        self._stop.clear()
        self._running = True
        with self._lock:
            self._video_times.clear()
            self._audio_clock_offset = None
            self._audio_expected_adc = None
            for state in self._status.values():
                state.update(status="starting", started_monotonic=time.monotonic())
                state.pop("error", None)
        self._threads = [threading.Thread(target=self._camera_worker, name="sunhacks-camera", daemon=True),
                         threading.Thread(target=self._audio_worker, name="sunhacks-microphone", daemon=True)]
        for thread in self._threads:
            thread.start()

    def _camera_worker(self) -> None:
        cap = None
        try:
            if self._stop.is_set():
                return
            permission = media_authorization_status()["camera"]
            with self._lock:
                self._status["camera"]["authorization"] = permission
            if permission in {"not_determined", "denied", "restricted"}:
                raise RuntimeError(_permission_message("camera", permission))
            import cv2
            backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, backend)
            self._cap = cap
            if self._stop.is_set():
                return
            if not cap.isOpened():
                raise RuntimeError("Camera unavailable or permission denied")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            with self._lock:
                if self._stop.is_set():
                    return
                self._status["camera"].update(status="running", reported_device_fps=cap.get(cv2.CAP_PROP_FPS),
                                               reported_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                               reported_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            failures = 0
            while not self._stop.is_set():
                ok, bgr = cap.read()
                stamp = time.monotonic()
                if self._stop.is_set():
                    break
                if not ok:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("Camera stopped returning frames")
                    self._stop.wait(0.05)
                    continue
                failures = 0
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                scale = min(1.0, self.max_frame_edge / max(rgb.shape[:2]))
                if scale < 1:
                    rgb = cv2.resize(rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
                                     interpolation=cv2.INTER_AREA)
                rgb = np.ascontiguousarray(rgb)
                rgb.setflags(write=False)
                with self._lock:
                    if self._stop.is_set():
                        break
                    if len(self._video) == self._video.maxlen:
                        self._status["camera"]["queue_dropped"] += 1
                    self._video.append(VideoFrame(stamp, rgb, self._sequence))
                    self._sequence += 1
                    self._video_times.append(stamp)
                    self._status["camera"]["frames_received"] += 1
                    self._status["camera"]["last_frame_monotonic"] = stamp
                for tap in _frame_taps():
                    try:
                        tap.offer_frame(rgb, stamp)
                    except Exception:
                        pass  # a display-only observer can never stop capture
        except Exception as exc:
            with self._lock:
                if not self._stop.is_set():
                    self._status["camera"].update(status="missing", error=f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
                if self._cap is cap:
                    self._cap = None
            for tap in _frame_taps():
                try:
                    tap.capture_stopped()
                except Exception:
                    pass
            with self._lock:
                if self._stop.is_set():
                    self._status["camera"]["status"] = "stopped"

    def _audio_worker(self) -> None:
        stream = None
        if not _AUDIO_BACKEND_LOCK.acquire(blocking=False):
            with self._lock:self._status["audio"].update(status="missing",error="Another microphone stream or reconnect is still active")
            return
        try:
            if self._stop.is_set():
                return
            permission = media_authorization_status()["audio"]
            with self._lock:
                self._status["audio"]["authorization"] = permission
            if permission in {"not_determined", "denied", "restricted"}:
                raise RuntimeError(_permission_message("audio", permission))
            import sounddevice as sd
            device = sd.query_devices(self.audio_device, kind="input")
            source_rate = int(round(device["default_samplerate"]))
            if source_rate < 8000:
                raise RuntimeError("Microphone has no valid native sample rate")
            if self._stop.is_set():
                return
            self._resampler = StreamingResampler(source_rate, self.audio_sample_rate)
            stream = sd.InputStream(device=self.audio_device, samplerate=source_rate,
                                          channels=1, dtype="float32", blocksize=max(1, source_rate // 10),
                                          callback=self._audio_callback)
            self._stream = stream
            if self._stop.is_set():
                return
            stream.start()
            with self._lock:
                if self._stop.is_set():
                    return
                self._status["audio"].update(status="running", native_sample_rate=source_rate,
                                              device_name=device["name"], latency_seconds=float(stream.latency),
                                              callback_block_seconds=0.1,
                                              recommended_av_alignment_delay_seconds=0.2)
            while not self._stop.wait(0.1):
                if not stream.active:
                    raise RuntimeError("Microphone stream became inactive")
        except Exception as exc:
            with self._lock:
                if not self._stop.is_set():
                    self._status["audio"].update(status="missing", error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if stream is not None:
                    try:stream.stop()
                    finally:
                        try:stream.close()
                        finally:
                            if self._stream is stream:self._stream=None
            finally:
                with self._lock:
                    if self._stop.is_set():self._status["audio"]["status"]="stopped"
                _AUDIO_BACKEND_LOCK.release()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # Callback only copies into the bounded queue; DSP runs in drain().
        if self._stop.is_set():
            return
        now = time.monotonic()
        rate = self._resampler.source_rate if self._resampler is not None else 48000
        current = float(time_info.currentTime)
        adc = float(time_info.inputBufferAdcTime)
        valid_adc = bool(np.isfinite(current) and np.isfinite(adc) and current > 0 and 0 <= adc <= current)
        overflow = bool(getattr(status, "input_overflow", bool(status))) if status is not None else False
        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        with self._lock:
            if self._stop.is_set():
                return
            audio_status = self._status["audio"]
            if valid_adc:
                if self._audio_clock_offset is None:
                    # Anchor exactly once. Recomputing this offset each callback
                    # would turn Python/GPU scheduling jitter into false gaps.
                    self._audio_clock_offset = now - current
                    audio_status["fixed_portaudio_to_monotonic_offset_seconds"] = self._audio_clock_offset
                stamp = adc + self._audio_clock_offset
                callback_delta = now - (current + self._audio_clock_offset)
                audio_status["maximum_callback_scheduling_delta_seconds"] = max(
                    audio_status["maximum_callback_scheduling_delta_seconds"], abs(callback_delta))
                if self._audio_expected_adc is not None and abs(adc - self._audio_expected_adc) > 0.005:
                    audio_status["adc_timeline_gap_callbacks"] += 1
                self._audio_expected_adc = adc + frames / rate
                audio_status["last_timestamp_quality"] = "ADC clock with fixed initial host offset"
            else:
                # No valid ADC clock is available: expose the receipt-based
                # estimate rather than silently inventing sample-clock timing.
                stamp = now - frames / rate
                self._audio_expected_adc = None
                audio_status["timestamp_fallback_callbacks"] += 1
                audio_status["last_timestamp_quality"] = "estimated from callback receipt; ADC timestamp unavailable"
            if len(self._audio) == self._audio.maxlen:
                audio_status["queue_dropped"] += 1
            self._audio.append(_NativeAudio(stamp, rate, mono, discontinuity=overflow))
            audio_status["native_samples_received"] += mono.size
            audio_status["clipped_samples"] += int(np.count_nonzero(np.abs(mono) >= 0.999))
            audio_status["last_chunk_monotonic"] = stamp
            if overflow:
                audio_status["overflow_callbacks"] += 1
            if status:
                audio_status["last_callback_status"] = str(status)

    def drain(self) -> tuple[list[VideoFrame], list[AudioChunk]]:
        with self._lock:
            video = list(self._video)
            audio_native = list(self._audio)
            self._video.clear()
            self._audio.clear()
        audio: list[AudioChunk] = []
        if self._resampler is not None:
            for native in audio_native:
                try:
                    chunk = self._resampler.push(native.samples, native.start, discontinuity=native.discontinuity)
                    if chunk is not None:
                        audio.append(chunk)
                except Exception as exc:
                    with self._lock:
                        self._status["audio"].update(status="error", error=f"Resampling failed: {type(exc).__name__}: {exc}")
        return video, audio

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = {key: dict(value) for key, value in self._status.items()}
            times = list(self._video_times)
            result["camera"]["queued_frames"] = len(self._video)
            result["audio"]["queued_chunks"] = len(self._audio)
            result["audio"]["resampling_discontinuities"] = self._resampler.discontinuities if self._resampler else 0
            result["camera"]["measured_capture_fps"] = ((len(times) - 1) / (times[-1] - times[0])
                                                          if len(times) > 1 and times[-1] > times[0] else None)
            result["camera"]["observed_fps"] = result["camera"]["measured_capture_fps"]
            for state in result.values():
                if state["status"] == "starting":
                    state["startup_elapsed_seconds"] = time.monotonic() - state["started_monotonic"]
                    state["note"] = "Opening capture device; if this stalls, use explicit main-thread authorization probe."
        return result

    def stop(self) -> None:
        self._stop.set()
        deadline = time.monotonic() + 2.0
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._running = False
        with self._lock:
            for key, state in self._status.items():
                if state["status"] in {"starting", "running"}:
                    worker_name = "sunhacks-camera" if key == "camera" else "sunhacks-microphone"
                    alive = any(thread.name == worker_name and thread.is_alive() for thread in self._threads)
                    state["status"] = "stopping" if alive else "stopped"
                    if alive:
                        state["note"] = "Device open/authorization is still blocked; no late frames are accepted. Restart process after resolving access."
            self._video.clear()
            self._audio.clear()
