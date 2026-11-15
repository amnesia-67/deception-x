"""Display-only remote photoplethysmography (rPPG) heart rate from the live camera.

The monitor taps frames that :class:`sunhacks.capture.LiveCapture` already
receives; it never opens a camera. Frames are handed over with an O(1) bounded
append and all work (face detection, FacePhys inference, POS fallback, spectral
BPM) runs on one daemon worker thread, off the capture and deception paths.

The BPM is a dashboard readout only. Nothing here is read by fusion, stack,
tabular or showcase scoring.

Model: FacePhys (Wang et al., arXiv:2512.06275) ONNX export from
https://github.com/KegangWangCCNU/FacePhys-Release, MIT with a Privacy
Protection Addendum: inference stays on this Mac and derived metrics are not
transmitted. Weights are fetched on first use into a local cache and verified
against the sha256 values in ``models_manifest.json``. When the weights are
unavailable (offline, disabled, hash mismatch, onnxruntime missing) the
unsupervised POS algorithm (Wang et al., IEEE TBME 2017) supplies the pulse.
"""
from __future__ import annotations

from collections import deque
import functools
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from typing import Any
import urllib.request

import numpy as np

BAND_HZ = (0.7, 3.5)          # 42-210 BPM
WINDOW_SECONDS = 10.0         # rolling analysis window
MIN_SECONDS = 6.0             # warm-up before the first BPM
RESAMPLE_HZ = 30.0            # uniform rate used for spectral analysis
UPDATE_SECONDS = 1.0          # ~1 Hz estimates
IDLE_SECONDS = 2.5            # no frames for this long => camera considered off
FACE_LOST_SECONDS = 1.5       # no face for this long => reset the window
DETECT_EVERY = 3              # run the face detector on every Nth frame
MAX_PENDING = 90              # frames queued between capture and worker (~3 s)
PULSE_SECONDS = 6.0
PULSE_POINTS = 90
FACEPHYS_INPUT = 36
NOTE = "Display only. Heart rate never feeds lie/truth scores."


# --------------------------------------------------------------------------
# Weights: manifest + cache + sha256-verified download
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def manifest() -> dict[str, Any]:
    data = json.loads(Path(__file__).with_name("models_manifest.json").read_text())
    return data["rppg"]


BUNDLED_DIR = Path(__file__).resolve().parent / "rppg_weights"


def cache_dir() -> Path:
    override = os.environ.get("SUNHACKS_RPPG_CACHE")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "sunhacks" / "rppg"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sunhacks" / "rppg"


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_asset(entry: dict[str, Any], *, directory: Path | None = None, timeout: float = 60.0,
                allow_download: bool = True) -> Path:
    """Return a cached, hash-verified asset, downloading it once if needed.

    Weights bundled in the repo (sunhacks/rppg_weights/) are used first, so no
    download is needed; the cache and download are only a fallback.
    """
    if directory is None:
        bundled = BUNDLED_DIR / entry["filename"]
        if bundled.is_file() and sha256_file(bundled) == entry["sha256"]:
            return bundled
    directory = Path(directory or cache_dir())
    target = directory / entry["filename"]
    expected = entry["sha256"]
    if target.is_file():
        if sha256_file(target) == expected:
            return target
        target.unlink()  # stale or corrupt cache entry
    if not allow_download:
        raise FileNotFoundError(f"{entry['filename']} is not cached; run `python -m sunhacks fetch-rppg`")
    directory.mkdir(parents=True, exist_ok=True)
    limit = int(entry.get("max_bytes", 32 * 1024 * 1024))
    digest = hashlib.sha256()
    size = 0
    fd, tmp = tempfile.mkstemp(prefix=".download-", dir=directory)
    try:
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "sunhacks-rppg/1"})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                block = response.read(1 << 16)
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise ValueError(f"{entry['filename']} exceeds {limit} bytes")
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{entry['filename']} sha256 mismatch: {digest.hexdigest()}")
        os.replace(tmp, target)
        return target
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fetch_all(*, directory: Path | None = None) -> dict[str, Any]:
    """Pre-download every rPPG asset (CLI: ``python -m sunhacks fetch-rppg``)."""
    info = manifest()
    result = {"cache_dir": str(directory or cache_dir()), "assets": {}}
    for key, entry in info["assets"].items():
        path = fetch_asset(entry, directory=directory)
        result["assets"][key] = {"path": str(path), "sha256": entry["sha256"], "bytes": path.stat().st_size}
    return result


# --------------------------------------------------------------------------
# Signal processing (pure numpy/scipy; unit-testable without a camera)
# --------------------------------------------------------------------------

def resample_uniform(timestamps, values, rate: float = RESAMPLE_HZ):
    t = np.asarray(timestamps, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    keep = np.concatenate(([True], np.diff(t) > 1e-6))
    t, v = t[keep], v[keep]
    if len(t) < 2:
        return np.empty(0), np.empty((0, v.shape[1]))
    grid = np.arange(t[0], t[-1], 1.0 / rate)
    out = np.stack([np.interp(grid, t, v[:, k]) for k in range(v.shape[1])], axis=1)
    return grid, out


def pos_bvp(rgb, fs: float):
    """Plane-Orthogonal-to-Skin pulse (Wang, den Brinker, Stuijk, de Haan 2017)."""
    c = np.asarray(rgb, dtype=np.float64)
    n = len(c)
    win = max(2, int(round(1.6 * fs)))
    h = np.zeros(n)
    if n < win:
        return h
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(0, n - win + 1):
        block = c[start:start + win]
        mean = block.mean(axis=0)
        if np.any(mean <= 1e-6):
            continue
        s = (block / mean) @ projection.T
        std1 = s[:, 1].std()
        p = s[:, 0] + (s[:, 0].std() / std1 if std1 > 1e-9 else 0.0) * s[:, 1]
        h[start:start + win] += p - p.mean()
    return h


def bandpass(signal, fs: float, band=BAND_HZ, order: int = 3):
    from scipy.signal import butter, sosfiltfilt, detrend
    x = detrend(np.asarray(signal, dtype=np.float64))
    sos = butter(order, [band[0], min(band[1], 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    if len(x) <= 3 * (2 * len(sos) + 1):
        return x - x.mean()
    return sosfiltfilt(sos, x)


def spectral_bpm(signal, fs: float, band=BAND_HZ) -> dict[str, Any]:
    """Dominant in-band frequency with an SNR-based quality in [0, 1]."""
    x = bandpass(signal, fs, band)
    n = len(x)
    if n < int(fs * 2) or not np.isfinite(x).all() or x.std() < 1e-12:
        return {"bpm": None, "quality": 0.0, "snr_db": None}
    x = (x - x.mean()) * np.hanning(n)
    nfft = max(8192, 1 << int(math.ceil(math.log2(n * 8))))
    power = np.abs(np.fft.rfft(x, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any() or power[in_band].sum() <= 0:
        return {"bpm": None, "quality": 0.0, "snr_db": None}
    idx = np.flatnonzero(in_band)
    k = idx[np.argmax(power[idx])]
    f0 = freqs[k]
    if 0 < k < len(power) - 1:  # parabolic peak interpolation
        a, b, c = np.log(power[k - 1:k + 2] + 1e-30)
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            f0 = freqs[k] + 0.5 * (a - c) / denom * (freqs[1] - freqs[0])
    # SNR after de Haan & Jeanne 2013: fundamental + first harmonic vs the rest.
    analysis = (freqs >= band[0]) & (freqs <= min(2 * band[1], fs / 2))
    tol = 0.1
    signal_bins = analysis & ((np.abs(freqs - f0) <= tol) | (np.abs(freqs - 2 * f0) <= 2 * tol))
    s, noise = power[signal_bins].sum(), power[analysis & ~signal_bins].sum()
    snr_db = 10 * math.log10(s / noise) if noise > 0 and s > 0 else (30.0 if s > 0 else -30.0)
    quality = float(np.clip((snr_db + 6.0) / 12.0, 0.0, 1.0))
    return {"bpm": float(f0 * 60.0), "quality": quality, "snr_db": float(snr_db)}


def quality_label(quality):
    if quality is None:
        return None
    return "good" if quality >= 0.6 else "fair" if quality >= 0.35 else "poor"


# --------------------------------------------------------------------------
# FacePhys ONNX streaming model
# --------------------------------------------------------------------------

class FacePhysStream:
    """Per-frame stateful FacePhys inference with onnxruntime on the CPU."""

    def __init__(self, model_path: Path, state_path: Path | None = None, threads: int = 1):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.image_name, self.dt_name = inputs[0].name, inputs[1].name
        self.state_names = [i.name for i in inputs[2:]]
        zeros = {i.name: np.zeros(i.shape, dtype=np.float32) for i in inputs[2:]}
        if state_path is not None:
            with gzip.open(state_path, "rb") as stream:
                loaded = json.loads(stream.read().decode())
            for name in self.state_names:
                if name in loaded:
                    value = np.asarray(loaded[name], dtype=np.float32)
                    if value.shape == zeros[name].shape:
                        zeros[name] = value
        self.initial_state = zeros
        self.state = {k: v.copy() for k, v in zeros.items()}
        self.parameters = None

    def reset(self) -> None:
        self.state = {k: v.copy() for k, v in self.initial_state.items()}

    def step(self, face_rgb36: np.ndarray, dt: float) -> float:
        image = np.asarray(face_rgb36, dtype=np.float32)
        if image.shape != (FACEPHYS_INPUT, FACEPHYS_INPUT, 3):
            raise ValueError("FacePhys expects a 36x36 RGB face crop")
        feeds = {self.image_name: (image / 255.0)[None, None],
                 self.dt_name: np.array([dt], dtype=np.float32), **self.state}
        outputs = self.session.run(None, feeds)
        self.state = dict(zip(self.state_names, outputs[1:]))
        return float(np.asarray(outputs[0]).reshape(-1)[0])


# --------------------------------------------------------------------------
# Face boxes
# --------------------------------------------------------------------------

class FaceTracker:
    """MediaPipe BlazeFace (the asset the app already uses), else OpenCV Haar."""

    def __init__(self, model_path: Path | None):
        self.kind = "none"
        self._detector = self._mp = self._haar = None
        if model_path is not None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mpp
                from mediapipe.tasks.python import vision
                options = vision.FaceDetectorOptions(
                    base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                    min_detection_confidence=0.5)
                self._mp, self._detector = mp, vision.FaceDetector.create_from_options(options)
                self.kind = "mediapipe_blazeface_short_range"
            except Exception:
                self._detector = None
        if self._detector is None:
            try:
                import cv2
                cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
                if not cascade.empty():
                    self._haar, self.kind = cascade, "opencv_haar"
            except Exception:
                self._haar = None

    def detect(self, rgb: np.ndarray):
        """Return the largest face as (x0, y0, x1, y1) floats, or None."""
        if self._detector is not None:
            import cv2
            for scale in (1.0, 2.0):  # same retry as local_faces: small faces need 2x
                image = rgb if scale == 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale,
                                                            interpolation=cv2.INTER_LINEAR)
                result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                              data=np.ascontiguousarray(image)))
                best = None
                for detection in result.detections:
                    bb = detection.bounding_box
                    area = bb.width * bb.height
                    if bb.width > 4 and bb.height > 4 and (best is None or area > best[0]):
                        best = (area, (bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height))
                if best is not None:
                    return tuple(float(v) / scale for v in best[1])
            return None
        if self._haar is not None:
            import cv2
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.15, 5, minSize=(48, 48))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return float(x), float(y), float(x + w), float(y + h)
        return None

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            finally:
                self._detector = None


def _clip_box(box, width, height):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(width, int(round(x1))), min(height, int(round(y1)))
    return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 8 else None


def face_crop_box(box, width, height):
    """FacePhys input: the whole face, slightly taller toward the forehead."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return _clip_box((x0 - 0.05 * w, y0 - 0.15 * h, x1 + 0.05 * w, y1 + 0.02 * h), width, height)


def skin_roi_box(box, width, height):
    """POS input: inner face (cheeks/nose/lower forehead), excluding hair and background."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return _clip_box((x0 + 0.2 * w, y0 + 0.1 * h, x1 - 0.2 * w, y1 - 0.15 * h), width, height)


# --------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------

class HeartRateMonitor:
    """Rolling-window BPM estimator fed by LiveCapture frames.

    ``offer_frame`` is the only call made on the capture thread. ``snapshot``
    is cheap and lock-protected; the HTTP state endpoint calls it about 1 Hz.
    """

    def __init__(self, config: dict[str, Any] | None = None, *, clock=time.monotonic,
                 start_thread: bool = True):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.method_preference = cfg.get("method", "facephys")
        if self.method_preference not in ("facephys", "pos"):
            raise ValueError("heart_rate.method must be facephys or pos")
        self.auto_download = bool(cfg.get("auto_download", True))
        self.face_model_path = cfg.get("face_model_path")
        self.expected_face_sha256 = cfg.get("expected_face_sha256")
        self.cache = Path(cfg["cache_dir"]).expanduser() if cfg.get("cache_dir") else None
        self.clock = clock
        self._start_thread = start_thread
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: deque = deque(maxlen=MAX_PENDING)
        self._closed = False
        self._thread: threading.Thread | None = None
        self._model_thread: threading.Thread | None = None
        self._model: FacePhysStream | None = None
        self._model_status = {"status": "disabled" if self.method_preference == "pos" else "not_loaded"}
        self._tracker: FaceTracker | None = None
        self._tracker_ready = False
        self._dropped = 0
        self._error = None
        self._reset_requested = False
        self._recorded_reading = None
        self._capture_generation = 0
        self._reset_signal_locked(full=True)

    # ---- capture-thread API -------------------------------------------------
    def offer_frame(self, rgb: np.ndarray, timestamp: float) -> None:
        if not self.enabled or self._closed:
            return
        with self._wake:
            if self._last_offer is None:
                self._recorded_reading = None
                self._capture_generation += 1
            if len(self._pending) == self._pending.maxlen:
                self._dropped += 1
            self._pending.append((timestamp, rgb))
            self._last_offer = self.clock()
            self._wake.notify()
            if self._start_thread and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="rppg-heart-rate", daemon=True)
                self._thread.start()

    def capture_stopped(self) -> None:
        """Camera released: go idle now; the worker drops its window."""
        with self._wake:
            self._pending.clear()
            self._last_offer = None
            self._reset_requested = True
            self._wake.notify()

    def clear_recorded_reading(self) -> None:
        """A new participant/reset must not display the previous recording."""
        with self._lock:
            self._recorded_reading = None
            self._capture_generation += 1

    # ---- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._pending.clear()
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._tracker is not None:
            self._tracker.close()

    def _reset_signal_locked(self, *, full: bool) -> None:
        self._samples: deque = deque()   # (t, facephys_bvp|nan, r, g, b)
        self._estimates: deque = deque(maxlen=5)
        self._box = None
        self._smoothed = None
        self._last_face = None
        self._last_frame_t = None
        self._frame_index = 0
        self._last_compute = None
        self._result = {"bpm": None, "bpm_raw": None, "quality": None, "snr_db": None,
                        "method": None, "pulse": []}
        if full:
            self._last_offer = None
            self._frame_times: deque = deque(maxlen=60)
        if self._model is not None:
            self._model.reset()

    # ---- model loading (background, never blocks frames) ---------------------
    def _ensure_model_loading(self) -> None:
        if self.method_preference != "facephys" or self._model is not None:
            return
        if self._model_thread is not None:
            return

        def load():
            with self._lock:
                self._model_status = {"status": "loading"}
            try:
                info = manifest()
                assets = info["assets"]
                model_path = fetch_asset(assets["facephys_onnx"], directory=self.cache, allow_download=self.auto_download)
                state_path = fetch_asset(assets["facephys_state"], directory=self.cache, allow_download=self.auto_download)
                model = FacePhysStream(model_path, state_path)
                model.step(np.zeros((FACEPHYS_INPUT, FACEPHYS_INPUT, 3), np.uint8), 1 / 30)  # warm kernels
                model.reset()
                with self._lock:
                    self._model = model
                    self._model_status = {"status": "ready", "sha256": assets["facephys_onnx"]["sha256"]}
            except Exception as exc:
                with self._lock:
                    self._model_status = {"status": "unavailable",
                                          "reason": f"{type(exc).__name__}: {str(exc)[:200]}; using POS fallback"}

        self._model_thread = threading.Thread(target=load, name="rppg-model-load", daemon=True)
        self._model_thread.start()

    def _ensure_tracker(self) -> None:
        if self._tracker_ready:
            return
        self._tracker_ready = True
        path = None
        try:
            if self.face_model_path and Path(self.face_model_path).expanduser().is_file():
                candidate = Path(self.face_model_path).expanduser().resolve()
                if self.expected_face_sha256 in (None, sha256_file(candidate)):
                    path = candidate
            if path is None:
                path = fetch_asset(manifest()["assets"]["blazeface_short_range"], directory=self.cache,
                                   allow_download=self.auto_download)
        except Exception:
            path = None
        self._tracker = FaceTracker(path)

    # ---- worker ---------------------------------------------------------------
    def _run(self) -> None:
        self._ensure_model_loading()
        while True:
            with self._wake:
                if self._closed:
                    return
                if not self._pending:
                    self._wake.wait(timeout=0.25)
                batch = list(self._pending)
                self._pending.clear()
                idle = self._last_offer is None or self.clock() - self._last_offer > IDLE_SECONDS
                if self._reset_requested or (idle and self._samples):
                    self._reset_signal_locked(full=self._reset_requested)
                    self._reset_requested = False
            for timestamp, rgb in batch:
                try:
                    self.process_frame(rgb, timestamp)
                except Exception as exc:  # never let rPPG break capture
                    with self._lock:
                        self._error = f"rPPG frame failed: {type(exc).__name__}: {str(exc)[:200]}"
            if batch:
                self.maybe_compute(batch[-1][0])

    def process_frame(self, rgb: np.ndarray, timestamp: float) -> None:
        """Detect/track the face, run FacePhys and collect POS RGB means."""
        import cv2
        self._ensure_tracker()
        height, width = rgb.shape[:2]
        with self._lock:
            if self._last_frame_t is not None and timestamp <= self._last_frame_t:
                return
            gap = None if self._last_frame_t is None else timestamp - self._last_frame_t
            if gap is not None and gap > 1.0:  # discontinuity: restart the window
                self._reset_signal_locked(full=False)
                gap = None
            self._frame_times.append(timestamp)
            index = self._frame_index
            self._frame_index += 1
        box = self._box
        if box is None or index % DETECT_EVERY == 0:
            detected = self._tracker.detect(rgb) if self._tracker is not None else None
            if detected is not None:
                self._last_face = timestamp
                box = self._stabilize(detected)
            elif self._last_face is None or timestamp - self._last_face > FACE_LOST_SECONDS:
                if self._box is not None or self._samples:
                    with self._lock:
                        self._reset_signal_locked(full=False)
                        self._frame_times.append(timestamp)
                box = None
        self._box = box
        if box is None:
            with self._lock:
                self._last_frame_t = timestamp
            return
        roi = skin_roi_box(box, width, height)
        crop = face_crop_box(box, width, height)
        if roi is None or crop is None:
            return
        x0, y0, x1, y1 = roi
        mean_rgb = rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
        value = float("nan")
        model = self._model
        if model is not None:
            cx0, cy0, cx1, cy1 = crop
            face = cv2.resize(np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1]), (FACEPHYS_INPUT, FACEPHYS_INPUT),
                              interpolation=cv2.INTER_AREA)
            dt = float(np.clip(gap if gap is not None else 1 / 30, 1 / 120, 0.25))
            value = model.step(face, dt)
        with self._lock:
            self._last_frame_t = timestamp
            self._samples.append((timestamp, value, *map(float, mean_rgb)))
            while self._samples and self._samples[0][0] < timestamp - WINDOW_SECONDS:
                self._samples.popleft()

    def _stabilize(self, detected):
        """EMA plus hysteresis so the crop does not jitter (jitter is pulse-band noise)."""
        d = np.asarray(detected, dtype=np.float64)
        if self._smoothed is None:
            self._smoothed = d
            return tuple(d)
        self._smoothed = 0.7 * self._smoothed + 0.3 * d
        if self._box is None:
            return tuple(self._smoothed)
        current = np.asarray(self._box)
        size = max(current[2] - current[0], current[3] - current[1], 1.0)
        centre_shift = np.abs((self._smoothed[:2] + self._smoothed[2:]) / 2 - (current[:2] + current[2:]) / 2).max() / size
        scale_shift = abs((self._smoothed[2] - self._smoothed[0]) - (current[2] - current[0])) / size
        return tuple(self._smoothed) if centre_shift > 0.04 or scale_shift > 0.06 else self._box

    def maybe_compute(self, now: float, *, force: bool = False) -> None:
        with self._lock:
            if not force and self._last_compute is not None and now - self._last_compute < UPDATE_SECONDS:
                return
            self._last_compute = now
            samples = list(self._samples)
            generation = self._capture_generation
        if len(samples) < 8:
            return
        data = np.asarray(samples, dtype=np.float64)
        t = data[:, 0]
        span = t[-1] - t[0]
        if span < MIN_SECONDS:
            return
        neural = np.isfinite(data[:, 1]).all()
        grid, uniform = resample_uniform(t, data[:, 1:], RESAMPLE_HZ)
        if len(grid) < RESAMPLE_HZ * 2:
            return
        if neural:
            pulse, method = uniform[:, 0], "facephys"
        else:
            pulse, method = pos_bvp(uniform[:, 1:4], RESAMPLE_HZ), "pos"
        estimate = spectral_bpm(pulse, RESAMPLE_HZ)
        filtered = bandpass(pulse, RESAMPLE_HZ)
        tail = filtered[-int(PULSE_SECONDS * RESAMPLE_HZ):]
        if len(tail) > 2:
            idx = np.linspace(0, len(tail) - 1, PULSE_POINTS)
            tail = np.interp(idx, np.arange(len(tail)), tail)
            peak = np.abs(tail).max()
            tail = tail / peak if peak > 0 else tail
        with self._lock:
            if generation != self._capture_generation or self._reset_requested or self._last_offer is None:
                return
            if estimate["bpm"] is not None and estimate["quality"] >= 0.15:
                self._estimates.append(estimate["bpm"])
            shown = float(np.median(self._estimates)) if self._estimates else None
            self._error = None
            self._result = {"bpm": shown, "bpm_raw": estimate["bpm"], "quality": estimate["quality"],
                            "snr_db": estimate["snr_db"], "method": method,
                            "pulse": [round(float(v), 3) for v in tail]}
            if shown is not None and estimate["quality"] >= 0.15:
                self._recorded_reading = {"bpm": round(shown, 1), "quality": round(estimate["quality"], 3),
                    "quality_label": quality_label(estimate["quality"]), "method": method,
                    "pulse": list(self._result["pulse"]), "window_seconds": round(float(span), 2),
                    "display_only": True, "source": "last_valid_estimate_in_this_recording"}

    # ---- state ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            frames = list(self._frame_times)
            span = (self._samples[-1][0] - self._samples[0][0]) if len(self._samples) > 1 else 0.0
            active = self._last_offer is not None and now - self._last_offer <= IDLE_SECONDS
            face = self._box is not None
            result = dict(self._result)
            recorded = dict(self._recorded_reading) if self._recorded_reading is not None else None
            fps = (len(frames) - 1) / (frames[-1] - frames[0]) if len(frames) > 1 and frames[-1] > frames[0] else None
            model_status = dict(self._model_status)
            error = self._error
        if not self.enabled:
            status = "disabled"
        elif not active:
            status = "idle"
        elif not face:
            status = "no_face"
        elif result["bpm"] is None:
            status = "acquiring"
        else:
            status = "tracking"
        live = status == "tracking"
        info = manifest()
        return {
            "status": status,
            "display_only": True,
            "recorded_reading": recorded if not active else None,
            "warming_up": status in ("acquiring", "no_face") or (active and span < MIN_SECONDS),
            "bpm": round(result["bpm"], 1) if live and result["bpm"] is not None else None,
            "bpm_raw": round(result["bpm_raw"], 1) if live and result["bpm_raw"] is not None else None,
            "quality": round(result["quality"], 3) if live and result["quality"] is not None else None,
            "quality_label": quality_label(result["quality"]) if live else None,
            "snr_db": round(result["snr_db"], 2) if live and result["snr_db"] is not None else None,
            "method": result["method"] if live else ("facephys" if self._model is not None else "pos"),
            "pulse": result["pulse"] if live else [],
            "window_seconds": round(span, 2),
            "target_window_seconds": WINDOW_SECONDS,
            "progress": round(min(1.0, span / MIN_SECONDS), 3) if active else 0.0,
            "face_detected": face and active,
            "face_detector": self._tracker.kind if self._tracker is not None else None,
            "frame_rate": round(fps, 1) if fps and active else None,
            "dropped_frames": self._dropped,
            "band_hz": list(BAND_HZ),
            "model": {"id": info["model_id"], "parameters": info["parameters"], **model_status},
            "error": error,
            "note": NOTE,
        }


def analyze_frames(frames, timestamps, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Offline helper: run the live monitor synchronously over a frame sequence."""
    monitor = HeartRateMonitor(config, start_thread=False, clock=lambda: float(timestamps[-1]))
    if monitor.method_preference == "facephys":
        monitor._ensure_model_loading()
        monitor._model_thread.join()
    with monitor._lock:
        monitor._last_offer = float(timestamps[-1])
    for frame, stamp in zip(frames, timestamps):
        monitor.process_frame(np.ascontiguousarray(frame), float(stamp))
        monitor.maybe_compute(float(stamp))
    monitor.maybe_compute(float(timestamps[-1]), force=True)
    result = monitor.snapshot()
    monitor.close()
    return result
