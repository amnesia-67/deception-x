"""Local reviewed BlazeFace preprocessing for the 89-field digest.

This is the inference-only geometry from friend revision53393582. The caller
already chose128 timestamped frames. No capture, network, training or data files
are opened. A real detector returning no face uses the training center fallback;
an absent detector or model asset is an error, never a simulated detection.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time

import numpy as np

MAX_SIDE = 960
OUT_RES = 384
FACE_SCALE = 1.6
N_FRAMES = 128


class LocalFaceUnavailable(ValueError):
    pass


def _create_detector(path):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.FaceDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(path)),
        min_detection_confidence=0.5,
    )
    return mp, vision.FaceDetector.create_from_options(opts)


def _fit(frame):
    import cv2
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    return frame if scale == 1.0 else cv2.resize(
        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def smooth_boxes(boxes, w, h):
    """Exact source interpolation/median5, with explicit no-detection fallback."""
    ok = ~np.isnan(boxes[:, 0])
    rate = float(ok.mean())
    out = np.zeros((len(boxes), 3), np.float32)
    if ok.sum() == 0:
        side = min(w, h)
        out[:] = (w / 2, h / 2, side / FACE_SCALE)
        return out, rate, True
    t = np.arange(len(boxes))
    for k in range(3):
        out[:, k] = np.interp(t, t[ok], boxes[ok, k])
    k = 5
    pad = np.pad(out, ((k // 2, k // 2), (0, 0)), mode="edge")
    out = np.stack([np.median(pad[i:i + k], axis=0) for i in range(len(out))]).astype(np.float32)
    return out, rate, False


def crop_frames(frames, boxes, res=OUT_RES):
    import cv2
    h, w = frames.shape[1:3]
    out = np.empty((len(frames), res, res, 3), np.uint8)
    for i, (frame, (cx, cy, side)) in enumerate(zip(frames, boxes)):
        size = max(16.0, side * FACE_SCALE)
        x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
        x1, y1 = x0 + int(round(size)), y0 + int(round(size))
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        src = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else frame
        crop = src[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
        out[i] = cv2.resize(crop, (res, res), interpolation=cv2.INTER_AREA if crop.shape[0] > res else cv2.INTER_CUBIC)
    return out


class LocalFacePreprocessor:
    def __init__(self, model_path, expected_sha256=None):
        self.model_path = Path(model_path).expanduser().resolve()
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or
                len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise LocalFaceUnavailable("Face model SHA256 must be64 lowercase hex characters")
        self.expected_sha256 = expected_sha256
        self._lock = threading.RLock()
        self._mp = self._detector = None
        self._closed = False
        self._asset_sha256 = None

    def prepare(self):
        """Load the local SDK/model; no image forward is performed."""
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise LocalFaceUnavailable("Face preprocessor is closed")
            cached = self._detector is not None
            if not cached:
                if not self.model_path.is_file() or not 0 < self.model_path.stat().st_size <= 8 * 1024 * 1024:
                    raise LocalFaceUnavailable("Local reviewed BlazeFace asset is missing or exceeds8MiB")
                with self.model_path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if self.expected_sha256 is not None and digest != self.expected_sha256:
                    raise LocalFaceUnavailable("BlazeFace asset SHA256 mismatch")
                mp, detector = _create_detector(self.model_path)
                self._mp, self._detector, self._asset_sha256 = mp, detector, digest
            return {"status":"ready", "model_id":"mediapipe/blaze_face_short_range", "device":"cpu",
                    "weights_loaded":True, "kernels_warmed":False, "cache_hit":cached,
                    "asset_sha256":self._asset_sha256,
                    "elapsed_ms":round((time.perf_counter()-started)*1000,3)}

    def _detect(self, frames):
        import cv2
        boxes = np.full((len(frames), 4), np.nan, np.float32)
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            best = None
            for scale in (1.0, 2.0):
                img = frame if scale == 1.0 else cv2.resize(frame, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
                result = self._detector.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                              data=np.ascontiguousarray(img)))
                for detection in result.detections:
                    bb = detection.bounding_box
                    side = max(bb.width, bb.height) / scale
                    candidate = ((bb.origin_x + bb.width/2)/scale, (bb.origin_y + bb.height/2)/scale,
                                 side, detection.categories[0].score)
                    if not np.isfinite(candidate).all() or side <= 0:
                        raise LocalFaceUnavailable("Face detector emitted invalid coordinates or score")
                    if best is None or candidate[2]*candidate[3] > best[2]*best[3]:
                        best = candidate
                if best is not None:
                    break
            if best is not None:
                boxes[i] = best
        return boxes

    def __call__(self, selected_rgb_frames, selected_timestamps):
        frames = list(selected_rgb_frames)
        timestamps = np.asarray(selected_timestamps, dtype=np.float64)
        if (len(frames) != N_FRAMES or timestamps.shape != (N_FRAMES,) or
                not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0) or
                timestamps[-1] <= timestamps[0]):
            raise LocalFaceUnavailable("Need128 selected frames and finite chronological source timestamps")
        shape = None
        checked = []
        for frame in frames:
            frame = np.asarray(frame)
            if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or
                    min(frame.shape[:2]) < 1 or max(frame.shape[:2]) > 4096):
                raise LocalFaceUnavailable("Selected camera frames must be bounded RGB uint8 images")
            if shape is None:
                shape = frame.shape
            if frame.shape != shape:
                raise LocalFaceUnavailable("Selected camera frames must have consistent dimensions")
            checked.append(_fit(frame))
        with self._lock:
            self.prepare()
            array = np.stack(checked)
            h, w = array.shape[1:3]
            detections = self._detect(array)
            boxes, rate, fallback = smooth_boxes(detections[:, :3], w, h)
            crops = crop_frames(array, boxes)
            return crops, {"boxes_cxcys":boxes.tolist(), "timestamps":timestamps.tolist(),
                           "face_detect_rate":rate, "centre_crop_fallback":fallback,
                           "preprocessing":"blazeface-largest-side-score-interp-median5-scale1.6-replicate-384"}

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                detector, self._detector, self._mp = self._detector, None, None
                if detector is not None:
                    detector.close()
        return {"status":"closed"}
