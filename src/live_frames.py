"""Live frame preparation shared by `analyze` and `jepa_video`, identical to training preprocessing.

Training (scripts/preprocess_rlt.py) decoded, for every clip, the frame nearest each of 128 uniformly spaced
times over the whole answer, resized so the long side is <= 960 px, ran MediaPipe BlazeFace, median-smoothed
the boxes and cut 384 x 384 face crops. Live requests are prepared the same way, by TIME (per-frame
timestamps or a constant fps are required), never by frame count.
"""

from __future__ import annotations

import base64
from typing import Dict, Tuple

import numpy as np

N_CACHE = 128   # frames per answer in training
RES = 384       # face-crop side in training
MIN_FRAMES = 16


def decode_image(b: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode a frame (expected JPEG/PNG bytes)")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return img if s == 1.0 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def frame_times(payload: Dict, n: int) -> np.ndarray:
    if "timestamps" in payload:
        ts = np.asarray(payload["timestamps"], dtype=np.float64)
        if ts.shape != (n,) or not np.isfinite(ts).all():
            raise ValueError("'timestamps' must have one finite entry (seconds) per frame")
        return ts
    fps = float(payload.get("fps", 0) or 0)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("send per-frame 'timestamps' (seconds) or a constant 'fps'")
    return np.arange(n) / fps


def select_by_time(ts_in: np.ndarray, n: int = N_CACHE) -> np.ndarray:
    """Index of the frame nearest each of n uniformly spaced times over the answer (as in training)."""
    order = np.argsort(ts_in, kind="stable")
    ts_sorted = ts_in[order]
    span = float(ts_sorted[-1] - ts_sorted[0]) or 1e-6
    targets = ts_sorted[0] + (np.arange(n) + 0.5) * span / n
    nxt = np.clip(np.searchsorted(ts_sorted, targets), 0, len(ts_sorted) - 1)
    prv = np.clip(nxt - 1, 0, len(ts_sorted) - 1)
    near = np.where(np.abs(ts_sorted[prv] - targets) < np.abs(ts_sorted[nxt] - targets), prv, nxt)
    return order[near]


def prepare_face_crops(payload: Dict) -> Tuple[np.ndarray, Dict, float, Dict]:
    """-> (crops (128, 384, 384, 3) uint8, meta {boxes_cxcys, timestamps, face_detect_rate}, duration_s,
    face diagnostics). Raw camera frames are expected; this machine crops the face."""
    from scripts.preprocess_rlt import MAX_SIDE, crop_frames, detect_faces, smooth_boxes

    frames_b64 = payload.get("frames")
    if not isinstance(frames_b64, list) or len(frames_b64) < MIN_FRAMES:
        raise ValueError(f"'frames' must be a list of >= {MIN_FRAMES} base64 JPEG/PNG images spanning the answer")
    frames = [decode_image(base64.b64decode(f)) for f in frames_b64]
    ts_in = frame_times(payload, len(frames))
    idx = select_by_time(ts_in)
    sel = np.stack([fit_max_side(frames[i], MAX_SIDE) for i in idx])
    ts = ts_in[idx]
    h, w = sel.shape[1:3]
    raw = detect_faces(sel)
    boxes, rate, fallback = smooth_boxes(raw[:, :3], w, h)
    crops = crop_frames(sel, boxes, RES)
    meta = {"boxes_cxcys": boxes.tolist(), "timestamps": ts.tolist(), "face_detect_rate": rate,
            "selected_indices": [int(i) for i in idx]}
    duration = float(ts_in.max() - ts_in.min()) + (float(np.median(np.diff(np.sort(ts_in)))) if len(ts_in) > 1 else 0.0)
    info = {"face_detect_rate": rate, "centre_crop_fallback": fallback,
            "selected_time_span_s": [round(float(ts.min()), 4), round(float(ts.max()), 4)],
            "unique_frames_used": int(len(set(idx.tolist())))}
    return crops, meta, duration, info
