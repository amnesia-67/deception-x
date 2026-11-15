#!/usr/bin/env python3
"""Decode Real-Life Trials clips into face-cropped frame stacks and 16 kHz audio.

Per clip, written to $DECEPTIONX_CACHE (default ~/deceptionx_cache, ext4 for speed):
  frames/<clip>.npy  uint8 (N_FRAMES, 384, 384, 3), N_FRAMES uniformly spaced over the clip
  audio/<clip>.npy   float32 mono waveform @ 16 kHz
  meta/<clip>.json   timestamps, fps, face-detection rate, smoothed face boxes
Plus contact sheets for identity review under data/review/.

Face crops use MediaPipe BlazeFace (models/mediapipe/blaze_face_short_range.tflite).
Clips where no face is found fall back to a centre crop, and that fallback is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rlt_data import CACHE_DIR, ROOT, index_clips  # noqa: E402

N_FRAMES = 128
OUT_RES = 384
SR = 16000
FACE_SCALE = 1.6  # crop side = 1.6 x max(face w, h): whole face + a little context
BLAZE = os.path.join(ROOT, "models", "mediapipe", "blaze_face_short_range.tflite")


MAX_SIDE = 960  # selected frames are downscaled to this before face detection (memory bound)


def _to_rgb(fr):
    w, h = fr.width, fr.height
    s = min(1.0, MAX_SIDE / max(w, h))
    if s < 1.0:
        return fr.to_ndarray(format="rgb24", width=int(round(w * s / 2) * 2), height=int(round(h * s / 2) * 2))
    return fr.to_ndarray(format="rgb24")


def _probe_timing(path: str):
    """First pass: timestamps only (cheap, no pixel conversion)."""
    import av

    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        ts = [float(f.pts * vs.time_base) for f in c.decode(vs) if f.pts is not None]
        fps = float(vs.average_rate or 25.0)
    if not ts:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.sort(np.asarray(ts)), fps


def decode_video_frames(path: str, n: int):
    """Return (frames[n,H,W,3] uint8, timestamps[n], fps, duration).

    Two streaming passes: the first collects timestamps, the second converts only
    the n frames nearest to n uniformly spaced targets, so memory stays bounded
    (an 80 s 1080p clip would otherwise need ~15 GB).
    """
    import av

    ts, fps = _probe_timing(path)
    duration = float(ts[-1] - ts[0]) + 1.0 / fps
    targets = ts[0] + (np.arange(n) + 0.5) * (duration / n)
    nearest = np.clip(np.searchsorted(ts, targets), 0, len(ts) - 1)
    prev = np.clip(nearest - 1, 0, len(ts) - 1)
    pick = np.where(np.abs(ts[prev] - targets) < np.abs(ts[nearest] - targets), prev, nearest)
    want_ts = ts[pick]
    wanted = {}
    for i, t in enumerate(want_ts):
        wanted.setdefault(round(float(t), 6), []).append(i)
    frames = [None] * n
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for fr in c.decode(vs):
            if fr.pts is None:
                continue
            key = round(float(fr.pts * vs.time_base), 6)
            if key in wanted:
                img = _to_rgb(fr)
                for i in wanted.pop(key):
                    frames[i] = img
                if not wanted:
                    break
    missing = [i for i, f in enumerate(frames) if f is None]
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} selected frames not re-decoded")
    return np.stack(frames), want_ts, fps, duration


def decode_audio(path: str, sr: int = SR) -> np.ndarray:
    import av

    with av.open(path) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        res = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for fr in c.decode(c.streams.audio[0]):
            for o in res.resample(fr):
                chunks.append(o.to_ndarray().reshape(-1))
        for o in res.resample(None):
            chunks.append(o.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        opts = vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=BLAZE),
            min_detection_confidence=0.5,
        )
        _DETECTOR = (mp, vision.FaceDetector.create_from_options(opts))
    return _DETECTOR


def detect_faces(frames: np.ndarray):
    """Largest confident face per frame -> (cx, cy, side) or NaN. Tries 2x upscale for small faces."""
    import cv2

    mp, det = _detector()
    boxes = np.full((len(frames), 4), np.nan, np.float32)  # cx, cy, side, score
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        best = None
        for scale in (1.0, 2.0):
            img = f if scale == 1.0 else cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
            for d in res.detections:
                bb = d.bounding_box
                side = max(bb.width, bb.height) / scale
                cand = ((bb.origin_x + bb.width / 2) / scale, (bb.origin_y + bb.height / 2) / scale, side,
                        d.categories[0].score)
                if best is None or cand[2] * cand[3] > best[2] * best[3]:
                    best = cand
            if best is not None:
                break
        if best is not None:
            boxes[i] = best
    return boxes


def smooth_boxes(boxes: np.ndarray, w: int, h: int):
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
    # temporal median smoothing to remove jitter
    k = 5
    pad = np.pad(out, ((k // 2, k // 2), (0, 0)), mode="edge")
    out = np.stack([np.median(pad[i:i + k], axis=0) for i in range(len(out))]).astype(np.float32)
    return out, rate, False


def crop_frames(frames: np.ndarray, boxes: np.ndarray, res: int = OUT_RES) -> np.ndarray:
    import cv2

    h, w = frames.shape[1:3]
    out = np.empty((len(frames), res, res, 3), np.uint8)
    for i, (f, (cx, cy, side)) in enumerate(zip(frames, boxes)):
        s = max(16.0, side * FACE_SCALE)
        x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
        x1, y1 = x0 + int(round(s)), y0 + int(round(s))
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        src = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else f
        crop = src[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
        out[i] = cv2.resize(crop, (res, res), interpolation=cv2.INTER_AREA if crop.shape[0] > res else cv2.INTER_CUBIC)
    return out


def process_clip(rec_dict: dict, cache: str, force: bool = False) -> dict:
    cid = rec_dict["clip_id"]
    fpath = os.path.join(cache, "frames", cid + ".npy")
    apath = os.path.join(cache, "audio", cid + ".npy")
    mpath = os.path.join(cache, "meta", cid + ".json")
    if not force and all(os.path.exists(p) for p in (fpath, apath, mpath)):
        return json.load(open(mpath))
    t0 = time.time()
    frames, ts, fps, dur = decode_video_frames(rec_dict["video_path"], N_FRAMES)
    h, w = frames.shape[1:3]
    raw = detect_faces(frames)
    boxes, rate, fallback = smooth_boxes(raw[:, :3], w, h)
    crops = crop_frames(frames, boxes)
    audio = decode_audio(rec_dict["video_path"])
    np.save(fpath, crops)
    np.save(apath, audio)
    meta = {
        "clip_id": cid, "label": rec_dict["label"], "fps": fps, "duration_s": dur,
        "src_wh": [w, h], "timestamps": ts.round(4).tolist(), "face_detect_rate": rate,
        "centre_crop_fallback": fallback, "boxes_cxcys": boxes.round(2).tolist(),
        "raw_scores": np.nan_to_num(raw[:, 3], nan=0.0).round(3).tolist(),
        "audio_seconds": len(audio) / SR, "seconds": round(time.time() - t0, 2),
    }
    json.dump(meta, open(mpath, "w"))
    return meta


def contact_sheets(recs, cache: str, out_dir: str, per_sheet: int = 42, cols: int = 7):
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    tile = 160
    paths = []
    for s in range(0, len(recs), per_sheet):
        chunk = recs[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * (tile + 22), cols * tile, 3), 255, np.uint8)
        for i, r in enumerate(chunk):
            fr = np.load(os.path.join(cache, "frames", r.clip_id + ".npy"), mmap_mode="r")
            img = cv2.resize(np.asarray(fr[len(fr) // 2]), (tile, tile))
            y, x = (i // cols) * (tile + 22), (i % cols) * tile
            sheet[y:y + tile, x:x + tile] = img
            cv2.putText(sheet, r.clip_id.replace("trial_", ""), (x + 4, y + tile + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        p = os.path.join(out_dir, f"contact_sheet_{s // per_sheet + 1}.jpg")
        cv2.imwrite(p, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for sub in ("frames", "audio", "meta"):
        os.makedirs(os.path.join(CACHE_DIR, sub), exist_ok=True)
    recs = index_clips()
    if args.limit:
        recs = recs[: args.limit]
    print(f"preprocessing {len(recs)} real clips -> {CACHE_DIR} with {args.workers} workers", flush=True)
    metas = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_clip, {"clip_id": r.clip_id, "video_path": r.video_path, "label": r.label},
                          CACHE_DIR, args.force): r.clip_id for r in recs}
        for i, fu in enumerate(as_completed(futs), 1):
            m = fu.result()
            metas.append(m)
            print(f"[{i:3d}/{len(recs)}] {m['clip_id']:<18} dur={m['duration_s']:5.1f}s "
                  f"face={m['face_detect_rate']*100:5.1f}% fallback={m['centre_crop_fallback']} "
                  f"audio={m['audio_seconds']:5.1f}s ({m['seconds']}s)", flush=True)
    rates = np.array([m["face_detect_rate"] for m in metas])
    summary = {
        "n_clips": len(metas),
        "mean_face_detect_rate": float(rates.mean()),
        "clips_face_rate_below_50pct": sorted(m["clip_id"] for m in metas if m["face_detect_rate"] < 0.5),
        "centre_crop_fallbacks": sorted(m["clip_id"] for m in metas if m["centre_crop_fallback"]),
    }
    os.makedirs(os.path.join(ROOT, "data", "review"), exist_ok=True)
    json.dump(summary, open(os.path.join(ROOT, "data", "review", "preprocess_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("contact sheets:", contact_sheets(recs, CACHE_DIR, os.path.join(ROOT, "data", "review")))


if __name__ == "__main__":
    main()
