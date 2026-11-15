# Exact geometry functions from amnesia-67/sunhacks-2026, 6e30a90,
# scripts/preprocess_rlt.py, Git blob 48ed141cedc48b224150b7f8eefbd28f8f61c952.
# Parsed as a reference fixture only; no capture/preprocessing entry point.

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
