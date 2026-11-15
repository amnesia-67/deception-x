"""Access to preprocessed real clips plus train/eval views (augmentations run on the GPU)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.rlt_data import CACHE_DIR, ClipRecord, index_clips

SR = 16000


class ClipStore:
    def __init__(self, records: Sequence[ClipRecord] = None, cache: str = CACHE_DIR):
        self.records = list(records) if records is not None else index_clips(require_groups=True)
        self.cache = cache
        self.by_id = {r.clip_id: r for r in self.records}
        for r in self.records:
            for sub, ext in (("frames", ".npy"), ("audio", ".npy"), ("meta", ".json")):
                p = os.path.join(cache, sub, r.clip_id + ext)
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{p} missing; run scripts/preprocess_rlt.py")

    @property
    def ids(self) -> List[str]:
        return [r.clip_id for r in self.records]

    def labels(self, ids: Sequence[str]) -> np.ndarray:
        return np.array([self.by_id[i].label for i in ids], dtype=np.int64)

    def groups(self, ids: Sequence[str]) -> np.ndarray:
        return np.array([self.by_id[i].group for i in ids], dtype=np.int64)

    def frames(self, cid: str) -> np.ndarray:
        return np.load(os.path.join(self.cache, "frames", cid + ".npy"), mmap_mode="r")

    def audio(self, cid: str) -> np.ndarray:
        return np.load(os.path.join(self.cache, "audio", cid + ".npy"), mmap_mode="r")

    def meta(self, cid: str) -> Dict:
        with open(os.path.join(self.cache, "meta", cid + ".json")) as f:
            return json.load(f)

    def transcript(self, cid: str) -> str:
        return self.by_id[cid].transcript


# ----------------------------------------------------------------------------- video
WINDOW = 64  # every video model sees a contiguous window of 64 cached frames (half the clip)


def frame_stride(n_out: int) -> int:
    return WINDOW // n_out  # vjepa2: 64 frames stride 1; vjepa2_1: 32 frames stride 2


def eval_starts(n_total: int = 128) -> list:
    return [0, n_total - WINDOW]  # first and second half of the clip, averaged at eval


def select_frames(n_total: int, n_out: int, start: int) -> np.ndarray:
    s = frame_stride(n_out)
    start = int(np.clip(start, 0, n_total - n_out * s))
    return start + np.arange(n_out) * s


def video_batch(store: ClipStore, ids: Sequence[str], n_out: int, train: bool, rng: np.random.Generator,
                start: int = 0, device="cuda") -> torch.Tensor:
    """(B, n_out, H, W, 3) uint8 on GPU.

    Contiguous window (keeps the frame spacing V-JEPA expects as close as the cache allows).
    Train: random window start, flip, resized crop, colour jitter. Eval: the given start.
    """
    out = []
    for cid in ids:
        fr = store.frames(cid)
        st = int(rng.integers(0, len(fr) - WINDOW + 1)) if train else start
        x = torch.from_numpy(np.ascontiguousarray(fr[select_frames(len(fr), n_out, st)])).to(device)
        if train:
            x = augment_video(x, rng)
        out.append(x)
    return torch.stack(out)


def augment_video(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """x: (T, H, W, 3) uint8 -> same shape uint8. Same transform for every frame (temporal consistency)."""
    T, H, W, _ = x.shape
    if rng.random() < 0.5:
        x = x.flip(2)
    scale = float(rng.uniform(0.80, 1.0))
    ch, cw = int(H * scale), int(W * scale)
    y0, x0 = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
    v = x[:, y0:y0 + ch, x0:x0 + cw].permute(0, 3, 1, 2).float()
    v = F.interpolate(v, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
    b = float(rng.uniform(-0.1, 0.1)) * 255.0
    c = float(rng.uniform(0.85, 1.15))
    mean = v.mean(dim=(1, 2, 3), keepdim=True)
    v = ((v - mean) * c + mean + b).clamp_(0, 255)
    return v.permute(0, 2, 3, 1).round().to(torch.uint8)


# ----------------------------------------------------------------------------- audio
def audio_tensor(store: ClipStore, cid: str, train: bool, rng: np.random.Generator,
                 max_train_s: float = 16.0, max_eval_s: float = 60.0, min_s: float = 1.0,
                 device="cuda") -> torch.Tensor:
    """(1, L) float32 on GPU. Train: random window <= max_train_s and +-6 dB gain."""
    a = np.array(store.audio(cid), dtype=np.float32, copy=True)  # mmap is read-only
    lim = int((max_train_s if train else max_eval_s) * SR)
    if len(a) > lim:
        s = int(rng.integers(0, len(a) - lim + 1)) if train else (len(a) - lim) // 2
        a = a[s:s + lim]
    if len(a) < int(min_s * SR):
        a = np.pad(a, (0, int(min_s * SR) - len(a)))
    x = torch.from_numpy(a).to(device).unsqueeze(0)
    if train:
        x = x * float(10 ** (rng.uniform(-6, 6) / 20))
    return x


# ----------------------------------------------------------------------------- text
def text_view(text: str, train: bool, rng: np.random.Generator, drop: float = 0.1) -> str:
    if not train or not text:
        return text
    words = text.split()
    keep = [w for w in words if rng.random() >= drop]
    return " ".join(keep) if keep else text
