#!/usr/bin/env python3
"""Frozen (no-LoRA) backbone features for every real clip -> $DECEPTIONX_CACHE/features/.

No labels are used, so these features carry no train/test leakage and can be
shared across CV folds by the fusion network and the tabular branch.

  <bb>.npy           (121, V, 32, D) temporal tokens per eval view (video V=2, audio/text V=1)
  qwen3_full.npy     (121, 1024)     full [EOS] embedding (intent head uses all dims)
  emotion_probs.npy  (121, 9)        emotion2vec+ 9-class probabilities
  ids.json           clip order
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backbones import BACKBONES  # noqa: E402
from src.clip_store import ClipStore, audio_tensor, eval_starts, video_batch  # noqa: E402
from src.rlt_data import CACHE_DIR  # noqa: E402
from src.tuning import limit_to_physical_vram  # noqa: E402

OUT = os.path.join(CACHE_DIR, "features")


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    limit_to_physical_vram()
    os.makedirs(OUT, exist_ok=True)
    st = ClipStore()
    ids = st.ids
    json.dump(ids, open(os.path.join(OUT, "ids.json"), "w"))
    rng = np.random.default_rng(0)
    for name in ["vjepa2", "vjepa2_1", "dasheng", "emotion2vec", "qwen3"]:
        path = os.path.join(OUT, f"{name}.npy")
        if os.path.exists(path):
            print(f"{name}: exists, skipping", flush=True)
            continue
        bb = BACKBONES[name](lora=None).eval()
        feats, extra, extra2 = [], [], []
        for i, cid in enumerate(ids):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if bb.modality == "video":
                    views = [bb(video_batch(st, [cid], bb.n_frames, False, rng, start=s))[0]
                             for s in eval_starts(len(st.frames(cid)))]
                    feats.append(torch.stack(views).float().cpu().numpy())
                elif bb.modality == "audio":
                    a = audio_tensor(st, cid, False, rng)
                    feats.append(bb(a).float().cpu().numpy())
                    if name == "emotion2vec":
                        extra.append(bb.emotion_probs(a)[0].float().cpu().numpy())
                else:
                    t = st.transcript(cid)
                    feats.append(bb([t]).float().cpu().numpy()[:, None, :].repeat(32, 1))
                    extra2.append(bb.embed_full([t])[0].float().cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  {name}: {i + 1}/{len(ids)}", flush=True)
        arr = np.stack(feats).astype(np.float16)
        if arr.ndim == 3:
            arr = arr[:, None]
        np.save(path, arr)
        if extra:
            np.save(os.path.join(OUT, "emotion_probs.npy"), np.stack(extra))
        if extra2:
            np.save(os.path.join(OUT, "qwen3_full.npy"), np.stack(extra2))
        print(f"{name}: saved {arr.shape}", flush=True)
        del bb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
