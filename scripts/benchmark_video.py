#!/usr/bin/env python3
"""Measure the served V-JEPA 2.1 ViT-L `jepa_video` path on this GPU (RTX 5070): cold and warm latency,
peak memory and request size, for one 7 s answer at several capture rates.

Uses the generated golden-fixture video (no person). Reports, separately:
  capture_fps            frames per second the client captured and sent
  model_input_unique_fps unique frames that reach the model per second of answer (128 time-uniform picks;
                         repeats when capture fps x duration < 128)
  inference_updates_per_s fresh scores per second if requests are sent back-to-back (1 / warm latency)

  bash scripts/run.sh scripts/benchmark_video.py
Writes reports/video_benchmark.json
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_mem_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0]) if out else float("nan")


def main():
    import cv2
    import torch

    from scripts.make_golden_fixture import DUR, make_frames
    from src.jepa_service import JEPAService

    frames, _ = make_frames()                      # 15 fps x 7 s, RGB 640x480
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    svc = JEPAService(device="cuda")
    load_s = time.perf_counter() - t0
    res = {"model": svc.info()["model"], "manifest_sha256": svc.info()["manifest_sha256"],
           "device": torch.cuda.get_device_name(0), "answer_seconds": DUR, "service_load_s": round(load_s, 2),
           "runs": []}
    for fps in (5.0, 15.0, 30.0):
        n = int(round(fps * DUR))
        idx = np.clip((np.arange(n) * 15.0 / fps).astype(int), 0, len(frames) - 1)
        enc = []
        for i in idx:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            enc.append(base64.b64encode(buf.tobytes()).decode())
        payload = {"frames": enc, "timestamps": (np.arange(n) / fps).tolist()}
        body = len(json.dumps(payload).encode())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        first = svc.predict(dict(payload))
        cold = time.perf_counter() - t
        warm = []
        for _ in range(5):
            t = time.perf_counter()
            r = svc.predict(dict(payload))
            warm.append(time.perf_counter() - t)
        uniq = r["face"]["unique_frames_used"]
        res["runs"].append({
            "capture_fps": fps, "frames_sent": n, "request_body_mb": round(body / 1e6, 2),
            "model_input_unique_frames": uniq, "model_input_unique_fps": round(uniq / DUR, 2),
            "first_request_s": round(cold, 3), "warm_median_s": round(float(np.median(warm)), 3),
            "warm_max_s": round(float(np.max(warm)), 3),
            "inference_updates_per_s": round(1.0 / float(np.median(warm)), 3),
            "prep_ms_last": r["prep_ms"], "model_ms_last": r["inference_ms"],
            "torch_peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "nvidia_smi_used_mb_after": gpu_mem_used_mb(),
            "scores_identical_across_warm_runs": first["per_fold_probability"] == r["per_fold_probability"]})
        print(res["runs"][-1], flush=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "reports", "video_benchmark.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
