#!/usr/bin/env python3
"""Generate the cross-machine golden parity fixture (no person, no private data) and its expected outputs.

Inputs are GENERATED, deterministic and redistributable: 7 s of synthetic 640x480 video at 15 fps (moving
shapes over a textured background - there is no face, so the documented centre-crop fallback path is
exercised), 7 s of synthetic 16 kHz mono audio (harmonic voiced segments with an F0 contour, pauses and a
low noise floor) and one fixed transcript with fillers. The exact transport bytes (JPEG q95 frames, little-
endian float32 PCM) are stored, so every machine decodes the same input.

Expected outputs are produced by the served code on Node 2:
  cuda_bf16  - exactly as served (autocast BF16), run 3 times to measure repeatability
  cuda_fp32  - every neural model cast to FP32, autocast off (the protocol's reference precision)
  video      - V-JEPA 2.1 ViT-L fold scores, if that export is loaded (--video)
plus the artifact hashes they were produced with. Every stage is recorded: selected frame indices and times,
face boxes / fallback, crop hashes and per-frame means, Qwen 1024-d instructed embedding, 33 raw intent
similarities, emotion2vec 9-class order, the 88 ordered digest values (null = missing), per-fold dasheng
logits, raw and Platt-transformed branch scores, final score, cutoff and decision.

  bash scripts/run.sh scripts/make_golden_fixture.py [--video]
Writes fixtures/golden_v1/{frames/*.jpg, audio_f32le_16k.bin, transcript.txt, timestamps.json, expected.json}
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fixtures", "golden_v1")
FPS, DUR, W, H, SR = 15.0, 7.0, 640, 480, 16000
TRANSCRIPT = ("Um, I was at the office until about six, and then I drove home. I didn't stop anywhere. "
              "Uh, I think I saw the red car outside, maybe, but I- I honestly don't remember the plate?")


def make_frames():
    import cv2

    rng = np.random.default_rng(20261111)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6)
    texture = (texture - texture.min()) / (np.ptp(texture) + 1e-9)
    n = int(FPS * DUR)
    out = []
    for i in range(n):
        t = i / FPS
        bg = np.stack([60 + 80 * xx / W + 40 * texture, 70 + 60 * yy / H + 30 * texture,
                       90 + 50 * texture + 20 * np.sin(2 * np.pi * (xx / W + 0.1 * t))], -1)
        img = np.clip(bg, 0, 255).astype(np.uint8)
        cx, cy = int(W / 2 + 120 * np.sin(2 * np.pi * 0.15 * t)), int(H / 2 + 40 * np.sin(2 * np.pi * 0.4 * t))
        cv2.ellipse(img, (cx, cy), (70, 95), 15 * np.sin(t), 0, 360, (200, 150, 110), -1)
        box = cv2.boxPoints(((W / 2 - 150 + 30 * np.cos(t), H - 90), (110, 50), 40 * t))
        cv2.fillPoly(img, [box.astype(np.int32)], (40, 180, 220))
        cv2.circle(img, (int(80 + 60 * t), 70), 18, (230, 230, 60), -1)
        out.append(img)  # RGB
    return out, (np.arange(n) / FPS).tolist()


def make_audio():
    rng = np.random.default_rng(7)
    n = int(SR * DUR)
    t = np.arange(n) / SR
    voiced = np.zeros(n, bool)
    for a, b in ((0.3, 1.6), (2.0, 3.4), (4.1, 5.2), (5.6, 6.7)):   # four phrases with pauses between
        voiced[int(a * SR):int(b * SR)] = True
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.35 * t) + 10 * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    harm = sum((0.6 ** k) * np.sin((k + 1) * phase) for k in range(8))
    env = np.convolve(voiced.astype(float), np.hanning(1601) / np.hanning(1601).sum(), mode="same")
    x = 0.25 * env * harm * (1 + 0.3 * np.sin(2 * np.pi * 4.0 * t)) + 0.002 * rng.normal(size=n)
    return x.astype("<f4")


def write_inputs():
    import cv2

    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
    frames, ts = make_frames()
    jpgs = []
    for i, f in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        open(os.path.join(OUT, "frames", f"{i:04d}.jpg"), "wb").write(buf.tobytes())
        jpgs.append(buf.tobytes())
    audio = make_audio()
    open(os.path.join(OUT, "audio_f32le_16k.bin"), "wb").write(audio.tobytes())
    open(os.path.join(OUT, "transcript.txt"), "w", encoding="utf-8", newline="\n").write(TRANSCRIPT)
    json.dump(ts, open(os.path.join(OUT, "timestamps.json"), "w"))
    return jpgs, audio, ts


def load_payload():
    names = sorted(os.listdir(os.path.join(OUT, "frames")))
    jpgs = [open(os.path.join(OUT, "frames", n), "rb").read() for n in names]
    audio = open(os.path.join(OUT, "audio_f32le_16k.bin"), "rb").read()
    payload = {"action": "analyze", "frames": [base64.b64encode(j).decode() for j in jpgs],
               "timestamps": json.load(open(os.path.join(OUT, "timestamps.json"))),
               "audio_f32_b64": base64.b64encode(audio).decode(),
               "transcript": open(os.path.join(OUT, "transcript.txt"), encoding="utf-8").read(),
               "request_id": "golden-v1", "source_id": "generated-fixture", "return_intermediates": True}
    files = {f"frames/{n}": hashlib.sha256(j).hexdigest() for n, j in zip(names, jpgs)}
    files["audio_f32le_16k.bin"] = hashlib.sha256(audio).hexdigest()
    files["transcript.txt"] = hashlib.sha256(payload["transcript"].encode()).hexdigest()
    files["timestamps.json"] = hashlib.sha256(open(os.path.join(OUT, "timestamps.json"), "rb").read()).hexdigest()
    return payload, files


def max_abs(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    assert (np.isfinite(a) == np.isfinite(b)).all(), "missingness differs"
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def compare(a, b):
    ia, ib = a["intermediates"], b["intermediates"]
    return {"final_score": abs(ia["final_score"] - ib["final_score"]),
            "dasheng_fold_logits": max_abs(ia["dasheng_fold_logits"], ib["dasheng_fold_logits"]),
            "xgb_digest": abs(ia["branch_raw"]["xgb_digest"] - ib["branch_raw"]["xgb_digest"]),
            "digest_88": max_abs([v for _, v in ia["digest_88_ordered"] if v is not None],
                                 [v for _, v in ib["digest_88_ordered"] if v is not None]),
            "qwen_embedding_1024": max_abs(ia["qwen_embedding_1024"], ib["qwen_embedding_1024"]),
            "emotion2vec_9": max_abs(list(ia["emotion2vec_9_ordered"].values()), list(ib["emotion2vec_9_ordered"].values())),
            "crops_identical": ia["crops_uint8_sha256"] == ib["crops_uint8_sha256"],
            "decision_identical": a["verdict"] == b["verdict"]}


def strip(r):
    keep = ("probability", "threshold", "verdict", "feature_version", "branches", "face", "missing_features",
            "intermediates", "duration_s")
    out = {k: r[k] for k in keep if k in r}
    for b in out.get("branches", {}).values():
        b.pop("top_deceptive_drivers", None), b.pop("top_truthful_drivers", None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="also record V-JEPA 2.1 ViT-L outputs (needs its export)")
    ap.add_argument("--regenerate-inputs", action="store_true")
    args = ap.parse_args()
    import torch

    from src.ensemble_service import EnsembleService
    from src.provenance import git_revision
    from src.v2_socket_server import server_revision

    if args.regenerate_inputs or not os.path.exists(os.path.join(OUT, "audio_f32le_16k.bin")):
        write_inputs()
    payload, files = load_payload()
    svc = EnsembleService(device="cuda")
    runs = [svc.predict(dict(payload)) for _ in range(3)]
    rep = [compare(runs[0], r) for r in runs[1:]]
    exp = {"fixture": "golden_v1", "generated_inputs": "synthetic, no person; see scripts/make_golden_fixture.py",
           "input_sha256": files, "request": {k: v for k, v in payload.items() if k not in ("frames", "audio_f32_b64")},
           "artifacts": server_revision(), "code_revision": git_revision(),
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda_bf16": strip(runs[0]),
           "cuda_bf16_repeatability_max_abs": {k: max(float(r[k]) for r in rep) for k in rep[0]
                                               if not k.endswith("identical")}}
    svc.to_fp32_reference()
    ref = svc.predict(dict(payload))
    exp["cuda_fp32"] = strip(ref)
    exp["bf16_vs_fp32_max_abs"] = compare(runs[0], ref)
    margin = abs(runs[0]["probability"] - runs[0]["threshold"])
    exp["decision_margin"] = {"bf16_score_minus_cutoff": runs[0]["probability"] - runs[0]["threshold"],
                              "near_cutoff": bool(margin < 5 * max(1e-6, exp["bf16_vs_fp32_max_abs"]["final_score"]))}
    exp["suggested_tolerances"] = {
        "note": "measured on Node 2: max(3 x BF16 repeat spread, 1.5 x BF16-vs-FP32 difference); Mac CPU FP32 / "
                "MPS FP16 should be compared to cuda_fp32 with these, and a decision flip is a failure unless "
                "decision_margin.near_cutoff is true",
        **{k: max(3 * exp["cuda_bf16_repeatability_max_abs"].get(k, 0.0), 1.5 * float(v))
           for k, v in exp["bf16_vs_fp32_max_abs"].items() if not isinstance(v, bool)}}
    del svc
    torch.cuda.empty_cache()
    if args.video:
        from src.jepa_service import JEPAService

        vid = JEPAService(device="cuda")
        vp = {k: payload[k] for k in ("frames", "timestamps")}
        v1, v2 = vid.predict(dict(vp)), vid.predict(dict(vp))
        exp["video_cuda_bf16"] = {k: v1[k] for k in ("model", "model_id", "manifest_sha256", "raw_score",
                                                      "per_fold_probability", "temporal_attention_32", "face")}
        exp["video_cuda_bf16_repeatability_max_abs"] = max_abs(v1["per_fold_probability"], v2["per_fold_probability"])
    json.dump(exp, open(os.path.join(OUT, "expected.json"), "w"), indent=1)
    print(json.dumps({k: exp[k] for k in ("cuda_bf16_repeatability_max_abs", "bf16_vs_fp32_max_abs",
                                          "decision_margin")}, indent=1))
    print("bf16:", exp["cuda_bf16"]["probability"], exp["cuda_bf16"]["verdict"], "| fp32:",
          exp["cuda_fp32"]["probability"], exp["cuda_fp32"]["verdict"])


if __name__ == "__main__":
    main()
