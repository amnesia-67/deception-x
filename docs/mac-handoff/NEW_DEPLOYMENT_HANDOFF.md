# Deployment handoff — status from the Windows / RTX 5070 node (Node 2)

> Updated after `MODEL_PIPELINE_REPAIR_HANDOFF.md`: see [REPAIR_RESPONSE.md](REPAIR_RESPONSE.md) for what changed.
> Numbers in the table below are current; the item-by-item section further down describes the earlier round.

Base: reviewed snapshot `a64490e` (which contains `6e30a90`) plus the changes listed here. Server:
`192.168.0.224:5002` (WSL 2 behind a Windows port proxy; inbound rule limited to LocalSubnet).
Nothing below asks for datasets, participant recordings or new training.

## Endpoints now served (one JSON object per line, newline-terminated)

| action | model | input | headline CV result (unseen people) |
|---|---|---|---|
| `analyze` | **stacked ensemble** = 0.54 × LoRA-tuned dasheng (5 folds) + 0.46 × 88-feature digest XGBoost, threshold 0.424 | frames + float32 16 kHz audio + transcript + timestamps | **AUROC 0.778** (95% CI 0.68–0.90), fully nested. Per-fold re-fitted thresholds: 87% of lies / 62% of truths; fixed exported cutoff on nested scores (reference only): 89% / 50% |
| `jepa_video` | **V-JEPA 2.1 ViT-L** LoRA, 5 folds (official 64-frame profile); ViT-B is historical and rejected | raw frames + timestamps | AUROC 0.655 (0.45–0.80), 5-fold OOF; about 4.2 s per 7 s answer on the RTX 5070; not in the verdict: the nested ablation found no gain (ΔAUROC −0.005, CI −0.019 to +0.010) |
| `features` / `schema` | 88-feature digest XGBoost (`v3-fixed-lag-motion`) | 88 named values | AUROC 0.763 (0.68–0.87) |
| `ping` | — | — | lists loaded models, CV metrics, caveats |

**Best output = `analyze`.** The Mac sends raw media; every model and feature extractor runs on Node 2,
so the Mac needs no model weights. Exact request/response contract: [MAC_CLIENT_CONTRACT.md](MAC_CLIENT_CONTRACT.md).

## Item-by-item response to the previous checklist

1. **V-JEPA 2.1 export — done.** `VJEPA21_SERVICE.patch` was reviewed and applied (CRLF-only differences),
   its 20 CPU tests pass, and the existing `runs/tune_vjepa2_1` folds were packaged with
   `scripts/export_deploy.py --jepa21-only` into `models/vjepa2_1_lora/` (manifest SHA256
   `d93d19e7…f03d`, base encoder `5fb693d3…b5be`, 5 distinct 608 873-byte folds). GPU parity verified:
   the served fold that held a clip out reproduces that clip's stored out-of-fold probability
   (`tests/test_v2_socket.py::test_jepa_video_reproduces_cross_validation_prediction`). Serving
   models are frozen (`requires_grad=False`). No V-JEPA 2 fallback; `models/vjepa2_lora/` (the old
   ViT-L folds) is not loaded by anything and is not in the public release.

2. **Manifests, safe states, strict loading — done for both served neural models.**
   - `models/ensemble_deploy.json` (SHA256 `281d0f01…1516`): format `sunhacks-ensemble-v1`, class 1 =
     deceptive, Platt A/B + convex weights, dropped-branch weights (all < 2e-6), CV metrics, caveats,
     and for each dasheng fold its SHA256, size, primitive config (r=8, alpha=16, last 4 blocks,
     `attn.qkv|attn.proj`) and full LoRA/head tensor schema.
   - Fold files contain only `{"config", "state"}` (test ids/labels/probabilities stripped), are
     loaded with `torch.load(weights_only=True)` from hash-checked bytes, and go through
     `jepa_deploy.validate_fold_state` + `load_fold_state_strict` (missing **and** unknown keys,
     shape/dtype, non-finite values all rejected before any mutation).
   - **A real stale-parameter bug was found and fixed here:** `torch.autocast` caches bf16 copies of
     `requires_grad` weights for the duration of an autocast region. Swapping fold weights inside one
     region silently reused the first fold's adapters. Fixes: serving models have
     `requires_grad=False`, and each fold runs in its own autocast region. Regression test:
     `test_analyze_from_raw_media_reproduces_training` (fold that held the clip out must reproduce its OOF value).
   - Artifact hashes: `branch2_xgboost.json ae232e41…c3ec`, `branch2_digest_schema.json 85779f9d…bd3bce`,
     dasheng folds `4739628e…`, `ab8d2c68…`, `450bed6e…`, `0c5aec1b…`, `82b982db…`; dasheng base
     `adaa439e…36c3`; BlazeFace `b4578f35…152f`; Qwen3-Embedding rev `97b0c614be…`, emotion2vec+ rev
     `6c303ba987…`. Full list: `models/MODEL_SOURCES.json` + the manifests above.
   - Node-2 stack: torch 2.11.0+cu130, transformers 5.22.2, peft 0.21.2, xgboost 3.4.1,
     mediapipe 1.2.2, opencv 5.0.0, numpy 2.4.4, safetensors 0.8.0.

3. **Temporal contract — explicit for `analyze`.** Send per-frame `timestamps` (seconds) or `fps`. The
   server selects, **by time**, the frame nearest each of 128 uniformly spaced times over the answer
   (as training did), crops faces at ≤ 960 px with MediaPipe, and computes head-motion speeds from the
   real timestamps. Duration defaults to the frame span (override with `duration_s`). Training clips
   were whole answers of 4.5–81 s (median 27 s): send one complete answer per request, not sliding
   windows. Answers longer than ~16 s retain fewer than 8 fps of unique frames; this matches training
   (128 frames per clip regardless of length). `jepa_video` still uses the patch's count-based
   resampling (documented there).

4. **Parity and throughput — partly done.** Parity fixtures use cached real clips (they need the local
   cache, so they run on Node 2 only); raw-camera preprocessing parity is checked by decoding the
   original video files through the same path the Mac uses. `analyze` returns `timings` with
   `audio_decode_ms`, `frame_decode_face_crop_ms`, `handcrafted_features_ms`, `gpu_queue_wait_ms`,
   `gpu_models_ms` and total `latency_ms`. Measured on the RTX 5070: ~0.6–0.7 s per answer warm
   (128 frames, ~25–35 s audio), ~2 s on the first request. Network encode/upload is on the Mac side
   and not measured here. **Still open:** a non-person synthetic fixture for the Mac, bounded
   queue/backpressure and request timestamps echoed in replies.

5. **Fusion network — intentionally not exported.** The stage-2 stacker (fit on out-of-fold predictions
   only) gave the ReDA/DARE/DMC fusion branch weight 1.9e-6 and the tuned V-JEPA 2, V-JEPA 2.1,
   emotion2vec and Qwen3 branches ≤ 1.2e-7, so they are dropped from `analyze` with negligible effect.
   No old synthetic head is substituted anywhere.

6. **89-field semantics — server-side.** For `analyze` the server computes all 89 features itself with
   `src/features_tabular.py` (`v2-no-within-z`), so the Mac does not need extractors. Parity: text,
   voice, emotion and intent features match the training cache to < 0.2 % relative error; face-motion
   features to ~5–7 % (JPEG + re-detection). Missing values are NaN (XGBoost default direction); the
   reply lists `missing_features`.

7. **Honest limits — retained.** Every reply carries `cv_metrics` and `caveats`. Scores are Platt-scaled
   on courtroom data, not calibrated for webcam input. Recording conditions alone predict the training
   labels with AUROC 0.669, the stacker's nesting is incomplete at the base level, 121 clips / 57
   people give wide intervals, and nothing here is evidence that a person is lying.

## What to run

Node 2 (Windows, this repo):
```
python main.py socket          # loads digest + ensemble + V-JEPA 2.1, binds 0.0.0.0:5002
```
After a reboot, re-run (elevated) `scripts\expose_socket_windows.ps1 -Port 5002 -Profile Public`
(the WSL IP changes). Mac: see [MAC_CLIENT_CONTRACT.md](MAC_CLIENT_CONTRACT.md).
