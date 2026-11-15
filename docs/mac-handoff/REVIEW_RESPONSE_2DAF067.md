# Response to `MODEL_PIPELINE_REPAIR_REVIEW_2DAF067.md` (Node 2)

## 1. V-JEPA 2.1 ViT-L: trained, exported and benchmarked

**Weights.** The only official source is `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt`
(Meta's README model table, commit `204698b`). That host's TLS certificate (`notAfter=Nov 11 23:59:59 2026 GMT`)
expired, and TLS verification was not bypassed. With the user's approval, the encoder was taken from Hugging Face
re-uploads and accepted only after cross-verification (`scripts/fetch_vitl_mirror.py`):
1. The same uploader's ViT-B is bit-identical to the ViT-B we fetched from Meta under valid TLS (158 tensors).
2. Each file matches Hugging Face's sha256 at a pinned revision.
3. Two independent uploads (mnmly, and apiantonio's Transformers-layout conversion mapped back to `qkv`) agree
   on all 302 tensors.
4. It loads strictly into the vendored `vit_large` (24 × 1024, 304.7M parameters).

No uploader code was run. Once Meta's renewed certificate was served (DigiCert, valid until 2027-04-25),
`scripts/verify_vitl_official.py` range-fetched the official `ema_encoder` under verified TLS. **All 302 tensors
are bit-identical to Meta's official checkpoint.** The result is recorded in
`models/vjepa2_1-vitl-384/manifest.json` and `models/MODEL_SOURCES.json`.

**5-fold LoRA result** (`runs/tune_vjepa2_1_l`, same identity-grouped folds as every other branch):

| | value |
|---|---|
| pooled OOF AUROC | **0.655** (95% CI 0.45–0.80) |
| per-fold AUROC | 0.704, 0.833, 0.519, 0.727, 0.800 |
| accuracy at 0.5 | 0.595; lies caught 48%, truths passed 72% |
| configs | r8 on the last 4 blocks (folds 0, 2, 3, 4); r16 on the last 8 (fold 1: all three ladder configs were sub-par on inner validation, at 0.42, 0.31 and 0.53, and the best one was refit, which is the standard rule; its test AUROC of 0.83 was not used to choose it) |
| training | micro-batch 2 × accumulation 4, 6.9 GB peak, 0.77 clips/s, GPU 99% |

This is better than ViT-B (0.528), but the interval still includes chance.

**Does video help the verdict? No.** In the fully nested, paired comparison (`scripts/nested_video_ablation.py`,
`reports/nested_video_ablation/`), ViT-L was LoRA-tuned inside every outer-train set (25 inner runs). Both
rules then scored the same unseen people:

| rule | AUROC (95% CI) | lies caught | truths passed |
|---|---|---|---|
| video alone (mean of fold models, at 0.5) | 0.648 (0.54–0.76) | 27/61 | 41/60 |
| dasheng + digest (served) | 0.778 (0.68–0.90) | 53/61 | 37/60 |
| dasheng + digest + ViT-L | 0.773 (0.66–0.90) | 53/61 | 37/60 |

Adding video changes AUROC by −0.005 (paired identity bootstrap 95% CI −0.019 to +0.010), and it is better in
only 28% of resamples. The stacker gave video weights of 0, 0.05, 0.24, 0 and 0 across the outer folds, and no
fold's AUROC moved. ViT-L carries a modest signal on its own, but nothing that audio and the digest do not
already capture. **Video stays out of the served verdict.** `jepa_video` remains available as a separate,
uncalibrated score.

**Benchmark** (`reports/video_benchmark.json`, RTX 5070, one 7 s answer, BF16):

| capture fps | request body | unique model frames | first request | warm median |
|---|---|---|---|---|
| 5 | 1.1 MB | 35 | 5.3 s | 4.1 s |
| 15 | 3.2 MB | 105 | 4.2 s | 4.2 s |
| 30 | 6.4 MB | 128 | 4.3 s | 4.3 s |

Preparation takes 0.5–0.6 s and the model 3.7 s. Each view has 18,432 tokens. Every fold adapts only its last
4 or 8 blocks, so the frozen base blocks run once per view and are shared. Only each fold's LoRA blocks run per
fold. `tests/test_golden_fixture.py` checks that this gives exactly the same per-fold scores and attention as the
full per-fold forward, which took 9.3 s. Peak allocation is 2.1 GB, and repeated runs are bit-identical. That is
about 0.24 scores per second: one score per answer.

**Ready and tested without the weights:**

| requirement | status |
|---|---|
| Genuine 2.1 L encoder | `src/backbones.py::VJEPA21LVideo` builds `vit_large` (1024 wide, 24 blocks, 16 heads; the constructor checks this) with Meta's builder kwargs, and loads `models/vjepa2_1-vitl-384/encoder.safetensors` (`ema_encoder`) strictly. The vendored encoder code equals upstream `204698b` except import paths. No B adapters are reused: L folds are trained from scratch. |
| Official L profile, not B's 32-frame path | 64 frames per view (`num_frames=64`, tubelet 2, 384 px, patch 16, RoPE with interpolation) → 32 native tubelets = 32 steps, with **no temporal interpolation**. Two views (first and second half of the 128 time-uniform frames, stride 1), logits averaged. Declared in `src/jepa_deploy.py::CONTRACT`. |
| Strict L identity, no fallback | Manifest format `sunhacks-vjepa21-vitl-classifier-v1`. Model `vjepa2_1_vitl_lora_5fold`, id `facebookresearch/vjepa2:vjepa2_1_vit_large_384`. The loader requires backbone `vjepa2_1_l`, 64 frames and 1024 width. `vjepa2_1` (B) and `vjepa2` folds are rejected by name. Base-encoder SHA256, fold hashes and full state schemas come from the actual L model. If the export is missing, `jepa_video` errors and `ping.jepa_unavailable_reason` explains why. The B export in `models/vjepa2_1_lora/` is historical and loaded by nothing. |
| Crop, normalisation and sampling | Shared with `analyze` and training (`src/live_frames.py`): the frame nearest each of 128 uniform times (timestamps or fps required), long side ≤ 960 px, BlazeFace, median-smoothed boxes, 384² crop, ImageNet normalisation. Replies echo `request_id` and `source_id` and carry `server_revision`. |
| RTX 5070 latency and memory | `scripts/benchmark_video.py` reports cold and warm latency, peak memory and request size at 5, 15 and 30 fps capture. It separates capture fps, unique model-input fps and inference updates per second. **Done** (above). |
| L-inclusive fusion | `scripts/nested_video_ablation.py`: a fully nested, paired with/without-ViT-L comparison on the same unseen people, using the same stacker and threshold recipe. Video is added to the exported verdict only if the paired ΔAUROC CI is above 0. Otherwise the report says it does not help. **Done:** ΔAUROC −0.005 (CI −0.019 to +0.010), so video is not added. |

## 2. Evaluation wording corrected

The 53/61 lies and 37/60 truths come from thresholds **re-fitted inside each outer fold** (0.448–0.485). They
evaluate the threshold-fitting procedure. The manifest now labels them `at_threshold_refit_per_outer_fold` and
adds `at_fixed_exported_cutoff_on_nested_scores`.

At the fixed 0.424 cutoff, the nested scores give **54/61 lies, 30/60 truths, 69% called deceptive**. That cutoff
was fitted on all clips, so this figure is reference only. The cutoff is lower than the per-fold values because
it was fitted on single-model OOF scores while serving averages five fold models; this is the audit's
aggregation mismatch, now measured. It is **not** re-tuned on these evaluation scores. Treat it as provisional
until there is target-domain calibration data. Updated in README, REPAIR_RESPONSE, NEW_DEPLOYMENT_HANDOFF and
MAC_CLIENT_CONTRACT.

The served weights, Platt parameters and cutoff are unchanged:

| field | value |
|---|---|
| weights | 0.5379349436053581 / 0.462065056394642 |
| cutoff | 0.4238521751969923 |
| digest model and schema | same SHA256 |
| dasheng fold bytes | same |

Only `models/ensemble_deploy.json` changed, because of the new `cv_metrics` labels and provenance, so its SHA256 changed.

The re-run of the nested evaluation reproduced every number exactly (AUROC 0.778, identical per-fold results).

## 3. Raw predictions, manifests, traces and immutable provenance

- **Published files.** `reports/nested_deploy/` now holds:
  - the per-clip OOF predictions (`oof.csv`: dataset clip ids, labels, served-aggregation scores, per-fold thresholds, 7 s diagnostic)
  - the 5 fold manifests, each with leakage asserts
  - all 25 inner runs, each with its config, attempts and **learning traces**: init and returned state SHA256, optimizer steps, per-epoch loss, grad norm and val AUROC, update L2, refit trace

  The inner-run cache now keeps these fields.
- **Provenance.** Every export and summary now records `source_tree_sha256`, a line-ending-normalised content hash of all
  51 source files that can affect training, export or serving, next to the commit and dirty flag. It
  identifies uncommitted code exactly. Fold files remain bound to the cache, code, base-weight and ladder fingerprints.

## 4. Golden cross-machine fixture

`fixtures/golden_v1/` (`scripts/make_golden_fixture.py`) holds generated inputs only: 7 s of synthetic 640×480 video at 15 fps
with moving shapes and no face (so it exercises the centre-crop fallback), synthetic 16 kHz audio with pauses,
and a fixed transcript with fillers. The exact JPEG and float32 bytes are stored with their SHA256.

`expected.json` records every stage, for both `cuda_bf16` (as served) and `cuda_fp32` (all models FP32,
autocast off):
- selected frame indices and times
- boxes and fallback
- crop hash and per-frame means
- Qwen 1024-d embedding
- 33 raw intent similarities
- emotion2vec 9-class order
- the 88 ordered digest values, with missingness
- per-fold dasheng logits
- raw and Platt-transformed branch scores
- final score, cutoff and decision

It also records the artifact hashes, `source_tree_sha256`, request and source ids, and measured tolerances:

| check | result |
|---|---|
| BF16 repeatability, 3 runs | bit-identical (0.0) |
| BF16 vs FP32, final score | 7.5e-5 |
| BF16 vs FP32, dasheng logits | 6.0e-3 |
| BF16 vs FP32, digest values | 3.9e-3 |
| Decision | identical, 0.109 from the cutoff |

The Mac compares its CPU FP32 or MPS FP16 run against `cuda_fp32` using `suggested_tolerances`. The ViT-L
section (`video_cuda_bf16`: per-fold scores, attention, frame use) is now included and bit-repeatable;
`tests/test_golden_fixture.py` checks the served ViT-L path against it. `tests/test_golden_fixture.py` checks that the
served code reproduces the fixture.

## 5. Privacy split acknowledged

`MAC_CLIENT_CONTRACT.md` now leads with the Mac team's split. Audio, ASR, the feature extractors and the
dasheng/digest/stacker run locally on the Mac from the exported artifacts, checked against the golden fixture.
Only video goes to Node 2 (`jepa_video`). `analyze` stays documented only for an explicit user decision to send
audio and transcript.

## Still open

- A frozen, consented, labelled 7-second webcam/ASR evaluation with unseen people.
- Whisper-vs-human transcript drift.
- Target-domain calibration.
- Licensed additional datasets.
