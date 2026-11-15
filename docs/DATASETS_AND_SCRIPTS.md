# Datasets, pretrained models and scripts used

This file records every input behind the results in `README.md`, `reports/` and the deployed models in
`models/`, and the exact order of scripts that produced them. Nothing was trained or evaluated on synthetic
data.

## 1. Datasets

### Used: Real-Life Trials (the only training and evaluation data)

| | |
|---|---|
| Paper | Pérez-Rosas, Abouelenien, Mihalcea, Burzo, "Deception Detection using Real-life Trial Data", ICMI 2015 |
| Source | Official University of Michigan download page: https://web.eecs.umich.edu/~mihalcea/downloads.html |
| File | `RealLifeDeceptionDetection.2016.zip`, 248 760 721 bytes, SHA256 `dfa2476ee4929214a1985e1f1f755ebb2115c17e67588c2e832a16ce3a9921dc` |
| Fetch | `bash scripts/fetch_data.sh` (downloads, checks size and SHA256, unzips to `data/raw/real_life_trials/`) |
| Content | 121 courtroom video clips: 61 deceptive, 60 truthful. Each has a human transcript, and there is a MUMIN gesture annotation CSV. |
| Labels | Folder name: `Clips/Deceptive` → 1, `Clips/Truthful` → 0 |
| Used for | Every training run, cross-validation, threshold and calibration fit, and every reported number |
| Not in git | Raw videos (`data/raw/`), the preprocessed cache (`~/deceptionx_cache`, about 7 GB of frames) and the contact sheets (`data/review/`) are gitignored. Redistribution follows the dataset's own terms. |

The MUMIN gesture codes are loaded for reference only. No model uses them as an input feature.

### Derived from it, committed

| file | what | how |
|---|---|---|
| `data/rlt_identity_groups.json` | 57 identity groups (RLT ships no speaker ids). Kept local and not published: it describes the people in the videos. | `scripts/build_identity_groups.py`: face contact sheets reviewed by eye, conservative merges. Five people appear with both labels. Used so that no person is in both train and test. |

### Named in Gemini.pdf but NOT used

| dataset | why not |
|---|---|
| DOLOS | Requires a signed release agreement and an account |
| MU3D | Requires a signed release agreement |
| Bag-of-Lies | Download page returned 404 |
| Box of Lies | Ships annotations and YouTube links only, not the media |

Because there is a single source domain, the PDF's multi-source domain-adaptation term is disabled (λ₃ = 0).

### No live or participant data

No Mac or webcam recordings, participant audio, images or transcripts were used for training, tuning,
calibration or thresholds. Webcam accuracy has not been measured.

## 2. Pretrained models (frozen bases, LoRA-tuned on Real-Life Trials)

All downloads are pinned and verified by `scripts/fetch_models.py`. Hashes, revisions and verification
details are in `models/MODEL_SOURCES.json`.

| model | source | verification | role |
|---|---|---|---|
| `mispeech/dasheng-base` | Hugging Face | git-blob / LFS sha256 match against the Hub | Audio backbone. LoRA-tuned. **Served** in `analyze`. |
| `emotion2vec/emotion2vec_plus_large` (rev `6c303ba987`) | Hugging Face | Encoder weights range-fetched (650 MB of 1.95 GB), zip CRC-checked, safe unpickler | 9-class emotion probabilities are digest features (frozen). LoRA-tuned branch evaluated, not served. |
| `Qwen/Qwen3-Embedding-0.6B` (rev `97b0c614be`) | Hugging Face | LFS sha256 | Frozen intent-similarity features for the digest. LoRA-tuned branch evaluated, not served. |
| `facebook/vjepa2-vitl-fpc64-256` (rev `b3c1679b7c`) | Hugging Face | LFS sha256 | Video backbone. LoRA-tuned and evaluated (AUROC 0.468), not served. |
| V-JEPA 2.1 ViT-B/16 @384 (`vjepa2_1_vitb_dist_vitG_384.pt`) | Meta CDN (`dl.fbaipublicfiles.com`) | `ema_encoder` subtree range-fetched (348 MB of 1.66 GB), CRC-checked | **Historical.** LoRA-tuned (AUROC 0.528). No longer served; its folds are rejected by the video loader. |
| V-JEPA 2.1 ViT-L/16 @384 (`vjepa2_1_vitl_dist_vitG_384.pt`, Meta `vjepa2_1_vit_large_384`, upstream commit `204698b`) | Official: Meta CDN (README model table). **Obtained from Hugging Face re-uploads** while Meta's TLS certificate was expired (2026-11-11 23:59:59 UTC). Meta publishes 2.1 nowhere else. User-approved, `scripts/fetch_vitl_mirror.py` | Accepted only after cross-verification. (1) The same uploader's ViT-B (`mnmly/vjepa2.1-vit-base-384@ee3ee8b`) is bit-identical to the ViT-B we fetched from Meta under valid TLS. (2) Hugging Face sha256 matches at pinned revisions. (3) `mnmly/vjepa2.1-vit-large-384@2594eca` equals the independently converted `apiantonio/vjepa2.1-vit-large-384@d6cfdbd` on all 302 tensors. (4) Strict load as 24 × 1024. No uploader code was run. (5) Once Meta's renewed certificate was served, `scripts/verify_vitl_official.py` confirmed all 302 tensors are bit-identical to Meta's official checkpoint. | Required video backbone (304.7M params, encoder SHA256 `3f0ae79e…`). LoRA-tuned with the official 64-frame profile and served as `jepa_video`. |
| MediaPipe BlazeFace short-range | Google model bucket | MD5 match | Face detection and crops, both in preprocessing and live. |

## 3. Scripts, in the order they were run

Every step runs inside WSL via `scripts/run.sh`, which sets the CUDA libraries, the cache path and PYTHONPATH. `python main.py <command>` wraps each one.

| # | command | script | reads → writes |
|---|---|---|---|
| 1 | — | `scripts/fetch_data.sh` | UMich zip → `data/raw/real_life_trials/` |
| 2 | `fetch-models` | `scripts/fetch_models.py` | Hugging Face / Meta / Google → `models/` + `models/MODEL_SOURCES.json` |
| 3 | `preprocess` | `scripts/preprocess_rlt.py` | Videos → `~/deceptionx_cache/{frames,audio,meta}`: 128 time-uniform frames, BlazeFace 384 px face crops, 16 kHz float audio |
| 4 | `groups` | `scripts/build_identity_groups.py` | Contact sheets → `data/rlt_identity_groups.json` |
| 5 | `tune` | `scripts/tune_backbones.py --backbones vjepa2 vjepa2_1 vjepa2_1_l dasheng emotion2vec qwen3` | LoRA-tunes each backbone under 5-fold identity-grouped CV. Has a watchdog and a retry ladder. Fold files are fingerprinted. Writes `runs/tune_<bb>/` and `reports/tune_*_report.md`. |
| 6 | `features` | `scripts/extract_frozen_features.py` | Frozen embeddings → `~/deceptionx_cache/features/` |
| 7 | `stage2` | `scripts/train_stage2.py` | Builds the 88-feature digest (`v3-fixed-lag-motion`), then XGBoost OOF, ReDA/DARE/DMC fusion, the stacker, the served-policy fit and the honesty checks (permutation null, recording-conditions probe, near-duplicate audit). Writes `runs/stage2/` and `reports/stage2_*`. |
| 7b | — | `scripts/make_golden_fixture.py` | Generated non-person fixture → `fixtures/golden_v1/` with BF16 and FP32 expected outputs at every stage |
| 8 | `nested` | `scripts/nested_deploy_eval.py` | Fully nested evaluation of the served `analyze` policy (25 inner LoRA runs) plus the 7-second diagnostic. Writes `runs/nested_deploy/` and `reports/nested_deploy_*`. |
| 9 | `export` | `scripts/export_deploy.py` (`--ensemble-only`, `--jepa21-only`) | Writes `models/branch2_xgboost.json`, `branch2_digest_schema.json`, `ensemble_deploy.json`, `dasheng_lora/` and `vjepa2_1_lora/` |
| 10 | `test` | `pytest tests/` | 98 passed, 1 skipped (no Branch-1 checkpoint) |
| 11 | `socket` | `run_v2_socket.py` → `src/v2_socket_server.py` | Serves `ping`, `schema`, `features`, `analyze` and `jepa_video` on `0.0.0.0:5002` |
| — | — | `scripts/expose_socket_windows.ps1` (elevated) | Windows port proxy and firewall rule (LocalSubnet only) into WSL |

Supporting modules:

- `src/rlt_data.py`: dataset index
- `src/clip_store.py`: cached clip access and views
- `src/backbones.py`: model wrappers and LoRA
- `src/tuning.py`: training loop, watchdog, metrics, classwise reporting, thresholds
- `src/features_tabular.py`: digest features
- `src/provenance.py`: fingerprints
- `src/ensemble_service.py` and `src/jepa_service.py`: live inference
- `src/jepa_deploy.py`: strict checkpoint loading
- `src/v2_socket_client.py`: client used by the Mac

## 4. Where the numbers live

| result | file |
|---|---|
| Served `analyze` policy, fully nested (AUROC 0.778) | `reports/nested_deploy/` (report, summary, per-clip OOF, fold manifests, inner runs with learning traces), `models/ensemble_deploy.json` → `cv_metrics` |
| Video ablation (with vs without ViT-L) | `runs/nested_video_ablation/` → `reports/nested_video_ablation/` |
| Cross-machine parity fixture | `fixtures/golden_v1/expected.json` (`scripts/make_golden_fixture.py`) |
| Per-branch OOF results and honesty checks | `reports/stage2_report.md`, `reports/stage2_summary.json` |
| Per-backbone tuning | `reports/tune_<backbone>_report.md` |
| Response to the repair audit | `docs/mac-handoff/REPAIR_RESPONSE.md` |
