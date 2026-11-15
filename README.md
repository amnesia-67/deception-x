# DeceptionX — tri-modal deception detection (Gemini.pdf spec), trained on real data

Implementation of the architecture in `Gemini.pdf`, trained and evaluated on **real** video,
audio and transcripts. An earlier version of this repo trained on a synthetic generator that
wrote the label into the features and reported 100% accuracy; that code, its checkpoints and
its reports have been removed.

## Data
Full list of datasets, pretrained models and the scripts that produced every result: `docs/DATASETS_AND_SCRIPTS.md`.

- **Real-Life Trials** (Pérez-Rosas et al., ICMI 2015): 121 courtroom clips, 61 deceptive /
  60 truthful, with human transcripts. Official University of Michigan download (249 MB).
- The other PDF datasets could not be used: DOLOS and MU3D require a signed release agreement
  and account; the Bag-of-Lies page is gone (404); Box-of-Lies ships annotations + YouTube links only.
- The dataset has **no speaker ids**. `data/rlt_identity_groups.json` (kept local, not published: it describes the people in the
  videos) assigns 57 identity groups,
  built by reviewing face contact sheets by eye (conservative merges). Five people appear with
  both labels.

## Models (all verified against official hashes — see `models/MODEL_SOURCES.json`)
| Branch-1 backbone | source | tuned how |
|---|---|---|
| V-JEPA 2 ViT-L `facebook/vjepa2-vitl-fpc64-256` | HF, sha256-verified | LoRA r8 on last 4 blocks, predictor skipped |
| V-JEPA 2.1 ViT-B/16 @384 (historical; not served) | Meta CDN (encoder weights range-fetched, CRC-checked) | LoRA |
| **V-JEPA 2.1 ViT-L/16 @384** (`vjepa2_1_vit_large_384`, required video model) | Hugging Face re-uploads of Meta's `vjepa2_1_vitl_dist_vitG_384.pt` encoder, cross-verified while Meta's CDN certificate was expired, then confirmed bit-identical to Meta's official file (see `docs/DATASETS_AND_SCRIPTS.md`) | LoRA, official 64-frame profile |
| `mispeech/dasheng-base` | HF | LoRA |
| `emotion2vec_plus_large` | HF (model weights range-fetched) | LoRA |
| `Qwen/Qwen3-Embedding-0.6B` (instruction prefix, MRL-128) | HF | LoRA |
| MediaPipe BlazeFace (face crops) | Google model bucket, MD5-verified | frozen |

## Results (out-of-fold, 5-fold identity-grouped CV, no person in both train and test)
**Served `analyze` policy, fully nested (base models, stacker and threshold all re-fitted inside every
outer fold; `reports/nested_deploy/report.md`): AUROC 0.778 (95% CI 0.68–0.90).** With the threshold re-fitted
inside each outer fold (0.448–0.485, which evaluates the threshold-fitting *procedure*), it catches 53/61 lies (87%)
and passes 37/60 truths (62%). The single cutoff exported for serving is 0.424; applied to the same nested scores
it catches 54/61 lies (89%) but passes only 30/60 truths (50%), calling 69% of answers deceptive. That cutoff was
fitted on all clips, so this figure is for reference, not an untouched estimate. At 0.5: 75% and 68%.
The model now errs towards "deceptive" rather than "truthful". Always-truthful baseline: balanced accuracy 0.50.

Per branch (stage 2; class 1 = deceptive; rates at 0.5):

| branch | AUROC (95% CI, group bootstrap) | accuracy@0.5 | lies caught | truths passed |
|---|---|---|---|---|
| dasheng (audio, LoRA) | **0.764** (0.63–0.92) | 0.694 | 56% | 83% |
| XGBoost behavioural digest (88 features, `v3-fixed-lag-motion`) | **0.763** (0.68–0.87) | 0.661 | 56% | 77% |
| ReDA + DARE + DMC fusion (frozen tokens) | 0.695 (0.60–0.81) | 0.620 | 41% | 83% |
| V-JEPA 2.1 ViT-L (video, LoRA, 64-frame profile) | 0.655 (0.45–0.80) | 0.595 | 48% | 72% |
| Qwen3-Embedding (text, LoRA) | 0.570 (0.46–0.69) | 0.545 | 49% | 60% |
| V-JEPA 2.1 ViT-B (video, LoRA; historical) | 0.528 (0.38–0.75) | 0.521 | 44% | 60% |
| V-JEPA 2 ViT-L (video, LoRA) | 0.468 (0.36–0.64) | 0.512 | 18% | 85% |
| emotion2vec+ (audio, LoRA) | 0.446 (0.34–0.56) | 0.446 | 44% | 45% |
| Ensemble, 7 branches stacked (meta-level nesting only — superseded) | 0.724 (0.61–0.93) | 0.636 | 52% | 75% |
| Ensemble, unweighted mean | 0.745 (0.65–0.88) | 0.628 | 39% | 87% |

Honesty checks (`reports/stage2_report.md`):
- **Label-permutation null** for the digest, run with exactly the real procedure (400 trees): shuffled
  labels give mean AUROC 0.496 (95th pct 0.610) vs 0.763 real, p ≈ 0.01.
- **Near-duplicate audit**: no clip pair from different identities has frozen audio or video embedding
  cosine ≥ 0.995 (max 0.967 / 0.836).
- **Duration shift**: dasheng on the central 7 s of each held-out clip (the live answer length) gives
  AUROC 0.736 vs 0.764 on the whole answer, and calls fewer answers deceptive (31% vs 36%).
- **Recording-conditions probe**: resolution, fps, duration, loudness, noise floor and crop-border
  colour *alone* reach AUROC 0.669. Part of any signal on this dataset may be courtroom/channel
  artefacts rather than deception. Clips were grouped by person, not by trial (no trial metadata).
- With 121 clips, per-fold AUROC swings widely (e.g. dasheng 0.44–1.00); trust the pooled CIs.
- The video backbones do not beat chance here; the stacker (fit on training folds only) gives them ~0 weight.

This is a research prototype. It is **not** suitable for real-world decisions about whether someone is lying.

## Protocol details
- Outer 5-fold `StratifiedGroupKFold` over identity groups; inner identity-balanced split
  (no identity > 25% of validation) for early stopping, the sub-par watchdog and choosing among
  3 retry configs; the chosen config is refit on all outer-train clips; test folds are used only to score.
- Every epoch writes `runs/<run>/metrics.jsonl` + live `status.json` (loss, AUROC, clips/s,
  nvidia-smi utilisation, VRAM). The watchdog terminates runs with non-finite loss, divergence,
  no learning, or <5% GPU utilisation, then retries the next config.
- Micro-batch is measured by a VRAM probe (video runs: 4 × 2 accumulation, 2 × 4 for the rank-16
  retry config; effective batch 8 everywhere) under a hard cap at the VRAM actually free, so WSL
  never spills into shared memory.

## Running (Windows; every command runs inside WSL on the RTX 5070)
```
python main.py fetch-models     # verified downloads
python main.py preprocess       # face crops + 16 kHz audio -> ~/deceptionx_cache
python main.py tune             # LoRA-tune all five backbones (~1.5 h)
python main.py features
python main.py stage2           # fusion, digest, honesty checks, ensemble
python main.py export           # deployable digest model + schema
python main.py socket           # model server on 0.0.0.0:5002
python main.py test             # GPU test suite
```
Data: download `RealLifeDeceptionDetection.2016.zip` from the official UMich page into
`data/raw/real_life_trials/` and unzip it. Training runs resume from finished folds (`--force` to redo).

## Socket server (Node 2) and reaching it from outside WSL
`python main.py socket` serves the real-data digest model. Protocol: one JSON object per line —
`{"action":"ping"}`, `{"action":"schema"}` (feature names, CV metrics, caveats), or
`{"features": {"<name>": value, ...}}` / `{"features": [..88 values..]}` / raw float32 bytes.
Replies carry the probability, verdict and TreeSHAP drivers.

**Best output: `analyze`** (stacked ensemble, fully nested AUROC 0.778): the Mac sends one answer's raw frames + timestamps, float32 16 kHz audio and the transcript; Node 2 computes every feature and runs the 5 tuned dasheng folds + the digest. Contract: `docs/mac-handoff/MAC_CLIENT_CONTRACT.md`; re-export with `python scripts/export_deploy.py --ensemble-only`.

**V-JEPA 2.1 ViT-L classifier** (`{"action": "jepa_video", "frames": [<base64 JPEG>, ...], "timestamps": [...]}`):
serves `models/vjepa2_1_l_lora` (5 LoRA folds of Meta's `vjepa2_1_vit_large_384`) and nothing else. ViT-B and
V-JEPA 2 folds are rejected, and if the ViT-L export is missing the endpoint is unavailable (`ping` gives
the reason), with no fallback. Frames are selected by time and face-cropped exactly as in training. Each answer
is scored on two 64-frame views (stride 1, the official L profile), giving 32 native tubelets of width 1024.
The reply is an uncalibrated classifier score with no verdict.

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

5-fold identity-grouped OOF AUROC **0.655** (95% CI 0.45–0.80; per fold 0.70, 0.83, 0.52, 0.73, 0.80). The
interval still includes chance. On the RTX 5070 one 7 s answer takes about **4.2 s** warm (0.5 s frame
preparation, 3.7 s model), with 2.1 GB peak allocation, so it gives one score per answer, not a live stream
(`reports/video_benchmark.json`). The frozen base blocks (0–15 or 0–19) are identical in every fold, so they
run once per view and only each fold's LoRA blocks run per fold. Scores are bit-identical to the full per-fold
forward (tested); the full path took 9.3 s.

```bash
python scripts/export_deploy.py --jepa21-only   # packages runs/tune_vjepa2_1_l into a FRESH models/vjepa2_1_l_lora
python scripts/benchmark_video.py               # cold/warm latency, memory, request size on this GPU
```

The export records full fold/base hashes and tensor/config schemas. The service requires all five
folds, restricted `weights_only=True` loading, exact complete LoRA/head keys, matching shapes/dtypes
and finite tensors. A mismatch fails startup; no previous-fold parameters are silently retained.
Scores remain experimental and must not be presented as evidence of deception.

Start without the video model: `python main.py socket --no-jepa`.

WSL 2 forwards only Windows *localhost* into Linux. To let other machines (Node 1) connect,
run once from an **elevated** PowerShell (re-run after reboots, the WSL IP changes):
```
powershell -ExecutionPolicy Bypass -File scripts\expose_socket_windows.ps1 -Port 5002
```
It adds a `netsh portproxy` (0.0.0.0:5002 → WSL IP) and an inbound firewall rule on the Private
profile; `-Remove` undoes both.

## Layout
`src/backbones.py` backbones + LoRA · `src/tuning.py` CV trainer, watchdog, reports ·
`src/clip_store.py` data/augmentation · `src/features_tabular.py` digest · `src/multimodal_fusion.py`
ReDA/DARE/DMC · `scripts/` pipeline steps · `third_party/` vendored V-JEPA 2.1 encoder and emotion2vec (MIT) ·
`reports/` result reports.

