# Mac model handoff (2026-11-13; webcam-pilot profile updated 2026-11-15)

For the agent setting up the Mac (Node 1). These are the latest model files the Mac runs locally. The default
(served) models are the same ones the Node 2 server on port 5002 serves, and they have not changed since
2026-11-11. The webcam-pilot models are an opt-in second profile, described below.

## Get them

```bash
git clone https://github.com/amnesia-67/deception-x.git && cd deception-x
git lfs pull                                 # models/dasheng-base/model.safetensors is an LFS object
```

## Files that run on the Mac

| file | what it is | sha256 of the committed bytes |
|---|---|---|
| `models/ensemble_deploy.json` | stacker (`sunhacks-ensemble-v2`): Platt calibration, weights 0.5379 / 0.4621, cutoff 0.4238521751969923, feature version `v3-fixed-lag-motion` | `6b7a3dd8394c873a99fe55f8ec3232b1ae00f2e25579f3d85594fb8fcf0217dc` |
| `models/dasheng_lora/fold0.pt` | tuned dasheng audio model, fold 0 | `4739628e4a4d5e97339257baaf1484cb21c5c74600218424f06cebb67ec8b11f` |
| `models/dasheng_lora/fold1.pt` | fold 1 | `ab8d2c688b86c7ff16cab0eac59024997b5b18c7843df05756ee2ff25331bc6c` |
| `models/dasheng_lora/fold2.pt` | fold 2 | `450bed6e4dc915561fd9403c597e9a65cec0af2c9ebed9b2935197d34e0f69d8` |
| `models/dasheng_lora/fold3.pt` | fold 3 | `0c5aec1b0854c4ca1eb6fb18e9c111b3b585525b101b6cfee8569d40d583b514` |
| `models/dasheng_lora/fold4.pt` | fold 4 | `82b982db2da144e8a670dd461cfcc0bb1ecef862abdd2db3b9fbd7ad4e1076de` |
| `models/branch2_xgboost.json` | 88-feature text/behaviour digest XGBoost | `f501ea925d0cc2f8d15776b239d97463963f2a83e4a155303420eaed36f563c1` |
| `models/branch2_digest_schema.json` | names and order of the 88 digest features | `bbed857c527031550a878e01e4be8670fcd2a169eea07a482389c4ead31b5356` |
| `src/v2_socket_client.py` | client for Node 2 (`jepa_video`, and `analyze` only with consent) | `7f292d84237be22bd33b41f23258e1e16230e870e09380db6b4b7136b7e0b5aa` |
| `fixtures/golden_v1/` | generated parity fixture (no real person); `expected.json` | `3ebb39d7c80673b3095b5cc2ae3c0c265dd7ac4e92e23cd601141fc588af9089` |

Hashes are of LF bytes as stored in git (`git show HEAD:<path> | shasum -a 256`), so they match a Mac clone.

Base models the digest and audio branch also need, pinned in `models/MODEL_SOURCES.json` (check the hashes there
before use): dasheng-base (in the repo via LFS), `Qwen/Qwen3-Embedding-0.6B`, `emotion2vec/emotion2vec_plus_large`,
and faster-whisper large-v3-turbo for the transcript.

## Check the setup

Run the golden fixture through the local pipeline and compare with `cuda_fp32` in `fixtures/golden_v1/expected.json`
using its `suggested_tolerances`. A flipped decision is a failure unless `decision_margin.near_cutoff` is true.

## Rules (full detail in `MAC_CLIENT_CONTRACT.md`)

- Only answer video goes to Node 2 (`192.168.0.224:5002`, `jepa_video`). Raw audio and transcripts stay on the Mac.
- Video is not part of the verdict. Show the `jepa_video` score separately as uncalibrated, and never add a video
  weight on the Mac (nested ablation: ΔAUROC −0.005, CI −0.019 to +0.010).
- The verdict is `probability >= 0.4238521751969923`, not 0.5.
- Show the result as an experimental score. Fully nested AUROC on unseen courtroom speakers is 0.778
  (CI 0.68–0.90). Webcam accuracy is not established: on the 206-clip SunHacks webcam pilot these served models
  score near chance (audio 0.57, video 0.47 AUROC).
- `ping` returns `server_revision`: the commit Node 2 serves (started with `--webcam-pilot`) and the hash of every
  served manifest. `docs/mac-handoff/SERVED_SOURCE_5002.md` says how to check it against this repository.

## Webcam-pilot profile (opt-in, not the default)

Models LoRA-tuned on the final SunHacks webcam export `sunhacks-webcam-pilot-850-v3-2026-11-15` (850 × 7 s clips,
5 people, self-reported labels), commit `143e924`. They replace the 206-clip models. The served courtroom models
above stay the default. Use this profile only when it is explicitly selected, and label its output as experimental.

How it was made (`scripts/pilot_final.py`; fixed before any training):
- **Split.** All 50 of P5's clips are held out as the **unseen-person test**: never trained on and never used
  for any choice. The other four people (P1, P2, P3, P4) use the export's 75/15/10 split: 600 train,
  120 validation, 80 test. The test is reported as the 40 clips used in earlier runs and the 40 new ones.
- **Training.** Each branch is a fresh LoRA on its base model, trained two ways: `bce` (the deployed recipe) and
  `pairwise` (same-person lie/truth pairs). The config ladder moves on only while validation is below 0.55, and runs
  that are not learning are stopped at epoch 4.
- **Selection.** The method per branch, then the branch combination, was chosen by validation within-person AUROC
  (ranking a person's lies above the same person's truths). Calibration (Platt) and the cutoff (prior-matched) were
  fitted on validation. Test and P5 were scored once, afterwards.
- **Chosen:** **dasheng alone** (validation within-person AUROC 0.704). The best combination scored 0.698
  (dasheng + emotion2vec), and all three 0.694.
- **Not finished:** ViT-L `pairwise` was stopped after epoch 3 for the presentation deadline (0.559 at epoch 3, against
  0.620 for the ViT-L `bce` model shipped).

Results of the fused verdict (= dasheng):

| set | clips | accuracy | AUROC (95% CI) | lies caught | truths passed | session-baseline accuracy |
|---|---|---|---|---|---|---|
| validation (selection set, optimistic) | 120 | 66.7% | 0.724 (0.63–0.81) | 40/60 | 40/60 | 63.9% (108 with a verdict) |
| **test, 4 familiar people** | 80 | **73.8%** | **0.802** (0.70–0.89) | 31/40 | 28/40 | 72.1% (68) |
| test, new 40 (never evaluated before) | 40 | 77.5% | 0.868 (0.74–0.97) | 18/20 | 13/20 | 71.4% (28) |
| test, previous 40 (used in earlier runs) | 40 | 70.0% | 0.734 (0.57–0.88) | 13/20 | 15/20 | 60.7% (28) |
| **P5, unseen person** | 50 | **50.0%** | **0.574** (0.42–0.74) | 3/25 | 22/25 | 53.2% (47) |

Per person (test rows; P5 = all 50): AUROC P1 0.780, P2 0.780, P3 0.590, P4 0.990, P5
0.574. The global cutoff calls Lie 45–60% of the time for the four familiar people, but only 12% for P5.

Each branch on its own (its own validation calibration and cutoff):

| branch | method, config, epoch | test AUROC / accuracy | P5 AUROC / accuracy / session accuracy |
|---|---|---|---|
| dasheng | `bce`, `r8_last4_lr2e-4`, epoch 9 | 0.802 / 73.8% | 0.574 / 50.0% / 53.2% |
| emotion2vec | `pairwise`, `r8_last4_lr2e-4`, epoch 9 | 0.652 / 57.5% | 0.522 / 50.0% / 44.7% |
| V-JEPA 2.1 ViT-L | `bce`, `r16_last8_lr3e-4`, epoch 10 | 0.654 / 61.3% | 0.651 / 50.0% / 63.8% |

ViT-L had the best P5 AUROC, but it was not chosen because selection uses validation only. Switching after
seeing P5's result would make it meaningless as an unseen-person test.

**What this means:** about 74% on new recordings of people the model was trained on, and close to chance on a
new person. On P5 the global cutoff calls almost every answer Truthful; the session baseline is a little better
(53%). Treat any webcam verdict as an experimental score, not evidence.

| file | what it is | sha256 (LF bytes as committed) |
|---|---|---|
| `models/pilot_webcam/dasheng/fold0.pt` | dasheng LoRA, in the fused verdict | `c7577bcd3b9eaaf4904c58d21035adb5ca909375f6a50f89a74441747f4f7afa` |
| `models/pilot_webcam/dasheng/deploy_meta.json` | manifest: hashes, config, state schema, calibration, threshold, metrics, reproduction, caveats | `5381b13f06d70cf4ad062f23998c50429eee03cf46ac43ab4b6e9b8b58c41cd2` |
| `models/pilot_webcam/emotion2vec/fold0.pt` | emotion2vec LoRA (shipped, not in the fused verdict) | `45ad793c95695b72ef1d50c39a6f651c62eabceb26c656feabb7f847739b77d1` |
| `models/pilot_webcam/emotion2vec/deploy_meta.json` | its manifest | `13f6839008396f774f852b60705bb7d45d63ad2d905b4b587e95813ac8705886` |
| `models/pilot_webcam/fusion.json` | validation-chosen combination (dasheng), standardisation, Platt, cutoff, metrics; bound to the branch manifest hashes | `891b32e0ac840f37d4a89a91b76ce893842055bc44b832767ced846bf0b36080` |
| `src/pilot_webcam.py` | `PilotAudioScorer`, `read_fusion`, `fuse`, `SessionBaseline` (Mac); `PilotVideoService` (Node 2) | `a77248089fa13b575b2885f01038182f9b946e5a9c2ace769f507fe8dde18bc0` |
| `fixtures/golden_v1/pilot_webcam_expected.json` | expected outputs on the golden fixture for every branch and the fusion | `a0576d966ef3c90e45c59d12c481ca562aa1b852d801993419f2305a5140bf41` |

On the Mac:

```python
from src.pilot_webcam import PilotAudioScorer, SessionBaseline, fuse, read_fusion

scorer = PilotAudioScorer(device="cpu", amp_dtype=None, backbone="dasheng")  # or device="mps"; FP32 off CUDA
fusion = read_fusion()                    # refuses unless the branch manifests on disk match its hashes
session = SessionBaseline(min_answers=4)  # one per interview session

r = scorer.score(audio_float32_16k_mono)  # logit, raw_score, probability, threshold, verdict, metrics, caveats
f = fuse({"dasheng": r["logit"]}, fusion) # fused_logit, probability, threshold, verdict (global cutoff)
s = session.add(f["fused_logit"])         # verdict None for the first 3 answers, then vs the session median
```

Show the session verdict when it exists (it avoids giving a new person the same verdict for every answer), otherwise
the global one, both labelled experimental. `fusion.json` names only dasheng, so `fuse` needs only its logit; if a
later fusion names `vjepa2_1_l`, its logit is `logit(raw_score)` from the `jepa_video` webcam-pilot reply.

Parity check: score `fixtures/golden_v1/audio_f32le_16k.bin` and compare `logit` with
`audio_branches[<branch>].cuda_fp32.logit` in `pilot_webcam_expected.json`, within its `suggested_tolerance_logit`
(dasheng −3.5597 ± 0.0042, emotion2vec −8.0221 ± 0.0331). Then `fuse` must give `fusion_cuda_bf16.fused_logit`
(−1.3304, Truthful) from `fusion_cuda_bf16.inputs_logit`. The old `audio_cuda_*` keys are kept and hold dasheng.

Numerics: the scores the metrics were computed from used TF32 matrix maths on the training GPU; the server and the Mac
run without it. dasheng probabilities differ by at most 0.011, which changes 3 of 120 validation verdicts and none of
the 130 test and P5 ones (each manifest's `reproduction.serving_numerics`). Reloading each saved model
reproduced its selected validation scores exactly (difference 0.0).

### Video (Node 2 only)

| file | what it is | sha256 |
|---|---|---|
| `models/pilot_webcam/vjepa2_1_l/fold0.pt` | V-JEPA 2.1 ViT-L LoRA (`r16_last8_lr3e-4`, epoch 10) | `f85aa149b6b6613a37b6fc531a198cd0247b981cf630a2d5f1fef2741cdf49a9` |
| `models/pilot_webcam/vjepa2_1_l/deploy_meta.json` | manifest (as above, plus the video `contract` and the base encoder hash) | `e0910ae6d979521db7f2af25a91d22647c0be87a462cd0ff5af9f686a5fdb9f8` |

- **Ladder:** `r8_last4_lr2e-4` and `r8_last4_lr5e-5` were stopped at epoch 4 as not learning. `r16_last8_lr3e-4`
  improved steadily to epoch 10 (validation within-person AUROC 0.620, pooled 0.647).
- **Calibration:** Platt A = 0.6998, B = −0.0047 on validation; threshold 0.5225, prior-matched on validation.
- **Results:** test AUROC 0.654 (accuracy 61.3%); P5 AUROC 0.651, but the global cutoff called all 50 of her
  answers Truthful (50.0%); session baseline 63.8%. It is not part of the fused verdict.

Node 2 serves it only when started with `--webcam-pilot`, and only for `jepa_video` requests with
`"profile": "webcam_pilot"`. Requests without `profile` get the served model as before. `ping` → `profiles` reports
whether it is loaded. Parity: `pilot_webcam_expected.json` → `video_cuda_bf16` (raw score 0.664105 on the fixture).

## Not included on purpose

- Anything derived from MU3D, DOLOS or Box of Lies (licence terms; none are in the served models).
- The V-JEPA 2.1 ViT-L weights: video runs only on Node 2.
