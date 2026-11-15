# Response to `MODEL_PIPELINE_REPAIR_HANDOFF.md` (Node 2, RTX 5070)

Base: `f602eb4` (the audit's target `5339358` plus the audit document). This work used only the existing
121 Real-Life Trials clips and the already-verified base models. No new data was downloaded or created, and
no participant recordings from the Mac were used. All numbers are source-domain (courtroom) estimates, **not
webcam accuracy**.

## Headline: the served policy, fully nested

`scripts/nested_deploy_eval.py` → `reports/nested_deploy/` (report, summary, per-clip `oof.csv`, per-fold
manifests and every inner run with its learning trace).

For each of the 5 outer identity-grouped folds, the whole served recipe is re-run on outer-train only:
- 5 inner dasheng LoRA tunings, 25 in total
- digest inner OOF predictions
- the stacker fit on the inner OOF matrix
- the threshold fit on the stacker's own inner CV

The untouched outer-test people are then scored exactly as served: the mean of the inner dasheng models plus a digest refit on all of outer-train. Each fold writes a manifest asserting that no test clip or identity entered training (`runs/nested_deploy/fold*_manifest.json`, all true).

| decision rule | AUROC | lies caught | truths passed | balanced acc | called deceptive | Brier |
|---|---|---|---|---|---|---|
| prior-matched threshold **re-fitted inside each outer fold** (0.448–0.485): evaluates the fitting procedure | **0.778** (0.68–0.90) | **53/61 (87%)**, CI 75–95% | 37/60 (62%), CI 45–80% | 0.743 (0.64–0.85) | 63% | 0.195 |
| fixed exported cutoff 0.424 on the same nested scores (cutoff fitted on all clips: reference only) | same | 54/61 (89%) | 30/60 (50%) | 0.693 | 69% | 0.195 |
| fixed 0.5 | same | 46/61 (75%) | 41/60 (68%) | 0.719 | 54% | 0.195 |
| always "truthful" baseline | 0.5 | 0% | 100% | 0.500 | 0% | 0.250 (prior) |

The old 7-branch table's stack (AUROC 0.711, 31/61 lies caught) is superseded. It had meta-level nesting only, and it was a different procedure from what was served.

**The truth bias is gone on source data, but the error has moved.** With per-fold thresholds the model calls
63% of answers deceptive and flags 38% of truthful answers. At the single exported cutoff (0.424) it flags
50% of them. That cutoff sits below the per-fold values because it was fitted on single-model OOF scores,
while serving averages five fold models. That is the aggregation mismatch the audit predicted, now measured.
The cutoff is provisional until there is target-domain calibration data. It is not re-tuned on these
evaluation scores, because that would fit to the test.

Residual limitation: the two served branches (dasheng and digest) were chosen from the earlier 7-branch stacker on all clips, and that choice is not re-made inside the outer folds. The prior-matched threshold rule was also picked after inspecting the full-data balanced-accuracy curve, which is flat between 0.66 and 0.70. Both choices were made before the nested evaluation was run.

## Item by item

| audit item | status | where |
|---|---|---|
| Classwise results for the actual pruned 2-branch deployment | **Done.** Confusion matrices, lie sensitivity, truth specificity, predicted-deceptive fraction, Brier, log loss, 10-bin ECE and baselines are reported for every branch and for the served policy, with identity-bootstrap CIs on sensitivity, specificity and balanced accuracy. The audit's reconstructed matrices match the saved OOF predictions exactly: dasheng 34/27/50/10, V-JEPA 2.1 27/34/36/24. | `src/tuning.py::classwise`, both reports |
| Fully nested base + meta validation | **Done** (see above). | `scripts/nested_deploy_eval.py` |
| Calibrate and evaluate the exact deployed aggregation (5-fold mean, full-data digest) | **Done as evaluation.** The nested run scores outer-test people with the served aggregation, so its numbers include the Platt / aggregation mismatch. Measured effect: averaging fold models shrinks the dasheng score spread (SD 0.12–0.22 against 0.16–0.29 for single-model OOF), which is why 0.5 already over-calls slightly. Separate calibration data does not exist on this node. | fold manifests: `score_sd_*` |
| Stacker refit for what is served | **Done.** The stacker is now refit on exactly the two served branches. Previously it was the 7-branch fit, pruned and renormalised. | `train_stage2.py::fit_deploy_policy`, `models/ensemble_deploy.json` (format v2) |
| Threshold not chosen from demos | **Done.** The prior-matched rule is fit on training-CV predictions only: the share called deceptive equals the base rate. It is re-fit inside every nested fold. The served threshold is 0.424. The balanced-accuracy argmax of 0.642 is logged for reference and rejected as noise, because it would have called only 25% of answers deceptive. | `tuning.py::prior_matched_threshold` |
| Duration-sensitive digest features | **Fixed and retrained.** Head speed, scale change and pixel motion are now rates over a fixed 0.7 s lag, not adjacent cached frames. Those frames were 55 ms apart on a 7 s answer and 640 ms on an 81 s clip, so detector jitter was divided by tiny gaps on short answers. Maxima became 95th percentiles. Raw word and question-mark counts became rates. The schema is now `v3-fixed-lag-motion` with 88 features, and the digest was retrained. Its AUROC went from 0.737 to 0.763. The service refuses to start if the schema, manifest and feature-code versions differ. | `src/features_tabular.py`, `tests/test_repair_handoff.py` |
| Seven-second contract | **Diagnostic only.** With the stage-1 dasheng folds on the central 7 s of held-out clips, AUROC is 0.736 against 0.764 on the whole answer, and 31% are called deceptive against 36%. Labels are inherited from the whole clip, so this is not a 7 s evaluation. **Open:** a frozen, consented, labelled 7 s webcam test set with unseen people, which is teammate-side data collection. | nested report |
| Train and evaluate with Whisper transcripts | **Open, blocked.** It needs `faster-whisper large-v3-turbo` (about 1.6 GB), which exceeds the 1 GB download guardrail for this node without explicit approval. The digest still uses human transcripts, and human-vs-ASR drift is unmeasured. | — |
| Permutation null matching the real procedure | **Fixed.** Same 400-tree model and parameters, with grouped folds rebuilt per permutation. The exchangeability unit is stated, along with why identity blocks are not exchangeable (sizes 1–20, five mixed-label identities). Null mean 0.496, 95th percentile 0.610, real 0.763, p = 0.0099. | `train_stage2.py::permutation_null` |
| Cache and fold versioning | **Done.** `tabular.npz` is bound to a fingerprint covering clip ids, labels, groups, transcripts, per-clip preprocessing meta, audio sample hashes, preprocessing code, feature code and frozen embedding hashes. `fold<k>.pt` is bound to cache, tuning code, base-weight sources, ladder, seed and k. Resume refuses a mismatch unless `--force` (retrain) or `--accept-stale` is given. | `src/provenance.py`, `tune_backbones.py` |
| Learning evidence | **Done for new runs.** Each run records init and returned state SHA256, optimizer step count, per-epoch loss, grad norm, val AUROC, update L2 and the actual refit epoch count. A run whose weights did not move is terminated. The 25 nested inner runs carry these traces. | `tuning.py::train_one` |
| Duplicate checks before splitting | **Done (embedding level).** No cross-identity clip pair reaches cosine 0.995 on frozen dasheng or V-JEPA 2.1 embeddings (maxima 0.967 and 0.836). Trial/source grouping is still unavailable, because RLT has no trial metadata. | stage-2 `near_duplicate_audit` |
| Request and source identity in replies | **Done.** `request_id`, `source_id`, `session_id`, `client_revision` and `capture_started_at` are echoed verbatim on success and on error. Replies also carry `server_revision` (git commit plus SHA256 of every served manifest and model), `received_at_unix`, `threshold`, `threshold_provenance` and `feature_version`. | `src/v2_socket_server.py`, `MAC_CLIENT_CONTRACT.md` |
| Group-balanced losses, simple baselines, modality ablations under the corrected protocol | **Partly.** The per-branch nested numbers (dasheng alone 0.759, digest alone 0.763) are the single-branch baselines. On AUROC, fusion does not beat either branch alone within the CIs. Group-balanced sampling was not tried. | nested report |
| Golden CPU FP32 / MPS FP16 versus CUDA BF16 parity trace | **Open.** It needs a redistributable or generated fixture run on both machines. Node 2 parity against training (real clips, raw-media path) is tested: `test_analyze_from_raw_media_reproduces_training`. | — |
| Additional licensed datasets (MU3D, DOLOS, Bag-of-Lies, Box of Lies) | **Open.** These need access agreements or are unavailable (see README). Multi-source adaptation stays disabled. | — |

## Re-running

```
python main.py stage2     # rebuilds tabular.npz if any fingerprinted input changed
python main.py nested     # ~17 min on the RTX 5070 (25 inner LoRA runs, resumable)
python main.py export     # refuses if stage2 / nested were made with another feature version
python main.py test       # 98 passed, 1 skipped on Node 2
```
