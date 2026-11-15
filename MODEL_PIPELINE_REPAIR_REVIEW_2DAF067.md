# Review of the repair push: ViT-L delivery still required

## Contradiction integration update — November 12, 2026

This supersedes the earlier local-arithmetic and shirt-only implementation described below. The user's requested contradiction path is now one **Drex Max request via OpenRouter**, taking the complete same-answer recognized transcript and two selected JPEG camera frames. No separate arithmetic pass, garment-rule pass, prose manager, retry or model/provider fallback runs.

The model selects `contradiction`, `no_contradiction` or `insufficient_evidence` against the images, logical consistency and basic arithmetic. Only a supported same-answer contradiction can change the game result to Lie. No-contradiction and insufficient evidence leave the trained classifier's prediction unchanged; neither verifies truth. Cancelled/stale outputs cannot affect a later answer.

The checkbox is disclosed and selected by default, with a persistent opt-out before capture. It authorizes recognized text plus two photos through OpenRouter and DeepInfra; no raw audio, private embeddings, behavioral scores or practice recordings are sent. The request starts from the complete local-ASR callback and overlaps remaining model work, with a five-second deadline. Model availability failures do not create evidence.

The route is `nace-ai/drex-max`, served by DeepInfra; the observed response reports `nace-ai/drex-max-20261016`. The user accepts the requested 1.1 designation as an assumption; retain the actual provider identifier in technical diagnostics. Keep credentials server-side in the explicitly configured env file; no keys belong in this repository.

Ten generated text-and-diagram cases exercised arithmetic, visible conflicts, unknown personal facts, outside-view claims, hypotheticals and transcript instruction injection. All produced the expected engineering outcomes. Median client request/collection time was 1.330 seconds, maximum 2.541 seconds. These are small constructed checks, not webcam deception accuracy, p95 timing or full-pipeline/L benchmarks. Mac regressions passed 758 Python tests and both Node UI suites.

The required genuine V-JEPA 2.1 ViT-L exports, compatible serving path and validated video-inclusive fusion remain separate outstanding requirements.

## Product decision update — November 12, 2026

The participant-facing result is **Truth** or **Lie** for a valid answer. The Mac explanation manager has been removed: no automatic or manual prose-summary call, no summary endpoint, and no generic manager keyframe buffer. Diagnostics stay in the technical dashboard. Failed input or missing models require another recording rather than a fabricated binary result.

The trained, validated fusion classifier is responsible for combining model evidence into the decision. Do not add an unvalidated LLM or decision-model vote over raw embeddings/scores as a substitute for the missing evaluated L-inclusive fusion.

Keep contradiction checks separate:
- Supported arithmetic: exact bounded local calculation on the recognized claim, with no API call.
- Supported visual claims: direct shirt-color claim compared with unambiguous observations in two opted-in images. Ambiguous/occluded/inconsistent evidence cannot assert a contradiction.
- No contradiction never verifies truth. A supported contradiction may map to the game label Lie; retain its evidence and original classifier output separately. This does not establish the speaker's intent.
- Drex Max 1.1 is only a candidate for a matched visual-check benchmark against DeepSeek, including insufficient-evidence cases. It has not been integrated or authorized as a new recipient of participant media. Do not assume general decision benchmarks establish deception accuracy.

The mandatory genuine V-JEPA 2.1 ViT-L and evaluation requirements below remain in force.

Reviewed November 11, 2026 (America/Phoenix). Source: deployment commit `2daf067e11fdca2d47351243a9780026115077b1`, compared with the mandatory-L handoff `5b17b33159c0395b8a76a4fe3abdd1779a9585bb`.

## Blocking requirement: actual V-JEPA 2.1 ViT-L, used in validated fusion

The latest user requirement is **V-JEPA 2.1 ViT-L only**. The earlier temporary permission for ViT-B is revoked. The Mac blocks live readiness before constructing or probing the B client. Do not request a B fallback or fix this by changing health labels.

This push still has the unchanged B implementation and all five unchanged B checkpoints:
- `src/backbones.py` calls `vt.vit_base`, hidden width 768, loading `vjepa2_1-vitb-384`.
- `src/jepa_deploy.py` and `models/vjepa2_1_lora/deploy_meta.json` specify `vjepa2_1_vit_base_384`, ViT-B, hidden width 768.
- Checkpoint schema declares head width 768 and qkv LoRA output width 2304. L requires compatible 1024-wide heads and 3072-wide qkv output adapters.
- The video loader, manifest and five fold Git blobs are byte-identical to the earlier audited `5339358` tree. No new L export appears in the complete inspected tree.

Required next delivery:
1. Train/export the genuine **2.1 L** base plus its matching LoRA and classifier heads; provide exact hashes, complete state schemas and non-private learning evidence. Do not reuse incompatible B adapters.
2. Serve the actual L model with strict model identity, base/source revisions and reviewed loader. Provide its exact crop, normalization, temporal frames/views/stride/tubelet/positional-interpolation contract. Explicitly validate any shorter-frame recipe against the official L profile; do not inherit B's 32-frame path silently.
3. Measure real RTX 5070 warmed and cold latency, memory and throughput with that L recipe. Distinguish capture fps, sampled unique model-input fps and fresh inference updates per second.
4. Train and evaluate an **L-inclusive final fusion path** if video is to influence the learned verdict. The current export still hard-codes only `tuned_dasheng` and `xgb_digest`. Do not add an arbitrary video weight on the Mac. If video-inclusive fusion does not improve held-out performance, report that openly rather than claiming video decides the answer.
5. Supply a generated or redistributable cross-machine golden fixture with expected preprocessing/features, fold scores, stack score and decision. Include the request/source IDs and artifact hashes in serving replies.

The original `MODEL_PIPELINE_REPAIR_HANDOFF.md` remains the full acceptance contract.

## Repairs recognized in this push

This review inspects submitted source and reports; training and evaluation metrics were not independently reproduced on the Mac.

The new nested code retrains base Dasheng models inside each outer-training split, fits the stacker on inner OOF predictions and tests outer people with fold-mean Dasheng plus refit digest. This addresses the specific earlier outer-fold contamination and evaluates the served aggregation.

The new fixed-lag motion features, rates, retrained digest, two-branch stack refit, cache provenance, classwise reporting and reply/source identifiers are substantive changes. The Mac adapter is being migrated as one versioned unit; new schema/weights are not interchangeable with the old deployment.

Exact new export contract:
- `sunhacks-ensemble-v2`, `v3-fixed-lag-motion`, **88 ordered features**.
- Dasheng weight 0.5379349436053581; digest weight 0.462065056394642.
- Platt A = [1.2428574190344905, 1.5430329189543752]; B = [0.18563188430181185, 0.48678425930137054].
- Exported cutoff **0.4238521751969923**, with training-rule provenance. It must travel with these exact model and calibration artifacts; do not apply it to old weights.
- Manifest SHA256 `0057caa6a7c6587e48b27900c726276681297875cc093ac204ec755ac7a1c1be`.
- Digest model SHA256 `f501ea925d0cc2f8d15776b239d97463963f2a83e4a155303420eaed36f563c1`.
- Digest schema SHA256 `bbed857c527031550a878e01e4be8670fcd2a169eea07a482389c4ead31b5356`.
- All five existing Mac Dasheng fold files match the new manifest's SHA256 entries; no new Dasheng fold bytes were required.

## Correct the evaluation wording and retain the remaining limits

The reported **53/61 lies caught and 37/60 truths passed** apply thresholds refitted separately inside each outer fold (roughly 0.448–0.485). They evaluate the threshold-fitting procedure. They are **not measured at the final fixed exported 0.4238521751969923 cutoff**. Correct phrases such as “at the served threshold” where they imply that fixed-cutoff result.

The nested estimate is improved development evidence, not untouched webcam confirmation. Branch selection and threshold-rule family were chosen after earlier full-data inspection; preserve that disclosure. Prior matching does not establish the webcam class prior or calibrate its probabilities.

Remaining training/evaluation work:
- Frozen, consented, correctly labelled seven-second webcam/ASR evaluation with unseen people. The central courtroom-audio crop with inherited whole-clip labels is a diagnostic, not this test.
- Human-transcript versus actual Whisper feature drift measurement.
- Independent target-domain calibration and source/recording-condition controls, with licensed data access handled on the teammate's side.
- Export non-private raw evaluation predictions, fold manifests and learning traces if reporting them as delivered; referenced ignored `runs/` artifacts are not in this push. Preserve training trace fields in cached summaries.
- Bind exports to immutable source/tree fingerprints. The current embedded `f602eb4`, `dirty: true` provenance alone cannot uniquely identify the training code.

## Privacy and division of work

The Mac retains raw audio, ASR, Qwen/emotion feature extraction and learned audio/behavioral inference locally. Only answer video may go to the reviewed GPU. Do not replace this with the combined raw-audio/transcript `analyze` request without renewed user instruction.

The contradiction-only guard checks supported arithmetic locally; shirt-color image checks require a separate per-answer opt-in. The DeepSeek text summary explains recognized text and diagnostic evidence and does not invent a new learned truth score. These checks do not substitute for training or validate deception detection.

No participant media, transcripts, scores, credentials or identifying screenshots are included in this document. No datasets were downloaded/created or models trained on the Mac for this review.
