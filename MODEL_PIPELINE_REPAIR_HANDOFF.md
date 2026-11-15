# Model and pipeline repair handoff

Prepared November 11, 2026 for the NVIDIA/training agent. Review target: `deploy/ensemble-and-vjepa21` at `53393582c2994025f2bbececc01eb7a893de1c96`. These are source/export audits and engineering checks, not a reproduced training run or a new accuracy measurement. No private Mac participant audio, images, transcript, demographic inference or per-person model scores are included.

## P0 architecture requirement: V-JEPA 2.1 ViT-L only

The latest user instruction supersedes the earlier temporary ViT-B approval. The required video backbone is **V-JEPA 2.1 ViT-L/16 at 384 px**, identified by Meta's `vjepa2_1_vit_large_384` entry point. **Do not use ViT-B, rename its weights, or fall back to V-JEPA 2 ViT-L.** The inspected `5339358` export is V-JEPA 2.1 **ViT-B**, not a compatible L export. The Mac will block that B service for the required live pipeline until the matching reviewed L export and serving protocol are delivered; old B results below are historical diagnostics only.

Meta's [official model table](https://github.com/facebookresearch/vjepa2/blob/main/README.md) distinguishes the 2.1 L checkpoint from both 2.1 B and V-JEPA 2 L. Its [2.1 encoder constructor](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py) defines L with **1024 hidden channels, 24 blocks and 16 attention heads**. The [official 2.1 Hub builder](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py) selects the L architecture and `ema_encoder`, with 384 px inputs, patch size 16, tubelet size 2 and a default 64-frame configuration. Pin the exact upstream revision and base-checkpoint SHA256; the actual trained input/view profile must be explicit in the new manifest. A reduced-frame profile requires its own training/quality validation rather than silently inheriting B's 32-frame recipe.

### NVIDIA agent implementation requirements

1. Build and train/tune the actual **2.1 ViT-L** encoder and a matching task head. Start from the reviewed official L base, not B adapters or the old V-JEPA 2 L checkpoint. Re-run the clean data, split and validation repairs below for this new model; L is now required, but its larger size is not an accuracy guarantee.
2. Export complete new LoRA/head states, base/source/package revisions, SHA256 hashes, parameter names/shapes/dtypes, label mapping and exact fold/deployment aggregation. B's 768-wide tensors cannot be renamed, padded or reshaped into L's 1024-wide tensors. For the same rank-r qkv/projection adapter layout, the L dimensions include A`[r,1024]`, qkv B`[3072,r]` and projection B`[1024,r]`; the current attention head would need 1024-wide normalization and input weights. Generate the expected schema from the actual L model and reject missing/extra/stale keys. A different head must publish its own matching schema.
3. Implement an actual L serving loader and new versioned identity/manifest. Startup and health replies must identify **V-JEPA 2.1 ViT-L**, the loaded base/export hashes and source revision; serving replies must carry matching request/session/window provenance. Do not report L in health while constructing B. Do not make the Mac accept the existing B manifest as an L alias.
4. Define and validate spatial normalization, source timestamp selection, frames per view, view overlap, native temporal tokens, pooling and any interpolation. Test one deterministic end-to-end L fixture against the NVIDIA reference. Measure real 5070 memory, latency, request size and sustained behavior; do not reuse the B timings or its unique-frame counts as L results.
5. If the system is presented as a full video/audio/text learned classifier, the new L features must actually contribute through a trained, exported and cleanly validated fusion rule. The historical deployment gives video zero influence. Do not force an arbitrary nonzero weight to satisfy a diagram: compare the L-inclusive rule against the same rule without L on untouched held-out data, report whether video helps, and do not claim a validated full-video verdict if it does not.
6. Keep the Mac's B blocker in place until the exact L artifact, contract and loader pass review and numerical parity. Missing L weights, an unknown profile or a B response must produce an unavailable/blocked state, never an automatic B or V-JEPA 2 fallback.

The currently published Hub source also contains a localhost checkpoint base URL. Verify the pinned loader and use the official checkpoint linked by Meta rather than treating that development URL as a valid production download location. This handoff did not download weights or run training.

## Read this first

The current complaint is a strong tendency to label both truth and lie demonstrations truthful. The exported validation itself has weak lie sensitivity; the seven-second webcam/Whisper setting also differs materially from training. Correctness/socket tests demonstrate execution, not predictive validity. Do not lower a threshold until selected demonstrations look convincing, reverse labels without a verified mapping, or claim that a larger backbone fixes the issue.

Five independent audit tracks examined local parity, capture/showcase integrity, NVIDIA video serving, training/validation and primary research. No local LoRA/state-loading, class inversion, Qwen/emotion ordering or Platt-formula defect was confirmed. Local stack formula differences were roundoff only. This does not establish numerical equivalence between Mac CPU FP32/MPS FP16 and CUDA BF16; a shared-input golden trace is still required.

### Immediate work order

0. Enforce the mandatory 2.1 ViT-L-only target above; replace the B training/export/loader and keep the Mac blocked until the reviewed L delivery exists.
1. Freeze the exact deployed artifacts and reproduce classwise predictions at the current threshold. Publish the actual pruned two-branch deployment's results, not only the original seven-branch CV table.
2. Repair fully nested base-plus-meta validation and calibrate the exact deployed five-fold-mean aggregation on separate calibration data.
3. Evaluate the exact seven-second, Whisper, camera/crop/transport contract on unseen people and recording sources; address truth-biased sensitivity before selecting architecture changes.
4. Align training and live ASR/duration/crop conditions, invalidate stale caches/folds and audit label/identity/source/duplicate mappings.
5. Compare simple baselines and modality ablations under the corrected protocol. Video must earn inclusion in a new trained/exported fusion rule; it is currently excluded from the verdict.
6. Deliver immutable contracts, parity fixtures, full held-out reports and explicit go/no-go limitations. If target-domain performance remains near chance, report that rather than hiding it behind a cleaner interface or a fact checker.

## Mac-owned repairs and engineering limits

The user-required ViT-L profile supersedes the previously tested B service. The following B-era engineering checks establish repaired transport/orchestration behavior only; they are not L activation, L throughput or permission to keep serving B.

The Mac repair implementation rejects mismatched component/ASR source IDs before reuse; rejects missing/nonfinite quality diagnostics, no detected face, digital silence and fully saturated audio; verifies every required classifier score; and does not expose stale scores/recognized text as current. These prevent invalid inputs from becoming confident-looking binary outputs. They do not cure the learned classifier's weak sensitivity.

The showcase exposes the raw ensemble/Dasheng/digest/independent-video scores, fixed cutoff, raw fold spread and local recognized words. It states that video does not contribute to the learned stack and webcam accuracy is unvalidated. A six-decimal score or fold range is not confidence or an accuracy interval. No arbitrary score abstention band or ad hoc video blending was introduced.

The split profile retains frames up to a maximum edge of 960 px, rather than silently reducing that maximum to 640 px, preserving the trained face-detail contract more closely. JPEG remains quality 90 with 4:4:4 sampling; optimized Huffman coding preserves decoded pixels and reduces payload without lowering image quality. The 16 MiB request cap remains enforced, with oversized inputs rejected. Source/transport geometry and remote face/fallback diagnostics are visible. The difference between the lossless RGB training cache and the additional JPEG serving transport still needs a paired ablation on representative target inputs.

A historical six-second generated non-person fixture was run three times per resolution, alternating 640/960, with real local models and the then-active ViT-B GPU video service. All configured branches completed.

| Maximum frame edge | Complete service trials (seconds) | Approximate request body |
|---|---|---|
| 640 px | 2.466, 2.498, 2.222 | 5.11 MB |
| 960 px | 3.248, 3.994, 2.878 | 12.38 MB |

Parent RSS high-water was about 3015 MiB; last MPS driver allocation was 3810 MiB and child RSS was 46 MiB. These are separate resource measurements; RSS and MPS allocations overlap in unified-memory accounting and must not be summed into a precise total. These are engineering observations, not predictive-quality data, p95 latency or a guarantee for arbitrary image detail. An initial unoptimized 960 fixture exceeded 16 MiB and was rejected; entropy optimization fixed that fixture without changing decoded pixels.

The Mac explicitly disables tokenizer cleanup in ASR decoding but does not invent omitted fillers. Training must still match real ASR output. No dataset was downloaded/created and no model was trained or tuned on the Mac. Current local changes do not replace the trained artifacts.

## NVIDIA-side training and validation repairs

### Why the current output can look like guessing

The exported reports already predict weak or inconsistent performance. In [stage2_summary.json](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/reports/stage2_summary.json), the historical V-JEPA2.1 ViT-B has AUROC 0.528 (95% identity-bootstrap CI 0.377–0.748), accuracy 0.521; the original seven-branch CV stack has AUROC 0.711 but accuracy 0.603 (CI 0.452–0.815). These are source-domain OOF estimates, not webcam accuracy. The confidence intervals are wide and the stack accuracy interval includes chance.

The historical exported local ensemble score does not use V-JEPA. [export_deploy.py:59](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/export_deploy.py#L59) keeps approximately 0.6 tuned Dasheng and 0.4 digest, drops the other branches, and [ensemble_service.py:209](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/src/ensemble_service.py#L209) applies their fixed Platt transforms and weighted sum. The required replacement of ViT-B by 2.1 ViT-L therefore also needs the matching fusion/export work if L is to affect the verdict; changing the video loader alone leaves the historical ensemble formula unchanged.

#### The reported classifier already misses many lies

The known source counts are 61 deceptive and 60 truthful clips. Full-precision accuracy and balanced accuracy uniquely determine the following integer confusion matrices. They were reconstructed algebraically, not measured from this Mac's private demo. The calculation enumerates integer TP in 0..61 and TN in 0..60, requiring `(TP+TN)/121` and `(TP/61+TN/60)/2` to match the full-precision reported accuracy and balanced accuracy. Each row has one matching solution; confirm it directly from saved OOF predictions before using it as a release metric.

| Reported OOF procedure | TP lie | FN lie | TN truth | FP truth | Lie sensitivity | Truth specificity | Predicted lie fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original seven-branch CV stack |31|30|42|18|50.8%|70.0%|40.5%|
| Tuned Dasheng |34|27|50|10|55.7%|83.3%|36.4%|
| XGBoost digest |36|25|50|10|59.0%|83.3%|38.0%|
| Historical V-JEPA2.1 ViT-B |27|34|36|24|44.3%|60.0%|42.1%|

The current user reports that both truths and lies are usually called truthful. This table supports a pre-existing asymmetry at 0.5 even on source validation; it does not quantify the user's live accuracy. Domain shift can worsen that asymmetry. Do not fix this by lowering the Mac's threshold until a few chosen demonstrations look correct.

Existing Platt signs are not a simple label inversion: the exported A/B make a Platt-transformed branch score of 0.5 occur at raw probability approximately 0.442 for Dasheng and 0.418 for XGBoost. Positive B values do not explain an always-truthful result on their own. Verify class mapping with actual fixtures before changing signs or thresholds.

### P0: produce a frozen seven-second target-domain evaluation

The current data is 121 courtroom clips, 57 manually inferred identities, and only five identities with both labels ([README:9–15](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/README.md#L9)). Recording-condition features alone obtain AUROC 0.669 ([README:40–47](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/README.md#L40)). Person grouping helps, but does not remove trial/channel/topic/device confounds or establish webcam generalization.

Alongside the required ViT-L migration, evaluate the exact newly exported model and actual seven-second capture/ASR contract on a frozen, consented, labeled webcam test set containing unseen people. Both classes should be represented for each participant, with matched topics, comparable speaking duration, counterbalanced order and capture conditions not correlated with the label. Record true labeling/compliance provenance; do not infer veracity from filename alone or assume a whole-video label applies to every seven-second subclip. Data collection and licensed-dataset acquisition belong on the teammate's side; no such dataset was created here.

Report: confusion matrix, sensitivity to lies, specificity to truths, balanced accuracy, AUROC, PR curves, score histograms by class/domain, predicted-positive fraction, log loss/Brier/reliability diagrams, and participant-clustered uncertainty intervals. Show the always-truthful and class-prior baselines. Predeclare acceptance criteria and any abstention policy; do not choose them from this final test. There is no evidence-based universal sample-count promise from this audit.

For broader data, inspect permitted access and modality/label contracts for [MU3D](https://pubmed.ncbi.nlm.nih.gov/29869221/), [Bag-of-Lies](https://arxiv.org/html/2508.04129v1), [DOLOS](https://arxiv.org/abs/2303.12745), and [Box of Lies](https://aclanthology.org/N19-1175/). Availability is not blanket redistribution permission. The [SVC 2025 challenge](https://arxiv.org/html/2508.04129v1) provides a useful cross-domain evaluation design: multiple training sources and a distinct held-out domain. It does not guarantee good webcam performance. Normalize each source's truth/lie mapping explicitly; published label conventions differ.

### P0: repair base-plus-meta nesting and evaluate the actual deployment policy

[train_stage2.py:262–269](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/train_stage2.py#L262) fits each outer-fold meta-model on rows of one global OOF matrix. For an outer test fold k, a training row from fold r was generated by a base model trained on every fold except r, including k. Thus the meta-training features depend on data and labels from the outer test fold. The script explicitly acknowledges this at [lines 331–333](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/train_stage2.py#L331). Calling this fully nested validation is incorrect.

Implement this scoped procedure for each outer split:

1. Freeze the outer test participants and source/trial groups.
2. Generate new inner OOF base predictions using only outer-training participants; fit all feature normalizers, early stopping, hyperparameters, calibration, stack weights and threshold/abstention choices inside that scope.
3. Fit the base deployment policy using only outer-training data. Predict outer-test data with that policy, then the inner-fitted calibration/stack. Never reuse the original global OOF matrix for these meta-training rows.
4. Save fold manifests asserting that every training-derived artifact excludes all outer-test IDs, source/trial groups and duplicates.

Deployment averages five Dasheng fold probabilities ([ensemble_service.py:187–210](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/src/ensemble_service.py#L187)), but calibration is fit on one held-out model's OOF score per training example ([train_stage2.py:287–299](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/train_stage2.py#L287)). Averaging changes the score distribution. The full-data XGBoost deployment also differs from each OOF model. This is a mismatch requiring evaluation, not proof of a sign bug. Validate/calibrate the exact deployed aggregation on separate representative calibration data and retain an untouched test set.

The headline seven-branch CV stack and final pruned two-branch export are also different fitted procedures. Publish exact exported-model held-out results, not only the original global CV table. Compare simple frozen-head/regularized classifiers and each strong branch against fusion using the same clean protocol; complexity is not evidence of gain. The current source table does not show a stack advantage over Dasheng in AUROC or over XGBoost in accuracy, but do not cherry-pick a replacement from that same test table.

### P0: train/evaluate with the deployed transcript and time contracts

Training uses human transcripts ([README:9–10](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/README.md#L9)); live text comes from Whisper large-v3-turbo. The digest explicitly measures fillers, repetitions, false starts, question marks and lexical diversity ([features_tabular.py:97–124](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/src/features_tabular.py#L97)). ASR may normalize or omit exactly those cues. Generate training/evaluation features with the same ASR revision, processor, language, decoding settings and transcript formatting used at inference. Quantify human-transcript versus actual-ASR feature/prediction drift and word/utterance errors. Do not invent missing fillers on the Mac.

The historical ViT-B training route takes 128 frames over the full variable-duration clip and selects two 32-frame stride-two views ([clip_store.py:54–85](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/src/clip_store.py#L54)). That historical 32-frame B recipe is not the new L contract. The new L export must declare and validate its own profile. Audio training uses up to 16 seconds and evaluation up to 60 ([clip_store.py:107–121](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/src/clip_store.py#L107)); live answers are seven seconds. Evaluate a fixed seven-second recipe explicitly. Do not merely crop long clips and trust inherited labels without statement-level validity.

Several digest fields are duration-sensitive. Pixel-motion differences use adjacent `frames[::2]` crops without division by elapsed time, unlike head velocities ([features_tabular.py:204–232](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/src/features_tabular.py#L204)). Shortening the answer changes motion lag even when Mac code matches exactly. Retrain the digest if changing these formulas or adding time-normalized alternatives; version the schema. Avoid silent feature changes under old XGBoost weights.

The current fusion explicitly sets multi-source domain adaptation to zero because only one source domain exists ([train_stage2.py:135–145](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/train_stage2.py#L135)). Additional licensed domains and a real cross-domain evaluation are prerequisites for claiming that part of the planned architecture is active. Multi-source adaptation is an experiment to evaluate, not an automatic reliability fix.

### P1: trustworthy evaluation and reproducible training

- Audit manual identity labels and source/trial grouping; add perceptual/video/audio duplicate checks before splitting. Grouping by person alone is not a source-independent test. Consider group-balanced losses/sampling so a many-clip individual does not dominate; validate that choice inside training only.
- Make the permutation test match the real procedure. Current `permutation_null` shuffles individual clip labels and uses 150 trees, while the real digest uses 400 ([train_stage2.py:90–130](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/train_stage2.py#L90)). State and justify the exchangeability unit and use the same model-selection/procedure under the null. This is a validation flaw; it does not prove fabricated training.
- Version cache keys with raw-source/ASR/preprocessing/schema/base/source hashes. Current tabular cache reuses IDs plus a version string only ([train_stage2.py:42–47](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/train_stage2.py#L42)). Current tuner skips existing fold files and validates test IDs, without binding them to current preprocessing/base/config revisions ([tune_backbones.py:49–81](https://github.com/amnesia-67/sunhacks-2026/blob/53393582c2994025f2bbececc01eb7a893de1c96/scripts/tune_backbones.py#L49)). Old folds/features must not silently survive a changed pipeline.
- Record initialization/final LoRA and head tensor hashes, actual optimizer steps, gradient/update norms, loss traces and actual refit epoch count. GPU utilization and finite tensors alone do not establish learning or quality.
- Preserve the honest weak-branch findings. V-JEPA 2.1 ViT-L is now required by the user; its effect on accuracy still needs clean evaluation after the data/protocol repairs. Frozen emotion categories are not lie labels; Qwen embeddings are not a chat model that fact-checks arbitrary personal claims. Do not force weak branches back into fusion because the project diagram contains them.
- Keep three-example warmup out of threshold fitting and feature Z-scoring. The current schema deliberately removes within-identity normalization; a reference-aware architecture needs its own training and held-out evaluation.

### Required delivery back to the Mac

Export one immutable **V-JEPA 2.1 ViT-L** version and its matching fusion/serving contract, with source/base/checkpoint hashes, complete LoRA/head key schemas, actual fold count and aggregation, explicit class mapping, ordered feature schema, ASR/crop/sampling/duration contracts, calibration/threshold provenance and package versions. Add expected intermediate tensors/features and final outputs for redistributable or generated parity fixtures, without raw private participant recordings. Include a separate frozen seven-second target-domain evaluation report with classwise diagnostics and uncertainty. Numerical parity tolerances must be measured for GPU BF16 versus Mac CPU FP32/MPS FP16; borderline outputs should not be silently reinterpreted.

Acceptance means: verified 2.1 ViT-L identity with no B or V-JEPA 2 fallback; strict artifact/schema loading; matching CPU/GPU preprocessing and scores within documented tolerances; no train/test identity/source/duplicate leakage; calibration fitted without final-test access; evaluation of the actual deployed ensemble on unseen seven-second users; and clear reporting of which error patterns remain. Passing socket/unit tests is necessary engineering evidence, not proof of reliable deception detection.


## Exact parity acceptance protocol

Use generated or explicitly redistributable inputs, not private demo recordings. Freeze timestamps, RGB channel order, raw 16 kHz mono PCM and one exact transcript. Compare original raw inputs, the new L contract's selected source IDs and timestamps, detector boxes/fallback, 384-square crops, ImageNet input tensors, the declared L frame/view layout, temporal/spatial pooling, every fold's LoRA/head state and output, Dasheng mel/BN/32-step reduction, Qwen full 1024-dimensional instructed embedding and 33 raw intent prototypes, emotion nine-class order, all 89 ordered digest values including legitimate missingness, raw classifier scores, branch Platt transforms and final score/threshold.

Produce reference outputs on CUDA FP32 first and BF16 separately, then compare CPU FP32/MPS FP16 on the Mac. Correctness tolerances should be justified by actual repeatability and measured differences, including whether any classification changes near the cutoff; do not assert arbitrary bit-identical equality across precisions. Fail closed on stale source IDs, incompatible shapes/hashes, missing modalities or changed preprocessing. Include claimed source revision and source IDs/timestamps in server responses for auditability; the current socket only binds a reply locally to its connection.

## Text-summary manager — explanation, not classification

The proposed default DeepSeek text-summary step explains the validated local result; it does not select a new truth/lie class, recalibrate a score or override the trained classifier. Its payload includes the recognized answer text and selected model/quality diagnostics. Raw audio and camera frames are excluded from this text-only summary. Application notices and the Start flow must accurately disclose this cloud transmission; the earlier statement that all recognized words remain local and that no automatic DeepSeek request runs must be removed wherever automatic summaries are enabled.

This is separate from the optional visual claim checker below. The sentence that the visual checker excludes the full transcript describes that checker only, not the text-summary manager. Both services need bounded deadlines, source/session matching, cancellation, validated responses and explicit unavailable states. Manager explanation latency is additional to local inference; a pending or failed explanation must not be reported as a completed new classifier.

## Optional contradiction-only guard — separate from learned deception

The requested credibility feature is a separate evidence guard, not a retrained classifier or new ensemble weight. Local deterministic arithmetic may flag an explicit false equation. An opt-in DeepSeek Flash visual check may inspect up to two same-answer frames plus an extracted shirt-color claim; audio and the full transcript are not sent. Its output may identify a clear contradiction or say that it cannot verify. It must never convert 'no contradiction found' into proof of truth, infer race/identity or other personal traits, treat absence outside the camera's view as evidence, or infer intent to deceive from a factual mistake.

[DeepSeek's official vision API](https://api-docs.deepseek.com/guides/vision/) supports selected images. Its latency must be measured separately; a prior text-only manager test on the Mac took about 4.2 s, which is not a vision-latency benchmark. The design runs claim checking in parallel after validated ASR. Preserve source/session IDs, explicit image opt-in, bounded deadlines, strict response validation and cancellation. Keep behavioral scores and contradiction evidence separately visible. A fact checker can catch obvious observable errors while the learned deception model remains weak; it must not be used to disguise that weakness.

## Primary research supporting the repair plan

- [Original RLT paper](https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf): courtroom data, average 28-second clips and 66-word transcripts; transcription instructions preserve repetitions, fillers and intentional-silence markers. This is not the same input distribution as seven-second Whisper speech.
- [Dataset Bias in Deception Detection, ICPR 2022](https://api.repository.cam.ac.uk/server/api/core/bitstreams/bb099bc7-c95f-4c9b-a9db-57df4270abad/content): source attributes and cross-domain generalization require explicit controls; benchmark scores need not demonstrate a portable deception signal.
- [SVC 2025 cross-domain challenge](https://arxiv.org/html/2508.04129v1): useful design for domain-separated evaluation and an example of why generalization must be measured directly. It is not a performance ceiling or an accuracy promise for our model.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) and [Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf): calibration needs a valid evaluation protocol and may degrade under distribution shift. Three practice statements do not establish webcam probability calibration.

The human-baseline and few-shot-personalization literature does not directly validate this exact fixed-model pipeline. Keep optional practice separate from calibration. The final artifact must identify 2.1 ViT-L accurately and state exactly what improved, what did not, and which limitations remain. Historical B artifacts and results must remain labeled as historical.
