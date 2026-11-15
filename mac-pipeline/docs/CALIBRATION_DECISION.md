# Warmup and calibration decision — November 11, 2026

The current showcase uses the unchanged trained decision rule and makes the 15-second warmup optional. Warmup provides practice and actual camera/microphone recording checks. It does not supply a personal threshold, change features or calibrate the prediction. Participants can go directly to the seven-second test.

Three independent research reviews covered the exact model contracts, personal baselines in deception research, and statistical calibration. This is an engineering decision from the available evidence, not a claim that personalization can never help. No located source validates two short identity statements and one invented lie as calibration for these exact weights.

## Evidence and limits

The deployed contract is decisive: the pinned teammate service processes one answer using fixed feature extraction, five Dasheng folds, exported branch Platt parameters and fixed stacking weights, followed by its existing 0.5 rule. The digest requires the ordered `v2-no-within-z` schema. V-JEPA is independent; Qwen and emotion2vec supply frozen features. None accepts a participant-reference profile. A new midpoint threshold, nearest-reference rule, baseline Z-scoring or agreement gate would change the inference procedure that was trained and evaluated.

[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) fit post-hoc calibration on held-out data under a shared-distribution assumption. Their results do not establish that three examples suffice here. [Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf) show that calibration can degrade under dataset shift. Courtroom training versus webcam demonstrations remains such a concern; exported Platt scaling is active, but its webcam calibration is unverified.

[Palena et al., 2018](https://pure.port.ac.uk/ws/portalfiles/portal/8457194/Detecting_deception_through_small_talk.pdf) found that small-talk baselines did not distinguish truthful and deceptive targets, while comparable baselines helped only a limited coded feature. These are interview/human-judgment findings, not direct tests of this ML pipeline. They caution against assuming that name/school introductions are comparable to arbitrary test statements.

There is relevant counterevidence: [Hsiao and Sun, 2023](https://arxiv.org/abs/2309.01383) studied individual adaptation using two truthful and two deceptive clips per person, training additional neural layers on a different architecture. Their individual-calibration experiment involved six people and longer clips. This supports investigating trained personalization; it does not validate fitting a threshold to our three five-second scores.

Official model references describe video/audio/text representations and task-specific adaptation, rather than automatic participant memory: [V-JEPA](https://github.com/facebookresearch/vjepa2), [Dasheng](https://github.com/RicherMans/Dasheng), [emotion2vec+](https://huggingface.co/emotion2vec/emotion2vec_plus_large), [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Loading a [LoRA adapter](https://arxiv.org/abs/2106.09685) supplies learned updates; it does not train those updates from subsequent warmup speech automatically.

## Implemented behavior

- Main action: seven-second test, with a server-owned stop timer.
- Optional practice: two true statements and one harmless lie over 15 seconds; local recording checks only. Practice raw media is discarded and no warmup media is sent to the GPU.
- Prediction: unchanged trained local ensemble and serving rule. The existing exported Platt parameters remain active. Video is checked separately and is not inserted into the learned stack.
- Inconclusive: missing, invalid or mismatched required inputs/model outputs. No new three-example agreement gate or personal threshold.
- Presentation: experimental model prediction, without a calibrated confidence percentage or a claim of reliable deception detection.

A future reference-aware model would need an explicit training/export contract and evaluation using matched reference/test topics, durations and capture conditions, with separate held-out answers and participants. The teammate could investigate an adapted head while retaining the backbones. The research here does not justify a universal minimum sample count or an accuracy-improvement promise. No data collection, training or dataset creation was performed for this decision.
