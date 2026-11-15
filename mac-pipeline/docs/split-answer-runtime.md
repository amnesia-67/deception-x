# Split whole-answer runtime

Current profile: `config.split-answer.local.json`, using the reviewed teammate export at `53393582c2994025f2bbececc01eb7a893de1c96`. The installed Mac is an M4 MacBook Air with 16 GB unified memory. This replaces the earlier blocked V-JEPA2 service profile.

## Execution and inputs

| Component | Runs on | Current role |
|---|---|---|
| Camera and Blue Snowball microphone | Mac | Timestamped RGB video and mono 16 kHz audio for one answer |
| Whisper large-v3-turbo | Mac, MPS FP16 | Same-answer transcript; recognition can omit fillers |
| Dasheng base + five trained LoRA/task heads | Mac, CPU FP32 | Mean of five sigmoid classifier scores; native temporal features resampled to 32×768 |
| Qwen3-Embedding-0.6B | Mac, MPS FP16 | Frozen instructed full 1024-wide testimony embedding and 33 intent prototypes for the digest |
| emotion2vec plus large | Mac, CPU FP32 | Frozen nine-class probabilities for the digest |
| BlazeFace and 89-field digest | Mac, CPU | Reviewed face crops/boxes/timestamps plus exact text, voice, emotion and intent features |
| XGBoost and exported Platt stack | Mac | Local digest score; approximately .6 tuned Dasheng + .4 digest after branch calibration |
| V-JEPA2.1 ViT-B, five trained folds | Friend's RTX 5070 at `192.168.0.224:5002` | Video-only independent classifier, 32 RGB frames at384×384 per view, width 768 |

**The learned local stack excludes the video classifier.** Its exported weights dropped the other candidate branches. ReDA/DARE fusion remains unavailable and its old synthetic checkpoint is not used. Frozen Qwen/emotion features match this digest's design; they are not advertised as tuned models. ViT-L remains a future separately trained/exported target.

Local artifacts, base weights, schemas and trainable states are hash checked. XGBoost runs in an owned persistent subprocess to avoid the native OpenMP crash observed when sharing the Torch/MediaPipe process. Only the actual ordered 89-field row crosses private local stdio; the worker receives no provider credentials. Undefined supported features remain NaN, while absent required modalities prevent scoring.

## Capture and lifecycle

Run `./scripts/run.sh`, then open `http://127.0.0.1:8765`. The configured launcher prepares in the background; **Prepare local models** rechecks readiness. Preparation loads local weights, computes static Qwen intent prototypes and verifies the remote classifier with generated non-person frames. Camera and microphone remain off.

Use **Start**, speak one complete answer for about **6–7 seconds**, then **Stop & analyze**. The implementation accepts 5–30 seconds with enough distinct captured frames. The top **Stop & analyze** control remains available throughout capture and stops even a short recording; insufficient input is reported without scoring. The alternate **Finish answer & analyze** control is gated until enough input exists. It stops camera/microphone before inference. One nearest-time selection produces the shared 128-frame raw cache used by both the remote video request and local face processing; it is not selected a second time. Sparse or gapped captures are rejected instead of padded. Only one answer is active. Stop cancels publication; late jobs cannot supply a later answer. Prepared models stay resident between answers and application shutdown releases them and the worker after active jobs finish.

## Frame rates mean different things

- **Capture rate:** 30 fps requested; actual delivery is measured.
- **Cache rate:** 128 distinct selected frames divided by answer duration.
- **Model input rate:** two stride-2 views each take 32 frames, so V-JEPA sees 64 unique source frames. Six to seven seconds gives approximately **9.1–10.7 unique model-input fps**. Eight seconds gives at most 8 fps; longer answers are sparser. Five folds repeat computation, not unique motion.
- **Service throughput:** unique consumed frames divided by processing time, excluding capture. This is not fresh-result frequency.
- **Fresh updates:** one result per capture-and-processing cycle. Source-end result age also includes stopping, waiting and processing.

This is a whole-answer interaction. Short webcam answers differ from the courtroom training distribution; denser motion does not establish equal predictive quality.

## Measured technical benchmark

[Benchmark summary](split-answer-benchmark-2026-11-12.json) uses six seconds of generated speech with explicit synthetic silence and 128 generated non-person frames. Real backbones and the remote classifier ran; no live sensors, participant data or manager call were used. The artificial images triggered the no-face crop fallback, so live face-crop cost/quality remains unmeasured.

| Measurement | Time |
|---|---:|
| Local runtime preparation, once | 10.384 s |
| First complete inference service trial | 3.863 s |
| Repeated trial 2 | 1.687 s |
| Repeated trial 3 | 1.636 s |

Preparation is separate from service time; the dashboard's additional remote verification can add time. Service includes local ASR/features/classifiers and remote video work, excluding capture, sensor stopping, UI delivery and the optional manager. These three trials do not establish sustained capacity, p95 latency or predictive accuracy. CPU FP32/MPS FP16 outputs have not been compared numerically with the teammate's training GPU BF16 outputs.

Observed parent-process RSS high-water mark was approximately 2693 MiB; last RSS 2135 MiB and MPS driver allocation 3810 MiB. These overlap in unified memory and must not be summed into a precise total. XGBoost worker RSS is not included.

## Data leaving the Mac and score meaning

Video alone goes to the configured GPU classifier. Audio, ASR, transcript, emotion features, digest and stack stay local. **Request manager summary** is a separate explicit DeepSeek API action; it sends the selected transcript and supported diagnostics. Keyframes are disabled in this profile; raw audio and full private embeddings are not sent. Inference never triggers that API automatically.

All task scores are experimental outputs, not calibrated webcam honesty probabilities. The teammate reports wide held-out intervals, a recording-condition confound and incomplete base-level nesting of the stacker. V-JEPA2.1's reported AUROC of .528 has a confidence interval including chance. A completed live run demonstrates pipeline mechanics, not reliable deception detection.

## First live check

The [live technical report](live-split-check-2026-11-11.json) excludes raw media, transcript text and classifier scores. The camera delivered about30fps and the Blue Snowball used48kHz mono, resampled continuously to16kHz. Both devices stopped; no audio overflow, queue drop, clipping or resampling discontinuity was reported. All available trained/frozen branches completed. Processing took3.129s and result age on completion was3.513s.

This answer lasted29.114s because of agent command delays before stopping. Its874 captured frames were selected into128 transported frames and64 native model frames, giving2.198 model-input fps. This does not demonstrate greater-than8fps live sampling. The direct Stop control avoids waiting for agent commands; a6–7-second user-controlled capture remains the next sampling check. No DeepSeek request ran.

## Short live sampling check

A subsequent [6.441-second live answer](live-short-answer-check-2026-11-11.json) captured194sourceframes at30.04fps, selected128transport frames and supplied64distinct native video frames. This measured9.936model-input fps inside that answer, exceeding8fps. All available model branches completed; processing took2.013seconds and source-end result age on completion was2.277seconds. Both sensors stopped. This resolves the earlier short-clip sampling check; it does not establish continuous coverage,9.94fresh inferences per second or predictive quality.
