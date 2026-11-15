# Fast runtime status — 2026-11-11

> Latest deployment update: the Windows endpoint is reachable, temporary2.1ViT-B is user-approved, and actual exports/serving identity await verification. See [NEW_TRAINING_AUDIT_9C11432.md](NEW_TRAINING_AUDIT_9C11432.md) and [REMOTE_HEALTH_2026-11-11.md](REMOTE_HEALTH_2026-11-11.md). Offline-state paragraphs below describe the earlier checkpoint.

The teammate service is off by the user's instruction while replacement models train on the document's intended datasets. Previous artifacts are execution-gated as `awaiting_retrained_weights` in both installed teammate profiles. No new network probes, participant capture, remote media or DeepSeek requests occurred after that update. Old files and historical reports remain for audit.

## Implemented

The default launcher uses the new remote-fast local profile. The dashboard shows remote offline/retraining state, local weight preparation, and replacement-artifact status. Live start is blocked before any sensor opens. A config toggle or a ping cannot activate it: reviewed raw-video source/protocol, real encoder identity, a non-person integration fixture and compatible replacement manifests are still required.

Local preparation loads Whisper large-v3-turbo and Qwen onto MPS FP16, Dasheng and emotion2vec onto CPU, then retains them across windows. It is weight loading only (`kernels_warmed:false`), with no capture/forward/network. The first actual forward is timed separately. The engine runs ASR→Qwen concurrently with Dasheng and remote video; emotion can complete separately. MPS work is serialized. Required jobs share a deadline, busy jobs cannot feed stale features to another window, and late emotion must match session, source window and stop state. Shutdown cleanup is reported finished only after both model and ASR caches are released.

Placeholder branches no longer trigger garbage collection/MPS cache eviction. The implementation supports selected branches, cache-hit telemetry and cancellation between branches. Active models are not freed under a running forward. Local preparation is restricted to the remote-fast profile to prevent accidental duplicate residency with a local-V-JEPA baseline.

## Actual Mac benchmark

Fresh Python process, Apple M4 MacBook Air/16 GB, two PyTorch CPU threads, interop one. Existing generated non-person speech fixture, 63,000 samples / 16 kHz = 3.9375 seconds, repeated five times. No sensors, remote vision, fusion or manager. Every local branch returned real official-base features with cache hits; ASR produced complete nonempty automatic text. This is a technical fixed-input timing check, not a dataset or accuracy evaluation.

| Measurement | Observed |
|---|---:|
| Local weight preparation | 7.919 s |
| First required-path forward, including first-use overhead | 3.203 s |
| Four later required-path forwards | 0.968 / 0.956 / 0.967 / 0.968 s |
| Warm Dasheng | 74.5–98.7 ms |
| Warm Qwen | 27.7–31.9 ms |
| Warm Whisper total | 928.4–939.7 ms |
| Supplemental emotion | 180.0–196.5 ms across all trials |
| Final current RSS | 1,591 MiB |
| Lifetime peak RSS | 2,737 MiB |
| Final MPS current / driver allocations | 2,689 / 3,801 MiB |

RSS and Metal allocation figures are different accounting views on unified memory and must not be added as a physical-memory total. Peak RSS includes preparation. This five-trial single-fixture test is too small for a meaningful production p95. Exact per-stage records, model revisions and limits: [benchmark JSON](fast-local-benchmark-2026-11-11.json).

The historical 44.188 s first window and ~32.550 s later window ran **local V-JEPA2.1**, with per-window loading. The new ~0.96 s figure excludes the entire video path and replacement classifier, so it is not a like-for-like end-to-end speedup. The prior teammate ~2.37 ms number also measured a small feature-input head, not raw video through V-JEPA.

## Regression verification

The integrated suite passed **277 tests**. This includes existing capture/alignment, ASR, model/head contract and manager checks; resident/concurrent-runtime and lifecycle races; quarantine and late-result publication; and mocked raw-video protocol validation. Dashboard syntax and six DOM-state checks passed. The protocol tests make no peer connections and do not establish actual GPU performance.

## Remaining verification

When replacements arrive, independently inspect source and hash/state schemas; pin exact encoder family, revision, preprocessing and feature transforms; compare training-machine and Mac fixture outputs. Then measure raw-video packing/bytes, network and GPU service, complete path latency/result age, memory and sustained capture under actual load. The proposed 64 frames at 16 sampled fps span 3.9375 seconds; a 4-second hop requires the sustained critical path to keep up or drops/gaps must be reported. Do not promise full latency before these measurements.

A new live demo remains blocked by the stopped service and absent compatible replacement weights/metadata. This is an operational blocker, not an approval rejection. Independent subagents reviewed scheduling, numerical interpretation, UI state, protocol validation and failure cases; a cleanup race and duplicate-residency path were corrected during review.
