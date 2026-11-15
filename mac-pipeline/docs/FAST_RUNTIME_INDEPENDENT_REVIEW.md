# Independent fast-pipeline integration review — 2026-11-11

This is a read-only review of the production orchestration/model wrappers and the staged resident/concurrent runtime. No endpoints, weights, sensors, GPU forwards, participant data or external APIs were used. The teammate is shutting the socket down and replacing the earlier invalid models with models trained on the document's actual datasets; the current deployment must therefore be explicitly unavailable for live task inference.

## Findings requiring integration changes

1. **Enforce artifact quarantine in execution.** The existing `Pipeline._infer` calls `run_teammate_fusion` whenever the old synthetic architecture string is configured, and the wrapper does not inspect a disabled/deployment state. UI warnings alone do not disable loading. Gate head execution before feature conversion/checkpoint loading, empty or disable old active artifact paths, and retain the prior files only for reproducible audit. Do not switch a new file into the old hardcoded hash contract or call it trained merely because state keys match. Add a regression test proving a disabled/quarantined head loader is never called even when genuine base embeddings are supplied.

2. **Model preparation is not remote readiness.** Staged `FastRuntime.prepare()` reports `ready` if local required weights load and any `vision_runner` callable exists. It does not test that callable or prove the service is reachable, executes an actual encoder, or matches the checkpoint/preprocessing expected by a head. The pipeline's live-start gate must independently require verified remote protocol/identity, a successful non-person real-encoder integration fixture and compatible replacement-artifact manifests. With the service off, show `offline_by_user`, keep live start blocked, and perform no automatic retries/media transfers.

3. **Cleanup completion has a race.** In the reviewed staging version, `_cleanup_if_ready` sets `_cleanup_done=True` before `registry.close()` and `clear_asr_cache()` finish. A concurrent `close()` can report `closed` during cleanup. An immediate replacement runtime can then load a new ASR cache that the old runtime clears. Use separate cleanup-started and cleanup-finished state; `closed` must mean cleanup finished, with a completion Event if callers need to wait. This was sent to the runtime author for a focused blocking-cleanup regression test. Stop the old runtime, wait for active jobs and cleanup, then create/load a replacement.

4. **Background and late results need session provenance.** A timed-out accelerator forward can keep running after the pipeline worker exits. Its slot must remain occupied until the forward actually completes. A late emotion callback must match the captured session ID, captured stop event and current window ID before patching state; numeric window IDs alone repeat across sessions. Stopping a session must not close resident models while forwards run. Preparation/reconfiguration must account for runtime jobs, not just the original pipeline thread's liveness.

5. **Preserve actual input and model provenance.** A 32×1024 V-JEPA2/256 sequence is not interchangeable with V-JEPA2.1/384. Preserve exact returned identity, revision/checkpoint hash, preprocessing, temporal sampling, spatial/token pooling and shape. Keep actual clip timestamps through upload/response checks. A `source_window_id` on a Qwen output derived from manual session-context text does not establish time alignment; preserve transcript scope/ranges and require the text scope specified by the new training manifest. The existing dense audio adapter and its adapt-before-interpolate order are architecture-specific; do not assume replacement training uses the same recipe.

## Sound parts of the staged design

- Required paths can overlap: remote video; local ASR then Qwen; CPU Dasheng. Emotion is supplemental and need not block this particular head.
- MPS work is serialized by a shared reentrant lock, avoiding an unsupported assumption that two MPS forwards improve throughput on a 16 GB unified-memory machine.
- Jobs are bounded by key; a busy or timed-out prior source window does not silently provide embeddings to the next source window.
- Resident local weights avoid repeated loading. Placeholder branches no longer trigger unconditional allocator eviction. Preparation explicitly reports `kernels_warmed: false` and uses no dummy silent-audio forward.
- Missing/partial ASR text is withheld from Qwen rather than presented as a complete transcript. RMS screening is still not speech VAD, and ASR hallucinations remain a limitation.

The staged event-based tests cover dependency overlap, ASR→Qwen order, late emotion, deadlines/busy slots, stop behavior, preparation exclusion, deferred model release, partial text and preservation of remote identity. These are scheduling/lifecycle tests; they do not measure actual GPU service, microphone capture health, memory fit or classifier quality.

## Defensible latency and capacity verification

Do not promise a specific full-pipeline latency while the remote service and replacement weights are unavailable. The historical 44.188 s first-window result and approximately 32.550 s later result came from local V-JEPA2.1, and are not remote baselines. The earlier 2.37 ms teammate result tested the small feature-input head, not video through V-JEPA.

While remote is off, use a fresh process and existing permitted non-person/in-memory technical fixtures to measure local preparation, first forward and repeated resident forwards separately. Record the exact fixture duration/text length, model revisions, device/dtype, thread settings, temperatures/load context if available, and sample count. Report the local required-path service time separately from optional emotion and any speculative remote path. No participant data need leave the Mac.

After the teammate service and compatible manifests are reviewed:

1. Verify a real non-person clip traverses JPEG packing, socket transfer, the actual GPU encoder and validated sequence output. Record per-request bytes, packing time, total client round trip and server decode/preprocessing/forward durations. Derive durations on each machine's own clock; never subtract unsynchronized remote timestamps from Mac monotonic timestamps.
2. Measure multiple warmed sequential windows and a sustained run under the actual capture load. Report distributions only with sample counts that make them meaningful; include p50/p95 when enough samples exist, plus failures, deadlines, queue/drop/rejection counts and gaps.
3. Separate camera capture fps, selected clip sampling fps, unique source frames actually processed per second, and fresh inference updates per second. A 64-frame clip at 16 sampled fps spans 63/16 = 3.9375 seconds between first and last timestamps. Higher sampling retains faster motion but shortens temporal coverage; it does not imply 16 new inference results per second.
4. Report service latency from dispatch to result, queue delay, and result age measured from the source window's end. The first result also requires collecting a clip. To sustain a requested hop without dropping, the critical path must keep up with that hop; reducing the hop cannot overcome inadequate capacity.
5. Measure current process RSS and lifetime peak separately, plus MPS current and driver allocations where available. `ru_maxrss` alone retains the peak from any earlier local-V-JEPA load, so it cannot establish the optimized runtime's current memory footprint. Monitor the 16 GB machine for pressure/swap during the sustained capture run.
6. Run a new live demo only after readiness is real, showing where each model runs and exactly which data are sent. A successful speed/plumbing test is not validation of a deception detector; task validity needs genuinely held-out evaluation matching the actual dataset and intended deployment, separate from tuning/calibration.

## Replacement artifact contract

Request the exact backbone family/version/checkpoint hash, preprocessing and sampling recipe, input dimensions and pooling, transcript scope/token limit, any LoRA target modules/rank/base revision, adapter/head source revision and state schema, ordered behavioral-field schema with units/normalization/missingness, ensemble parameters, and held-out split/calibration provenance. Validate hashes and strict keys, run deterministic fixture comparisons against the training machine, then activate an explicit new profile. Do not fabricate missing 250-field values or preserve the earlier label-derived calibration. Dataset acquisition/training remains the teammate's work.

## Remaining blockers

Remote throughput, full end-to-end latency and replacement-model compatibility cannot be established while the teammate socket is off and the actual replacement artifacts/manifests are absent. These are operational/missing-artifact blockers, not an approval denial. Local runtime/interface work and no-sensor local readiness checks can proceed independently.

## Second pass on integrated staging

Reviewed `/tmp/sunhacks-integration-stage` after the parent applied the changes. The shared deployment gate now runs before reference artifact reads and before the friend wrapper's array conversion/checkpoint loading. The new `remote_fast` live start is unconditionally blocked while offline. Background emotion checks the captured stop event, session ID, current window ID and result source ID. Cleanup now separates started and finished state and reports `closing` while either registry release or ASR-cache release is still running. These resolve the critical issues above for the current offline profile.

One remaining optional-path guard was sent to the parent: `prepare_runtime()` also accepts `local_baseline` configurations. Starting a local baseline afterward creates a separate registry beside retained fast-runtime ASR/Qwen/Dasheng and may additionally load the large local V-JEPA model. Restrict preparation to `remote_fast`, or require complete fast-runtime release before a local-baseline live start, to avoid duplicate residency on the 16 GB Mac. This does not block the default offline `remote_fast` profile, whose live entry point is already disabled. No additional probes, weights, sensors or GPU operations were used for this second pass.
