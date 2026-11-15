# Current teammate/DeepSeek integration — November11,2026

> Latest deployment update: the Windows endpoint is reachable, temporary2.1ViT-B is user-approved, and actual exports/serving identity await verification. See [NEW_TRAINING_AUDIT_9C11432.md](NEW_TRAINING_AUDIT_9C11432.md) and [REMOTE_HEALTH_2026-11-11.md](REMOTE_HEALTH_2026-11-11.md). Offline-state paragraphs below describe the earlier checkpoint.

## Latest offline fast-runtime update

The integrated updated suite passed277 tests; dashboard syntax and six DOM-state checks also passed. The teammate service is off and prior artifacts are quarantined. Local resident-model technical timings, new execution gates and remaining blockers are in [FAST_RUNTIME_STATUS.md](FAST_RUNTIME_STATUS.md). Earlier live and single-model runs below are historical and do not establish remote throughput.

The reviewed teammate fusion implementation, its saved audio adapter, real per-tubelet V-JEPA and native Dasheng sequences, Qwen128 live-demo transformation, local Whisper ASR and DeepSeek Flash manager are installed.120tests pass in the installed environment (3.56seconds); pip check passes. Strict loading/forward parity was checked against the actual uploaded checkpoint. Secret-file parsing never executes a shell, and a scoped scan found no API-key value in project code, configs or reports.

The provided trainer generates synthetic features with label offsets, and its simulated cloud calibration leaks known labels. Therefore the head is explicitly marked synthetic-trained/unvalidated and the tabular/calibration outputs remain inactive. See [the updated pinned-source audit](FRIEND_HANDOFF_REVIEW_2026-11-11.md).

DeepSeek Flash authenticated successfully: a text-only, non-participant diagnostic request completed in1.594seconds with thinking disabled. Camera keyframes remain off. The API key remains only in the user's authorized Desktop env file and server process environment. Gemini is not selected by the live profile.

Official local Whisper large-v3-turbo passed real MPS execution using a temporary synthesized-speech fixture:4.399seconds of audio transcribed correctly,6.851seconds including load and2.689seconds generation. This was a technical check, not a participant recording or transcription quality benchmark. Current main-thread camera and MacBook Air microphone probes passed; Blue Snowball was absent from the current device list.

The participant live demo completed after explicit readiness. All four local base encoders, local Whisper and the uploaded head/audio adapter returned valid outputs. The first window took44.188seconds including loads and ASR; its head took21.636ms and returned raw score0.1647. This is not a validated deception probability. After explicit approval to transmit transcript/diagnostics, DeepSeek Flash summarized a retained live window in4.215seconds with zero images or raw audio sent. The camera and microphone are now stopped. See friend-live-demo-2026-11-11.json and deepseek-live-summary-2026-11-11.json. No connection to the friend's GPU was used. The earlier baseline report below is historical; its ASR/manager/fusion absence statements are superseded by this update.

---

# Verification and remaining work

Verified on Apple M4 MacBook Air,16GB unified memory, macOS15.3, Python3.12.11. Exact installed packages are in `requirements-macos.lock.txt`; primary-source interfaces and artifact hashes are in `MODEL_CONTRACTS.md`, `MODEL_DOWNLOADS.json` and the manifest. Official base snapshots total about8.1GiB locally, including Meta source and training-state-containing checkpoint packages. Models and the1.6GiB virtual environment are ignored by Git.

## Implemented and checked

Camera and microphone capture, bounded timestamp buffers, distinct-frame sampling, overlapping64-frame windows, anti-aliased audio resampling, gap/missingness propagation, serialized offline model adapters, local manual transcript/optional ASR interface, strict LoRA and downstream handoff checks, reference fusion/tabular/stack interfaces, opt-in manager client, and a loopback dashboard. Mocks are explicit; missing models never silently become mocks.

52 tests passed, including resampling continuity/anti-aliasing; scheduling jitter versus realADC gaps; device startup/stop races; deferred audio alignment; sparse-video rejection; latest-window queue/drop accounting; model/adapter/FPS mismatch rejection; safe trained-head shapes; manager full-response deadline/fallback/schema validation; late-result/session protection; and actual localhost Host/Origin/body checks. No camera footage or microphone recordings are stored by these tests. The final installed environment also passed all52tests (15.24seconds), and pip check found no broken requirements. The dashboard was verified against that installed revision and left stopped, with camera and microphone off.

The dashboard was checked in a browser: live API state updates, synthetic labels, transcript submission, manager-disabled response, diagnostics layout and explicit stop behavior. It has no external UI assets, does not receive the API key, and sends no sensor data by default.

## Measurements

Single-branch synthetic tests use official weights, one separately timed load, two forward trials and accelerator synchronization. They measure preprocessing/inference/output transfer, not task accuracy. There is no separately discarded warm-up. All reported shapes were finite.

| Model/path | Load | First / second forward | Pipeline width | Peak process RSS |
|---|---:|---:|---:|---:|
|V-JEPA2.1 ViT-L64×384 MPS FP32|4.715s|23.257s /23.075s|1024|2682MiB|
|V-JEPA2.1 ViT-L64×384 MPS FP16/autocast|6.028s|18.359s /18.761s|1024|2638MiB|
|Dasheng CPU FP32|1.639s|553ms /40ms|768|735MiB|
|emotion2vec plus large CPU FP32|2.407s|118ms /105ms|1024|1723MiB|
|Qwen3-Embedding MPS FP16, short test text|2.728s|596ms /32ms|1024|2793MiB|

MPS driver allocations were sampled after the FP16 vision trials at2665MiB; this is not a measured GPU peak. ProcessRSS and MPS memory overlap in unified memory and must not be added. The earlier FP16 path without autocast failed with a query/key/value dtype mismatch; autocast resolved execution. Equivalent task quality across dtypes has not been established.

Exploratory same-weight ViT-L inputs also executed:8frames1.622/0.843s;16frames2.127/1.994s;32frames5.697/6.054s. Their first-to-last spans at24fps are0.292/0.625/1.292s. The live contract remains64frames. No dataset was used for a quality comparison, so shorter clips are not declared equivalent or selected as a trained interface.

A seven-second actual device test captured159camera frames, measuring29.979fps during active capture, and received Blue Snowball48kHz mono audio, resampled to16kHz. Startup occupies part of this test. No queue drops, overflows or clipping occurred. Its early clock-mapping implementation had two resampler resets; the later fix is verified below.

The first all-four live24fps-sampling run produced a complete audio/video window, all four `base` branches, and no inference errors. It took31.861s including loads/releases and other work, peakRSS3631MiB. The rolling camera rate fell to25.143fps;21windows were rejected for timing/sampling gaps and2pending windows were replaced. Thus local high-density sampling is sparse rather than continuous coverage. Its24audio timestamp resets prompted the clock fix, not an assumption of microphone data loss.

After the fixed PortAudio-to-host offset change, a separate explicit16fps/64-frame all-four live run passed:32.336s window service, peakRSS3467MiB, camera25.142fps, no inference errors, all four `base` branches, complete aligned audio. Blue Snowball had **zero** resampling discontinuities, ADC gaps, timestamp fallbacks, overflows, clipping or queue drops despite a144ms maximum callback scheduling delta.15video windows were rejected. This is one short stability test, not sustained/thermal/tail-latency validation. Capture timestamps remain host receipt times; the fixed audio offset can have initial scheduling bias. Hardware A/V synchronization has not been measured.

See the raw `benchmark-*.json`, `capture-host.json`, `camera-probe.json`, `live-integration.json`, and `live-integration-fixed-clock-16fps.json`. `live-integration-16fps-before-clock-fix.json` records an earlier comparison and is deliberately labeled before the fix.

## Remaining gaps and decisions

- No teammate LoRA, fusion, XGBoost or calibration artifacts have been supplied. Reference architectures and strict manifests are concrete; no learned task result is fabricated. Base encoders alone do not establish deception detection.
- The plan's250 physiological/behavioral features, MediaPipe face/iris tracking, Py-Feat action units, rPPG/HRV, validated VAD and SHAP are not implemented. Current diagnostics are measured signal quality and motion. A short diagnostic baseline is not a physiological calibration.
- Local ASR loader is present, but ASR weights and verbatim-transcription quality are unverified; manual transcripts are clearly scoped as session context.
- No manager key/provider model is configured and no real provider call was made. Timeout, fallback and schema behavior were tested with local stubs. Provider model availability, streaming/cache semantics, price and latency remain unverified. The client returns summaries/limitations rather than unsupported calibrated verdicts.
- The local automatic scheduler currently requires camera windows. A missing camera is reported; it does not synthesize video or start an independent audio-only classifier. There is no automatic device reconnect; stop/restart after device changes.
-24fps target sampling needs more stable capture than achieved under heavy local V-JEPA load.16fps was executed but also rejected timing-gapped windows. Sustained continuous analysis is not established. Do not hide this by reporting camera fps as model fps.
- A remote RTX5070 video service over the internet is a reviewed option, **not implemented or connected**. The user's priority is higher motion coverage with an acceptable visible delay. Continuous upload and independent result handling can support stable lag if both upload and GPU capacity keep up. CUDA support in the benchmark helper has not been run on real CUDA hardware. Obtain the same-checkpoint GPU benchmark and measured encoded upload sizes first; then define codec/preprocessing stage, authentication, endpoint and delay budget.

Independent agents researched interfaces, reviewed architecture/number interpretation, built/tested capture, inspected dashboard behavior, and implemented/tested handoff rejection. No datasets were downloaded or created, no training/tuning ran, and nothing was published or sent to the teammate. Sensor media was discarded in memory. Official base weight downloads were authorized and retained locally with license/provenance evidence.
