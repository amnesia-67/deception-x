# Independent performance and temporal-coverage review

Reviewed on 2026-11-11 against the saved benchmark JSON and the reviewed Meta source checkout (`204698b45b3712590f06245fbfba32d3be539812`). This review audits measurement interpretation and runtime policy. It does not add a new training or inference contract and does not evaluate behavioral accuracy.

## What the measurements establish

The saved `benchmark-vjepa-fp32.json` records official V-JEPA 2.1 ViT-L weights on this Mac, MPS, batch size one, 64 RGB frames at 384×384, with 304,680,960 encoder parameters. Python is3.12.11, PyTorch2.15.0 and macOS15.3. The timed path includes preprocessing, accelerator synchronization and pooled output transfer, after a separately timed model load.

| Configuration | Separate load | Two preprocessing + inference trials | Process lifetime peak RSS | MPS driver allocation sampled after trials |
|---|---:|---:|---:|---:|
| Float32 | 4.715s | 23.257s /23.075s | 2682MiB | 4137MiB |
| Float16 with MPS autocast | 6.028s | 18.359s /18.761s | 2638MiB | 2665MiB |

Both successful paths produced finite 1024-dimensional pooled embeddings. The earlier float16 run without autocast failed because rotary computations produced float32 queries/keys with float16 values. Its failure remains separate from the successful autocast report.

These are two-trial synthetic single-branch smoke measurements. They do not establish a stable tail latency, sustained thermal performance, full-pipeline memory, sensor rate, behavioral quality or equivalence of floating-point formats. The first timed inference may include cold kernel costs; there is no separately discarded warm-up. The benchmark loads the model once and reuses it for the trials. Default live operation with `keep_loaded:false` reloads weights for every selected window, so its service time additionally includes model loading, release and other configured branches.

`process_peak_rss_mb` is a process-lifetime high-water mark including load and inference, expressed in MiB despite its field suffix. MPS current and driver metrics are sampled after the trials; they are not a continuously measured GPU peak. The driver metric includes allocations beyond tensor memory. RSS and MPS driver allocations must not be added as disjoint physical memory on a unified-memory Mac.

## Honest runtime policy

Retain the selected `sunhacks-input-v1` video contract:64 distinct timestamp-selected frames at24fps target sampling and384px preprocessing. A clip has2.625seconds between its first and last timestamps. Keep camera capture independent, with30fps requested and its actual achieved rate measured.

At the measured float16 speed, a new result needs roughly18–19seconds of preprocessing/inference after a model is loaded. Float32 requires roughly23seconds. Neither sustains the intended1.333-second hop. The queue-of-one latest-window policy preserves recent input and bounded memory; it deliberately skips intervening windows. Dense24fps sampling inside the clips does not provide continuous24fps model coverage. Ignoring all other work, disjoint2.625-second clips processed every18.56seconds would cover only about14% of elapsed source time. At23.17seconds that estimate is about11%. These are calculations, not measured full-pipeline coverage.

A longer hop does not fix this limitation. Setting it to25seconds may reduce dropped-window counts but can make the next available input older. Keep the1.333-second hop as a *candidate freshness schedule*, expose rejected/dropped windows, and explain that only selected clips reach the model. For an operator who requires every moment to be analyzed, this Mac/configuration has not met that requirement. Do not silently lower input sampling or repeat frames to disguise the gap.

Real-model inference should be an explicit configured option. The unconfigured workbench can exercise capture and clearly marked mocks without claiming real-model performance. For a real configuration, float16 with autocast is a candidate based on successful execution and lower measured latency; it still needs numerical comparison to the float32 path and a full mixed-model stability run. Keeping the vision model resident avoids a4.7–6.0second reload, but enable that only with a measured residency budget for the actual model mix. Do not load all models concurrently by assumption or disable MPS allocator protection.

The dashboard should show these separately:

- Actual camera receipt rate, with timestamp provenance and device status.
- Requested/achieved unique-frame sampling inside the selected clip.
- Completed inference windows per second and distinct source frames processed per second.
- Source start/end of the last result, time since that clip ended, and the next in-progress clip's source interval where available.
- Completed, rejected and replaced windows; missing or stale audio/text/head branches.

Recompute result age from the monotonic clock whenever state is read. A value calculated only when inference finishes freezes while a subsequent18-second operation runs and understates staleness. A manager result must retain its source window/session identity rather than appear to summarize whatever current window happens to be displayed.

## Can fewer frames run through the same official encoder?

The official encoder's `forward` reads the input tensor's temporal dimension and divides it by the two-frame tubelet size. Its3D patch convolution is not tied to64 input frames, and attention constructs positions from the actual token grid. This makes even frame counts such as8,16 and32 structurally plausible using the same encoder weights. An actual forward pass still must confirm shape, finite output and device compatibility.

The flag `interpolate_rope:true` in the reviewed implementation rescales height and width positions. It does **not** interpolate the shorter time axis onto64-frame temporal coordinates. Call this variable-length forward support, not temporal interpolation that preserves the original clip's meaning. Meta's published ViT-L2.1 evaluation configurations use16 frames for K400/COIN/SSv2 and32 for Jester/Diving48. Those task configurations show that64 is not an intrinsic architecture-only input limit; they do not validate this pipeline's pooled features, dense24fps sampling, or task performance. The reviewed evaluation wrappers also have their own aggregation and checkpoint-loading behavior, so the config files are not a substitute for strict loading and local smoke tests.

At unchanged24fps sampling, the duration changes:

| Frames | Tubelet time steps | Token count at384px | First-to-last span |
|---:|---:|---:|---:|
| 8 | 4 | 2304 | 0.2917s |
| 16 | 8 | 4608 | 0.6250s |
| 32 | 16 | 9216 | 1.2917s |
| 64 | 32 | 18432 | 2.6250s |

Reducing frames while preserving a2.625-second span would reduce the actual sampling rate; that is a different experiment and can conflict with the user's preference for more than8fps. Shorter contiguous clips preserve dense sampling but lose temporal context. Mean pooling still yields1024 numbers, but the matching output dimension alone does not make the distributions or trained adapters interchangeable.

## Acceptance criteria for an experimental shorter clip

1. Use exactly the same official encoder/checkpoint, strict state loading,384px crop and normalization. Test8/16/32/64 as separately labeled experiments without changing live defaults.
2. Check full token shape, finite tokens/pooled features and output dimension. Record input length, sampling rate, source duration, dtype/autocast, model/source revisions and actual load/inference measurements.
3. Compare float16 and float32 on the same synthetic input for numerical consistency, reporting the metric and values rather than interpreting finiteness as equivalence. Cross-length pooled embeddings are not expected to match because their context changes.
4. Run enough repeated trials to distinguish cold from warm timing and reveal instability; record how memory was sampled. Later test the actual capture/preprocessing/model mix, queue bounds, dropped intervals, ASR/audio alignment and stop behavior.
5. Give any chosen shorter contract a new explicit identifier and share it with the teammate before tuning. Match frame count, sampling/crop, pooling and downstream normalization across training and inference.
6. Require task-specific validation from the teammate before claiming preserved quality. No datasets, training or quality evaluations are performed here.

## Evidence

Local measured reports: `benchmark-vjepa-fp32.json`, `benchmark-vjepa-fp16-autocast.json`, and the earlier failed `benchmark-vjepa.json`.

Primary implementation/configuration references at the reviewed revision:

- [Encoder forward and dynamic temporal shape](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/vision_transformer.py)
- [RoPE coordinate construction and spatial interpolation](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/utils/modules.py)
- [Two-frame3D patch embedding](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1/models/utils/patch_embed.py)
- [Official ViT-L2.1 K400 evaluation configuration:16frames](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/eval_2_1/vitl-384/k400.yaml)
- [Official ViT-L2.1 Jester evaluation configuration:32frames](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/eval_2_1/vitl-384/jester.yaml)
- [PyTorch MPS synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html), [tensor allocations](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html), [driver allocations](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html)

## Follow-up: actual live capture with all four base encoders

The later `live-integration.json` records an authorized37.433-second live capture run with all four official base branches enabled. One complete result was produced, with no inference errors in the report. Its total window service time was31.861seconds and process lifetime peak RSS was3631MiB. Branch times were25.508seconds for vision,1.467seconds for Dasheng,2.194seconds for emotion2vec and1.955seconds for Qwen; these live branch times include loading and are not interchangeable with warm synthetic inference timings. The sum of branch times does not include every release/diagnostic/orchestration cost.

The accepted video window contained64 distinct frames with actual first-to-last span2.617seconds and reported effective sampling24.074fps; its largest selected-frame gap was68.57milliseconds. Matching audio had coverage1.0 and no missing samples. The text input was an explicitly manual smoke-test transcript, not demonstrated local ASR. Fusion, tabular and calibration-stack artifacts remained unavailable, and no manager API was requested. This is working base-model integration evidence, not a trained end-to-end task classifier.

Camera capture requested30fps; the final rolling camera measurement was25.143fps under load. There were21 rejected candidate windows and2 pending-window replacements during the37.4-second session. A rolling25fps average does not prove that every24fps64-frame window has enough distinct well-spaced source frames. The strict selector checks uniqueness and per-target timestamp error, so jitter can reject windows even when average capture rate is nominally above the requested sampling rate. Those rejections must remain visible rather than be filled by duplicate frames. The capture's21 rejections are separate from its zero camera queue drops.

The microphone reported zero overflow callbacks and zero queue drops, but24 resampler discontinuities. The complete first-window audio is a positive integration result; it does not establish uninterrupted audio alignment throughout the run. Investigate timestamp mapping/callback scheduling under model load and preserve missing-data reporting before claiming sustained audiovisual stability. The report's source-end-to-result age was31.916seconds at publication; the dashboard's current-age field correctly continues to increase afterward.

### Local availability profile versus dense-motion target

Recommend a named, explicit local **16fps sampling profile** for availability on this Mac, retaining the selected ViT-L2.1 model,64 frames,384px preprocessing and a32-frame hop. It gives3.9375seconds first-to-last span and a2-second nominal hop. The observed camera rate leaves more headroom for distinct timestamp matching than a24fps target, so it is a reasonable next configuration to test. It is a recommendation based on the24fps run's rejection evidence; no16fps live availability rate has been measured in this report.

Keep the24fps/64-frame profile as the denser-motion target, especially for evaluating the optional GPU path. Neither profile changes64-frame compute size: switching sampling from24 to16fps extends source context but does not remove the roughly18–19second measured vision compute or31.9-second first full-window service cost. A longer response delay is acceptable if coverage remains sustained; the Mac's current latest-window behavior instead discards intervening coverage when service capacity is insufficient. A selected16fps clip retains twice as many time samples per second as an8fps profile, but some briefer events can still be missed and temporal resolution differs from24fps.

The two profiles must have explicit sampling metadata and distinguishable preprocessing/settings identity in the teammate handoff. Record frame count, sampling fps, hop, crop/color/normalization, pooling, dtype and timestamp/missingness policy in a canonical settings object and verify its hash at adapter/head loading. A fixed1024-wide output or shared high-level contract name does not establish training compatibility across16fps and24fps. Do not silently switch a tuned24fps pipeline to16fps or attach matching-looking weights without the agreed profile.
