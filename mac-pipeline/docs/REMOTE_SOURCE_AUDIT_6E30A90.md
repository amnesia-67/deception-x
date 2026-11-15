# Raw-video service audit — `6e30a90391984e8dec392e41de1f6ddb25b3ed6d`

Read-only source inspection of `/tmp/sunhacks-friend-6e30a90`; commit pin supplied by root. No source import/execution, endpoint calls, media transmission, encoder invocation or production writes. An independent child reviewer confirmed the main findings. No V-JEPA2 fixture client was created or run: the user clarified that **V-JEPA2.1 is required**.

## What is now implemented

This commit finally adds a real raw-image path:

- `v2_socket_server.py:202` routes `action:"jepa_video"` to `JEPAService.predict` (`:423`).
- The request is one NDJSON object `{action:"jepa_video",frames:[base64_JPEG_or_PNG,...],face_crop:bool}`.
- `jepa_service.py` decodes at least 16 frames, uniformly resamples/repeats to 128, optionally performs face detection/cropping, creates two 64-frame views, runs each of 5 tuned fold models, and averages classifier logits.
- The response returns an ensemble score, per-fold scores, averaged 32-bin attention and timings. It **does not return [32,1024] encoder features** for the Mac fusion path.

This changes the prior “no raw-video route” finding for the older committed entry point. The existing strict `sunhacks-video-encoder-v1` client remains incompatible with this different action/output contract and must not be weakened or auto-enabled.

## Blocking identity mismatch

The new service is unambiguously **V-JEPA2 ViT-L**, not V-JEPA2.1:

- `JEPAService.__init__` rejects every fold whose `cfg.backbone != "vjepa2"`.
- The deployment directory is `models/vjepa2_lora` and model label is `vjepa2_vitl_lora_5fold`.
- `export_deploy.py:63` copies `runs/tune_vjepa2/fold*.pt`; its metadata explicitly reports V-JEPA2 results.
- That wrapper uses `facebook/vjepa2-vitl-fpc64-256`, rather than a 2.1 checkpoint/architecture.

The user's latest instruction requires 2.1 and is uncertain whether the friend's intended variant is B or L. Do not benchmark or deploy this 2 service under a 2.1 name. A matching 2.1 server loader/export and exact variant identity are required. The supplied fetch command names MediaPipe, Qwen3 and emotion2vec assets; it does not itself request a V-JEPA2.1 model or switch this service's loader.

## Loader and deployment issues

- `torch.load(...,weights_only=False)` loads arbitrary pickle checkpoints; no file hashes or source/base revision manifest are checked.
- Any nonempty `fold*.pt` glob is accepted; it does not require exactly fold0..fold4, reject stale/extra folds or verify a complete matching export. `info` still labels the model 5fold.
- Fold config is filtered by known TuneConfig fields, silently ignoring unknown metadata. Models are reused by only `(lora_r,lora_blocks)`; a robust deployment identity must also include every architecture/head setting that affects instantiation.
- Existing `load_trainable_state`/LoRA loading requires a separate strict review: the known backbone loader uses `strict=False` without requiring the complete expected adapter key set. Missing adapter values can leave prior/initial state, especially when swapping folds in a reused model.
- Export copies files into an existing directory without atomic generation selection, full hashes or pruning stale folds. Keep old and new artifacts isolated and review a complete immutable manifest before starting a service.
- Startup can select CPU, but prediction unconditionally uses CUDA autocast and `torch.cuda.synchronize`; this is not a functional CPU fallback.

## Protocol and timing limits

The service has no explicit version/checkpoint/preprocessing digest, UUID/window/timestamp echo, sampling-rate contract or auth/TLS. Input bounds are only a minimum 16 frames: no maximum frame count, message/image byte limit, decoded pixel cap, strict base64 validation or bounded GPU queue. Its lock serializes GPU work after preprocessing, so many threads can prepare large 128-frame arrays and wait indefinitely. A client timeout cannot cancel queued/running work.

Uniform count-based resampling repeats frames when fewer than 128 are supplied and discards original capture timing. It can make 128 model inputs from 16 unique camera frames; this must never be reported as 128 captured/unique frames or higher motion coverage. The 32 attention bins average the corresponding within-view bins of two different clip halves, so they cannot be directly mapped to one global 32-bin answer timeline.

`prep_ms` measures decode/crop preparation. `inference_ms` includes waiting for the GPU lock, adapter swaps, ten forward passes and completion synchronization; it is not an isolated single-backbone latency. Health roundtrip remains a separate metric. The published CV metadata reports AUROC 0.468, 95% CI 0.362–0.640 for this 2 model; the service's “Truthful/Deceptive” strings must not be presented as reliable findings about a person.

## Next required source

Push the actual **V-JEPA2.1** service and deployment bundle with explicit B/L identity, frame count, native/interpolated output dimensions, preprocessing/timestamps, complete strict adapter loading and full hashes. Decide whether the remote interface returns features for local fusion or an independently labeled classifier score; they are different contracts. Then review a bounded versioned protocol and verify it using non-person fixtures before any separately authorized live test.
