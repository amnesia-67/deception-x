# Current teammate deployment audit — 6e30a90

Reviewed `amnesia-67/sunhacks-2026` commit `6e30a90391984e8dec392e41de1f6ddb25b3ed6d` from the materialized source and recursive Git tree. This supersedes the deployment blockers in the historical 9c11432 audit where explicitly resolved below. No new model code was imported or executed; no training, dataset download, participant upload, endpoint call or production edit was performed by this audit.

## Concrete progress

- **Raw-video inference now exists.** [JEPAService](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/src/jepa_service.py#L82) accepts base64 JPEG/PNG frames, optionally applies the existing face detector/crop helpers, creates 128 cached-style face frames, then runs two 64-frame views through each of five LoRA-tuned visual fold models. This replaces the former missing/head-only video route. The new NDJSON action is `jepa_video`; the old `jepa` action is a different legacy head route.
- **Visual fold artifacts are committed.** The reviewed Git tree contains `models/vjepa2_lora/fold0.pt` through `fold4.pt`, totaling 11,696,829 bytes, plus `deploy_meta.json`. [The export](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/scripts/export_deploy.py#L60) copies these states from the real-data `runs/tune_vjepa2` training route. Artifact presence is established; parent performs byte hashes and tensor-only inspection separately.
- **Source now explains the live health identity.** The advertised `vjepa2_vitl_lora_5fold` matches this source's service. It is no longer accurate to say the service implementation is wholly absent from GitHub. A runtime revision/hash attestation is still stronger than a matching model-name string.
- **The real-data training route remains unchanged.** Git blob hashes of `src/tuning.py`, `src/backbones.py`, `src/clip_store.py`, `scripts/preprocess_rlt.py`, `scripts/train_stage2.py` and `models/MODEL_SOURCES.json` are unchanged from the previously reviewed 9c11432 version. The prior synthetic generator/head path must not be substituted for this deployment.
- **LAN connectivity and the tabular artifact were already resolved by the parent.** Windows `192.168.0.224:5002` returned health/schema, and the 89-field booster loaded locally. Neither the old WSL address nor absent tabular weights remains a current blocker.

## Exact identity: remaining mismatch

This service is **V-JEPA 2 ViT-L**, not V-JEPA **2.1** ViT-B. [Its loader](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/src/jepa_service.py#L63) explicitly requires `cfg.backbone == "vjepa2"`. The unchanged backbone constructs `facebook/vjepa2-vitl-fpc64-256`: 64 frames at 256 px and 1024-wide tubelets. The current export only copies `runs/tune_vjepa2`; the recursive tree contains no `vjepa2_1` LoRA/head deployment artifacts.

The user already approved **V-JEPA2.1 ViT-B/16 temporarily**, 32 frames at 384 px, width 768, until ViT-L retraining. Do not seek approval for that size choice again. The remaining delivery request is the trained **2.1** adapter/head export and a service that selects its correct 32-frame profile. The supplied **2** model can be evaluated as an accurately labeled alternative only if the user chooses that change; renaming its health field does not make it 2.1.

## Preprocessing and FPS

The backend correctly reuses `TunedModel`, the original backbone normalization, `select_frames` and `eval_starts`, and averages two view logits, then five fold logits. These are source-compatible choices for the exported V-JEPA2 fold models.

Important remaining contract differences:

1. **No timestamp input.** [Resampling](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/src/jepa_service.py#L46) is uniform over frame indices. Training sampled by clip timestamps. The two match only if transmitted frames are equally spaced; dropped or irregular camera frames break that assumption. Supply timestamp-uniform preselection on the Mac, retain original timestamps/window bounds in local evidence, and test the exact selection with irregular-frame fixtures.
2. **128 frames are a cap, not a retained-FPS guarantee.** For answer duration D seconds and N transmitted frames, retained unique frames are at most `min(N, 128)`, giving at most `min(N,128)/D` unique sampled fps. Answers longer than 16 seconds cannot retain more than 8 fps through this route. Accepted 16-frame inputs are repeated to 128, adding no motion. Use bounded timestamped windows if faster motion is the objective; quality for short live windows remains unverified against training's full-answer recipe.
3. **Five folds do not multiply unique frames.** Five models × two 64-frame views executes 640 model frame slots per request but uses at most 128 unique source frames. The README's approximately 3-second RTX5070 figure is a teammate-reported claim, not an independent measurement. Fresh result rate, unique-frame throughput, capture rate, network transfer and end-to-end lag require separate measurements.
4. **Face-crop flags matter.** `face_crop:false` assumes already-cropped faces and simply resizes each image to 384 square; passing raw full-scene frames under that flag is wrong. `face_crop:true` reuses training crop helpers after downscaling a raw frame's longest side to at most 960. Different camera geometry, JPEG encoding, frame spacing and full-answer duration remain possible distribution shifts. The raw crop helper's Git hash is unchanged, but its file was not materialized for this latest incremental reread.
5. **Attention is not localized across the whole answer.** The service averages 32-element attention arrays across the two disjoint half-answer views. Index i in view1 and view2 denotes different times. `peak_tubelets` therefore cannot honestly be mapped to a single global timestamp. Keep view-specific weights and map their tubelet spans to timestamps if the UI needs temporal locations. Attention weights alone are not a causal explanation of deception.

## Loader and service reliability findings

- **Require complete adapter key sets.** The unchanged `load_lora_state_dict` rejects unknown keys but uses `strict=False` without rejecting missing expected LoRA keys. The service swaps fold states into shared models, so a partial state can retain parameters from the preceding fold. Parent should require exact expected LoRA names, shapes, finite values and strict head state before serving any artifact.
- **Use safe loading.** [The service](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/src/jepa_service.py#L63) uses `torch.load(...weights_only=False)`. Tensor/primitive checkpoints should use `weights_only=True`; there is no need to execute pickle code to inspect or load these exports. Verify source/base/artifact hashes and class-1 semantics.
- **Validate exactly five distinct folds.** It accepts every `fold*.pt`, permits one or more files while reporting a fixed `5fold` identity, and export silently skips missing files. Require fold0–4 exactly or label the actual ensemble explicitly; reject stale/extra/missing artifacts.
- **CPU fallback is broken.** Although server startup can select CPU, [predict](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/src/jepa_service.py#L124) unconditionally uses CUDA autocast and `torch.cuda.synchronize()`. This is an RTX service; do not treat it as a tested Mac or CPU backend.
- **Bound request resources and queueing.** The socket has no total request-byte cap, and `_prepare` decodes all supplied images before discarding most and before acquiring the GPU lock. Add bounds on frame count/encoded bytes/decoded dimensions plus one pending request or explicit stale-request replacement. The 120-second timeout is not a queue-length guarantee.
- **Timing names are approximate.** `prep_ms` includes decode/crop; `inference_ms` includes lock wait, adapter copying, GPU work and attention conversion. It excludes encode/upload/return. Retain a client end-to-end measurement and expose queue wait separately before using these fields as backbone speed.
- **Version/provenance metadata is incomplete.** `deploy_meta.json` supplies honest CV metrics but not a complete manifest of source revision, base SHA, per-fold SHA, LoRA key lists, package versions and preprocessing profile. Parent can pin verified files locally; a formal export manifest is preferable for handoff.

## Validation evidence and limits

[The added test](https://github.com/amnesia-67/sunhacks-2026/blob/6e30a90391984e8dec392e41de1f6ddb25b3ed6d/tests/test_v2_socket.py#L125) sends a previously cached, already-cropped dataset clip at JPEG quality100 and checks that its held-out fold reproduces the stored probability within .03. This is a useful parity design. This audit did not run it, and the dataset/cache/OOF files are not transferred here. It does not test raw-camera face-crop parity, irregular timestamps, missing-frame behavior, sustained throughput or the entire five-model ensemble's unseen-person accuracy. A deterministic non-person fixture can test transport/shape/load mechanics without transferring the dataset.

The supplied metadata accurately reports V-JEPA2 AUROC **0.468**, 95% CI **0.362–0.640**, accuracy **0.512**, and says its verdicts are not evidence of deception. These are repository-reported identity-grouped results, not independently reproduced measurements. The API's sigmoid output is an uncalibrated model score despite being called `probability`. The five-fold ensemble is reasonable for new people, but the supplied single-held-out-fold CV number is not itself a separate validation of that deployment ensemble.

## Still pending for the intended complete pipeline

1. Correct V-JEPA2.1 temporary ViT-B fold/head export + matching service identity/input profile, unless the user explicitly changes the family choice.
2. Strict, hash-verified tensor-only load and deterministic preprocessing/output parity; measured live capture/upload/inference/result throughput without accumulating lag.
3. The actual 89-feature extraction contract before activating Branch2; do not zero-pad Mac diagnostics into it.
4. Trained stage2 fusion states, per-fold normalizers and seed/view aggregation if multimodal fusion is to run. This commit does not export them. Tuned Dasheng/emotion/Qwen fold artifacts are also not committed.
5. Preserve the experimental/chance-level caveats in the live UI and manager evidence. A successful live test demonstrates mechanics, not truthfulness detection.

No new permission request is implied by routine local loader/test work already authorized. No third-party message was sent.
