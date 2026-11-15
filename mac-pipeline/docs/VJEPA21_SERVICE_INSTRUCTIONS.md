# Reviewable V-JEPA2.1 ViT-B service fix

Draft patch: [VJEPA21_SERVICE.patch](VJEPA21_SERVICE.patch) against friend repository commit `6e30a90391984e8dec392e41de1f6ddb25b3ed6d`. Nothing was published, applied to either production repository, or invoked on the endpoint. The patch only uses existing tuned folds; it performs no training or dataset creation/download.

## Proposed behavior

- Load **V-JEPA2.1 ViT-B**, exact model ID `facebookresearch/vjepa2:vjepa2_1_vit_base_384`, from the existing `runs/tune_vjepa2_1/fold0.pt` through `fold4.pt`.
- Instantiate the existing `TunedModel` / `BACKBONES['vjepa2_1']` class, with its dynamic **768-dimensional head**. No V-JEPA2 or ViT-L fallback.
- Reproduce training: 128 cached frames → two 64-cache-frame views →32 input frames at stride 2 in each view →16 native temporal tubelets → interpolation to 32. The B model remains 384px.
- Keep `action:'jepa_video'` a **classifier-only** endpoint. It returns `raw_score`, `calibrated:false`, explicit true model identity and contract, plus numerical legacy score fields. It no longer emits a Truthful/Deceptive verdict. No encoder embeddings or fusion compatibility are claimed.
- Require exactly five folds, full hashes and byte sizes for folds/base encoder, matching manifest/config/state schemas, restricted `weights_only=True`, and complete finite LoRA/head keysets with matching shapes/dtypes. Validate both state groups before mutating a shared model; strictly load the head and every expected LoRA value. Missing keys fail rather than retaining prior-fold/default values.
- Add a no-training export-only command that strips training/test metadata and packages primitive config plus tensors into a fresh immutable directory. Existing published directories are never overwritten.

## Friend-agent application

Review the patch against the stated commit, then apply in the friend's checkout:

```bash
git apply --check /path/to/VJEPA21_SERVICE.patch
git apply /path/to/VJEPA21_SERVICE.patch
python -m pytest -q tests/test_jepa21_deploy.py
```

When the existing five 2.1 folds and base `models/vjepa2_1-vitb-384/encoder.safetensors` are present, package them **without training or dataset access**:

```bash
python scripts/export_deploy.py --jepa21-only
```

Do not use the old plain `main.py export` / export script main for this packaging task: its unrelated tabular path refits XGBoost. If `models/vjepa2_1_lora` already exists, the command refuses overwriting it. Stop the service and preserve/archive the complete previous generation before intentionally selecting a replacement; alternatively package to a fresh directory with `--jepa21-output PATH` and explicitly select that immutable generation. Do not modify a base checkpoint while any loader/service is using it.

Start the reviewed 2.1 service after packaging, using the existing launcher:

```bash
python run_v2_socket.py --device cuda
```

A metadata ping should now identify `vjepa2_1_vitb_lora_5fold`, five folds, model ID ending `vit_base_384`, native width 768, frames-per-view 32, and classifier-only status. Return the pushed source revision and full exported manifest/hashes to the Mac agent. Independently verify the real GPU load/forward and existing held-out parity test before a user-authorized live test. The Mac pipeline must stay disabled until the actual updated service/artifacts are reviewed; applying this patch alone does not enable it.

## Validation completed here

- `git apply --check` passes against a temporary copy of 6e30a90.
- Applying in that temporary copy reproduces all seven staged files exactly.
- **20 CPU-only fake-backbone tests pass**, independently rerun by another agent. They cover correct B geometry, no-CUDA fake inference, complete fold replacement, missing/unknown/shape/dtype/nonfinite state rejection before mutation, hash-before-deserialization, explicit restricted loading, identity/extra/missing-fold rejection, no-training export stripping and overwrite refusal.
- Changed Python files compile.
- No actual model, GPU, datasets, sensors, network or endpoint was used for verification.

## Limits retained intentionally

This is a focused model/export correction. The legacy server still lacks bounded/authenticated/versioned transport, bounded queue admission and source-window/timestamp echoes. Its count-based 128-frame resampler repeats short inputs; it does not establish camera fps or 128 unique frames. Attention averages bins from separate temporal halves. These limitations remain explicit in the source/README and must be resolved before treating it as a robust live pipeline. It is not a ViT-L checkpoint, encoder-feature endpoint, accuracy validation or measured GPU benchmark.

Generic packaging for Dasheng, emotion2vec and Qwen tuned folds is not added by this focused patch; downloading their official bases alone does not load those tuned adapters. They need their own reviewed manifests/strict loaders and matching input contracts.
