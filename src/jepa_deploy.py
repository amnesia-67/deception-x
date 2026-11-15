"""Small, strict deployment helpers for the V-JEPA 2.1 ViT-L classifier folds (ViT-L ONLY).

ViT-B (`vjepa2_1`) and V-JEPA 2 folds are historical and are rejected here: there is no fallback.

No training, datasets, downloads or base-model instantiation occurs here.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch

MODEL_LABEL = "vjepa2_1_vitl_lora_5fold"
MODEL_ID = "facebookresearch/vjepa2:vjepa2_1_vit_large_384"
BACKBONE = "vjepa2_1_l"
HIDDEN = 1024
MANIFEST_FORMAT = "sunhacks-vjepa21-vitl-classifier-v1"
BASE_RELATIVE_PATH = "models/vjepa2_1-vitl-384/encoder.safetensors"
HISTORICAL_REJECTED = {"vjepa2_1": "V-JEPA 2.1 ViT-B (historical)", "vjepa2": "V-JEPA 2 ViT-L (historical)"}
CONTRACT = {
    "backbone": BACKBONE, "model_id": MODEL_ID, "variant": "ViT-L",
    "encoder": {"embed_dim": HIDDEN, "depth": 24, "heads": 16, "patch_size": 16, "tubelet_size": 2,
                "checkpoint_key": "ema_encoder", "upstream_commit": "204698b45b3712590f06245fbfba32d3be539812"},
    "classifier_only": True, "cache_frames": 128, "views": 2,
    "cache_window_frames": 64, "frames_per_view": 64, "frame_stride": 1,
    "spatial_resolution": 384, "hidden_width": HIDDEN,
    "native_temporal_tubelets": 32, "spatial_pooling": "mean_over_24x24_patches",
    "temporal_steps": 32, "temporal_interpolation": "none (32 native tubelets)",
    "official_profile": "Meta hub builder vjepa2_1_vit_large_384: num_frames=64, tubelet_size=2, img_size=384, "
                        "patch_size=16, use_rope=True, interpolate_rope=True; encoder built with exactly these",
    "preprocessing": "rlt-face384-square-bilinear-antialias-div255-imagenet-v1",
    "sampling": "time_uniform_128_nearest_frame; timestamps or fps required; blazeface crop on node 2",
    "fold_aggregation": "mean of the 5 fold logits, each the mean over the 2 views",
}
MAX_FOLD_BYTES = 64 * 1024 * 1024


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_checkpoint(path, *, expected_sha256=None, expected_size=None):
    """Hash and restricted-load the SAME bounded bytes; never use pickle fallback."""
    path = Path(path)
    with path.open("rb") as stream:
        data = stream.read(MAX_FOLD_BYTES + 1)
    if not data or len(data) > MAX_FOLD_BYTES:
        raise ValueError("Fold checkpoint is empty or exceeds 64 MiB")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path.name}")
    if expected_size is not None and len(data) != expected_size:
        raise ValueError(f"Checkpoint size mismatch: {path.name}")
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Fold requires a plain config dictionary and trainable state")
    family = checkpoint["config"].get("backbone")
    if family in HISTORICAL_REJECTED:
        raise ValueError(f"{HISTORICAL_REJECTED[family]} fold rejected: only V-JEPA 2.1 ViT-L is served")
    if family != BACKBONE:
        raise ValueError("Only V-JEPA 2.1 ViT-L (vjepa2_1_l) fold checkpoints are accepted")
    state_schema(checkpoint.get("state"))
    return checkpoint, digest, len(data)


def state_schema(state):
    if not isinstance(state, dict) or set(state) != {"lora", "head"}:
        raise ValueError("Trainable state must contain exactly lora and head dictionaries")
    result = {}
    for group in ("lora", "head"):
        values = state[group]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Missing {group} parameters")
        result[group] = {}
        for name, value in values.items():
            if (not isinstance(name, str) or not isinstance(value, torch.Tensor) or
                    not value.is_floating_point() or value.is_complex() or not torch.isfinite(value).all()):
                raise ValueError(f"{group} must contain named finite real floating tensors only")
            result[group][name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return result


def validate_fold_state(model, state):
    """Validate BOTH groups before either shared module can be mutated."""
    state_schema(state)
    expected = {
        "lora": {key: value for key, value in model.backbone.state_dict().items() if "lora_" in key},
        "head": model.head.state_dict(),
    }
    if not expected["lora"]:
        raise ValueError("Configured backbone has no LoRA parameters")
    for group in ("lora", "head"):
        if set(state[group]) != set(expected[group]):
            raise ValueError(f"Incomplete or unknown {group} keys")
        for key, value in state[group].items():
            if value.shape != expected[group][key].shape or value.dtype != expected[group][key].dtype:
                raise ValueError(f"Wrong {group} shape/dtype: {key}")


def load_fold_state_strict(model, state):
    validate_fold_state(model, state)
    result = model.backbone.load_state_dict(state["lora"], strict=False)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Backbone did not load the exact complete LoRA state")
    model.head.load_state_dict(state["head"], strict=True)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
