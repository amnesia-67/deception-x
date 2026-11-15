"""Reviewed teammate head for an explicitly unvalidated live integration demo.

The exported head was trained on generated features. Real encoder outputs are
outside its demonstrated training distribution. This wrapper reports its numeric
output, never a calibrated probability or a judgment about a person's honesty.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from .deployment import artifact_unavailable_reason

ARCHITECTURE = "friend-reda-dare-synthetic-v1"
SOURCE_REVISION = "6a16734c1af7d4a5ff24400b06849241b285d5fb"
CHECKPOINT_SHA256 = "cdeeb2f19c51aa9c614665e9c36554e9b1efe03fca367842ee94beec95820e78"
FEATURE_CONTRACT = "friend-reda-dare-live-demo-v1"
MODEL_IDS = {
    "vision": "facebookresearch/vjepa2:vjepa2_1_vit_large_384",
    "dasheng": "mispeech/dasheng-base",
    "text": "Qwen/Qwen3-Embedding-0.6B",
}
_CACHE = {}


class TeammateFusionUnavailable(ValueError):
    pass


def _array(item, name, field, shape):
    if item.get("status") not in ("base", "trained"):
        raise TeammateFusionUnavailable(f"Real {name} encoder output is required; missing, mock and error branches are rejected.")
    if item.get("model_id") != MODEL_IDS[name]:
        raise TeammateFusionUnavailable(f"Unexpected {name} base encoder identity.")
    raw = item.get(field)
    if raw is None:
        raise TeammateFusionUnavailable(f"{name} is missing its required private encoder features.")
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape)):
        raise TeammateFusionUnavailable(f"Incompatible {name} feature shape; expected {shape}.")
    if not value.size or not np.isfinite(value).all():
        raise TeammateFusionUnavailable(f"{name} features must be nonempty and finite.")
    if value.ndim == 2 and (value.shape[0] > 4096 or np.any(~np.any(value != 0, axis=-1))):
        raise TeammateFusionUnavailable(f"{name} features contain zero-filled rows or exceed the bounded sequence length.")
    if value.ndim == 1 and not np.any(value != 0):
        raise TeammateFusionUnavailable(f"{name} features cannot be a fabricated zero vector.")
    return np.ascontiguousarray(value)


def _load(cfg):
    """Strictly load both parts of this exact reviewed checkpoint on CPU."""
    if cfg.get("architecture") != ARCHITECTURE:
        raise TeammateFusionUnavailable("Configure the reviewed teammate architecture explicitly.")
    if cfg.get("source_revision") != SOURCE_REVISION:
        raise TeammateFusionUnavailable("Teammate source revision does not match the reviewed forward implementation.")
    if cfg.get("sha256") != CHECKPOINT_SHA256:
        raise TeammateFusionUnavailable("Teammate checkpoint SHA256 does not match the inspected upload.")
    if not cfg.get("path"):
        raise TeammateFusionUnavailable("The local teammate checkpoint has not been configured.")
    path = Path(cfg["path"]).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise TeammateFusionUnavailable("Expected the reviewed small local fusion checkpoint file.")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise TeammateFusionUnavailable("Teammate checkpoint file bytes differ from the inspected upload.")
    key = (str(path), path.stat().st_mtime_ns, actual)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    from .vendor.friend_fusion import Branch1MultimodalNetwork

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"network_state_dict", "adapter_state_dict"}:
        raise TeammateFusionUnavailable("The checkpoint must contain exactly the inspected network and audio-adapter state dictionaries.")
    for state in checkpoint.values():
        if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
            raise TeammateFusionUnavailable("Checkpoint states must contain finite tensors only.")
    model = Branch1MultimodalNetwork(visual_dim=1024, audio_dim=768, text_dim=128, d=256).cpu()
    adapter = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.LayerNorm(768), torch.nn.GELU()).cpu()
    model.load_state_dict(checkpoint["network_state_dict"], strict=True)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    # The original export is BF16. CPU FP32 preserves every stored value while
    # avoiding unsupported half-precision kernels on different Mac versions.
    _CACHE.clear()
    _CACHE[key] = (model, adapter)
    return model, adapter


def run_teammate_fusion(branches, config):
    """Return an explicitly synthetic-trained numeric head result or a reason."""
    started = time.perf_counter()
    result = {
        "status": "unavailable", "model_id": "teammate/reda-dare-synthetic",
        "architecture": ARCHITECTURE, "feature_contract": FEATURE_CONTRACT,
        "calibrated": False, "experimental": True,
        "training_domain": "generated_synthetic_features",
        "training_feature_match_verified": False,
        "used_modalities": ["vision", "dasheng", "text"],
        "not_used_modalities": ["emotion"],
        "raw_score": None, "embedding_shape": None, "reason": "",
        "source_revision": SOURCE_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    try:
        reason = artifact_unavailable_reason(config, "fusion")
        if reason:
            raise TeammateFusionUnavailable(reason)
        cfg = config.get("models", {}).get("fusion", {})
        visual = _array(branches.get("vision", {}), "vision", "_sequence_embedding", (32, 1024))
        audio = _array(branches.get("dasheng", {}), "dasheng", "_sequence_embedding", (None, 768))
        text = _array(branches.get("text", {}), "text", "_embedding", (1024,))
        # This is an explicit live-demo feature recipe, not an assertion that
        # the synthetic training run used Qwen or any real visual encoder.
        text128 = text[:128].copy()
        norm = float(np.linalg.norm(text128.astype(np.float64)))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise TeammateFusionUnavailable("The first128 Qwen dimensions cannot be normalized.")
        text128 /= norm
        model, adapter = _load(cfg)
        import torch
        with torch.inference_mode():
            fv = torch.from_numpy(visual).unsqueeze(0)
            native_audio = torch.from_numpy(audio).unsqueeze(0)
            # Ordering is essential: GELU/LayerNorm are nonlinear, so adapting
            # after interpolation would not reproduce the reviewed audio path.
            adapted = adapter(native_audio)
            fa = torch.nn.functional.interpolate(adapted.transpose(1, 2).float(), size=32,
                                                 mode="linear", align_corners=False).transpose(1, 2)
            ft = torch.from_numpy(text128).unsqueeze(0)
            outputs = model(fv, fa, ft)
        required = ("p_neural", "logit_fused", "alpha", "D_va", "D_vl", "D_al")
        if any(key not in outputs or not torch.isfinite(outputs[key]).all() for key in required):
            raise TeammateFusionUnavailable("The head emitted missing or nonfinite results.")
        score = float(outputs["p_neural"].item())
        gates = outputs["alpha"][0].float().cpu().numpy()
        if not 0 <= score <= 1 or gates.shape != (3,) or np.any(gates < 0) or abs(float(gates.sum()) - 1) > 1e-5:
            raise TeammateFusionUnavailable("The head emitted an invalid score or modality gate vector.")
        distances = {key: float(outputs[key].item()) for key in ("D_va", "D_vl", "D_al")}
        if any(not -1e-5 <= value <= 2.00001 for value in distances.values()):
            raise TeammateFusionUnavailable("The head emitted an invalid cosine feature distance.")
        result.update(
            status="synthetic_trained", raw_score=score,
            raw_logit=float(outputs["logit_fused"].item()),
            modality_gates=dict(zip(("visual", "audio", "text"), map(float, gates))),
            feature_distances=distances, device="cpu", dtype="float32",
            head_input_shapes={"vision": [1, 32, 1024], "audio": [1, 32, 768], "text": [1, 128]},
            text_scope=config.get("runtime", {}).get("text_scope", "unspecified"),
            reason="Synthetic-feature-trained head. This raw class-1 sigmoid score is unvalidated on people and is not a probability of deception.",
            limitations=[
                "Training generated random features with label-dependent offsets; it did not tune the V-JEPA/Qwen encoders on real recordings.",
                "Live-demo features use V-JEPA2.1 spatial means per tubelet, native Dasheng sequences with the exported adapter, and first128 Qwen dimensions followed by L2 normalization.",
                "Audio is adapted before linear resizing to32 steps; this is not independently calibrated hardware A/V synchronization.",
                "Emotion2vec is not an input to this exported head. Branch2 and the leaked synthetic calibration ensemble are not used.",
            ],
        )
    except (TeammateFusionUnavailable, OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        result["reason"] = str(exc)[:450]
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result
