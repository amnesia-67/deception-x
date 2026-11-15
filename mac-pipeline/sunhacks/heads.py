"""Offline trained-head interfaces. No downloads, training, implicit imputation or labels.

These reference contracts are an integration target, not the plan's unspecified
ReDA/DARE network. Artifacts are absent until the teammate supplies them.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .models import CONTRACT, DIMS, MODEL_IDS, POOLING
from .deployment import artifact_unavailable_reason

BRANCH_ORDER = ("vision", "dasheng", "emotion", "text")
FEATURE_NAMES = (
    "audio.rms_dbfs", "audio.peak_dbfs", "audio.clipping_fraction",
    "audio.activity_fraction_proxy", "audio.duration_seconds",
    "video.brightness_mean", "video.mean_absolute_frame_difference",
)


class ArtifactUnavailable(ValueError):
    pass


def _cfg(config, name):
    return config.get("models", {}).get(name, {})


def _read_artifact(config, name, architecture, filename):
    reason = artifact_unavailable_reason(config, name)
    if reason:
        raise ArtifactUnavailable(reason)
    cfg = _cfg(config, name)
    if not cfg.get("path"):
        raise ArtifactUnavailable(f"Teammate {name} artifact has not been configured.")
    path = Path(cfg["path"]).expanduser().resolve()
    if not path.is_dir():
        raise ArtifactUnavailable(f"{name}.path must be an existing local artifact directory.")
    manifest = json.loads((path / "handoff.json").read_text())
    if manifest.get("input_contract") != CONTRACT or manifest.get("architecture") != architecture:
        raise ArtifactUnavailable("Head input contract or architecture does not match this implementation.")
    expected = manifest.get("sha256", {}).get(filename)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactUnavailable(f"Manifest requires SHA256 for {filename}.")
    model_path = path / filename
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected:
        raise ArtifactUnavailable(f"Artifact SHA256 mismatch: {filename}.")
    return cfg, manifest, model_path


def _base_result(name):
    return {"status": "unavailable", "model_id": f"teammate/{name}",
            "input_contract": CONTRACT, "calibrated": False,
            "experimental": True, "embedding_shape": None,
            "reason": "", "elapsed_ms": 0.0}


def _finish(result, started, exc=None):
    if exc is not None:
        result["reason"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def fusion_module():
    """Exact reference architecture/state-key contract for the teammate export."""
    import torch
    return torch.nn.Sequential(
        torch.nn.LayerNorm(sum(DIMS.values())),
        torch.nn.Linear(sum(DIMS.values()), 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.0),
        torch.nn.Linear(256, 2),
    )


def _class_names(manifest):
    names = manifest.get("class_names")
    if not isinstance(names, list) or len(names) != 2 or any(not isinstance(x, str) or not x or len(x) > 100 for x in names) or len(set(names)) != 2:
        raise ArtifactUnavailable("Exactly two distinct explicit class_names are required; no default task labels are inferred.")
    return names


def _adapter_sha(config, branch):
    path = _cfg(config, branch).get("adapter_path")
    if not path:
        return None
    return hashlib.sha256((Path(path).expanduser() / "adapter_model.safetensors").read_bytes()).hexdigest()


def run_fusion(branches, config):
    """Run only with all four real, compatible pooled vectors and matching scope."""
    started = time.perf_counter()
    result = _base_result("fusion")
    try:
        cfg, manifest, path = _read_artifact(config, "fusion", "sunhacks-fusion-mlp-v1", "model.safetensors")
        from .models import input_settings
        if manifest.get("preprocessing_settings") != input_settings(config):
            raise ArtifactUnavailable("Fusion preprocessing settings differ from runtime sampling, spatial, audio or text contract.")
        expected_inputs = []
        arrays = []
        for branch in BRANCH_ORDER:
            item = branches.get(branch, {})
            if item.get("status") not in ("base", "trained"):
                raise ArtifactUnavailable(f"Fusion requires real {branch} features; missing/error/mock inputs are rejected.")
            revision = item.get("revision")
            if not isinstance(revision, str) or not revision:
                raise ArtifactUnavailable(f"Exact {branch} base revision is required.")
            identity = {"branch": branch, "model_id": MODEL_IDS[branch], "base_revision": revision,
                        "output_dim": DIMS[branch], "pooling": POOLING[branch],
                        "adapter_sha256": _adapter_sha(config, branch)}
            if item.get("model_id") != identity["model_id"] or item.get("pooling") != identity["pooling"]:
                raise ArtifactUnavailable(f"{branch} model identity or pooling does not match the input contract.")
            expected_inputs.append(identity)
            vector = np.asarray(item.get("_embedding"), dtype=np.float32)
            if vector.shape != (DIMS[branch],) or not np.isfinite(vector).all():
                raise ArtifactUnavailable(f"{branch} features must be finite and exactly {DIMS[branch]} wide.")
            arrays.append(vector)
        if manifest.get("inputs") != expected_inputs:
            raise ArtifactUnavailable("Fusion training inputs do not match runtime encoder revision, adapter, dimension, order or pooling.")
        text_scope = config.get("runtime", {}).get("text_scope")
        allowed_scope = "session_context" if cfg.get("allow_session_context") else "video_window"
        if text_scope != allowed_scope or manifest.get("text_scope") != allowed_scope:
            raise ArtifactUnavailable("Text timing scope differs from fusion training contract; manual session text is not window-aligned.")
        classes = _class_names(manifest)
        import torch
        from safetensors.torch import load_file
        model = fusion_module()
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(np.concatenate(arrays)).unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise ArtifactUnavailable("Fusion emitted nonfinite outputs.")
        result.update(status="trained", architecture=manifest["architecture"],
                      class_names=classes, logits=logits.tolist(), probabilities=probabilities.tolist(),
                      reason="Teammate reference fusion head; output is experimental and uncalibrated. No reliability claim.")
    except (ArtifactUnavailable, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        return _finish(result, started, exc)
    return _finish(result, started)


def run_tabular(features, config):
    """Load native XGBoost JSON; reject missing values, schema drift and feature reordering."""
    started = time.perf_counter()
    result = _base_result("tabular")
    if _cfg(config,"tabular").get("disabled_reason"):
        return _finish(result,started,ArtifactUnavailable(_cfg(config,"tabular")["disabled_reason"]))
    try:
        _, manifest, path = _read_artifact(config, "tabular", "sunhacks-xgboost-diagnostics-v1", "model.json")
        names = manifest.get("feature_names")
        if not isinstance(names, list) or not names or any(not isinstance(name,str) or name not in FEATURE_NAMES for name in names) or len(set(names)) != len(names):
            raise ArtifactUnavailable("Tabular schema must list unique implemented diagnostic feature names in training order.")
        if manifest.get("missingness_policy") != "reject":
            raise ArtifactUnavailable("Only explicit reject-on-missing diagnostic schema is supported.")
        values = []
        for name in names:
            group, field = name.split(".")
            value = features.get(group, {}).get(field)
            if features.get(group, {}).get("status") != "ok" or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ArtifactUnavailable(f"Missing or invalid diagnostic: {name}; no silent zero filling.")
            values.append(value)
        classes = _class_names(manifest)
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(path))
        if model.feature_names != names:
            raise ArtifactUnavailable("XGBoost embedded feature names do not match handoff training order.")
        model_cfg = json.loads(model.save_config())
        if model_cfg["learner"]["objective"]["name"] != "binary:logistic":
            raise ArtifactUnavailable("Tabular reference contract requires binary:logistic objective.")
        scores = np.asarray(model.predict(xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=names))).reshape(-1)
        if scores.size != 1 or not np.isfinite(scores).all() or not 0 <= scores[0] <= 1:
            raise ArtifactUnavailable("Tabular model emitted invalid binary probability.")
        result.update(status="trained", architecture=manifest["architecture"], class_names=classes,
                      probabilities=[float(1-scores[0]),float(scores[0])], feature_names=names,
                      reason="Teammate diagnostics classifier; experimental, uncalibrated output. These are signal-quality features, not physiology.")
    except (ArtifactUnavailable, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        return _finish(result, started, exc)
    return _finish(result, started)


def run_stack(branch_scores, config):
    """Explicit complete-input logistic/Platt parameters; never renormalize missing branches."""
    started = time.perf_counter()
    result = _base_result("stack")
    if _cfg(config,"stack").get("disabled_reason"):
        return _finish(result,started,ArtifactUnavailable(_cfg(config,"stack")["disabled_reason"]))
    try:
        _, manifest, path = _read_artifact(config, "stack", "sunhacks-stack-logistic-v1", "parameters.json")
        parameters = json.loads(path.read_text())
        names = parameters.get("inputs")
        coefficients = parameters.get("coefficients")
        if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names) or len(set(names)) != len(names):
            raise ArtifactUnavailable("Stack input names must be explicit and unique.")
        if not isinstance(coefficients, list) or len(coefficients) != len(names):
            raise ArtifactUnavailable("Stack coefficient count does not match input count.")
        numbers = coefficients + [parameters.get("intercept"), parameters.get("platt_a"), parameters.get("platt_b")]
        if any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in numbers):
            raise ArtifactUnavailable("Stack coefficients and Platt parameters must be finite numbers.")
        if parameters.get("input_semantics") != "class_1_probability" or manifest.get("missingness_policy") != "reject":
            raise ArtifactUnavailable("Stack requires complete class-1 probabilities and explicit reject missingness.")
        probabilities = []
        for name in names:
            item = branch_scores.get(name,{})
            values = item.get("probabilities",[])
            if item.get("status") != "trained" or not isinstance(values,list) or len(values) != 2:
                raise ArtifactUnavailable(f"Stack input {name} is missing or untrained; no fallback renormalization.")
            value = values[1]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1) > 1e-5:
                raise ArtifactUnavailable(f"Stack input {name} is not a finite probability.")
            if item.get("class_names") != manifest.get("class_names"):
                raise ArtifactUnavailable(f"Stack class order differs from {name}.")
            probabilities.append(value)
        classes = _class_names(manifest)
        raw = float(np.dot(coefficients, probabilities) + parameters["intercept"])
        calibrated_logit = parameters["platt_a"] * raw + parameters["platt_b"]
        score = float(1/(1+math.exp(-max(-700,min(700,calibrated_logit)))))
        result.update(status="trained", architecture=manifest["architecture"], class_names=classes,
                      probabilities=[1-score,score], calibration_status="parameters_provided_validation_unverified",
                      reason="Explicit logistic and Platt parameters applied; validation/calibration quality has not been independently established.")
    except (ArtifactUnavailable, OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        return _finish(result, started, exc)
    return _finish(result, started)
