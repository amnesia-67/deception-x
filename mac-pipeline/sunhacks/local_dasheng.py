"""Strict local inference for the teammate's real-data Dasheng fold export.

CPU float32 kernels use the trainer's BF16-rounded base values. Numerical
parity is measured on generated fixtures, not assumed for participant clips. No download, training, microphone, network or manager call
occurs here. The default factory uses only the already reviewed local Dasheng
source and weight files. A factory seam permits lightweight contract tests.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_REVISION = "53393582c2994025f2bbececc01eb7a893de1c96"
MANIFEST_SHA256 = "281d0f016737a9b92f3a3623a581300346303224d8adc877e078267c12531516"
BASE_SHA256 = "adaa439ebec13933501242364a29b7912c2695d0354061b278c873438b2736c3"
SOURCE_HASHES = {
    "configuration_dasheng.py": "7a524b9d37c0e0392b75bbfb0fe5e2d1f60f62a8f2c46917bb186b638580966b",
    "modeling_dasheng.py": "a7908ca04b9e3ec3c6fb04b3719d5ae180d555383cbe8ab4bd2f35be272cbe99",
    "feature_extraction_dasheng.py": "78139c9219ff9bc48f9c5f86fb3fba9e727cb765679a92db69066d5caeb9875c",
    "config.json": "b9ff40da4167dd776cb2fe084c9a80c0fdd6568c3b995e0255b46a9d435e7b80",
}
CONFIG_FIELDS = {
    "backbone", "lora_r", "lora_blocks", "lora_dropout", "lr_lora", "lr_head",
    "weight_decay", "head_dropout", "label_smoothing", "max_epochs", "min_epochs",
    "patience", "eff_batch", "micro_batch", "seed", "tag",
}
SR = 16000
MAX_FOLD_BYTES = 64 * 1024 * 1024


class LocalDashengUnavailable(ValueError):
    pass


def _sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LocalDashengUnavailable("Duplicate manifest key")
        result[key] = value
    return result


def _bad_constant(value):
    raise LocalDashengUnavailable("Nonfinite manifest number")


def _json_bytes(data):
    return json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)


def tensor_schema(state):
    if not isinstance(state, dict) or set(state) != {"lora", "head"}:
        raise LocalDashengUnavailable("State requires exactly lora and head groups")
    schema = {}
    for group, values in state.items():
        if not isinstance(values, dict) or not values:
            raise LocalDashengUnavailable("Missing trainable state group")
        schema[group] = {}
        for key, value in values.items():
            if (not isinstance(key, str) or not isinstance(value, torch.Tensor) or
                    value.layout != torch.strided or not value.is_floating_point() or
                    value.is_complex() or not torch.isfinite(value).all()):
                raise LocalDashengUnavailable("Only finite real floating tensor states are accepted")
            schema[group][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return schema


def validate_fold_state(model, state):
    tensor_schema(state)
    expected = {
        "lora": {k: v for k, v in model.backbone.state_dict().items() if "lora_" in k},
        "head": model.head.state_dict(),
    }
    if not expected["lora"]:
        raise LocalDashengUnavailable("Configured model has no LoRA tensors")
    for group in expected:
        if set(state[group]) != set(expected[group]):
            raise LocalDashengUnavailable("Incomplete or unknown " + group + " keys")
        for key, value in state[group].items():
            if value.shape != expected[group][key].shape or value.dtype != expected[group][key].dtype:
                raise LocalDashengUnavailable("Wrong state shape/dtype: " + key)


def load_fold_state(model, state):
    """Validate every tensor before changing either shared module."""
    validate_fold_state(model, state)
    result = model.backbone.load_state_dict(state["lora"], strict=False)
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise LocalDashengUnavailable("Backbone did not load its complete LoRA state")
    model.head.load_state_dict(state["head"], strict=True)


def eval_audio(waveform, sample_rate=SR):
    """Same center-60-second / minimum-one-second rule as the deployed model."""
    a = np.asarray(waveform)
    if sample_rate != SR or a.ndim != 1 or len(a) < SR // 2:
        raise LocalDashengUnavailable("Need at least 0.5 seconds of mono 16000 Hz audio")
    if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all() or np.max(np.abs(a)) > 1.0:
        raise LocalDashengUnavailable("Audio must be finite floating PCM in [-1, 1]")
    a = np.ascontiguousarray(a, dtype=np.float32)
    if len(a) > 60 * SR:
        start = (len(a) - 60 * SR) // 2
        a = a[start:start + 60 * SR]
    if len(a) < SR:
        a = np.pad(a, (0, SR - len(a)))
    return np.ascontiguousarray(a)


def interp_time(tokens, steps=32):
    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise LocalDashengUnavailable("Expected nonempty B,T,D temporal features")
    if tokens.shape[1] == steps:
        return tokens
    x = tokens.transpose(1, 2).float()
    y = (F.adaptive_avg_pool1d(x, steps) if tokens.shape[1] > steps else
         F.interpolate(x, size=steps, mode="linear", align_corners=False))
    return y.transpose(1, 2).to(tokens.dtype)


class AttnPoolHead(nn.Module):
    """Exact teammate head: LayerNorm, scalar attention, weighted pool, logit."""
    def __init__(self, dim=768, dropout=.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.norm(x.float())
        if x.ndim == 2:
            return self.out(self.drop(x)).squeeze(-1)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return self.out(self.drop((weights.unsqueeze(-1) * x).sum(1))).squeeze(-1)


class LocalDashengBackbone(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model
        import torchaudio.transforms as AT

        rank, blocks = config["lora_r"], config["lora_blocks"]
        if (type(rank) is not int or type(blocks) is not int or rank < 1 or
                not 1 <= blocks <= len(base_model.encoder.blocks)):
            raise LocalDashengUnavailable("Invalid LoRA rank or block count")
        indices = "|".join(str(i) for i in range(len(base_model.encoder.blocks) - blocks,
                                               len(base_model.encoder.blocks)))
        spec = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=config["lora_dropout"],
                          target_modules=rf"encoder\.blocks\.({indices})\.attn\.(qkv|proj)", bias="none")
        for param in base_model.parameters():
            param.requires_grad_(False)
        self.model = inject_adapter_in_model(spec, base_model).float().cpu().eval()
        self.model.encoder.init_bn.float().eval()
        self.mel = AT.MelSpectrogram(sample_rate=SR, n_fft=512, win_length=512, hop_length=160,
                                     f_min=0, f_max=8000, n_mels=64, center=True)
        self.to_db = AT.AmplitudeToDB(top_db=120)

    def forward(self, wav):
        # CPU float32 throughout. Apply frozen BN exactly once, before encoder.
        with torch.autocast("cpu", enabled=False):
            mel = self.to_db(self.mel(wav.float()))
            encoder = self.model.encoder
            bn = encoder.init_bn
            x = bn(mel.unsqueeze(1))
            encoder.init_bn = None
            try:
                tokens = encoder(x.float())
            finally:
                encoder.init_bn = bn
        return interp_time(tokens)


class LocalDashengFold(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.backbone = LocalDashengBackbone(base_model, config)
        self.head = AttnPoolHead(dropout=config["head_dropout"])


def reviewed_base_factory(base_path, source_path):
    """Return a local-only factory after source/base hash checks; no auto-code loader."""
    base_path, source_path = Path(base_path).resolve(), Path(source_path).resolve()
    for name, expected in SOURCE_HASHES.items():
        path = source_path / name
        if not path.is_file() or _sha_file(path) != expected:
            raise LocalDashengUnavailable("Reviewed Dasheng source hash differs: " + name)
    if (_sha_file(base_path / "model.safetensors") != BASE_SHA256 or
            _sha_file(base_path / "config.json") != SOURCE_HASHES["config.json"]):
        raise LocalDashengUnavailable("Dasheng base/config differs from the training export")

    def create(config):
        from .models import _local_module
        module = _local_module(source_path, "modeling_dasheng")
        base = module.DashengModel.from_pretrained(str(base_path), local_files_only=True,
                                                   use_safetensors=True, outputdim=None)
        from .training_precision import bf16_base_in_fp32
        base = bf16_base_in_fp32(base)
        return LocalDashengFold(base, config)
    return create


class TunedDashengEnsemble:
    def __init__(self, manifest_path, fold_dir, *, model_factory=None, base_path=None,
                 source_path=None, expected_manifest_sha256=MANIFEST_SHA256):
        raw = Path(manifest_path).read_bytes()
        if not raw or len(raw) > 4 * 1024 * 1024:
            raise LocalDashengUnavailable("Manifest empty or too large")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest_sha256 != expected_manifest_sha256:
            raise LocalDashengUnavailable("Ensemble manifest hash differs from the reviewed export")
        self.meta = _json_bytes(raw)
        from .local_digest import ensemble_policy
        try:self.policy=ensemble_policy(self.meta)
        except ValueError as exc:raise LocalDashengUnavailable("Wrong ensemble manifest format/feature policy") from exc
        entries = self.meta.get("dasheng_folds")
        expected_names = {f"fold{i}.pt" for i in range(5)}
        fold_dir = Path(fold_dir)
        if (not isinstance(entries, dict) or set(entries) != expected_names or
                {p.name for p in fold_dir.glob("fold*.pt")} != expected_names):
            raise LocalDashengUnavailable("Need exactly the five hashed Dasheng fold files")
        self._closed = False
        self._lock = threading.Lock()
        self.models, self.folds = {}, []
        if model_factory is None:
            if base_path is None or source_path is None:
                raise LocalDashengUnavailable("Reviewed local Dasheng base/source paths are required")
            model_factory = reviewed_base_factory(base_path, source_path)
        for name in sorted(expected_names):
            entry = entries[name]
            with (fold_dir / name).open("rb") as stream:
                data = stream.read(MAX_FOLD_BYTES + 1)
            if (not data or len(data) > MAX_FOLD_BYTES or len(data) != entry.get("size") or
                    hashlib.sha256(data).hexdigest() != entry.get("sha256")):
                raise LocalDashengUnavailable("Fold bytes/hash differ: " + name)
            ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if (not isinstance(ck, dict) or set(ck) != {"config", "state"} or
                    not isinstance(ck["config"], dict) or set(ck["config"]) - CONFIG_FIELDS or
                    ck["config"].get("backbone") != "dasheng" or ck["config"] != entry.get("config") or
                    tensor_schema(ck["state"]) != entry.get("state_schema")):
                raise LocalDashengUnavailable("Fold config or tensor schema differs: " + name)
            config = ck["config"]
            key = tuple(config[k] for k in ("lora_r", "lora_blocks", "lora_dropout", "head_dropout"))
            if key not in self.models:
                model = model_factory(config).float().cpu().eval()
                for param in model.parameters():
                    param.requires_grad_(False)
                self.models[key] = model
            validate_fold_state(self.models[key], ck["state"])
            self.folds.append((name, key, ck["state"]))

    def prepare(self):
        if self._closed:
            return {"status": "stopped", "weights_loaded": False, "kernels_warmed": False}
        return {"status": "ready", "weights_loaded": True, "kernels_warmed": False,
                "model_id": "mispeech/dasheng-base", "classifier": "tuned_dasheng_lora_5fold",
                "device": "cpu", "dtype": "float32", "folds": len(self.folds),
                "manifest_sha256": self.manifest_sha256, "gpu_bf16_parity_verified": False}

    @torch.inference_mode()
    def infer(self, waveform, *, sample_rate=SR, source_window_id=None, stop_event=None):
        started = time.perf_counter()
        audio = eval_audio(waveform, sample_rate)
        wav = torch.from_numpy(audio).unsqueeze(0)
        wait_started = time.perf_counter()
        def cancelled():
            return self._closed or (stop_event is not None and stop_event.is_set())
        with self._lock:
            queue_ms = (time.perf_counter() - wait_started) * 1000
            if cancelled():
                raise LocalDashengUnavailable("Dasheng inference cancelled")
            logits, sequences = [], []
            for _, key, state in self.folds:
                if cancelled():
                    raise LocalDashengUnavailable("Dasheng inference cancelled")
                model = self.models[key]
                load_fold_state(model, state)
                # Separate region per fold; frozen parameters and no bf16 cache.
                with torch.autocast("cpu", enabled=False):
                    tokens = model.backbone(wav)
                    if tuple(tokens.shape) != (1, 32, 768) or not torch.isfinite(tokens).all():
                        raise LocalDashengUnavailable("Dasheng emitted invalid temporal tokens")
                    value = float(model.head(tokens).float())
                if not math.isfinite(value):
                    raise LocalDashengUnavailable("Dasheng emitted a nonfinite logit")
                logits.append(value)
                sequences.append(tokens[0].float().cpu().numpy())
            if cancelled():
                raise LocalDashengUnavailable("Dasheng inference cancelled")
        scores = [_sigmoid(value) for value in logits]
        sequence = np.mean(np.stack(sequences), axis=0, dtype=np.float32)
        return {
            "status": "trained", "model_id": "mispeech/dasheng-base",
            "classifier": "tuned_dasheng_lora_5fold", "raw_score": float(np.mean(scores)),
            "per_fold_scores": scores, "per_fold_logits": logits, "calibrated": False,
            "score_semantics": "experimental_class1_score_not_honesty_probability",
            "trained_classifier": {"status": "trained", "raw_score": float(np.mean(scores)),
                                   "calibrated": False, "per_fold_scores": scores,
                                   "score_semantics": "experimental_class1_score_not_honesty_probability"},
            "class_1_meaning": self.meta["class_1_meaning"],
            "source_revision": self.policy["source_revision"], "manifest_sha256": self.manifest_sha256,
            "device": "cpu", "dtype": "float32", "gpu_bf16_parity_verified": False,
            "source_window_id": source_window_id,
            "audio_eval_samples": len(audio), "sample_rate": SR,
            "_sequence_embedding": sequence, "_embedding": sequence.mean(0),
            "embedding_shape": [768], "sequence_shape": [32, 768],
            "feature_aggregation": "mean_of_five_fold_temporal_features_not_fusion_input",
            "score_aggregation": "mean_of_five_sigmoid_fold_logits",
            "queue_wait_ms": round(queue_ms, 3), "elapsed_ms": round((time.perf_counter()-started)*1000, 3),
        }

    def close(self):
        self._closed = True
        with self._lock:
            self.models.clear()
            self.folds.clear()


def _sigmoid(value):
    return (1 / (1 + math.exp(-value)) if value >= 0 else
            math.exp(value) / (1 + math.exp(value)))


def combine_exported_scores(p_dasheng, p_digest, manifest):
    """Exact exported two-branch Platt + convex stack; no webcam-calibration claim."""
    branches = manifest.get("branches")
    if branches != ["tuned_dasheng", "xgb_digest"]:
        raise LocalDashengUnavailable("Only the reviewed Dasheng + digest export is supported")
    weights = np.asarray(manifest.get("weights"), dtype=np.float64)
    A, B = (np.asarray(manifest.get(k), dtype=np.float64) for k in ("platt_A", "platt_B"))
    probs = np.asarray([p_dasheng, p_digest], dtype=np.float64)
    if (any(v.shape != (2,) for v in (weights, A, B, probs)) or
            not all(np.isfinite(v).all() for v in (weights, A, B, probs)) or
            np.any(weights < 0) or abs(weights.sum()-1) > 1e-8 or
            np.any(probs < 0) or np.any(probs > 1)):
        raise LocalDashengUnavailable("Invalid exported calibration weights or branch scores")
    clipped = np.clip(probs, 1e-4, 1-1e-4)
    calibrated = [_sigmoid(float(z)) for z in (np.log(clipped/(1-clipped))*A+B)]
    return {"status": "experimental", "model": "stacked_ensemble",
            "raw_score": float(np.dot(weights, calibrated)), "calibrated": False,
            "score_semantics": "courtroom_platt_stack_not_webcam_honesty_probability",
            "branches": dict(zip(branches, probs.tolist())),
            "platt_branch_scores": dict(zip(branches, calibrated)),
            "weights": dict(zip(branches, weights.tolist()))}
