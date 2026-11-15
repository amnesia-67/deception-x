"""Reject semantically incompatible training exports before loading adapter code."""
import builtins
import hashlib
import json

import pytest

from sunhacks.models import (
    CONTRACT, DIMS, MODEL_IDS, POOLING, ModelUnavailable,
    _apply_adapter, input_settings,
)


def adapter_export(tmp_path, training_config=None):
    branch = "text"
    path = tmp_path / "adapter"
    path.mkdir()
    hashes = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        content = b"metadata compatibility fixture; never loaded"
        (path / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "model_id": MODEL_IDS[branch], "base_revision": "reviewed-base-revision",
        "input_contract": CONTRACT, "output_dim": DIMS[branch],
        "pooling": POOLING[branch], "sha256": hashes,
        "preprocessing_settings": input_settings(training_config or {}),
    }
    (path / "handoff.json").write_text(json.dumps(manifest))
    cfg = {"adapter_path": str(path), "revision": "reviewed-base-revision"}
    return branch, cfg, path, manifest


@pytest.fixture
def forbid_adapter_code(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise AssertionError("Incompatible artifact reached PEFT import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_24fps_training_export_rejected_at_16fps_runtime(tmp_path, forbid_adapter_code):
    branch, cfg, _, _ = adapter_export(tmp_path, {"sample_fps": 24})
    with pytest.raises(ModelUnavailable, match="preprocessing_settings"):
        _apply_adapter(object(), cfg, branch, input_settings({"sample_fps": 16}))


def test_changed_text_truncation_rejected(tmp_path, forbid_adapter_code):
    branch, cfg, _, _ = adapter_export(tmp_path, {"models": {"text": {"max_tokens": 512}}})
    with pytest.raises(ModelUnavailable, match="preprocessing_settings"):
        _apply_adapter(object(), cfg, branch, input_settings({"models": {"text": {"max_tokens": 256}}}))


def test_wrong_base_revision_rejected_before_loading(tmp_path, forbid_adapter_code):
    branch, cfg, _, _ = adapter_export(tmp_path)
    cfg["revision"] = "different-base-revision"
    with pytest.raises(ModelUnavailable, match="revision"):
        _apply_adapter(object(), cfg, branch, input_settings({}))


def test_legacy_manifest_without_preprocessing_rejected(tmp_path, forbid_adapter_code):
    branch, cfg, path, manifest = adapter_export(tmp_path)
    del manifest["preprocessing_settings"]
    (path / "handoff.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelUnavailable, match="preprocessing_settings"):
        _apply_adapter(object(), cfg, branch, input_settings({}))


def test_numeric_config_spellings_have_same_behavior_identity():
    assert input_settings({"sample_fps": 24}) == input_settings({"sample_fps": 24.0})
    # Training on NVIDIA must not be rejected just for changing the backend.
    assert input_settings({"models": {"text": {"device": "cuda"}}}) == input_settings({"models": {"text": {"device": "mps"}}})
