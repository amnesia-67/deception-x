"""CPU integration checks for the exact reviewed export; no training or dataset."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from sunhacks import teammate_fusion as subject

CHECKPOINT = Path(__file__).resolve().parents[1] / ".models/teammate-20261111/branch1_fusion.pt"


@pytest.fixture
def config():
    return {"models": {"fusion": {
        "architecture": subject.ARCHITECTURE, "source_revision": subject.SOURCE_REVISION,
        "path": str(CHECKPOINT), "sha256": subject.CHECKPOINT_SHA256,
    }}, "runtime": {"text_scope": "video_window"}}


@pytest.fixture
def branches():
    rng = np.random.default_rng(81)
    result = {name: {"status": "base", "model_id": model_id} for name, model_id in subject.MODEL_IDS.items()}
    result["vision"]["_sequence_embedding"] = rng.normal(size=(32, 1024)).astype(np.float32)
    result["dasheng"]["_sequence_embedding"] = rng.normal(size=(71, 768)).astype(np.float32)
    result["text"]["_embedding"] = rng.normal(size=(1024,)).astype(np.float32)
    return result


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="Exact uploaded checkpoint is not available on this machine")
def test_actual_checkpoint_strict_load_and_forward_parity(branches, config):
    result = subject.run_teammate_fusion(branches, config)
    assert result["status"] == "synthetic_trained", result["reason"]
    assert result["calibrated"] is False
    assert result["training_feature_match_verified"] is False
    assert 0 <= result["raw_score"] <= 1
    assert abs(sum(result["modality_gates"].values()) - 1) < 1e-5
    assert all(np.isfinite(v) for v in result["feature_distances"].values())
    assert not any(key in result for key in ("probabilities", "classification", "verdict", "confidence"))
    model, adapter = subject._load(config["models"]["fusion"])
    text = branches["text"]["_embedding"][:128].copy()
    text /= np.linalg.norm(text.astype(np.float64))
    with torch.inference_mode():
        adapted = adapter(torch.from_numpy(branches["dasheng"]["_sequence_embedding"])[None])
        aligned = torch.nn.functional.interpolate(adapted.transpose(1, 2).float(), size=32,
                                                  mode="linear", align_corners=False).transpose(1, 2)
        expected = model(torch.from_numpy(branches["vision"]["_sequence_embedding"])[None], aligned,
                         torch.from_numpy(text)[None])
    assert result["raw_score"] == pytest.approx(float(expected["p_neural"].item()), abs=1e-7)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    for key, value in checkpoint["adapter_state_dict"].items():
        assert torch.equal(adapter.state_dict()[key], value.float())


@pytest.mark.parametrize("name", ["vision", "dasheng", "text"])
@pytest.mark.parametrize("status", ["mock", "unavailable", "error"])
def test_missing_or_mock_modality_is_never_replaced(branches, config, name, status):
    branches[name]["status"] = status
    result = subject.run_teammate_fusion(branches, config)
    assert result["status"] == "unavailable"
    assert result["raw_score"] is None


@pytest.mark.parametrize("name,field,shape", [
    ("vision", "_sequence_embedding", (32, 1024)),
    ("dasheng", "_sequence_embedding", (9, 768)),
    ("text", "_embedding", (1024,)),
])
def test_zero_padding_is_rejected(branches, config, name, field, shape):
    branches[name][field] = np.zeros(shape, dtype=np.float32)
    assert subject.run_teammate_fusion(branches, config)["status"] == "unavailable"


def test_temporal_shape_mismatch_is_rejected(branches, config):
    branches["vision"]["_sequence_embedding"] = branches["vision"]["_sequence_embedding"][:16]
    result = subject.run_teammate_fusion(branches, config)
    assert result["status"] == "unavailable" and "shape" in result["reason"]


def test_nonfinite_values_are_rejected(branches, config):
    branches["dasheng"]["_sequence_embedding"][0, 0] = np.nan
    result = subject.run_teammate_fusion(branches, config)
    assert result["status"] == "unavailable" and "finite" in result["reason"]


def test_fake_model_identity_is_rejected(branches, config):
    branches["vision"]["model_id"] = "facebook/vjepa2-vitl-fpc64-256"
    result = subject.run_teammate_fusion(branches, config)
    assert result["status"] == "unavailable" and "identity" in result["reason"]


def test_source_revision_must_match_before_import_or_load(config):
    config["models"]["fusion"]["source_revision"] = "other"
    with pytest.raises(subject.TeammateFusionUnavailable, match="source revision"):
        subject._load(config["models"]["fusion"])


def test_hash_configuration_must_match_before_import_or_load(config):
    config["models"]["fusion"]["sha256"] = "0" * 64
    with pytest.raises(subject.TeammateFusionUnavailable, match="SHA256"):
        subject._load(config["models"]["fusion"])


def test_tampered_weights_are_rejected(config, tmp_path):
    path = tmp_path / "branch1.pt"
    path.write_bytes(b"not the reviewed weights")
    config["models"]["fusion"]["path"] = str(path)
    with pytest.raises(subject.TeammateFusionUnavailable, match="file bytes"):
        subject._load(config["models"]["fusion"])


def test_qwen_prefix_cannot_be_fabricated_zeros(branches, config):
    branches["text"]["_embedding"][:128] = 0
    result = subject.run_teammate_fusion(branches, config)
    assert result["status"] == "unavailable" and "normalized" in result["reason"]
