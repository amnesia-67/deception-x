"""Lifecycle and scheduling tests with fake loaders; no weights or GPU forwards."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sunhacks import asr, models


def valid_inputs():
    return np.zeros((64, 1, 1, 3), np.uint8), np.ones(64, np.float32), "Spoken fixture."


def fake_accelerators(monkeypatch):
    calls = {"gc": 0, "mps": 0, "cuda": 0}
    def count(name):
        def invoke(): calls[name] += 1
        return invoke
    monkeypatch.setattr(models.gc, "collect", count("gc"))
    fake = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(empty_cache=count("mps")),
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=count("cuda")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return calls


def install_encoder(registry, monkeypatch, *, device="mps"):
    loads = []
    def infer(branch, clip, audio, transcript):
        if branch not in registry._loaded:
            loads.append(branch)
            registry._loaded[branch] = (object(), None, device, None, False)
        return np.ones(models.DIMS[branch], np.float32), False, device
    monkeypatch.setattr(registry, "_infer", infer)
    return loads


def test_resident_encoders_survive_placeholders_and_repeat_window(monkeypatch):
    cleanup = fake_accelerators(monkeypatch)
    registry = models.ModelRegistry({"models": {
        name: {"keep_loaded": True} for name in models.DIMS
    }})
    loads = install_encoder(registry, monkeypatch)
    first = registry.run(*valid_inputs())
    second = registry.run(*valid_inputs())
    assert loads == ["vision", "dasheng", "emotion", "text"]
    assert first["fusion"]["status"] == second["tabular"]["status"] == "unavailable"
    assert first["text"]["load_cache_hit"] is False
    assert second["text"]["load_cache_hit"] is True
    assert cleanup == {"gc": 0, "mps": 0, "cuda": 0}
    assert set(registry._loaded) == set(models.DIMS)


def test_close_collects_once_per_device_and_is_idempotent(monkeypatch):
    cleanup = fake_accelerators(monkeypatch)
    registry = models.ModelRegistry({})
    registry._loaded = {
        "vision": (object(), None, "mps", None, False),
        "text": (object(), None, "mps", None, False),
        "dasheng": (object(), None, "cpu", None, False),
        "emotion": (object(), None, "cuda", None, False),
    }
    registry._sequences["vision"] = np.ones((1, 1024))
    registry.close()
    registry.close()
    assert registry._loaded == registry._sequences == {}
    assert cleanup == {"gc": 1, "mps": 1, "cuda": 1}


def test_unloaded_and_mock_branches_do_not_collect_or_flush(monkeypatch):
    cleanup = fake_accelerators(monkeypatch)
    registry = models.ModelRegistry({})
    assert registry._release("vision") is False
    assert registry.run(*valid_inputs(), demo=True)["vision"]["status"] == "mock"
    assert cleanup == {"gc": 0, "mps": 0, "cuda": 0}


def test_nonresident_cpu_branch_releases_without_flushing_mps(monkeypatch):
    cleanup = fake_accelerators(monkeypatch)
    registry = models.ModelRegistry({})
    install_encoder(registry, monkeypatch, device="cpu")
    result = registry.run(*valid_inputs(), branches=("dasheng",))
    assert result["dasheng"]["status"] == "base"
    assert not registry._loaded
    assert cleanup == {"gc": 1, "mps": 0, "cuda": 0}


def test_cancellation_skips_remaining_branches_without_loading(monkeypatch):
    registry = models.ModelRegistry({"models": {"dasheng": {"keep_loaded": True}}})
    loads = install_encoder(registry, monkeypatch)
    result = registry.run(*valid_inputs(), branches=("dasheng", "text", "emotion"),
                          should_stop=lambda: bool(loads))
    assert loads == ["dasheng"]
    assert result["dasheng"]["status"] == "base"
    assert result["text"]["status"] == result["emotion"]["status"] == "cancelled"
    assert "_embedding" not in result["text"]


@pytest.mark.parametrize("branches", [("unknown",), ("text", "text"), "text"])
def test_unknown_or_repeated_branch_rejected_before_inference(branches):
    with pytest.raises(ValueError):
        models.ModelRegistry({}).run(*valid_inputs(), branches=branches)


@pytest.fixture
def asr_cfg(tmp_path):
    native = {"model_type": "whisper", "d_model": 1280, "encoder_layers": 32,
              "decoder_layers": 4, "num_mel_bins": 128}
    (tmp_path / "config.json").write_text(json.dumps(native))
    (tmp_path / "model.safetensors").write_bytes(b"fixture-never-loaded")
    asr._CACHE.clear()
    yield {"path": str(tmp_path), "device": "cpu", "keep_loaded": True,
           "revision": asr.OFFICIAL_REVISION}
    asr._CACHE.clear()


def test_prepare_asr_loads_once_without_forward_and_reports_not_warmed(monkeypatch, asr_cfg):
    loads = []
    def load(cfg):
        loads.append(cfg)
        return object(), object(), "cpu", "float32"
    monkeypatch.setattr(asr, "_load_model", load)
    first = asr.prepare_asr(asr_cfg)
    second = asr.prepare_asr(asr_cfg)
    assert len(loads) == 1
    assert first["status"] == second["status"] == "ready"
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert first["kernels_warmed"] is second["kernels_warmed"] is False
    asr.clear_asr_cache()
    assert not asr._CACHE


def test_prepare_asr_requires_explicit_residency(monkeypatch, asr_cfg):
    monkeypatch.setattr(asr, "_load_model", lambda cfg: pytest.fail("Unexpected load"))
    result = asr.prepare_asr({**asr_cfg, "keep_loaded": False})
    assert result["status"] == "error" and "keep_loaded=true" in result["reason"]
    assert not asr._CACHE


def test_prepare_asr_load_failure_is_reported_without_cache(monkeypatch, asr_cfg):
    def fail(cfg): raise RuntimeError("Fixture load failed")
    monkeypatch.setattr(asr, "_load_model", fail)
    result = asr.prepare_asr(asr_cfg)
    assert result["status"] == "error" and "Fixture load failed" in result["reason"]
    assert not asr._CACHE


def test_empty_asr_shutdown_is_noop(monkeypatch, asr_cfg):
    monkeypatch.setattr(asr.gc, "collect", lambda: pytest.fail("Unexpected collection"))
    asr.clear_asr_cache()
