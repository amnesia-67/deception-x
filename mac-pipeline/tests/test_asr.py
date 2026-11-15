"""Local validation/generation contract tests. No weights or GPU forward loaded."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sunhacks.asr as asr


@pytest.fixture
def cfg(tmp_path):
    native = {"model_type": "whisper", "d_model": 1280, "encoder_layers": 32, "decoder_layers": 4, "num_mel_bins": 128}
    (tmp_path / "config.json").write_text(json.dumps(native))
    (tmp_path / "model.safetensors").write_bytes(b"fixture-placeholder-never-loaded")
    return {"path": str(tmp_path), "device": "cpu", "revision": asr.OFFICIAL_REVISION}


@pytest.fixture(autouse=True)
def no_cache():
    asr._CACHE.clear()
    yield
    asr._CACHE.clear()


def speech():
    return (.03 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)


def install_fake(monkeypatch, *, eos=True):
    calls = {"loads": 0, "processor": [], "generation": []}
    class Processor:
        def __call__(self, values, **kwargs):
            calls["processor"].append(kwargs)
            return {"input_features": torch.zeros(1, 128, 3000), "attention_mask": torch.ones(1, 3000, dtype=torch.long)}
        def batch_decode(self, ids, **kwargs): return ["Actual stub-decoded speech."]
    class Model:
        generation_config = SimpleNamespace(lang_to_id={"<|en|>": 50259}, eos_token_id=50257)
        def generate(self, **kwargs):
            calls["generation"].append(kwargs)
            return torch.tensor([[50258, 50259, 50360, 50364, 42, 50257 if eos else 42]])
    def load(cfg):
        calls["loads"] += 1
        return Model(), Processor(), "cpu", torch.float32
    monkeypatch.setattr(asr, "_load_model", load)
    return calls


def test_real_contract_uses_audio_mask_and_transcribe_no_word_timestamps(monkeypatch, cfg):
    calls = install_fake(monkeypatch)
    result = asr.transcribe_local(speech(), cfg)
    assert result["status"] == "base" and result["text"] == "Actual stub-decoded speech."
    assert result["language"] == "en" and result["device"] == "cpu"
    assert result["word_timestamps"] is None and result["timestamp_granularity"] == "window_only"
    assert calls["processor"][0]["sampling_rate"] == 16000
    assert calls["processor"][0]["return_attention_mask"] is True
    generation = calls["generation"][0]
    assert generation["task"] == "transcribe" and generation["return_timestamps"] is False
    assert generation["max_new_tokens"] == 128 and generation["num_beams"] == 1
    assert generation["attention_mask"].dtype == torch.long
    assert generation["input_features"].dtype == torch.float32
    assert not asr._CACHE


def test_configured_language_and_explicit_cache(monkeypatch, cfg):
    calls = install_fake(monkeypatch)
    cfg.update(language="english", keep_loaded=True)
    asr.transcribe_local(speech(), cfg); asr.transcribe_local(speech(), cfg)
    assert calls["loads"] == 1 and calls["generation"][0]["language"] == "english"
    asr.clear_asr_cache(); assert not asr._CACHE


@pytest.mark.parametrize("audio", [None, np.empty(0), np.ones((2, 30)), np.array([np.nan]), np.ones(480001)])
def test_invalid_missing_or_overlong_input_never_loads(monkeypatch, cfg, audio):
    monkeypatch.setattr(asr, "_load_model", lambda cfg: pytest.fail("Unexpected model load"))
    result = asr.transcribe_local(audio, cfg)
    assert result["status"] == "error" and result["text"] == ""


def test_silence_does_not_invent_speech(monkeypatch, cfg):
    monkeypatch.setattr(asr, "_load_model", lambda cfg: pytest.fail("Unexpected model load"))
    result = asr.transcribe_local(np.zeros(16000, np.float32), cfg)
    assert result["status"] == "missing" and result["text"] == ""


def test_missing_snapshot_is_unavailable(monkeypatch):
    monkeypatch.setattr(asr, "_load_model", lambda cfg: pytest.fail("Unexpected model load"))
    result = asr.transcribe_local(speech(), {"path": "/not/a/real/model"})
    assert result["status"] == "unavailable"


def test_wrong_model_architecture_rejected_before_load(monkeypatch, cfg):
    install_fake(monkeypatch)
    p = Path(cfg["path"]) / "config.json"; data = json.loads(p.read_text()); data["decoder_layers"] = 32; p.write_text(json.dumps(data))
    result = asr.transcribe_local(speech(), cfg)
    assert result["status"] == "error" and "large-v3-turbo" in result["reason"]


@pytest.mark.parametrize("limit", [0, 441, True, "128"])
def test_generation_bound_rejected_before_load(monkeypatch, cfg, limit):
    monkeypatch.setattr(asr, "_load_model", lambda cfg: pytest.fail("Unexpected model load"))
    assert asr.transcribe_local(speech(), {**cfg, "max_new_tokens": limit})["status"] == "error"


def test_failed_forward_has_no_fake_transcript_and_releases_cache(monkeypatch, cfg):
    calls = install_fake(monkeypatch)
    original = asr._load_model
    def load(cfg):
        model, processor, device, dtype = original(cfg)
        def fail(**kwargs): raise RuntimeError("test forward failure")
        model.generate = fail
        return model, processor, device, dtype
    monkeypatch.setattr(asr, "_load_model", load)
    result = asr.transcribe_local(speech(), cfg)
    assert result["status"] == "error" and not result["text"] and not asr._CACHE


def test_truncated_generation_marked_partial(monkeypatch, cfg):
    install_fake(monkeypatch, eos=False)
    result = asr.transcribe_local(speech(), {**cfg, "max_new_tokens": 4})
    assert result["status"] == "partial" and result["truncated"]


def test_loader_is_offline_safetensors_and_strict(monkeypatch, cfg):
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    calls = {}
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000, feature_size=128))
    class Model:
        def to(self, **kwargs): calls["to"] = kwargs; return self
        def eval(self): return self
    def load_processor(path, **kwargs): calls["processor"] = kwargs; return processor
    def load_model(path, **kwargs): calls["model"] = kwargs; return Model(), {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(AutoModelForSpeechSeq2Seq, "from_pretrained", load_model)
    asr._load_model(cfg)
    assert calls["processor"] == {"local_files_only": True, "trust_remote_code": False}
    assert calls["model"]["use_safetensors"] is True and calls["model"]["weights_only"] is True
    assert calls["model"]["local_files_only"] is True and calls["model"]["trust_remote_code"] is False
    assert calls["model"]["output_loading_info"] is True
    assert calls["to"] == {"device": "cpu", "dtype": torch.float32}


def test_missing_inference_weights_fail_strict_load(monkeypatch, cfg):
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000, feature_size=128))
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(AutoModelForSpeechSeq2Seq, "from_pretrained", lambda *a, **kw: (object(), {"missing_keys": ["encoder.layer.weight"]}))
    with pytest.raises(ValueError, match="strictly"):
        asr._load_model(cfg)
