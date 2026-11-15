"""Offline Whisper large-v3-turbo for bounded, real captured audio windows.

No download is performed here. No transcript is invented for missing/silent audio.
Output is automatic speech recognition, not guaranteed verbatim transcription.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import threading
import time

import numpy as np

MODEL_ID = "openai/whisper-large-v3-turbo"
OFFICIAL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
_CACHE = {}
_LOCK = threading.RLock()


def _sync(torch, device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _validate_path(cfg):
    if not cfg.get("path"):
        raise FileNotFoundError("Official local ASR model path is not configured")
    path = Path(cfg["path"]).expanduser().resolve()
    if not path.is_dir() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError("Local Whisper safetensors snapshot is missing")
    native = json.loads((path / "config.json").read_text())
    expected = {"model_type": "whisper", "d_model": 1280, "encoder_layers": 32,
                "decoder_layers": 4, "num_mel_bins": 128}
    if any(native.get(k) != v for k, v in expected.items()):
        raise ValueError("ASR path is not the selected Whisper large-v3-turbo architecture")
    return path


def _load_model(cfg):
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    path = _validate_path(cfg)
    device = cfg.get("device", "mps")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("ASR device must be cpu, mps or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no implicit CPU fallback was used")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no implicit CPU fallback was used")
    dtype_name = cfg.get("dtype", "float32" if device == "cpu" else "float16")
    if dtype_name not in ("float32", "float16"):
        raise ValueError("ASR dtype must be float32 or float16")
    if device == "cpu" and dtype_name != "float32":
        raise ValueError("The tested CPU ASR path requires float32")
    dtype = getattr(torch, dtype_name)
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if (getattr(processor.feature_extractor, "sampling_rate", None) != SAMPLE_RATE or
            getattr(processor.feature_extractor, "feature_size", None) != 128):
        raise ValueError("Whisper processor must produce128 mel features from16000Hz audio")
    model, loading = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, weights_only=True, dtype=dtype,
        attn_implementation="sdpa", output_loading_info=True)
    if any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Whisper checkpoint does not strictly match the instantiated architecture")
    model = model.to(device=device, dtype=dtype).eval()
    _sync(torch, device)
    return model, processor, device, dtype


def _detected_language(model, token_ids, configured):
    mapping = getattr(model.generation_config, "lang_to_id", {}) or {}
    by_id = {value: token.removeprefix("<|").removesuffix("|>") for token, value in mapping.items()}
    for value in token_ids[:8]:
        if value in by_id:
            return by_id[value]
    return configured or None


def clear_asr_cache():
    """Drop explicitly retained models, for stop/config transitions."""
    with _LOCK:
        if not _CACHE:
            return
        devices = {bundle[2] for bundle in _CACHE.values()}
        _CACHE.clear()
        gc.collect()
        if "mps" in devices:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        if "cuda" in devices:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _cache_key(cfg, path):
    device = cfg.get("device", "mps")
    return (str(path), device,
            cfg.get("dtype", "float32" if device == "cpu" else "float16"),
            cfg.get("revision"))


def prepare_asr(cfg: dict) -> dict:
    """Load explicitly resident ASR before capture; perform no forward or capture.

    Loading weights is not a kernel warmup. A caller that wants to measure a
    first forward must use an explicitly labelled fixture and discard its text.
    clear_asr_cache() releases the retained model on shutdown/reconfiguration.
    """
    started = time.perf_counter()
    result = {"status": "unavailable", "model_id": MODEL_ID,
              "revision": cfg.get("revision"), "kernels_warmed": False}
    try:
        if not cfg.get("keep_loaded", False):
            raise ValueError("ASR preparation requires keep_loaded=true")
        path = _validate_path(cfg)
        key = _cache_key(cfg, path)
        with _LOCK:
            cache_hit = key in _CACHE
            if not cache_hit:
                _CACHE[key] = _load_model(cfg)
            _, _, device, dtype = _CACHE[key]
            result.update(status="ready", cache_hit=cache_hit,
                          device=device, dtype=str(dtype),
                          reason="Official local ASR weights are resident; first-forward latency remains to be measured.")
    except Exception as exc:
        result.update(status="unavailable" if isinstance(exc, OSError) else "error",
                      reason=type(exc).__name__ + ": " + str(exc)[:300])
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def transcribe_local(audio: np.ndarray, cfg: dict) -> dict:
    """Transcribe one mono16kHz window, rejecting gaps/silence before model load.

    Word timestamps are not produced. The caller retains the source window range.
    keep_loaded defaults false to avoid overlapping all encoder weight footprints.
    """
    started = time.perf_counter()
    result = {"status": "unavailable", "text": "", "language": None,
              "model_id": MODEL_ID, "revision": cfg.get("revision"),
              "source": "local_asr", "word_timestamps": None,
              "timestamp_granularity": "window_only", "sample_rate": SAMPLE_RATE}
    try:
        if audio is None:
            raise ValueError("ASR audio is missing")
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("ASR requires nonempty finite mono audio")
        if values.size > SAMPLE_RATE * MAX_AUDIO_SECONDS:
            raise ValueError("ASR window exceeds30seconds; explicit segmentation is required")
        result["duration_seconds"] = values.size / SAMPLE_RATE
        minimum_rms = float(cfg.get("minimum_rms", .0001))
        if not np.isfinite(minimum_rms) or not 0 <= minimum_rms <= .1:
            raise ValueError("Invalid ASR silence threshold")
        rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        if rms < minimum_rms:
            result.update(status="missing", reason="Audio is below the configured RMS floor; no speech transcript was invented.")
            return result
        tokens = cfg.get("max_new_tokens", 128)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= 440:
            raise ValueError("ASR max_new_tokens must be within1..440")
        language = cfg.get("language")
        if language is not None and (not isinstance(language, str) or not language.strip() or len(language) > 80):
            raise ValueError("Invalid ASR language hint")
        path = _validate_path(cfg)
    except (OSError, ValueError, TypeError) as exc:
        result.update(status="unavailable" if isinstance(exc, OSError) else "error",
                      reason=type(exc).__name__ + ": " + str(exc)[:240])
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result

    with _LOCK:
        import torch
        device = cfg.get("device", "mps")
        key = _cache_key(cfg, path)
        model = processor = inputs = generated = bundle = None
        try:
            loaded = time.perf_counter()
            result["load_cache_hit"] = key in _CACHE
            if key not in _CACHE:
                _CACHE[key] = _load_model(cfg)
            bundle = _CACHE[key]
            model, processor, device, dtype = bundle
            result["load_or_cache_ms"] = round((time.perf_counter() - loaded) * 1000, 3)
            inputs = processor(values, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                               truncation=False, padding="max_length", return_attention_mask=True)
            if "input_features" not in inputs:
                raise ValueError("Whisper processor did not return input_features")
            inputs = {name: tensor.to(device=device, dtype=dtype) if name == "input_features" else tensor.to(device=device)
                      for name, tensor in inputs.items()}
            _sync(torch, device)
            forward_started = time.perf_counter()
            generation = {"task": "transcribe", "return_timestamps": False,
                          "max_new_tokens": tokens, "num_beams": 1, "do_sample": False,
                          "condition_on_prev_tokens": False, "return_dict_in_generate": False}
            if language:
                generation["language"] = language
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation)
            _sync(torch, device)
            ids = generated.detach().cpu()
            text = processor.batch_decode(ids, skip_special_tokens=True, decode_with_timestamps=False, clean_up_tokenization_spaces=False)[0].strip()
            sequence = ids[0].tolist()
            eos = getattr(model.generation_config, "eos_token_id", None)
            eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
            truncated = bool(sequence and sequence[-1] not in eos_ids and len(sequence) >= tokens)
            result.update(status="partial" if truncated else ("base" if text else "missing"), text=text,
                          language=_detected_language(model, sequence, language), device=device,
                          dtype=str(dtype), truncated=truncated,
                          inference_ms=round((time.perf_counter() - forward_started) * 1000, 3),
                          reason="Local automatic transcript; recognition may normalize or omit disfluencies. Verify against the captured speech.",
                          limitations=["No word-level timestamps or guaranteed verbatim disfluency preservation."])
            if truncated:
                result["limitations"].append("Generation token bound reached; transcript may be incomplete.")
        except Exception as exc:
            result.update(status="error", text="", reason=type(exc).__name__ + ": " + str(exc)[:300])
        finally:
            inputs = generated = model = processor = bundle = None
            if not cfg.get("keep_loaded", False):
                released = _CACHE.pop(key, None)
                if released is not None:
                    del released
                    gc.collect()
                    if device == "mps" and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result
