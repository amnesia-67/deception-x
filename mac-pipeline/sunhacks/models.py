"""Local, lazy model adapters. Nothing in this module downloads weights or trains.

Embeddings are private return fields; public fields describe provenance and health.
Base embeddings and illustrative mocks are never deception probabilities.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

CONTRACT = "sunhacks-input-v1"
MODEL_IDS = {
    "vision": "facebookresearch/vjepa2:vjepa2_1_vit_large_384",
    "dasheng": "mispeech/dasheng-base",
    "emotion": "emotion2vec/emotion2vec_plus_large",
    "text": "Qwen/Qwen3-Embedding-0.6B",
    "fusion": "teammate/fusion-pending",
    "tabular": "teammate/tabular-pending",
}
DIMS = {"vision": 1024, "dasheng": 768, "emotion": 1024, "text": 1024}
POOLING = {"vision": "mean_all_tokens", "dasheng": "mean_hidden_states", "emotion": "utterance_mean", "text": "last_token_l2"}


def input_settings(config: dict) -> dict:
    """Canonical, JSON-safe preprocessing identity for every trained artifact.

    Capture maximum edge is currently fixed in LiveCapture's pipeline usage.
    Device/dtype are deliberately excluded: compatible NVIDIA training exports
    must run on the Mac. Requested and actual sampling rates remain separate.
    """
    text_cfg = config.get("models", {}).get("text", {})
    return {
        "video_frames": 64,
        "sample_fps": float(config.get("sample_fps", 24)),
        "hop_seconds": float(config.get("hop_seconds", 32 / 24)),
        "capture_requested_fps": float(config.get("capture_fps", 30)),
        "capture_max_edge": 640,
        "vision_preprocessing": "opencv-linear-short438-rounded-center384-rgb-div255-imagenet-v1",
        "audio_sample_rate": 16000,
        "audio_channels": 1,
        "audio_alignment": "video-first-to-last-timestamp-half-open",
        "audio_missingness": "reject-any-missing-sample",
        "dasheng_preprocessing": {
            "n_fft": 512, "win_size": 512, "hop_size": 160,
            "feature_size": 64, "f_min": 0, "f_max": 8000,
            "center": True, "top_db": 120,
        },
        "emotion_waveform_normalization": "layer_norm-entire-window",
        "max_tokens": int(text_cfg.get("max_tokens", 512)),
        "text_policy": "raw-transcript-no-instruction",
        "embedding_pooling": dict(POOLING),
    }


class ModelUnavailable(RuntimeError):
    pass


def _sync(torch, device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _collect_model_memory(devices):
    """Collect released models once, without flushing unrelated empty branches."""
    gc.collect()
    if "torch" not in sys.modules:
        return
    torch = sys.modules["torch"]
    if "mps" in devices and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if "cuda" in devices and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reviewed_source(cfg):
    source = Path(cfg.get("source_path", "")).expanduser()
    if not cfg.get("allow_reviewed_code") or not cfg.get("source_path") or not source.is_dir():
        raise ModelUnavailable("Requires inspected official local source_path and allow_reviewed_code=true; remote code is never enabled implicitly.")
    if not cfg.get("revision"):
        raise ModelUnavailable("Record the reviewed source/checkpoint revision before enabling custom model code.")
    return source.resolve()


def _local_module(source, filename):
    """Load explicitly opted-in local Dasheng code with its relative imports."""
    namespace = "sunhacks_reviewed_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(source)]
        sys.modules[namespace] = package
    name = f"{namespace}.{filename}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / f"{filename}.py")
        if spec is None or spec.loader is None:
            raise ModelUnavailable(f"Missing reviewed local {filename}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _apply_adapter(model, cfg, branch, settings):
    if not cfg.get("adapter_path"):
        return model, False
    path = Path(cfg["adapter_path"]).expanduser().resolve()
    if not path.is_dir():
        raise ModelUnavailable("adapter_path must be an existing local directory")
    manifest = json.loads((path / "handoff.json").read_text())
    required = {"model_id": MODEL_IDS[branch], "base_revision": cfg.get("revision"),
                "input_contract": CONTRACT, "output_dim": DIMS[branch], "pooling": POOLING[branch]}
    if not cfg.get("revision") or any(manifest.get(k) != v for k, v in required.items()):
        raise ModelUnavailable("Adapter handoff does not match exact model/revision/input contract/dimension/pooling.")
    if manifest.get("preprocessing_settings") != settings:
        raise ModelUnavailable("Adapter preprocessing_settings do not match runtime sampling, windows, text limit or preprocessing.")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        expected = manifest.get("sha256", {}).get(filename)
        if not expected or hashlib.sha256((path / filename).read_bytes()).hexdigest() != expected:
            raise ModelUnavailable(f"Adapter hash missing or mismatched: {filename}")
    from peft import LoraConfig, inject_adapter_in_model
    from peft.utils.save_and_load import set_peft_model_state_dict
    from safetensors.torch import load_file
    lora = LoraConfig.from_pretrained(str(path), local_files_only=True)
    model = inject_adapter_in_model(lora, model)
    loaded = set_peft_model_state_dict(model, load_file(str(path / "adapter_model.safetensors")))
    if loaded.unexpected_keys or any("lora_" in k for k in loaded.missing_keys):
        raise ModelUnavailable("Adapter keys are incompatible with instantiated base model")
    return model, True


class ModelRegistry:
    """Each branch fails independently; live mode never silently becomes demo mode."""
    def __init__(self, config: dict):
        self.config = config
        self.models = config.get("models", {})
        self._loaded = {}
        self._sequences = {}
        self.on_stage = None

    def _load(self, branch):
        if branch in self._loaded:
            return self._loaded[branch]
        cfg = self.models.get(branch, {})
        if not cfg.get("path"):
            raise ModelUnavailable("Official local weights or compatible teammate artifact have not been configured.")
        path = Path(cfg["path"]).expanduser().resolve()
        if not path.exists():
            raise ModelUnavailable("Configured local model path does not exist.")
        import torch
        device = cfg.get("device", "cpu")
        if device not in ("cpu", "mps", "cuda"):
            raise ModelUnavailable("Adapter device must be cpu, mps or cuda; no remote connection is created.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ModelUnavailable("MPS is unavailable in this Python/PyTorch environment.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailable("CUDA is unavailable; this option is for a separate GPU-machine benchmark.")
        dtype_name = cfg.get("dtype", "float32")
        if dtype_name not in ("float32", "float16", "bfloat16"):
            raise ModelUnavailable("dtype must be float32, float16 or bfloat16")
        dtype = getattr(torch, dtype_name)
        processor = None
        if branch == "vision":
            source = _reviewed_source(cfg)
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            vit = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
            model = vit.vit_large(img_size=(384, 384), patch_size=16, num_frames=64,
                tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                uniform_power=False, use_rope=True, img_temporal_dim_size=1,
                interpolate_rope=True)
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint.get("ema_encoder")
            if state is None:
                raise ModelUnavailable("V-JEPA2.1 distilled ViT-L checkpoint must contain ema_encoder.")
            state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            del checkpoint, state
        elif branch == "dasheng":
            source = _reviewed_source(cfg)
            module = _local_module(source, "modeling_dasheng")
            extractor = _local_module(source, "feature_extraction_dasheng")
            model = module.DashengModel.from_pretrained(str(path), local_files_only=True,
                                                       use_safetensors=True, outputdim=None)
            processor = extractor.DashengFeatureExtractor.from_pretrained(str(path), local_files_only=True)
            expected_extractor = input_settings(self.config)["dasheng_preprocessing"]
            if any(getattr(processor, key, None) != value for key, value in expected_extractor.items() if key != "top_db"):
                raise ModelUnavailable("Dasheng local extractor settings differ from the fixed v1 preprocessing contract.")
        elif branch == "emotion":
            if not cfg.get("license_reviewed"):
                raise ModelUnavailable("Emotion2vec+ large uses a custom model license; record license_reviewed=true after retaining the applicable agreement.")
            # FunASR contains the built-in Emotion2vec architecture. No custom
            # remote model.py is enabled. Passing model_conf directly bypasses
            # Hub lookup and the SDK's permissive checkpoint loader entirely.
            from funasr import AutoModel
            import yaml
            native_cfg = yaml.safe_load((path / "config.yaml").read_text())
            if native_cfg.get("model") != "Emotion2vec" or native_cfg.get("model_conf", {}).get("embed_dim") != 1024:
                raise ModelUnavailable("Expected official Emotion2vec large configuration with1024-dimensional features.")
            if native_cfg["model_conf"].get("normalize") is not True:
                raise ModelUnavailable("Emotion2vec v1 requires official waveform normalization enabled.")
            tokenizer_cfg = dict(native_cfg.get("tokenizer_conf", {}))
            tokenizer_cfg["token_list"] = str(path / "tokens.txt")
            model = AutoModel(model="Emotion2vec", model_path=str(path),
                              model_conf=native_cfg["model_conf"],
                              tokenizer="CharTokenizer", tokenizer_conf=tokenizer_cfg,
                              init_param=None, device=device, disable_update=True,
                              trust_remote_code=False, hub="hf", log_level="ERROR")
            # FunASR's generic loader tolerates absent parameters. This official
            # checkpoint omits its unused pretraining decoder; remove that
            # module, then require every inference parameter to match strictly.
            model.model.modality_encoders["AUDIO"].decoder = None
            checkpoint = torch.load(str(path / "model.pt"), map_location="cpu",
                                    weights_only=True, mmap=True)
            state = checkpoint["model"]
            state = {k.removeprefix("module.").removeprefix("d2v_model."): v
                     for k, v in state.items()}
            model.model.load_state_dict(state, strict=True)
            if cfg.get("training_base_precision") == "bf16_roundtrip_fp32":
                if dtype != torch.float32 or cfg.get("adapter_path"):
                    raise ModelUnavailable("The reviewed BF16-base/FP32 runtime profile requires a frozen FP32 model")
                from .training_precision import bf16_base_in_fp32
                model.model = bf16_base_in_fp32(model.model)
            model.model.eval()
            del state, checkpoint
            if cfg.get("adapter_path"):
                model.model, trained = _apply_adapter(model.model, cfg, branch, input_settings(self.config))
            else:
                trained = False
            self._loaded[branch] = model, None, device, dtype, trained
            return self._loaded[branch]
        elif branch == "text":
            from transformers import AutoModel, AutoTokenizer
            processor = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                      trust_remote_code=False, padding_side="left")
            model = AutoModel.from_pretrained(str(path), local_files_only=True,
                trust_remote_code=False, use_safetensors=True, torch_dtype=dtype,
                attn_implementation="sdpa")
        else:
            raise ModelUnavailable("Teammate fusion/tabular model and ordered feature schema are missing; no classifier is fabricated.")
        model, trained = _apply_adapter(model, cfg, branch, input_settings(self.config))
        model = model.to(device=device, dtype=dtype).eval()
        self._loaded[branch] = model, processor, device, dtype, trained
        return self._loaded[branch]

    def _infer(self, branch, clip, audio, transcript):
        import torch
        model, processor, device, dtype, trained = self._load(branch)
        _sync(torch, device)
        sequence = None
        want_sequence = self.models.get("fusion",{}).get("architecture")=="friend-reda-dare-synthetic-v1"
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            if branch == "vision":
                # Official single-view evaluation: bilinear short side438,
                # center crop384, then RGB/255 and ImageNet normalization.
                import cv2
                height, width = clip.shape[1:3]
                short_side = int(384 * 256 / 224)
                if width < height:
                    new_width, new_height = short_side, int(short_side * height / width)
                else:
                    new_width, new_height = int(short_side * width / height), short_side
                left, top = int(round((new_width - 384)/2)), int(round((new_height - 384)/2))
                frames = np.stack([cv2.resize(frame, (new_width,new_height), interpolation=cv2.INTER_LINEAR)[top:top+384,left:left+384] for frame in clip])
                x = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).unsqueeze(0)
                x = x.to(device=device, dtype=dtype) / 255.0
                mean = torch.tensor([.485, .456, .406], device=device, dtype=dtype).view(1,3,1,1,1)
                std = torch.tensor([.229, .224, .225], device=device, dtype=dtype).view(1,3,1,1,1)
                tokens = model((x - mean) / std)
                expected_tokens = (clip.shape[0]//2)*24*24
                if tuple(tokens.shape[1:]) != (expected_tokens, 1024):
                    raise ValueError(f"Unexpected V-JEPA token shape {tuple(tokens.shape)}")
                out = tokens.mean(dim=1)[0]
                if want_sequence:
                    sequence = tokens.reshape(1,clip.shape[0]//2,24*24,1024).mean(dim=2)[0]
            elif branch == "dasheng":
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device=device, dtype=dtype) for k, v in inputs.items()}
                # logits applies sigmoid even without a classifier; pool raw features.
                hidden = model(**inputs).hidden_states
                out = hidden.mean(dim=1)[0]
                if want_sequence: sequence = hidden[0]
            elif branch == "emotion":
                result = model.generate(input=audio, fs=16000, granularity="utterance",
                                        extract_embedding=True, disable_pbar=True)
                out = torch.as_tensor(result[0]["feats"])
            else:
                max_tokens = int(self.models.get(branch, {}).get("max_tokens", 512))
                if not 1 <= max_tokens <= 32768:
                    raise ValueError("text.max_tokens must be within1..32768")
                inputs = processor([transcript], padding=True, truncation=True,
                                   max_length=max_tokens, return_tensors="pt")
                inputs = {k: v.to(device) for k,v in inputs.items()}
                hidden = model(**inputs).last_hidden_state
                if (inputs["attention_mask"][:, -1] == 1).all():
                    pooled = hidden[:, -1]
                else:
                    positions = inputs["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0], device=device), positions]
                out = torch.nn.functional.normalize(pooled, p=2, dim=1)[0]
            _sync(torch, device)
            embedding = out.detach().float().cpu().numpy().reshape(-1)
            if sequence is not None:
                self._sequences[branch] = sequence.detach().float().cpu().numpy()
        if embedding.size != DIMS[branch] or not np.isfinite(embedding).all():
            raise ValueError("Model emitted nonfinite features or an incompatible embedding dimension")
        return embedding, trained, device

    def _release(self, branch):
        """Release an actual loaded model; unavailable placeholders are no-ops."""
        bundle = self._loaded.pop(branch, None)
        if bundle is None:
            return False
        device = bundle[2]
        del bundle
        _collect_model_memory({device})
        return True

    def close(self):
        """Release all resident encoders once, after their worker has stopped.

        The caller owns lifecycle synchronization: do not close a registry while
        one of its branches is still executing. Repeated close calls are no-ops.
        """
        devices = {bundle[2] for bundle in self._loaded.values()}
        had_models = bool(self._loaded)
        self._loaded.clear()
        self._sequences.clear()
        if had_models:
            _collect_model_memory(devices)

    def run(self, clip: np.ndarray, audio: np.ndarray, transcript: str, demo: bool = False,
            *, branches=None, should_stop=None) -> dict:
        """Run selected independent branches, with cancellation between branches.

        A running accelerator forward cannot be interrupted here. Cancelled
        branches have no fabricated embedding and do not load model weights.
        Separate callers may schedule distinct resident branches independently;
        callers must not execute the same branch concurrently on this registry.
        """
        selected = tuple(MODEL_IDS) if branches is None else tuple(branches)
        if any(branch not in MODEL_IDS for branch in selected):
            raise ValueError("Unknown model branch requested")
        if len(set(selected)) != len(selected):
            raise ValueError("Model branches must not be repeated")
        result = {}
        for branch in selected:
            model_id = MODEL_IDS[branch]
            if should_stop is not None and should_stop():
                result[branch] = {"status": "cancelled", "model_id": model_id,
                                  "elapsed_ms": 0.0, "embedding_shape": None,
                                  "reason": "Inference stopped before this branch started.",
                                  "input_contract": CONTRACT}
                self._sequences.pop(branch, None)
                continue
            if self.on_stage is not None: self.on_stage(branch)
            self._sequences.pop(branch,None)
            started = time.perf_counter()
            item = {"status": "unavailable", "model_id": model_id, "elapsed_ms": 0.0,
                    "embedding_shape": None, "reason": "", "input_contract": CONTRACT}
            if branch not in ("fusion", "tabular"):
                item["load_cache_hit"] = branch in self._loaded
            try:
                if branch in ("fusion", "tabular"):
                    raise ModelUnavailable("Teammate trained artifact, ordered feature schema and validation evidence are missing.")
                if branch == "vision" and (clip is None or clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[0] != 64 or clip.shape[-1] != 3 or min(clip.shape[1:3]) < 1):
                    raise ModelUnavailable("Requires64 RGB uint8 frames shaped[64,H,W,3]; no synthetic padding of missing video.")
                if branch in ("dasheng", "emotion") and (audio is None or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all()):
                    raise ModelUnavailable("Requires nonempty finite mono audio at16000Hz; missing audio is not silence.")
                if branch == "text" and not transcript.strip():
                    raise ModelUnavailable("Transcript unavailable; supply manual text or configure local ASR.")
                if demo:
                    # Stable shape demonstration; carries no semantic interpretation.
                    vector = np.zeros(DIMS[branch], dtype=np.float32)
                    item.update(status="mock", reason="Illustrative zero embedding; no model weights were used.")
                    device = "mock"
                else:
                    vector, trained, device = self._infer(branch, clip, np.asarray(audio,dtype=np.float32), transcript)
                    item.update(status="trained" if trained else "base", reason="Local LoRA encoder features; classifier unvalidated." if trained else "Official base encoder features; no task-trained classifier.")
                item.update(_embedding=vector, embedding_shape=list(vector.shape), device=device,
                            pooling=POOLING[branch], revision=self.models.get(branch, {}).get("revision"))
                if branch in self._sequences:
                    item["_sequence_embedding"] = self._sequences.pop(branch)
            except ModelUnavailable as exc:
                item["reason"] = str(exc)
            except ImportError as exc:
                item["reason"] = f"Optional local model dependency unavailable: {exc}"
            except Exception as exc:
                item.update(status="error", reason=f"{type(exc).__name__}: {str(exc)[:350]}")
            finally:
                item["elapsed_ms"] = round((time.perf_counter() - started)*1000, 3)
                if not self.models.get(branch, {}).get("keep_loaded", False):
                    self._release(branch)
            result[branch] = item
        return result

    def transcribe(self, audio: np.ndarray) -> dict:
        cfg = self.models.get("asr", self.config.get("asr", {}))
        if cfg.get("backend")=="transformers":
            from .asr import transcribe_local
            return transcribe_local(audio,cfg)
        if not cfg.get("path"):
            return {"status":"unavailable", "text":"", "reason":"Local ASR path not configured; manual transcript remains available."}
        try:
            path = Path(cfg["path"]).expanduser().resolve()
            if not path.is_dir():
                raise ModelUnavailable("ASR path must be an existing local CTranslate2 model directory")
            from faster_whisper import WhisperModel
            model = WhisperModel(str(path), device="cpu", compute_type=cfg.get("compute_type","int8"), local_files_only=True)
            segments, info = model.transcribe(np.asarray(audio,dtype=np.float32), beam_size=1,
                                              vad_filter=False, language=cfg.get("language"))
            text = " ".join(segment.text.strip() for segment in segments)
            return {"status":"base", "text":text, "language":info.language, "device":"cpu",
                    "reason":"Local ASR transcription; verify recognition errors."}
        except Exception as exc:
            return {"status":"error", "text":"", "reason":f"{type(exc).__name__}: {str(exc)[:300]}"}
