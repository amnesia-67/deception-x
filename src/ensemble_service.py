"""Best-available live prediction: the stage-2 stacked ensemble, served from Node 2.

Node 1 (the Mac) sends one spoken answer: video frames, 16 kHz mono audio and the transcript.
Everything else runs here, computed exactly as in training:

  tuned_dasheng  5 LoRA-tuned dasheng fold models (OOF AUROC 0.764) on the audio
  xgb_digest     89 behavioural features -> XGBoost (OOF AUROC 0.737):
                   text      src.features_tabular.text_features(transcript, duration)
                   voice     voice_features(audio)
                   face      MediaPipe face boxes + 384px crops -> face_features
                   emotion   frozen emotion2vec+ 9-class probabilities
                   intent    frozen Qwen3-Embedding similarity to 33 forensic dialogue acts
  stacker        Platt-scaled convex combination fit on out-of-fold predictions (models/ensemble_deploy.json)
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np
import torch

from src.live_frames import N_CACHE, RES  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
SR = 16000
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]


def eval_audio(a: np.ndarray, max_s: float = 60.0, min_s: float = 1.0) -> np.ndarray:
    """Identical to src.clip_store.audio_tensor(train=False): centre 60 s, padded to >= 1 s."""
    lim = int(max_s * SR)
    if len(a) > lim:
        s = (len(a) - lim) // 2
        a = a[s:s + lim]
    if len(a) < int(min_s * SR):
        a = np.pad(a, (0, int(min_s * SR) - len(a)))
    return np.ascontiguousarray(a, dtype=np.float32)


def freeze_for_serving(model: torch.nn.Module) -> torch.nn.Module:
    """eval() + requires_grad=False everywhere: no autograd state, and autocast never caches weights."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def decode_audio_payload(payload: Dict) -> np.ndarray:
    if "audio_pcm16_b64" in payload:
        return np.frombuffer(base64.b64decode(payload["audio_pcm16_b64"]), dtype="<i2").astype(np.float32) / 32768.0
    if "audio_f32_b64" in payload:
        return np.frombuffer(base64.b64decode(payload["audio_f32_b64"]), dtype="<f4").astype(np.float32)
    raise ValueError("send 16 kHz mono audio as 'audio_pcm16_b64' (int16 little-endian) or 'audio_f32_b64'")


class EnsembleService:
    def __init__(self, device: str = "cuda"):
        from src.backbones import Emotion2vecAudio, Qwen3Text
        from src.explainer import ForensicTreeSHAPExplainer
        from src.features_tabular import INTENTS
        from src.tabular_branch import TabularLOSOXGBoost
        from src.tuning import TuneConfig, TunedModel

        from src.features_tabular import FEATURE_VERSION

        self.meta = json.load(open(os.path.join(MODELS, "ensemble_deploy.json")))
        self.schema = json.load(open(os.path.join(MODELS, "branch2_digest_schema.json")))
        self.names = self.schema["feature_names"]
        # Never run old XGBoost weights on features computed by changed formulas (or vice versa).
        for what, v in (("digest schema", self.schema.get("feature_version")),
                        ("ensemble manifest", self.meta.get("feature_version"))):
            if v != FEATURE_VERSION:
                raise ValueError(f"{what} was exported for feature version {v!r}, but src/features_tabular.py "
                                 f"computes {FEATURE_VERSION!r}; re-run train_stage2 / nested_deploy_eval / export")
        self.threshold = float(self.meta["threshold"])
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("ensemble_deploy.json threshold must be in (0, 1)")
        self.device = device
        # digest model + SHAP
        self.xgb = TabularLOSOXGBoost(device=device)
        self.xgb.load_model(os.path.join(MODELS, "branch2_xgboost.json"))
        self.explainer = ForensicTreeSHAPExplainer(self.xgb, self.names)
        # frozen feature extractors (exactly as scripts/extract_frozen_features.py: lora=None)
        self.e2v = Emotion2vecAudio(lora=None, device=device).eval()
        self.qwen = Qwen3Text(lora=None, device=device).eval()
        self.amp_dtype = torch.bfloat16      # as trained and served; None = FP32 parity reference
        self._intents = INTENTS
        self.intent_emb = self._intent_embeddings()
        # tuned dasheng folds (one model per adapter shape; fold weights swapped in)
        import hashlib
        import io

        from src.jepa_deploy import load_fold_state_strict, validate_fold_state

        self._load_strict = load_fold_state_strict
        valid = {f.name for f in fields(TuneConfig)}
        entries = self.meta.get("dasheng_folds") or {}
        if set(entries) != {f"fold{k}.pt" for k in range(5)}:
            raise ValueError("ensemble_deploy.json must list exactly fold0.pt..fold4.pt with hashes")
        self.dasheng_folds, self.dasheng_models = [], {}
        for k in range(5):
            name = f"fold{k}.pt"
            data = open(os.path.join(MODELS, "dasheng_lora", name), "rb").read()
            if hashlib.sha256(data).hexdigest() != entries[name]["sha256"] or len(data) != entries[name]["size"]:
                raise ValueError(f"dasheng {name} does not match its recorded SHA256/size")
            ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if set(ck) != {"config", "state"} or set(ck["config"]) - valid or ck["config"] != entries[name]["config"]:
                raise ValueError(f"dasheng {name}: unexpected checkpoint layout/config")
            cfg = TuneConfig(**ck["config"])
            key = (cfg.lora_r, cfg.lora_blocks)
            if key not in self.dasheng_models:
                self.dasheng_models[key] = freeze_for_serving(TunedModel(cfg, device=device))
            validate_fold_state(self.dasheng_models[key], ck["state"])  # complete keys/shapes/dtypes, finite
            self.dasheng_folds.append((key, ck["state"]))
        self.lock = threading.Lock()

    def _amp(self):
        return torch.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None)

    @torch.no_grad()
    def _intent_embeddings(self) -> np.ndarray:
        with self._amp():
            batch = self.qwen.tok(self._intents, padding=True, truncation=True, max_length=64,
                                  return_tensors="pt").to(self.device)
            e = self.qwen.model(**batch).last_hidden_state[:, -1].float()
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def to_fp32_reference(self) -> "EnsembleService":
        """Cast every neural model to FP32 and disable autocast: the CUDA FP32 reference of the parity
        protocol (served outputs are BF16). Fold states are FP32 already and load unchanged."""
        for m in [self.e2v, self.qwen, *self.dasheng_models.values()]:
            m.float()
        self.amp_dtype = None
        self.intent_emb = self._intent_embeddings()
        return self

    def info(self) -> Dict:
        return {"model": "stacked_ensemble", "branches": self.meta["branches"], "weights": self.meta["weights"],
                "threshold": self.threshold, "threshold_provenance": self.meta["threshold_provenance"],
                "feature_version": self.meta["feature_version"],
                "cv_metrics": self.meta["cv_metrics"], "caveats": self.meta["caveats"]}

    # ------------------------------------------------------------------ features
    def _face(self, payload: Dict):
        from src.live_frames import prepare_face_crops

        crops, meta, duration, info = prepare_face_crops(payload)
        return crops, meta, duration, info

    @torch.no_grad()
    def predict(self, payload: Dict) -> Dict:
        from src.features_tabular import face_features, text_features, voice_features

        t0 = time.perf_counter()
        timings = {}
        transcript = str(payload.get("transcript", "")).strip()
        if not transcript:
            raise ValueError("'transcript' (verbatim text of the answer) is required")
        audio = decode_audio_payload(payload)
        if len(audio) < SR // 2:
            raise ValueError("audio shorter than 0.5 s")
        timings["audio_decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        t1 = time.perf_counter()
        crops, fmeta, duration, face_info = self._face(payload)
        timings["frame_decode_face_crop_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        t1 = time.perf_counter()
        duration = float(payload.get("duration_s", duration))
        row = {}
        row.update(text_features(transcript, duration))
        row.update(voice_features(audio))
        row.update(face_features(crops, fmeta))
        timings["handcrafted_features_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        a_eval = torch.from_numpy(eval_audio(audio)).to(self.device).unsqueeze(0)
        t_wait = time.perf_counter()
        with self.lock:
            timings["gpu_queue_wait_ms"] = round((time.perf_counter() - t_wait) * 1000, 1)
            t1 = time.perf_counter()
            with self._amp():
                emo = self.e2v.emotion_probs(a_eval)[0].float().cpu().numpy()
                q = self.qwen.embed_full([transcript])[0].float().cpu().numpy()
            fold_logits = []
            for key, state in self.dasheng_folds:
                m = self.dasheng_models[key]
                self._load_strict(m, state)  # rejects missing/unknown keys: no stale fold parameters
                # A fresh autocast region per fold: autocast caches bf16 copies of weights inside a region,
                # so swapping fold weights within one region would silently reuse the first fold's copy.
                with self._amp():
                    fold_logits.append(float(m.head(m.backbone(a_eval)).float()))
            torch.cuda.synchronize()
            timings["gpu_models_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        for k, e in enumerate(EMOTIONS):
            row[f"emo_{e}"] = float(emo[k])
        qn = q / np.linalg.norm(q)
        sims = qn @ self.intent_emb.T
        ip = np.exp((sims - sims.max()) / 0.05)
        ip /= ip.sum()
        for k in range(len(ip)):
            row[f"intent_{k:02d}"] = float(ip[k])
        missing = [n for n in self.names if n not in row]
        x = np.array([[row.get(n, np.nan) for n in self.names]], dtype=np.float32)
        p_xgb = float(self.xgb.predict_proba(x)[0])
        shap = self.explainer.explain(x, top_k=6)
        per_fold = [1 / (1 + np.exp(-l)) for l in fold_logits]
        p_dash = float(np.mean(per_fold))  # matches stage 2: OOF probs of each fold model; here averaged
        probs = {"tuned_dasheng": p_dash, "xgb_digest": p_xgb}
        lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
        cal = [1 / (1 + np.exp(-(lg(probs[b]) * A + B)))
               for b, A, B in zip(self.meta["branches"], self.meta["platt_A"], self.meta["platt_B"])]
        p = float(np.dot(self.meta["weights"], cal))
        extra = {}
        if payload.get("return_intermediates"):
            import hashlib

            extra["intermediates"] = {
                "precision": "cuda_bf16_autocast" if self.amp_dtype is not None else "cuda_fp32",
                "selected_frame_indices": fmeta.get("selected_indices"),
                "selected_timestamps_s": fmeta["timestamps"], "face_boxes_cxcys": fmeta["boxes_cxcys"],
                "crops_uint8_sha256": hashlib.sha256(np.ascontiguousarray(crops).tobytes()).hexdigest(),
                "crop_mean_rgb_per_frame": np.asarray(crops, np.float64).mean((1, 2)).round(4).tolist(),
                "audio_eval_samples": int(a_eval.shape[-1]),
                "qwen_embedding_1024": [float(v) for v in q],
                "intent_similarity_raw_33": [float(v) for v in sims],
                "emotion2vec_9_ordered": {e: float(v) for e, v in zip(EMOTIONS, emo)},
                "digest_88_ordered": [[n, None if not np.isfinite(v) else float(v)] for n, v in zip(self.names, x[0])],
                "dasheng_fold_logits": fold_logits,
                "branch_raw": probs, "branch_platt": dict(zip(self.meta["branches"], map(float, cal))),
                "final_score": p, "cutoff": self.threshold,
            }
        return {**extra,
            "model": "stacked_ensemble", "probability": round(p, 6),
            "threshold": self.threshold, "threshold_provenance": self.meta["threshold_provenance"],
            "verdict": "Deceptive" if p >= self.threshold else "Truthful",
            "feature_version": self.meta["feature_version"],
            "branches": {
                "tuned_dasheng": {"probability": round(p_dash, 6), "per_fold": [round(v, 6) for v in per_fold],
                                  "weight": self.meta["weights"][self.meta["branches"].index("tuned_dasheng")]},
                "xgb_digest": {"probability": round(p_xgb, 6),
                               "weight": self.meta["weights"][self.meta["branches"].index("xgb_digest")],
                               "top_deceptive_drivers": shap["top_positive_drivers"],
                               "top_truthful_drivers": shap["top_negative_drivers"]},
            },
            "emotion2vec": {e: round(float(v), 4) for e, v in zip(EMOTIONS, emo)},
            "face": face_info, "duration_s": round(duration, 2), "missing_features": missing,
            "digest_features": {n: (None if not np.isfinite(v) else round(float(v), 6)) for n, v in zip(self.names, x[0])},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "timings": timings,
            "n_frames_received": len(payload["frames"]),
            "cv_metrics": self.meta["cv_metrics"], "caveats": self.meta["caveats"],
        }
