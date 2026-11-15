"""Opt-in "webcam_pilot" profile: dasheng and emotion2vec (audio) and V-JEPA 2.1 ViT-L (video) branches LoRA-tuned on
the SunHacks webcam pilot (7 s clips, self-reported labels; the manifests' `trained_on` names the exact export), plus
an optional validation-chosen fusion of branch logits (fusion.json) and a per-session baseline verdict.

Nothing here changes the served default. The served courtroom models stay the default everywhere; a client must
ask for this profile by name (Mac: PilotAudioScorer; Node 2: `jepa_video` with "profile": "webcam_pilot", only
when the server was started with --webcam-pilot).

Layout (written by `scripts/pilot_finetune.py export`, never by hand):
    models/pilot_webcam/<backbone>/fold0.pt          {"config": TuneConfig fields, "state": trainable tensors}
    models/pilot_webcam/<backbone>/deploy_meta.json  format sunhacks-pilot-webcam-v1: hashes, config, state schema,
                                                     calibration (Platt fitted on the 31 validation clips only),
                                                     threshold, held-out metrics and caveats

Preprocessing is the served contract for each backbone: audio = 16 kHz mono float32 (src.ensemble_service.eval_audio),
video = raw frames + timestamps, face-cropped by src.live_frames exactly as the served ViT-L.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import fields
from typing import Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_DIR = os.path.join(ROOT, "models", "pilot_webcam")
PROFILE = "webcam_pilot"
MANIFEST_FORMAT = "sunhacks-pilot-webcam-v1"
LABELS = {"dasheng": "dasheng_lora_webcam_pilot", "emotion2vec": "emotion2vec_lora_webcam_pilot",
          "vjepa2_1_l": "vjepa2_1_vitl_lora_webcam_pilot"}
AUDIO_BRANCHES = ("dasheng", "emotion2vec")
VIDEO_AGGREGATION = "single model (no folds): mean of the 2 view logits"
FUSION_FILE = "fusion.json"
FUSION_FORMAT = "sunhacks-pilot-webcam-fusion-v1"


def video_contract() -> Dict:
    from src.jepa_deploy import CONTRACT

    return {**CONTRACT, "fold_aggregation": VIDEO_AGGREGATION}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def calibrate(raw_probability: float, calibration: Dict) -> float:
    """Platt scaling as exported: sigmoid(A * logit(raw) + B)."""
    return float(_sigmoid(calibration["A"] * _logit(raw_probability) + calibration["B"]))


def read_manifest(backbone: str, root: str = PILOT_DIR) -> tuple:
    """Return (manifest, manifest_sha256, checkpoint) after checking format, hash, size, config and state schema."""
    import torch

    from src.jepa_deploy import state_schema
    from src.tuning import TuneConfig

    folder = os.path.join(root, backbone)
    raw = open(os.path.join(folder, "deploy_meta.json"), "rb").read()
    meta = json.loads(raw)
    if meta.get("format") != MANIFEST_FORMAT or meta.get("profile") != PROFILE or meta.get("backbone") != backbone \
            or meta.get("model") != LABELS[backbone]:
        raise ValueError(f"{folder}: not a {MANIFEST_FORMAT} manifest for {backbone}")
    entries = meta.get("folds") or {}
    if set(entries) != {"fold0.pt"} or {n for n in os.listdir(folder) if n.endswith(".pt")} != {"fold0.pt"}:
        raise ValueError(f"{folder}: the webcam-pilot export is exactly one model, fold0.pt")
    entry = entries["fold0.pt"]
    data = open(os.path.join(folder, "fold0.pt"), "rb").read()
    if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["size"]:
        raise ValueError(f"{folder}/fold0.pt does not match its recorded SHA256/size")
    ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    valid = {f.name for f in fields(TuneConfig)}
    if set(ck) != {"config", "state"} or set(ck["config"]) - valid or ck["config"] != entry["config"]:
        raise ValueError(f"{folder}/fold0.pt: unexpected checkpoint layout/config")
    if ck["config"]["backbone"] != backbone or state_schema(ck["state"]) != entry["state_schema"]:
        raise ValueError(f"{folder}/fold0.pt: backbone or trainable-state schema differs from the manifest")
    cal = meta.get("calibration") or {}
    if not all(np.isfinite(cal.get(k, np.nan)) for k in ("A", "B")) or not 0.0 < meta.get("threshold", -1) < 1.0:
        raise ValueError(f"{folder}: calibration A/B and a threshold in (0, 1) are required")
    return meta, hashlib.sha256(raw).hexdigest(), ck


def _public(meta: Dict) -> Dict:
    return {"profile": PROFILE, "metrics": meta.get("metrics"), "trained_on": meta.get("trained_on"),
            "caveats": meta.get("caveats")}


class PilotAudioScorer:
    """Webcam-pilot audio branch (dasheng or emotion2vec) on 16 kHz mono float32 audio of one answer (Mac or Node 2).

    amp_dtype: torch.bfloat16 on CUDA reproduces how it was trained/selected; None = FP32 (CPU / MPS parity runs).
    """

    def __init__(self, device: str = "cuda", amp_dtype="auto", root: str = PILOT_DIR, backbone: str = "dasheng"):
        import torch

        from src.ensemble_service import freeze_for_serving
        from src.jepa_deploy import load_fold_state_strict, validate_fold_state
        from src.tuning import TuneConfig, TunedModel

        if backbone not in AUDIO_BRANCHES:
            raise ValueError(f"audio backbone must be one of {AUDIO_BRANCHES}")
        self.backbone = backbone
        self.meta, self.manifest_sha256, ck = read_manifest(backbone, root)
        self.device = device
        self.amp_dtype = (torch.bfloat16 if torch.device(device).type == "cuda" else None) if amp_dtype == "auto" \
            else amp_dtype
        self.model = freeze_for_serving(TunedModel(TuneConfig(**ck["config"]), device=device))
        validate_fold_state(self.model, ck["state"])
        load_fold_state_strict(self.model, ck["state"])
        self.threshold = float(self.meta["threshold"])

    def info(self) -> Dict:
        return {"model": LABELS[self.backbone], "manifest_sha256": self.manifest_sha256, "device": str(self.device),
                **_public(self.meta)}

    def score(self, audio_f32_16k: np.ndarray) -> Dict:
        import torch

        from src.ensemble_service import eval_audio

        a = torch.from_numpy(eval_audio(np.array(audio_f32_16k, dtype=np.float32))).to(self.device)[None]
        with torch.no_grad(), torch.autocast(torch.device(self.device).type, dtype=self.amp_dtype or torch.float32,
                                             enabled=self.amp_dtype is not None):
            logit = float(self.model.head(self.model.backbone(a)).float())
        if not np.isfinite(logit):
            raise ValueError(f"webcam-pilot {self.backbone} emitted a nonfinite logit")
        raw = float(_sigmoid(logit))
        p = calibrate(raw, self.meta["calibration"])
        return {"model": LABELS[self.backbone], "profile": PROFILE, "manifest_sha256": self.manifest_sha256,
                "logit": logit, "raw_score": raw, "probability": p, "threshold": self.threshold,
                "verdict": "Deceptive" if p >= self.threshold else "Truthful",
                "threshold_provenance": self.meta.get("threshold_provenance"), **_public(self.meta)}


class PilotVideoService:
    """Webcam-pilot V-JEPA 2.1 ViT-L behind the same strict loader and forward as the served JEPAService."""

    def __init__(self, device: str = "cuda", root: str = PILOT_DIR):
        from src.jepa_service import JEPAService

        meta, _, _ = read_manifest("vjepa2_1_l", root)  # hashes/schema checked here and again by JEPAService
        self.svc = JEPAService(os.path.join(root, "vjepa2_1_l"), device=device, n_folds=1,
                               manifest_format=MANIFEST_FORMAT, model_label=LABELS["vjepa2_1_l"],
                               contract=video_contract())
        self.meta, self.threshold = meta, float(meta["threshold"])

    def info(self) -> Dict:
        return {**self.svc.info(), **_public(self.meta)}

    def predict(self, payload: Dict) -> Dict:
        out = self.svc.predict(payload)
        p = calibrate(out["raw_score"], self.meta["calibration"])
        out.update({"profile": PROFILE, "calibrated": "platt_on_pilot_validation",
                    "calibrated_probability": round(p, 6), "threshold": self.threshold,
                    "threshold_provenance": self.meta.get("threshold_provenance"), **_public(self.meta)})
        out.pop("cv_auroc", None)
        return out


def read_fusion(root: str = PILOT_DIR) -> Dict:
    """The validation-selected branch combination; refuses it unless every branch manifest it names is on disk unchanged."""
    raw = open(os.path.join(root, FUSION_FILE), "rb").read()
    fu = json.loads(raw)
    if fu.get("format") != FUSION_FORMAT or fu.get("profile") != PROFILE or not fu.get("branches"):
        raise ValueError(f"{root}/{FUSION_FILE}: not a {FUSION_FORMAT} manifest")
    for bb, b in fu["branches"].items():
        sha = hashlib.sha256(open(os.path.join(root, bb, "deploy_meta.json"), "rb").read()).hexdigest()
        if sha != b["manifest_sha256"]:
            raise ValueError(f"fusion names {bb} manifest {b['manifest_sha256'][:12]}, but {sha[:12]} is on disk")
        if not (np.isfinite(b["logit_mean"]) and b["logit_sd"] > 0):
            raise ValueError(f"fusion: bad standardisation for {bb}")
    if not 0.0 < fu.get("threshold", -1) < 1.0:
        raise ValueError("fusion: threshold must be in (0, 1)")
    fu["fusion_sha256"] = hashlib.sha256(raw).hexdigest()
    return fu


def fuse(branch_logits: Dict[str, float], fusion: Dict) -> Dict:
    """Fused verdict from the raw (uncalibrated) logits of the branches the fusion manifest names.

    Each logit is standardised with its validation mean/sd, the standardised logits are averaged, then Platt and the
    validation cutoff are applied. Video's logit is logit(raw_score) from the `jepa_video` webcam_pilot reply.
    """
    missing = set(fusion["branches"]) - set(branch_logits)
    if missing:
        raise ValueError(f"fusion needs logits for {sorted(missing)}")
    z = {bb: (float(branch_logits[bb]) - b["logit_mean"]) / b["logit_sd"] for bb, b in fusion["branches"].items()}
    fused = float(np.mean(list(z.values())))
    p = float(_sigmoid(fusion["calibration"]["A"] * fused + fusion["calibration"]["B"]))
    return {"profile": PROFILE, "fused_logit": fused, "standardised_logits": z, "probability": p,
            "threshold": fusion["threshold"], "verdict": "Deceptive" if p >= fusion["threshold"] else "Truthful",
            "fusion_sha256": fusion.get("fusion_sha256")}


class SessionBaseline:
    """Per-session verdict relative to that session's own earlier answers (label-free).

    The pilot showed that one global cutoff gives a new person nearly the same verdict for every answer, because
    the scores shift per person/session. This compares each fused logit with the median of the session's answers so
    far (including it); no verdict until `min_answers` answers have been seen.
    """

    def __init__(self, min_answers: int = 4):
        self.min_answers, self.history = min_answers, []

    def add(self, fused_logit: float) -> Dict:
        self.history.append(float(fused_logit))
        if len(self.history) < self.min_answers:
            return {"verdict": None, "reason": f"baseline needs {self.min_answers} answers ({len(self.history)} so far)"}
        rel = float(fused_logit - np.median(self.history))
        return {"verdict": "Deceptive" if rel >= 0 else "Truthful", "relative_to_session_median": rel,
                "answers_in_baseline": len(self.history)}


def available(root: Optional[str] = None) -> Dict[str, bool]:
    root = root or PILOT_DIR
    return {bb: os.path.exists(os.path.join(root, bb, "deploy_meta.json")) for bb in LABELS}
