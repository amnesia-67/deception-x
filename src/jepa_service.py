"""Live V-JEPA 2.1 ViT-L classifier inference for the socket server (Branch-1 visual model, Gemini.pdf).

Serves the five LoRA-tuned V-JEPA 2.1 ViT-L/16 @384 fold models (models/vjepa2_1_l_lora/fold<k>.pt) as an
ensemble (mean of fold logits). ViT-B and V-JEPA 2 artifacts are rejected; there is no fallback: when the
ViT-L export is missing the endpoint is unavailable.

Input (from Node 1): raw camera frames of one spoken answer (JPEG/PNG) with per-frame timestamps or fps.
Exactly as in training, the frame nearest each of 128 uniformly spaced times is face-cropped to 384 px
(src.live_frames), and scored on two 64-frame views (stride 1, the official ViT-L profile): 32 native tubelets
x 24x24 patches, width 1024, spatially mean-pooled, no temporal interpolation.

Speed: every fold adapts only its last 4 or 8 blocks with LoRA; the blocks below are the identical frozen base
encoder in every fold. Those frozen blocks are therefore run once per view and shared, and each fold runs only
its own adapted blocks (`share_prefix=True`). This is the same arithmetic in the same order, and
tests/test_jepa21_deploy.py checks that it equals the full per-fold forward exactly. Returns an uncalibrated
classifier score, not a verdict; its identity-grouped CV result is in the manifest's cv_metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.jepa_deploy import (BACKBONE, BASE_RELATIVE_PATH, CONTRACT, HIDDEN, MANIFEST_FORMAT, MODEL_ID,
                             MODEL_LABEL, file_sha256, load_fold_state_strict, read_checkpoint, state_schema,
                             validate_fold_state)
from src.backbones import interp_time, normalize_video
from src.live_frames import MIN_FRAMES, N_CACHE, RES, decode_image  # noqa: F401  (re-exported)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_DIR = os.path.join(ROOT, "models", "vjepa2_1_l_lora")


def first_adapted_block(model) -> int:
    idx = {int(name.split("blocks.", 1)[1].split(".", 1)[0])
           for name, _ in model.backbone.model.named_parameters() if "lora_" in name}
    if not idx:
        raise ValueError("Fold model has no LoRA blocks")
    if idx != set(range(min(idx), len(model.backbone.model.blocks))):
        raise ValueError("LoRA must cover a contiguous run of final blocks to share the frozen prefix")
    return min(idx)


def check_encoder_supports_split(enc):
    """The split forward replicates vision_transformer.VisionTransformer.forward for the served case only."""
    if not (enc.use_rope and enc.out_layers is None and enc.handle_nonsquare_inputs and not enc.attn_out):
        raise ValueError("Encoder configuration not supported by the shared-prefix forward")


def embed_video(enc, x):
    """Patch embedding + video modality embedding, as VisionTransformer.forward (RoPE, no masks, video mode)."""
    _, _, T, H, W = x.shape
    if enc.check_temporal_dim(x.shape):
        raise ValueError("Expected video input, not an image")
    dims = {"T": T // enc.tubelet_size, "H_patches": H // enc.patch_size, "W_patches": W // enc.patch_size}
    h = enc.patch_embed(x)
    if enc.modality_embedding:
        h += enc.video_mod_embed.repeat(h.shape[0], 1, 1)
    return h, dims


def run_blocks(enc, h, lo, hi, dims):
    for blk in enc.blocks[lo:hi]:
        h, _ = blk(h, mask=None, return_attn=enc.attn_out, mode="video", **dims)
    return h


class JEPAService:
    def __init__(self, deploy_dir: str = DEPLOY_DIR, device: str = "cuda", *, n_folds: int = 5,
                 manifest_format: str = MANIFEST_FORMAT, model_label: str = MODEL_LABEL, contract: Dict = None):
        """Defaults load the served 5-fold export. src.pilot_webcam passes its own single-model manifest
        (format, label, contract) for the opt-in webcam-pilot profile; nothing else changes."""
        from src.tuning import TuneConfig, TunedModel

        contract = dict(CONTRACT) if contract is None else dict(contract)
        self.model_label, self.contract = model_label, contract

        deploy_path = Path(deploy_dir)
        meta_path = deploy_path / "deploy_meta.json"
        with meta_path.open("rb") as stream:
            manifest_bytes = stream.read(4 * 1024 * 1024 + 1)
        if len(manifest_bytes) > 4 * 1024 * 1024:
            raise ValueError("Deployment manifest exceeds 4 MiB")
        self.meta = json.loads(manifest_bytes)
        if (self.meta.get("format") != manifest_format or self.meta.get("contract") != contract or
                self.meta.get("model") != model_label):
            raise ValueError("Expected a reviewed V-JEPA 2.1 ViT-L classifier manifest; ViT-B / V-JEPA 2 are not accepted")
        expected_names = {f"fold{k}.pt" for k in range(n_folds)}
        if {path.name for path in deploy_path.glob("fold*.pt")} != expected_names:
            raise ValueError(f"Deployment must contain exactly fold0.pt through fold{n_folds - 1}.pt")
        entries = self.meta.get("folds")
        if not isinstance(entries, dict) or set(entries) != expected_names:
            raise ValueError(f"Manifest must describe all {n_folds} folds exactly")
        base = self.meta.get("base_encoder", {})
        base_path = Path(ROOT) / BASE_RELATIVE_PATH
        if (base.get("path") != BASE_RELATIVE_PATH or not isinstance(base.get("sha256"), str) or
                len(base["sha256"]) != 64 or not base_path.is_file() or
                base.get("size") != base_path.stat().st_size or file_sha256(base_path) != base["sha256"]):
            raise ValueError("V-JEPA 2.1 ViT-L base encoder hash/size does not match manifest")
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.folds = []
        self.models: Dict[tuple, "TunedModel"] = {}
        valid = {field.name for field in fields(TuneConfig)}
        for name in sorted(expected_names):
            entry = entries[name]
            if (not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or
                    len(entry["sha256"]) != 64 or type(entry.get("size")) is not int):
                raise ValueError("Every fold requires a SHA256 and byte size")
            ck, _, _ = read_checkpoint(deploy_path / name, expected_sha256=entry["sha256"],
                                       expected_size=entry["size"])
            if (set(ck["config"]) - valid or ck["config"] != entry.get("config") or
                    state_schema(ck["state"]) != entry.get("state_schema")):
                raise ValueError("Unknown/mismatched fold config or trainable-state schema")
            cfg = TuneConfig(**ck["config"])
            if cfg.backbone != BACKBONE:
                raise ValueError("Only vjepa2_1_l (V-JEPA 2.1 ViT-L) folds are accepted")
            key = (cfg.backbone, cfg.lora_r, cfg.lora_blocks, cfg.lora_dropout, cfg.head_dropout)
            if key not in self.models:
                model = TunedModel(cfg, device=device).eval()
                for prm in model.parameters():  # serving only: autocast must not cache fold weights
                    prm.requires_grad_(False)
                bb = model.backbone
                if (bb.name, bb.n_frames, bb.res, bb.out_dim) != (BACKBONE, contract["frames_per_view"], RES, HIDDEN):
                    raise ValueError("Loader must instantiate V-JEPA 2.1 ViT-L / 64 frames / 1024 wide with a 1024 head")
                self.models[key] = model
            validate_fold_state(self.models[key], ck["state"])
            self.folds.append({"name": name, "key": key, "state": ck["state"]})
        self.n_frames = next(iter(self.models.values())).backbone.n_frames
        # Shared frozen prefix (a pure speed path with identical results): needs the real vendored encoder.
        self.share_prefix = all(hasattr(getattr(m.backbone, "model", None), "blocks") for m in self.models.values())
        if self.share_prefix:
            # First LoRA-adapted block of each model, read from the modules actually present (not the config).
            self.first_adapted = {key: first_adapted_block(model) for key, model in self.models.items()}
            # Prefix runner: the model with the longest frozen prefix; its blocks below that are pure base.
            self.prefix_key = max(self.first_adapted, key=self.first_adapted.get)
            for model in self.models.values():
                check_encoder_supports_split(model.backbone.model)
        self.lock = threading.Lock()
        self.device = device

    def info(self) -> Dict:
        return {"model": self.model_label, "model_id": MODEL_ID, "folds": len(self.folds),
                "adapter_shapes": sorted({(key[1], key[2]) for key in self.models}),
                "contract": dict(self.contract), "manifest_sha256": self.manifest_sha256,
                "device": str(self.device), "cv_metrics": self.meta.get("cv_metrics"),
                "caveat": self.meta.get("caveat")}

    def _windows(self, crops):
        from src.clip_store import eval_starts, select_frames
        # 128 cache frames -> two 64-frame views at stride 1 (first and second half).
        return [torch.from_numpy(np.ascontiguousarray(crops[select_frames(N_CACHE, self.n_frames, start)]))
                for start in eval_starts(N_CACHE)]

    def _prepare(self, payload: Dict):
        from src.live_frames import prepare_face_crops

        crops, _, _, face = prepare_face_crops(payload)
        return crops, face

    def _shared_prefixes(self, windows):
        """Run the frozen base blocks once per view: {first adapted block s: hidden state after block s-1}."""
        bb = self.models[self.prefix_key].backbone
        enc = bb.model
        starts = sorted(set(self.first_adapted.values()))
        out = []
        for win in windows:
            x = normalize_video(win.to(self.device).unsqueeze(0), bb.res).to(torch.bfloat16)
            h, dims = embed_video(enc, x.permute(0, 2, 1, 3, 4).contiguous())
            cache, done = {}, 0
            for s in starts:
                h = run_blocks(enc, h, done, s, dims)
                cache[s], done = h, s
            out.append((cache, dims))
        return out

    def _fold_tokens(self, model, key, prefix):
        """Finish one view for one fold from the shared prefix, exactly as VJEPA21Video.forward would."""
        bb = model.backbone
        enc = bb.model
        cache, dims = prefix
        s = self.first_adapted[key]
        h = enc.norms_block[-1](run_blocks(enc, cache[s], s, len(enc.blocks), dims))
        t, hh, ww = bb.grid
        return interp_time(h.view(1, t, hh * ww, -1).mean(2))

    @torch.no_grad()
    def predict(self, payload: Dict) -> Dict:
        t0 = time.perf_counter()
        crops, face = self._prepare(payload)                # (128, 384, 384, 3) uint8 face crops
        t_prep = time.perf_counter() - t0
        windows = self._windows(crops)
        device_type = torch.device(self.device).type
        with self.lock:
            per_fold, attn = [], []
            prefixes = None
            if self.share_prefix:
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                    prefixes = self._shared_prefixes(windows)
            for f in self.folds:
                model = self.models[f["key"]]
                load_fold_state_strict(model, f["state"])
                logits = []
                for v, win in enumerate(windows):
                    x = win.to(self.device).unsqueeze(0)
                    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                        if prefixes is not None:
                            tokens = self._fold_tokens(model, f["key"], prefixes[v])
                        else:
                            tokens = model.backbone(x)           # (1, 32, 1024): 32 native tubelets
                        if tuple(tokens.shape) != (1, 32, HIDDEN) or not torch.isfinite(tokens).all():
                            raise ValueError("V-JEPA 2.1 ViT-L emitted invalid/nonfinite 1024-wide tokens")
                        value = float(model.head(tokens).float())
                        if not np.isfinite(value):
                            raise ValueError("V-JEPA2.1 classifier emitted a nonfinite logit")
                        logits.append(value)
                        h = model.head.norm(tokens.float())
                        attention = torch.softmax(model.head.score(h).squeeze(-1), 1)[0].float().cpu().numpy()
                        if attention.shape != (32,) or not np.isfinite(attention).all():
                            raise ValueError("V-JEPA2.1 classifier emitted invalid/nonfinite attention")
                        attn.append(attention)
                per_fold.append(float(np.mean(logits)))
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(torch.device(self.device))
        logit = float(np.mean(per_fold))
        if not np.isfinite(logit):
            raise ValueError("V-JEPA2.1 ensemble emitted a nonfinite logit")
        sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value)) if value >= 0 else np.exp(value) / (1.0 + np.exp(value))
        p = sigmoid(logit)
        a = np.mean(attn, 0)  # temporal attention over 32 tubelets, averaged over folds and windows
        out = {
            "model": self.model_label, "model_id": MODEL_ID, "branch": 1,
            "classifier_only": True, "contract": dict(self.contract),
            "manifest_sha256": self.manifest_sha256, "device": str(self.device),
            "raw_score": round(float(p),6), "calibrated": False,
            "probability": round(float(p),6), "p_neural": round(float(p),6),
            "score_semantics": "experimental_sigmoid_classifier_score_not_honesty_probability",
            "per_fold_probability": [round(float(sigmoid(value)), 6) for value in per_fold],
            "temporal_attention_32": [round(float(v), 5) for v in a],
            "peak_tubelets": [int(i) for i in np.argsort(-a)[:3]],
            "n_frames_received": len(payload["frames"]),
            "shared_frozen_prefix": bool(self.share_prefix),
            "prep_ms": round(t_prep * 1000, 1),
            "inference_ms": round((time.perf_counter() - t0 - t_prep) * 1000, 1),
            "cv_auroc": (self.meta.get("cv_metrics") or {}).get("auroc"),
            "caveat": self.meta.get("caveat", "uncertain under identity-grouped CV; not evidence of deception"),
        }
        if face is not None:
            out["face"] = face
        return out

