"""Pure local inputs for the teammate's 89-feature digest and two-branch stacker.

No model loading, camera access, network or training occurs here. Callers must
provide the frozen emotion2vec nine-class probabilities, the full Qwen1024
testimony embedding, and its 33 normalized intent prototype embeddings. Prepared
faces must use the reviewed BlazeFace/crop recipe; this module does not replace
that detector with invented boxes. Vendor formulas are preserved byte for byte.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_REVISION = "53393582c2994025f2bbececc01eb7a893de1c96"
FEATURE_SOURCE_SHA256 = "f80c3c079a42bdb9bcee793ee1e41f495e7565cd8b8d520384f3fe44732881ec"
SCHEMA_SHA256 = "85779f9ddb30a42af7a7157642e34d7fcf7c5ebe58c35473ccc74932dcfd3bce"
LEGACY_FEATURE_VERSION = "v2-no-within-z"
NEW_FEATURE_VERSION = "v3-fixed-lag-motion"
NEW_SOURCE_REVISION = "2daf067e11fdca2d47351243a9780026115077b1"
NEW_FEATURE_SOURCE_SHA256 = "332b0b1aec833d888f6bd7d5c1dd735fe656d5d819dcaadd379dd4907d06ed0e"
NEW_SCHEMA_SHA256 = "bbed857c527031550a878e01e4be8670fcd2a169eea07a482389c4ead31b5356"
FEATURE_CONTRACTS = {
    LEGACY_FEATURE_VERSION: {"feature_count":89,"source_file":"friend_features_tabular.py",
        "source_sha256":FEATURE_SOURCE_SHA256,"schema_file":"friend_digest_schema.json",
        "schema_sha256":SCHEMA_SHA256,"source_revision":SOURCE_REVISION},
    NEW_FEATURE_VERSION: {"feature_count":88,"source_file":"friend_features_tabular_v3.py",
        "source_sha256":NEW_FEATURE_SOURCE_SHA256,"schema_file":"friend_digest_schema_v3.json",
        "schema_sha256":NEW_SCHEMA_SHA256,"source_revision":NEW_SOURCE_REVISION},
}


def digest_contract(feature_version=LEGACY_FEATURE_VERSION):
    if feature_version not in FEATURE_CONTRACTS:
        raise DigestInputError("Unsupported reviewed digest feature version")
    return dict(FEATURE_CONTRACTS[feature_version])
EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")
QWEN_TESTIMONY_INSTRUCTION = (
    "Given a verbatim spoken testimony transcript, represent its psycholinguistic, "
    "disfluency and reality-monitoring content for deception analysis"
)
QWEN_TESTIMONY_MAX_TOKENS = 512
QWEN_INTENT_MAX_TOKENS = 64


class DigestInputError(ValueError):
    """Actual compatible model outputs or synchronized inputs are unavailable."""


def _verify_vendor(feature_version=LEGACY_FEATURE_VERSION):
    vendor = Path(__file__).parent / "vendor"
    contract=digest_contract(feature_version)
    for name, digest in ((contract["source_file"],contract["source_sha256"]),
                         (contract["schema_file"],contract["schema_sha256"])):
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != digest:
            raise DigestInputError(f"Reviewed vendor file changed: {name}")
    return vendor


def _feature_module(feature_version):
    import importlib
    _verify_vendor(feature_version)
    name=digest_contract(feature_version)["source_file"][:-3]
    return importlib.import_module(".vendor."+name,__package__)


_verify_vendor()  # verify exact reviewed source bytes before importing its formulas
from .vendor import friend_features_tabular as features  # noqa: E402
INTENTS = tuple(features.INTENTS)


def format_testimony(transcript):
    """Exact instructed query; intent prototype documents remain uninstructed."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise DigestInputError("A final verbatim testimony transcript is required")
    return f"Instruct: {QWEN_TESTIMONY_INSTRUCTION}\nQuery: {transcript}"


def _finite(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise DigestInputError(f"{name} must have shape {shape} and finite values")
    return a


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise DigestInputError(f"{name} must be positive and finite")
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise DigestInputError(f"{name} must be positive and finite")
    return v


def load_digest_schema(feature_version=LEGACY_FEATURE_VERSION):
    """Load one of the two byte-pinned reviewed schema/formula contracts."""
    contract=digest_contract(feature_version)
    vendor = _verify_vendor(feature_version)
    schema = json.loads((vendor / contract["schema_file"]).read_bytes())
    names=schema.get("feature_names",[])
    count=contract["feature_count"]
    if schema.get("feature_version") != feature_version or len(names)!=count or len(set(names))!=count:
        raise DigestInputError("Reviewed digest feature version/count mismatch")
    return schema


def select_timestamp_frames(timestamps, *, n_cache=128):
    """Indices nearest 128 uniform times, matching the new analyze endpoint.

    Capture input must be chronological and finite. Duplicate selected frames
    remain duplicates; this function never labels them as additional unique fps.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if (ts.ndim != 1 or len(ts) < 16 or not np.isfinite(ts).all() or
            np.any(np.diff(ts) <= 0) or type(n_cache) is not int or n_cache != 128):
        raise DigestInputError("Need >=16 strictly increasing capture timestamps and the 128-frame profile")
    span = ts[-1] - ts[0]
    targets = ts[0] + (np.arange(n_cache) + 0.5) * span / n_cache
    nxt = np.clip(np.searchsorted(ts, targets), 0, len(ts) - 1)
    prv = np.clip(nxt - 1, 0, len(ts) - 1)
    near = np.where(np.abs(ts[prv] - targets) < np.abs(ts[nxt] - targets), prv, nxt)
    return near, ts[near]


def prepare_digest_faces(frames, timestamps, preprocessor, *, already_selected=False):
    """Select times, then invoke a reviewed crop adapter supplied by the caller.

    ``preprocessor(selected_rgb_frames, selected_timestamps)`` must return
    ``(crops_uint8_128x384x384x3, metadata)``. Metadata contains the original-pixel
    smoothed ``boxes_cxcys``, selected timestamps and actual detection rate. The
    callback owns the exact <=960px fit, BlazeFace, smoothing and 1.6x crop.
    Set ``already_selected=True`` for a shared 128-frame cache already selected
    by the pipeline. Its frame order and actual timestamps are preserved; repeated
    selected timestamps are permitted, but a positive chronological span remains
    required. The default selects the cache from original capture frames.
    """
    if not callable(preprocessor):
        raise DigestInputError("A reviewed face preprocessor is required")
    if len(frames) != len(timestamps):
        raise DigestInputError("One capture timestamp is required per frame")
    if type(already_selected) is not bool:
        raise DigestInputError("already_selected must be a boolean")
    if already_selected:
        ts = _finite(timestamps, (128,), "already-selected face timestamps")
        if len(frames) != 128 or np.any(np.diff(ts) < 0) or ts[-1] <= ts[0]:
            raise DigestInputError("Need exactly 128 already-selected chronological frames with positive span")
        idx = np.arange(128)
    else:
        idx, ts = select_timestamp_frames(timestamps)
    selected = []
    for i in idx:
        frame = np.asarray(frames[int(i)])
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not min(frame.shape[:2]):
            raise DigestInputError("Camera frames must be RGB uint8 HxWx3")
        selected.append(frame)
    crops, meta = preprocessor(selected, ts.copy())
    validate_prepared_faces(crops, meta)
    if not np.array_equal(np.asarray(meta["timestamps"]), ts):
        raise DigestInputError("Face adapter changed the selected capture timestamps")
    return crops, meta


def validate_prepared_faces(crops, meta):
    a = np.asarray(crops)
    if a.shape != (128, 384, 384, 3) or a.dtype != np.uint8:
        raise DigestInputError("Need 128 reviewed 384px RGB uint8 face crops")
    if not isinstance(meta, dict):
        raise DigestInputError("Actual face metadata is required")
    b = _finite(meta.get("boxes_cxcys"), (128, 3), "smoothed face boxes")
    ts = _finite(meta.get("timestamps"), (128,), "selected face timestamps")
    if np.any(b[:, 2] <= 0) or np.any(np.diff(ts) < 0) or ts[-1] <= ts[0]:
        raise DigestInputError("Face boxes need positive sides and chronological timestamps")
    rate = float(meta.get("face_detect_rate", np.nan))
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise DigestInputError("Actual face detection rate must be in [0,1]")


@dataclass(frozen=True)
class DigestRow:
    feature_names: tuple[str, ...]
    values: np.ndarray
    missing_features: tuple[str, ...]
    feature_version: str = LEGACY_FEATURE_VERSION

    def public_features(self):
        return {name: (float(v) if np.isfinite(v) else None)
                for name, v in zip(self.feature_names, self.values[0])}


def build_digest_row(*, transcript, audio, duration_s, face_crops, face_metadata,
                     emotion_probabilities, qwen_embedding, intent_embeddings, sample_rate=16000,
                     feature_version=LEGACY_FEATURE_VERSION):
    """Compute the exact ordered row. Undefined voice fields stay NaN.

    Pass original full-answer float audio here, not the separate center-60s,
    minimum-1s model waveform. Intent prototypes must be frozen Qwen embeddings
    of ``features.INTENTS`` in that exact order, left-padded/max64, L2-normalized.
    The testimony vector uses its reviewed instruction prefix/max512/full1024.
    """
    if sample_rate != 16000:
        raise DigestInputError("Digest audio must be 16kHz mono")
    if not isinstance(transcript, str) or not transcript.strip():
        raise DigestInputError("A final verbatim testimony transcript is required")
    duration = _positive(duration_s, "answer duration")
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim != 1 or len(wav) < 8000 or not np.isfinite(wav).all():
        raise DigestInputError("Need >=0.5s finite mono audio")
    validate_prepared_faces(face_crops, face_metadata)
    emo = _finite(emotion_probabilities, (9,), "frozen emotion2vec nine-class probabilities")
    if np.any(emo < 0) or np.any(emo > 1) or not np.isclose(emo.sum(), 1, atol=1e-3):
        raise DigestInputError("Emotion probabilities must be a normalized nine-class distribution")
    q = _finite(qwen_embedding, (1024,), "full frozen Qwen testimony embedding")
    intents = _finite(intent_embeddings, (33, 1024), "frozen Qwen intent prototypes")
    qnorm = np.linalg.norm(q)
    if qnorm <= 0 or not np.allclose(np.linalg.norm(intents, axis=1), 1, atol=2e-3):
        raise DigestInputError("Need nonzero Qwen testimony and normalized intent prototypes")
    schema = load_digest_schema(feature_version)
    extractor=_feature_module(feature_version)
    names = tuple(schema["feature_names"])
    row = extractor.text_features(transcript, duration)
    row.update(extractor.voice_features(wav))
    row.update(extractor.face_features(face_crops, face_metadata))
    row.update({f"emo_{name}": float(emo[i]) for i, name in enumerate(EMOTIONS)})
    sims = (q / qnorm) @ intents.T
    ip = np.exp((sims - sims.max()) / 0.05)
    ip /= ip.sum()
    row.update({f"intent_{i:02d}": float(ip[i]) for i in range(33)})
    if set(row) - set(names):
        raise DigestInputError("Feature extractor produced unknown schema fields")
    values = np.asarray([[row.get(name, np.nan) for name in names]], dtype=np.float32)
    if np.isinf(values).any():
        raise DigestInputError("Feature extractor emitted infinite values")
    missing = tuple(name for name, value in zip(names, values[0]) if np.isnan(value))
    return DigestRow(names, values, missing,feature_version)


def ensemble_policy(manifest):
    """Return only reviewed serving policies; never infer a replacement threshold."""
    if not isinstance(manifest,dict):raise DigestInputError("Expected an ensemble deployment manifest")
    fmt=manifest.get("format")
    if fmt=="sunhacks-ensemble-v1":
        version=manifest.get("feature_version",LEGACY_FEATURE_VERSION)
        if version!=LEGACY_FEATURE_VERSION:raise DigestInputError("Legacy manifest requires v2-no-within-z features")
        if "threshold" in manifest and (isinstance(manifest["threshold"],bool) or manifest["threshold"]!=.5):
            raise DigestInputError("Legacy export has a fixed 0.5 serving threshold")
        return {"threshold":.5,"threshold_provenance":"Legacy reviewed serving rule: class-1 score >= 0.5.",
                "feature_version":version,"source_revision":SOURCE_REVISION}
    if fmt!="sunhacks-ensemble-v2" or manifest.get("feature_version")!=NEW_FEATURE_VERSION:
        raise DigestInputError("Unsupported ensemble format/feature version pair")
    threshold=manifest.get("threshold")
    if (isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or
            not np.isfinite(threshold) or not 0<threshold<1):
        raise DigestInputError("Exported decision threshold must be finite and within (0,1)")
    provenance=manifest.get("threshold_provenance")
    if not isinstance(provenance,str) or not provenance.strip() or len(provenance)>4096:
        raise DigestInputError("Exported threshold provenance is required")
    return {"threshold":float(threshold),"threshold_provenance":provenance,
            "feature_version":NEW_FEATURE_VERSION,"source_revision":NEW_SOURCE_REVISION}


def combine_ensemble_scores(dasheng_score, digest_score, manifest):
    """Apply the exact exported Platt-on-logit then convex-weight formula.

    The caller verifies the manifest hash before passing it. This function does
    not turn a courtroom-calibrated score into a calibrated webcam probability.
    """
    ensemble_policy(manifest)
    branches = manifest.get("branches")
    if not isinstance(branches, list) or len(branches) != 2 or set(branches) != {"tuned_dasheng", "xgb_digest"}:
        raise DigestInputError("Only the exported Dasheng/digest two-branch stack is supported")
    w = _finite(manifest.get("weights"), (2,), "convex weights")
    A = _finite(manifest.get("platt_A"), (2,), "Platt slopes")
    B = _finite(manifest.get("platt_B"), (2,), "Platt offsets")
    if np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-8):
        raise DigestInputError("Stack weights must be nonnegative and sum to one")
    probs = {"tuned_dasheng": dasheng_score, "xgb_digest": digest_score}
    p = _finite([probs[name] for name in branches], (2,), "branch scores")
    if np.any(p < 0) or np.any(p > 1):
        raise DigestInputError("Branch scores must be in [0,1]")
    p = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p)) * A + B
    calibrated = np.empty_like(z)
    positive = z >= 0
    calibrated[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_negative = np.exp(z[~positive])
    calibrated[~positive] = exp_negative / (1 + exp_negative)
    return float(w @ calibrated)
