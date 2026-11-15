"""Explicit user-selected pilot blend; never modify the learned export."""
import math
import time

NAMES = ("dasheng", "vision", "emotion")


def checked_weights(settings):
    if settings is None:
        return None
    weights = settings.get("weights") if isinstance(settings, dict) else None
    if not isinstance(weights, dict) or set(weights) != set(NAMES):
        raise ValueError("Pilot blend requires dasheng, vision and emotion weights")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Pilot blend weights must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1., abs_tol=1e-9):
        raise ValueError("Pilot blend weights must sum to one")
    return {k: float(weights[k]) for k in NAMES}


def blend(branches, weights, window_id):
    """Average calibrated branch margins relative to their own exported cutoffs.

    A fixed blend has no fitted joint calibration. Missing inputs block it;
    weights are never silently renormalized to whichever models finish.
    """
    started = time.perf_counter()
    weights = checked_weights({"weights": weights})
    result = {"status": "unavailable", "profile": "webcam_pilot", "manual_blend": True,
              "model_id": "manual_weighted_pilot_blend", "source_window_id": window_id,
              "weights": weights, "raw_score": None, "threshold": None, "calibrated": False,
              "threshold_provenance": "User-selected weighted average of branch export cutoffs; not a fitted joint threshold.",
              "calibration_scope": "Individual branch calibrations only; this blend has not been validated.",
              "caveats": ["Manual weights have not been validated for accuracy."],
              "used_modalities": [], "inputs": {}, "elapsed_ms": 0.}
    for name, weight in weights.items():
        if not weight:
            continue
        outer = branches.get(name, {})
        if outer.get("source_window_id") != window_id:
            result["reason"] = "Missing same-answer " + name + " result; no partial blend."
            return result
        if name == "dasheng":
            item = outer.get("_pilot_decision", {})
            valid = outer.get("status") in ("trained", "trained_classifier") and item.get("status") == "trained_classifier" and item.get("profile") == "webcam_pilot" and item.get("source_window_id") == window_id
            score = item.get("raw_score")
        elif name == "emotion":
            item = outer.get("trained_classifier", {})
            valid = outer.get("status") == "base" and item.get("status") == "trained" and item.get("profile") == "webcam_pilot" and item.get("source_window_id", window_id) == window_id
            score = item.get("calibrated_probability")
        else:
            item = outer
            valid = item.get("status") == "trained_classifier"
            score = item.get("calibrated_probability")
        threshold = item.get("threshold")
        if not valid or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in (score, threshold)):
            result["reason"] = "Missing valid calibrated " + name + " result; no partial blend."
            return result
        result["inputs"][name] = {"score": float(score), "threshold": float(threshold),
                                  "model_id": item.get("model_id"), "manifest_sha256": item.get("manifest_sha256"),
                                  "source_window_id": window_id}
        result["used_modalities"].append(name)
    result.update(status="experimental_blend",
                  raw_score=sum(weights[k] * v["score"] for k, v in result["inputs"].items()),
                  threshold=sum(weights[k] * v["threshold"] for k, v in result["inputs"].items()),
                  elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                  reason="Manual weighted pilot blend; accuracy has not been validated.")
    return result
