"""Signal quality diagnostics, deliberately separate from learned behavior scores.

These values are not physiological measurements, deception indicators or the
plan's unimplemented 250-feature digest. Missing input remains None.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def audio_diagnostics(samples: np.ndarray | None, sample_rate: int = 16000,
                      activity_threshold_dbfs: float = -35.0) -> dict[str, Any]:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if samples is None or np.asarray(samples).size == 0:
        return {"status": "missing", "rms_dbfs": None, "clipping_fraction": None,
                "activity_fraction_proxy": None, "duration_seconds": 0.0}
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Audio must be finite, mono PCM")
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    rms_db = float(20 * np.log10(max(rms, 1e-8)))
    block = max(1, round(sample_rate * .02))
    energies = [float(np.sqrt(np.mean(values[i:i + block].astype(np.float64) ** 2)))
                for i in range(0, len(values), block)]
    threshold = 10 ** (activity_threshold_dbfs / 20)
    return {"status": "ok", "rms_dbfs": rms_db,
            "peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(values))), 1e-8))),
            "clipping_fraction": float(np.mean(np.abs(values) >= .999)),
            "activity_fraction_proxy": float(np.mean(np.asarray(energies) >= threshold)),
            "activity_method": "20ms RMS threshold; not voice activity detection",
            "activity_threshold_dbfs": activity_threshold_dbfs,
            "duration_seconds": len(values) / sample_rate, "sample_rate": sample_rate}


def video_diagnostics(frames: Iterable[Any]) -> dict[str, Any]:
    frames = list(frames)
    if not frames:
        return {"status": "missing", "brightness_mean": None,
                "mean_absolute_frame_difference": None, "frames": 0}
    # Duck-type frames as VideoFrame or HWC RGB arrays for standalone use.
    arrays = [np.asarray(getattr(frame, "rgb", frame)) for frame in frames]
    if any(array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Video frames must be finite HWC RGB arrays")
    brightness = float(np.mean([np.mean(array, dtype=np.float64) / 255 for array in arrays]))
    differences = [float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255)
                   for a, b in zip(arrays, arrays[1:]) if a.shape == b.shape]
    return {"status": "ok", "brightness_mean": brightness,
            "mean_absolute_frame_difference": float(np.mean(differences)) if differences else None,
            "frames": len(arrays),
            "interpretation": "Image quality/motion diagnostics only; no face, emotion or physiology inference"}


def signal_diagnostics(video_frames: Iterable[Any] = (), audio_samples: np.ndarray | None = None,
                       sample_rate: int = 16000) -> dict[str, Any]:
    return {"audio": audio_diagnostics(audio_samples, sample_rate), "video": video_diagnostics(video_frames)}
