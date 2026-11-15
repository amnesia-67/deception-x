"""
Tests for MediaPipe 3D Geometric Overrides & Iris Gaze Pitch
Accredited to Gemini.pdf Specifications (Pages 4, 15, 38).
"""

import numpy as np
import pytest

from src.mediapipe_override import (
    compute_mediapipe_geometric_overrides,
    apply_geometric_overrides_to_z,
    generate_synthetic_mediapipe_landmarks,
)


def test_mediapipe_neutral_landmarks():
    # Neutral mesh with nominal ratios
    pts = generate_synthetic_mediapipe_landmarks(lip_stretch=0.70, lip_drop=0.12, gaze_up=0.0)
    overrides = compute_mediapipe_geometric_overrides(pts)
    
    assert "AU20_override" in overrides
    assert "AU15_override" in overrides
    assert "Gaze_Pitch_override" in overrides
    
    # Neutral AU20 should be near baseline (~0.16)
    assert 0.0 <= overrides["AU20_override"] <= 0.35
    # Neutral AU15 should be near baseline (0.0)
    assert overrides["AU15_override"] <= 0.15
    # Neutral gaze pitch should be near 0 degrees
    assert abs(overrides["Gaze_Pitch_override"]) < 5.0


def test_mediapipe_stretched_lips_au20():
    # Stretched mouth corners (AU20 lip stretcher activation)
    pts = generate_synthetic_mediapipe_landmarks(lip_stretch=0.92, lip_drop=0.12, gaze_up=0.0)
    overrides = compute_mediapipe_geometric_overrides(pts)
    
    # AU20 intensity should be elevated (> 0.8)
    assert overrides["AU20_override"] > 0.80


def test_mediapipe_depressed_corners_au15():
    # Depressed mouth corners (AU15 corner depressor activation)
    pts = generate_synthetic_mediapipe_landmarks(lip_stretch=0.70, lip_drop=0.25, gaze_up=0.0)
    overrides = compute_mediapipe_geometric_overrides(pts)
    
    # AU15 intensity should be elevated (> 0.7)
    assert overrides["AU15_override"] > 0.70


def test_mediapipe_upward_gaze_pitch():
    # Looking upward (+up canonical pitch)
    pts = generate_synthetic_mediapipe_landmarks(lip_stretch=0.70, lip_drop=0.12, gaze_up=1.5)
    overrides = compute_mediapipe_geometric_overrides(pts)
    
    # Gaze pitch should be positive (+up)
    assert overrides["Gaze_Pitch_override"] > 10.0


def test_apply_geometric_overrides_to_z():
    z_orig = np.zeros(250, dtype=np.float32)
    overrides = {
        "AU20_override": 0.88,
        "AU15_override": 0.74,
        "Gaze_Pitch_override": 18.5,
    }
    z_corr = apply_geometric_overrides_to_z(z_orig, overrides)
    
    # Check AU15 indices (11, 31)
    assert np.isclose(z_corr[11], 0.74)
    assert np.isclose(z_corr[31], 0.74 * 1.2)
    
    # Check AU20 indices (13, 33)
    assert np.isclose(z_corr[13], 0.88)
    assert np.isclose(z_corr[33], 0.88 * 1.2)
    
    # Check Gaze Pitch index (65)
    assert np.isclose(z_corr[65], 18.5)
