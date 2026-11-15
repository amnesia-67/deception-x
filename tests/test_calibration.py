"""
Unit tests for the two-stage dynamic baseline calibration engine.
Accredited to Gemini.pdf Pages 8-15.
"""

import numpy as np
import pytest

from src.calibration import BaselineCalibrator

def test_calibrator_synthetic_initialization():
    calibrator = BaselineCalibrator()
    assert not calibrator.calibrated
    calibrator.calibrate_synthetic_default()
    assert calibrator.calibrated
    assert calibrator.window_a_completed
    assert calibrator.window_b_completed
    assert calibrator.mu_base.shape == (250,)
    assert calibrator.sigma_base.shape == (250,)
    assert np.all(calibrator.epsilon_min > 0)

def test_calibrator_two_stage_feeding():
    calibrator = BaselineCalibrator()
    rng = np.random.RandomState(42)
    
    # Window A (silent visual/autonomic): 150 frames @ 30fps
    window_a = rng.randn(150, 250).astype(np.float32) + 2.0
    calibrator.feed_window_a(window_a)
    assert calibrator.window_a_completed
    assert not calibrator.calibrated
    # Check that visual/autonomic features (0..90) were set
    assert np.allclose(calibrator.mu_base[:90], np.mean(window_a[:, :90], axis=0))
    
    # Window B (verifiable trivial truth): 50 samples
    window_b = rng.randn(50, 250).astype(np.float32) + 5.0
    calibrator.feed_window_b(window_b)
    assert calibrator.window_b_completed
    assert calibrator.calibrated
    # Check that acoustic/linguistic features (90..250) were set
    assert np.allclose(calibrator.mu_base[90:], np.mean(window_b[:, 90:], axis=0))

def test_z_score_normalization_and_clipping():
    calibrator = BaselineCalibrator()
    calibrator.calibrate_synthetic_default()
    
    # Create extreme feature value to test [-10, +10] clipping
    raw = calibrator.mu_base.copy()
    raw[0] += 5000.0 * calibrator.sigma_base[0]  # extreme high
    raw[1] -= 5000.0 * calibrator.sigma_base[1]  # extreme low
    raw[2] = calibrator.mu_base[2]               # exact mean
    
    z = calibrator.normalize(raw)
    assert z.shape == (250,)
    assert z[0] == pytest.approx(10.0)
    assert z[1] == pytest.approx(-10.0)
    assert z[2] == pytest.approx(0.0)
    assert np.all(z >= -10.0) and np.all(z <= 10.0)

def test_epsilon_min_prevents_zero_division():
    calibrator = BaselineCalibrator()
    # Force zero standard deviation
    calibrator.mu_base = np.zeros(250, dtype=np.float32)
    calibrator.sigma_base = np.zeros(250, dtype=np.float32)
    
    raw = np.ones(250, dtype=np.float32)
    z = calibrator.normalize(raw)
    # Check no NaN or Inf
    assert not np.any(np.isnan(z))
    assert not np.any(np.isinf(z))

def test_active_emotion_amplification():
    # 7-class distribution: [Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger]
    # Neutral dominant: 0.85, subtle Fear: 0.08, subtle Disgust: 0.07
    p_emo = np.array([0.85, 0.0, 0.0, 0.0, 0.08, 0.07, 0.0], dtype=np.float32)
    
    amplified = BaselineCalibrator.amplify_active_emotions(p_emo, gamma=0.75, tau_act=0.05)
    assert amplified.shape == (6,)  # 6 active emotions
    # Fear and Disgust were active (>0.05)
    assert amplified[3] > 0  # Fear index in active (index 4 in raw -> 3 in active)
    assert amplified[4] > 0  # Disgust index in active (index 5 in raw -> 4 in active)
    # Happy, Sad, Surprise, Anger should be 0
    assert amplified[0] == 0.0
    assert amplified[1] == 0.0
    # Sum should normalize to 1.0
    assert np.sum(amplified) == pytest.approx(1.0, rel=1e-3)
