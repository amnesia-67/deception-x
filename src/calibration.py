"""
Stage 0: 15-Second Two-Stage Dynamic Baseline Calibration Engine
Accredited to Gemini.pdf Specifications (Pages 8-15).

Eliminates inter-subject variance, demographic markers, and anatomical confounders
via within-subject Z-score deviation normalization:
    z_k = clip((x_k - mu_base,k) / max(sigma_base,k, epsilon_min,k), -10.0, +10.0)

Includes:
1. Window A (0-5s): Silent Visual & Autonomic Fixation
2. Window B (5-15s): Verifiable Trivial Truth Elicitation
3. Neutral-Stripped Active Emotion Non-Linear Amplification (gamma=0.75, tau_act=0.05)
4. Modality-specific noise floors epsilon_min preventing zero-variance divergence
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from src.feature_schema import FEATURE_NAMES, DOMAIN_SLICES


class BaselineCalibrator:
    """
    Two-Stage Dynamic Baseline Calibration Engine for within-subject Z-scoring.
    """
    def __init__(self, feature_dim: int = 250):
        self.feature_dim = feature_dim
        self.mu_base = np.zeros(feature_dim, dtype=np.float32)
        self.sigma_base = np.ones(feature_dim, dtype=np.float32)
        self.epsilon_min = self._build_epsilon_min()
        
        self.window_a_completed = False
        self.window_b_completed = False
        self.calibrated = False

    def _build_epsilon_min(self) -> np.ndarray:
        """
        Builds modality-specific noise floor epsilon_min vector:
        - AU probabilities: 0.015
        - Continuous affect: 0.020
        - Kinematics (EAR/MAR): 0.010
        - Autonomic rPPG variance: 0.010
        - F0 frequency: 1.50 Hz
        - VAD latency: 25.0 ms
        - Linguistic rates: 0.050
        - Default fallback: 1e-3
        """
        eps = np.full(self.feature_dim, 1e-3, dtype=np.float32)
        
        # 0..40 FACS AUs
        s_au, e_au = DOMAIN_SLICES["FACS_Action_Units"]
        eps[s_au:e_au] = 0.015
        
        # 40..56 Emotions & Affect
        s_em, e_em = DOMAIN_SLICES["Stripped_Emotions_Affect"]
        eps[s_em:e_em] = 0.020
        
        # 56..76 Ocular & Kinematics
        s_k, e_k = DOMAIN_SLICES["Ocular_Gaze_HeadPose"]
        eps[s_k:e_k] = 0.010
        
        # 76..90 Autonomic rPPG
        s_r, e_r = DOMAIN_SLICES["Autonomic_rPPG_Tremor"]
        eps[s_r:e_r] = 0.010
        
        # 90..178 Suprasegmental Acoustic (F0 semitone index)
        s_ac, e_ac = DOMAIN_SLICES["Suprasegmental_Acoustic"]
        eps[s_ac:e_ac] = 1.50
        
        # 178..192 Vocal stress & VAD latency
        s_vs, e_vs = DOMAIN_SLICES["VocalFoldStress_VAD"]
        eps[s_vs:e_vs] = 0.020
        # VAD latency is the last feature in this slice
        eps[e_vs - 1] = 25.0  # 25 ms
        
        # 192..217 Linguistic & RM
        s_lg, e_lg = DOMAIN_SLICES["Psycholinguistic_RealityMonitoring"]
        eps[s_lg:e_lg] = 0.050
        
        # 217..250 Speech Intents
        s_in, e_in = DOMAIN_SLICES["Speech_Intent_DialogueActs"]
        eps[s_in:e_in] = 0.010
        
        return eps

    def feed_window_a(self, visual_autonomic_samples: np.ndarray):
        """
        Processes Window A (t in [0, 5] s): Silent Visual & Autonomic Fixation.
        visual_autonomic_samples: (N_frames, feature_dim) or (N_frames, visual_autonomic_dims)
        """
        assert visual_autonomic_samples.ndim == 2, "Expected 2D array of samples (N, D)"
        n_samples, d = visual_autonomic_samples.shape
        
        # Calculate mean and std for visual/autonomic features (0..90)
        max_idx = min(d, 90)
        self.mu_base[:max_idx] = np.mean(visual_autonomic_samples[:, :max_idx], axis=0)
        self.sigma_base[:max_idx] = np.std(visual_autonomic_samples[:, :max_idx], axis=0)
        self.window_a_completed = True

    def feed_window_b(self, acoustic_linguistic_samples: np.ndarray):
        """
        Processes Window B (t in (5, 15] s): Verifiable Trivial Truth Elicitation.
        acoustic_linguistic_samples: (N_samples, feature_dim)
        """
        assert acoustic_linguistic_samples.ndim == 2, "Expected 2D array of samples (N, D)"
        n_samples, d = acoustic_linguistic_samples.shape
        
        # Calculate mean and std for acoustic/linguistic/intent features (90..250)
        min_idx = 90
        max_idx = min(d, self.feature_dim)
        if max_idx > min_idx:
            self.mu_base[min_idx:max_idx] = np.mean(acoustic_linguistic_samples[:, min_idx:max_idx], axis=0)
            self.sigma_base[min_idx:max_idx] = np.std(acoustic_linguistic_samples[:, min_idx:max_idx], axis=0)
            
        self.window_b_completed = True
        self.calibrated = True

    def calibrate_synthetic_default(self, seed: int = 42):
        """
        Initializes an empirical resting baseline calibrated for offline simulation / testing.
        """
        rng = np.random.RandomState(seed)
        self.mu_base = np.zeros(self.feature_dim, dtype=np.float32)
        # Populate realistic resting baselines
        # AUs have low baseline probability ~ 0.05
        self.mu_base[0:40] = 0.05
        # Emotions baseline: mostly low active emotions
        self.mu_base[40:56] = 0.08
        # Kinematics: EAR resting ~ 0.30, MAR ~ 0.15, Gaze near 0
        self.mu_base[56] = 0.30  # EAR mean
        self.mu_base[61] = 0.15  # MAR mean
        # Autonomic: Heart rate delta baseline 0, pNN50 ~ 20%
        self.mu_base[82] = 20.0  # pNN50
        # Acoustic F0 mean ~ 130 Hz semitone baseline
        self.mu_base[90] = 130.0
        # VAD latency baseline ~ 350 ms
        self.mu_base[191] = 350.0
        
        # Standard deviations
        self.sigma_base = np.ones(self.feature_dim, dtype=np.float32)
        self.sigma_base[0:40] = 0.08
        self.sigma_base[40:56] = 0.10
        self.sigma_base[56:76] = 0.05
        self.sigma_base[76:90] = 2.0
        self.sigma_base[90:178] = 5.0
        self.sigma_base[178:192] = 15.0
        self.sigma_base[191] = 85.0  # VAD latency std
        self.sigma_base[192:250] = 0.25
        
        self.window_a_completed = True
        self.window_b_completed = True
        self.calibrated = True

    def normalize(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Applies dynamic within-subject Z-score normalization:
            z_k = clip((x_k - mu_base,k) / max(sigma_base,k, epsilon_min,k), -10.0, +10.0)
        Accepts shape (250,) or (N, 250).
        """
        denom = np.maximum(self.sigma_base, self.epsilon_min)
        if raw_features.ndim == 1:
            z = (raw_features - self.mu_base) / denom
        else:
            z = (raw_features - self.mu_base[np.newaxis, :]) / denom[np.newaxis, :]
            
        return np.clip(z, -10.0, 10.0).astype(np.float32)

    @staticmethod
    def amplify_active_emotions(raw_7class_probs: np.ndarray, gamma: float = 0.75, tau_act: float = 0.05) -> np.ndarray:
        """
        Neutral-Stripped Active Emotion Non-Linear Amplification (Gemini.pdf Pages 13-15).
        
        raw_7class_probs: (7,) or (N, 7) ordered:
        [Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger]
        
        Zeros out p_Neutral, amplifies active emotions above tau_act=0.05 with power gamma=0.75,
        and re-normalizes over the active classes.
        """
        is_1d = raw_7class_probs.ndim == 1
        p = np.atleast_2d(raw_7class_probs).copy()
        
        # Zero out Neutral (index 0)
        p[:, 0] = 0.0
        active_p = p[:, 1:7]
        
        # Indicator mask: I(p_c > 0.05)
        mask = (active_p > tau_act).astype(np.float32)
        amplified = mask * np.power(np.maximum(active_p, 0.0), gamma)
        
        denom = np.sum(amplified, axis=1, keepdims=True) + 1e-6
        normalized_active = amplified / denom
        
        if is_1d:
            return normalized_active[0]
        return normalized_active
