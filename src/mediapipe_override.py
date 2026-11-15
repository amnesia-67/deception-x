"""
MediaPipe Geometric Overrides & Ocular Kinematics
Accredited to Gemini.pdf Specifications (Pages 4, 15, 38).

Bypasses Detectorv2 face_multitask_v28 F1 limitations for AU20 and AU15
by computing exact 3D Euclidean geometric ratios from MediaPipe 478-point mesh:
- AU20 (Lip Stretcher): Horizontal mouth corner distance (Landmarks 61, 291)
  normalized by outer-canthus inter-ocular distance (Landmarks 33, 263).
- AU15 (Lip Corner Depressor): Vertical displacement of mouth corners
  relative to upper lip center (Landmark 0) and philtrum (Landmark 164).
- Ocular Gaze Pitch: Derived from 3D iris centers (Landmarks 468, 473)
  relative to upper/lower palpebral fissure anchors (159/145 and 386/374),
  canonically negated to +up to match py-feat >= 2.1.1 standard.
"""

from typing import Dict, Any, Union, Optional
import numpy as np

# MediaPipe 478 Landmark Indices
LM_MOUTH_CORNER_LEFT = 291
LM_MOUTH_CORNER_RIGHT = 61
LM_LIP_UPPER_CENTER = 0
LM_LIP_LOWER_CENTER = 17
LM_PHILTRUM = 164
LM_LEFT_EYE_OUTER = 263
LM_RIGHT_EYE_OUTER = 33
LM_LEFT_EYE_UPPER = 386
LM_LEFT_EYE_LOWER = 374
LM_RIGHT_EYE_UPPER = 159
LM_RIGHT_EYE_LOWER = 145
LM_LEFT_IRIS_CENTER = 473
LM_RIGHT_IRIS_CENTER = 468


def compute_mediapipe_geometric_overrides(landmarks_3d: np.ndarray) -> Dict[str, float]:
    """
    Computes AU20, AU15, and Gaze Pitch from 3D facial landmarks [478, 3] or [N >= 478, 3].
    
    Args:
        landmarks_3d: np.ndarray of shape (478, 3) with (x, y, z) coordinates.
        
    Returns:
        Dictionary containing:
        - "AU20_override": normalized lip stretch intensity [0, 1]
        - "AU15_override": normalized lip corner depression intensity [0, 1]
        - "Gaze_Pitch_override": canonical vertical gaze pitch in degrees (+up)
    """
    if landmarks_3d.shape[0] < 474:
        raise ValueError(f"Expected at least 474 landmarks for 3D iris computation, got {landmarks_3d.shape[0]}")
    
    # 1. Outer Canthus Inter-Ocular Reference Distance (Scale Invariant)
    iod = float(np.linalg.norm(landmarks_3d[LM_LEFT_EYE_OUTER] - landmarks_3d[LM_RIGHT_EYE_OUTER]))
    iod_safe = max(iod, 1e-6)
    
    # 2. AU20 (Lip Stretcher): Horizontal mouth corner distance normalized by IOD
    mouth_width = float(np.linalg.norm(landmarks_3d[LM_MOUTH_CORNER_LEFT] - landmarks_3d[LM_MOUTH_CORNER_RIGHT]))
    mouth_stretch_ratio = mouth_width / iod_safe
    # Neutral mouth stretch ratio is ~0.65 - 0.70; stretch can reach ~0.95+
    au20_val = float(np.clip((mouth_stretch_ratio - 0.65) / 0.30, 0.0, 1.0))
    
    # 3. AU15 (Lip Corner Depressor): Vertical drop of corners relative to upper lip center
    upper_lip_y = landmarks_3d[LM_LIP_UPPER_CENTER, 1]
    corner_y_avg = (landmarks_3d[LM_MOUTH_CORNER_LEFT, 1] + landmarks_3d[LM_MOUTH_CORNER_RIGHT, 1]) / 2.0
    # In screen coordinates, positive y is downward
    corner_drop = (corner_y_avg - upper_lip_y) / iod_safe
    # Baseline corner drop is ~0.10 - 0.15; depressed corners reach 0.25+
    au15_val = float(np.clip((corner_drop - 0.12) / 0.15, 0.0, 1.0))
    
    # 4. Gaze Pitch: Iris center vertical displacement relative to palpebral fissures
    # Right eye
    r_upper_y = landmarks_3d[LM_RIGHT_EYE_UPPER, 1]
    r_lower_y = landmarks_3d[LM_RIGHT_EYE_LOWER, 1]
    r_iris_y = landmarks_3d[LM_RIGHT_IRIS_CENTER, 1]
    r_fissure_height = max(abs(r_lower_y - r_upper_y), 1e-6)
    r_norm_y = (r_iris_y - r_upper_y) / r_fissure_height  # 0.5 is centered
    
    # Left eye
    l_upper_y = landmarks_3d[LM_LEFT_EYE_UPPER, 1]
    l_lower_y = landmarks_3d[LM_LEFT_EYE_LOWER, 1]
    l_iris_y = landmarks_3d[LM_LEFT_IRIS_CENTER, 1]
    l_fissure_height = max(abs(l_lower_y - l_upper_y), 1e-6)
    l_norm_y = (l_iris_y - l_upper_y) / l_fissure_height
    
    avg_norm_y = (r_norm_y + l_norm_y) / 2.0
    # Canonical negation: in screen coords down is positive y, so looking up is smaller y
    # Negate so that upward gaze is positive pitch (+up) matching py-feat >= 2.1.1
    gaze_pitch_deg = float(-(avg_norm_y - 0.5) * 60.0)
    
    return {
        "AU20_override": au20_val,
        "AU15_override": au15_val,
        "Gaze_Pitch_override": gaze_pitch_deg,
    }


def apply_geometric_overrides_to_z(
    z_vector: np.ndarray,
    overrides: Dict[str, float]
) -> np.ndarray:
    """
    Applies the MediaPipe geometric overrides into the 250-dimensional z-vector.
    
    z indices mapping:
    - AU15: z[11] (AU15_mean) and z[31] (AU15_max)
    - AU20: z[13] (AU20_mean) and z[33] (AU20_max)
    - Gaze_Pitch: z[65] (Gaze_Pitch_mean)
    """
    z_corrected = z_vector.copy()
    
    # AU15 is index 11 in FACS_AUS list
    if "AU15_override" in overrides:
        z_corrected[11] = overrides["AU15_override"]
        z_corrected[31] = overrides["AU15_override"] * 1.2
        
    # AU20 is index 13 in FACS_AUS list
    if "AU20_override" in overrides:
        z_corrected[13] = overrides["AU20_override"]
        z_corrected[33] = overrides["AU20_override"] * 1.2
        
    # Gaze Pitch mean is index 65 (56 + 9)
    if "Gaze_Pitch_override" in overrides:
        z_corrected[65] = overrides["Gaze_Pitch_override"]
        
    return z_corrected


def generate_synthetic_mediapipe_landmarks(
    lip_stretch: float = 0.70,
    lip_drop: float = 0.12,
    gaze_up: float = 0.0
) -> np.ndarray:
    """
    Generates a synthetic 478-point 3D landmark mesh for testing and validation.
    """
    pts = np.zeros((478, 3), dtype=np.float32)
    
    # Eye reference points
    pts[LM_LEFT_EYE_OUTER] = [0.15, 0.20, 0.0]
    pts[LM_RIGHT_EYE_OUTER] = [-0.15, 0.20, 0.0]
    iod = 0.30
    
    # Fissures
    pts[LM_RIGHT_EYE_UPPER] = [-0.10, 0.18, 0.0]
    pts[LM_RIGHT_EYE_LOWER] = [-0.10, 0.22, 0.0]
    pts[LM_LEFT_EYE_UPPER] = [0.10, 0.18, 0.0]
    pts[LM_LEFT_EYE_LOWER] = [0.10, 0.22, 0.0]
    
    # Iris centers (gaze_up > 0 shifts iris up -> smaller y)
    iris_y_shift = -gaze_up * 0.02
    pts[LM_RIGHT_IRIS_CENTER] = [-0.10, 0.20 + iris_y_shift, 0.02]
    pts[LM_LEFT_IRIS_CENTER] = [0.10, 0.20 + iris_y_shift, 0.02]
    
    # Upper lip center & Philtrum
    pts[LM_PHILTRUM] = [0.0, 0.32, 0.05]
    pts[LM_LIP_UPPER_CENTER] = [0.0, 0.35, 0.05]
    pts[LM_LIP_LOWER_CENTER] = [0.0, 0.38, 0.05]
    
    # Mouth corners
    half_width = (iod * lip_stretch) / 2.0
    corner_y = 0.35 + (iod * lip_drop)
    pts[LM_MOUTH_CORNER_RIGHT] = [-half_width, corner_y, 0.03]
    pts[LM_MOUTH_CORNER_LEFT] = [half_width, corner_y, 0.03]
    
    return pts
