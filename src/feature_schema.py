"""
DeceptionX 250-Dimensional Feature Digest Schema & Metadata
Accredited to Gemini.pdf Specifications (Pages 12-20, 26-29, 43).

Domain breakdown:
- z[0:40]    : FACS Action Units (20 AUs x [Mean, Max Peak])
- z[40:56]   : Active Stripped Emotions & Continuous Affect (16 dims)
- z[56:76]   : Ocular, Gaze & Head Pose Kinematics (20 dims)
- z[76:90]   : Autonomic rPPG & Facial Micro-Tremor (14 dims)
- z[90:178]  : Suprasegmental Acoustic Functionals (88 eGeMAPSv02 dims)
- z[178:192] : Vocal Fold Stress & Silero VAD Latency (14 dims)
- z[192:217] : Psycholinguistic, Disfluency & Reality Monitoring (RM) (25 dims)
- z[217:250] : Speech Intent & Dialogue Act Probabilities (33 dims)
Total: 250 features
"""

from typing import Dict, List, Tuple

FACS_AUS = [
    ("AU01", "Inner Brow Raiser"),
    ("AU02", "Outer Brow Raiser"),
    ("AU04", "Brow Lowerer"),
    ("AU05", "Upper Lid Raiser"),
    ("AU06", "Cheek Raiser"),
    ("AU07", "Lid Tightener"),
    ("AU09", "Nose Wrinkler"),
    ("AU10", "Upper Lip Raiser"),
    ("AU11", "Nasolabial Deepener"),
    ("AU12", "Lip Corner Puller"),
    ("AU14", "Dimpler"),
    ("AU15", "Lip Corner Depressor"),
    ("AU17", "Chin Raiser"),
    ("AU20", "Lip Stretcher"),
    ("AU23", "Lip Tightener"),
    ("AU24", "Lip Pressor"),
    ("AU25", "Lips Part"),
    ("AU26", "Jaw Drop"),
    ("AU28", "Lip Suck"),
    ("AU43", "Eyes Closed"),
]

ACTIVE_EMOTIONS = ["Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

OCULAR_GAZE_HEAD_FEATURES = [
    "EAR_mean", "EAR_std", "EAR_max_velocity",
    "Blink_Rate_bpm", "Blink_Duration_ms",
    "MAR_mean", "LipPress_Compression_Ratio",
    "Gaze_Yaw_mean", "Gaze_Yaw_dispersion",
    "Gaze_Pitch_mean", "Gaze_Pitch_dispersion",
    "Gaze_Aversion_Ratio_gt10deg",
    "HeadPose_Pitch_velocity", "HeadPose_Yaw_velocity", "HeadPose_Roll_velocity",
    "HeadPose_Tx_velocity", "HeadPose_Ty_velocity", "HeadPose_Tz_velocity",
    "HeadPose_Rotational_Rigidity", "HeadPose_Translational_Rigidity"
]

AUTONOMIC_RPPG_FEATURES = [
    "Forehead_Cheek_POS_BVP_Variance",
    "Forehead_Cheek_CHROM_BVP_Variance",
    "POS_CHROM_CrossCorrelation",
    "Welch_PSD_HeartRate_DeltaBPM",
    "IBI_SDNN",
    "IBI_RMSSD",
    "IBI_pNN50",
    "LF_HF_Sympathetic_Ratio",
    "BVP_Systolic_Peak_Drop",
    "Facial_MicroTremor_8_12Hz_Patch1",
    "Facial_MicroTremor_8_12Hz_Patch2",
    "Facial_MicroTremor_8_12Hz_Patch3",
    "Facial_MicroTremor_8_12Hz_Patch4",
    "Facial_MicroTremor_8_12Hz_Patch5"
]

VOCAL_FOLD_STRESS_FEATURES = [
    "Praat_Local_Jitter",
    "Praat_Local_Absolute_Jitter",
    "Praat_RAP",
    "Praat_PPQ5",
    "Praat_Local_Shimmer",
    "Praat_APQ3",
    "Praat_APQ5",
    "Praat_APQ11",
    "Praat_Mean_Autocorrelation",
    "Praat_HNR",
    "CPP_Mean",
    "CPP_Std",
    "Spectral_Shannon_Entropy",
    "Silero_VAD_Response_Latency_ms"
]

PSYCHOLINGUISTIC_RM_FEATURES = [
    "Disfluency_Rate_per_100w",
    "Word_Repetition_Rate",
    "False_Start_Ratio",
    "Silent_Pause_Freq_gt250ms",
    "Mean_Pause_Duration_sec",
    "MATTR_25",
    "Word_Count",
    "Speaking_Rate_wps",
    "First_Person_Singular_Pronoun_Rate",
    "Third_Person_Pronoun_Rate",
    "Negation_Density",
    "Cognitive_Hedging_Density",
    "RM_External_Sensory_Ratio",
    "RM_Spatial_Detail_Density",
    "RM_Temporal_Sequence_Density",
    "RM_Cognitive_Operations_Ratio",
    "RM_External_to_Internal_Net_Score",
    "Pronoun_Distance_Score",
    "Verbal_Uncertainty_Index",
    "Adverbial_Intensifier_Rate",
    "Sensory_Visual_Density",
    "Sensory_Auditory_Density",
    "Affective_Word_Density",
    "Cognitive_Load_Lexical_Index",
    "Speech_Fluidity_Ratio"
]

SPEECH_INTENT_CLASSES = [
    "Narrative_Reconstruction", "Appeasement", "Deflection", "Alibi_Elaboration",
    "Self_Correction", "Minimization", "Denial", "Counter_Accusation",
    "Distancing", "Hedging_Qualification", "Emphatic_Assertion", "Plea_for_Belief",
    "Clarification_Request", "Topic_Shift", "Stalling_Repetition", "Over_Explanation",
    "Ambiguity_Injection", "Admission_of_Triviality", "Aggressive_Rebuttal", "Passive_Agreement",
    "Sensory_Exaggeration", "Evasion", "Blame_Externalization", "Affective_Appeal",
    "Memory_Disavowal", "Hypothetical_Framing", "Contextual_Reframing", "Concession",
    "Preemptive_Defense", "Vagueness", "Exculpatory_Narrative", "Discredit_Source",
    "Neutral_Statement"
]


def generate_feature_names() -> List[str]:
    """Generates the ordered list of 250 feature names."""
    names: List[str] = []
    
    # 0..39: 20 FACS Action Units (Mean, Max)
    for code, desc in FACS_AUS:
        names.append(f"{code}_{desc.replace(' ', '')}_Mean_z")
        names.append(f"{code}_{desc.replace(' ', '')}_Max_z")
    assert len(names) == 40
    
    # 40..55: Active Stripped Emotions & Continuous Affect (16 dims)
    for emo in ACTIVE_EMOTIONS:
        names.append(f"Emotion_{emo}_Mean_z")
        names.append(f"Emotion_{emo}_Max_z")
    names.append("Affect_Valence_Mean_z")
    names.append("Affect_Valence_Std_z")
    names.append("Affect_Arousal_Mean_z")
    names.append("Affect_Arousal_Std_z")
    assert len(names) == 56
    
    # 56..75: Ocular, Gaze & Head Pose Kinematics (20 dims)
    for feat in OCULAR_GAZE_HEAD_FEATURES:
        names.append(f"Kinematics_{feat}_z")
    assert len(names) == 76
    
    # 76..89: Autonomic rPPG & Facial Micro-Tremor (14 dims)
    for feat in AUTONOMIC_RPPG_FEATURES:
        names.append(f"Autonomic_{feat}_z")
    assert len(names) == 90
    
    # 90..177: Suprasegmental Acoustic Functionals (88 dims eGeMAPSv02)
    for i in range(88):
        names.append(f"eGeMAPS_Acoustic_Functional_{i:02d}_z")
    assert len(names) == 178
    
    # 178..191: Vocal Fold Stress & Silero VAD Latency (14 dims)
    for feat in VOCAL_FOLD_STRESS_FEATURES:
        names.append(f"VocalStress_{feat}_z")
    assert len(names) == 192
    
    # 192..216: Psycholinguistic, Disfluency & Reality Monitoring (25 dims)
    for feat in PSYCHOLINGUISTIC_RM_FEATURES:
        names.append(f"LinguisticRM_{feat}_z")
    assert len(names) == 217
    
    # 217..249: Speech Intent & Dialogue Act Probabilities (33 dims)
    for intent in SPEECH_INTENT_CLASSES:
        names.append(f"SpeechIntent_{intent}_Prob_z")
    assert len(names) == 250
    
    return names

FEATURE_NAMES = generate_feature_names()

DOMAIN_SLICES: Dict[str, Tuple[int, int]] = {
    "FACS_Action_Units": (0, 40),
    "Stripped_Emotions_Affect": (40, 56),
    "Ocular_Gaze_HeadPose": (56, 76),
    "Autonomic_rPPG_Tremor": (76, 90),
    "Suprasegmental_Acoustic": (90, 178),
    "VocalFoldStress_VAD": (178, 192),
    "Psycholinguistic_RealityMonitoring": (192, 217),
    "Speech_Intent_DialogueActs": (217, 250)
}

def get_feature_domain(index: int) -> str:
    """Returns the domain category for a feature index in 0..249."""
    for domain, (start, end) in DOMAIN_SLICES.items():
        if start <= index < end:
            return domain
    return "Unknown"
