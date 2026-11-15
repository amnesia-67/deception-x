"""
Unit tests for the 250-dimensional feature schema and metadata.
Accredited to Gemini.pdf Pages 12-20.
"""

from src.feature_schema import (
    FEATURE_NAMES,
    DOMAIN_SLICES,
    get_feature_domain,
    FACS_AUS,
    ACTIVE_EMOTIONS,
    SPEECH_INTENT_CLASSES
)

def test_feature_count():
    assert len(FEATURE_NAMES) == 250, f"Expected exactly 250 features, got {len(FEATURE_NAMES)}"

def test_domain_slices_coverage():
    # Verify continuous non-overlapping 0..250 coverage
    prev_end = 0
    total_coverage = 0
    for domain, (start, end) in DOMAIN_SLICES.items():
        assert start == prev_end, f"Domain {domain} starts at {start}, expected {prev_end}"
        assert end > start, f"Domain {domain} has non-positive range ({start}, {end})"
        total_coverage += (end - start)
        prev_end = end
    assert prev_end == 250, f"Last domain ends at {prev_end}, expected 250"
    assert total_coverage == 250

def test_facs_action_units():
    # 20 AUs x 2 (Mean, Max) = 40
    assert len(FACS_AUS) == 20
    s, e = DOMAIN_SLICES["FACS_Action_Units"]
    assert e - s == 40
    assert FEATURE_NAMES[0].startswith("AU01_")
    assert FEATURE_NAMES[39].startswith("AU43_")

def test_emotions_and_affect():
    # 6 emotions x 2 (Mean, Max) + 4 valence/arousal = 16
    assert len(ACTIVE_EMOTIONS) == 6
    s, e = DOMAIN_SLICES["Stripped_Emotions_Affect"]
    assert e - s == 16
    assert "Emotion_Happy_Mean_z" in FEATURE_NAMES[s:e]
    assert "Affect_Valence_Mean_z" in FEATURE_NAMES[s:e]

def test_suprasegmental_acoustics():
    # 88 eGeMAPSv02 features
    s, e = DOMAIN_SLICES["Suprasegmental_Acoustic"]
    assert e - s == 88

def test_speech_intents():
    # 33 Intent classes
    assert len(SPEECH_INTENT_CLASSES) == 33
    s, e = DOMAIN_SLICES["Speech_Intent_DialogueActs"]
    assert e - s == 33

def test_get_feature_domain():
    assert get_feature_domain(10) == "FACS_Action_Units"
    assert get_feature_domain(45) == "Stripped_Emotions_Affect"
    assert get_feature_domain(65) == "Ocular_Gaze_HeadPose"
    assert get_feature_domain(80) == "Autonomic_rPPG_Tremor"
    assert get_feature_domain(120) == "Suprasegmental_Acoustic"
    assert get_feature_domain(185) == "VocalFoldStress_VAD"
    assert get_feature_domain(200) == "Psycholinguistic_RealityMonitoring"
    assert get_feature_domain(230) == "Speech_Intent_DialogueActs"
    assert get_feature_domain(300) == "Unknown"
