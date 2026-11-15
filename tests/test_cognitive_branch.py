"""Unit tests for Branch 3 Cognitive Reasoning Harness & Schema."""

import pytest
import asyncio
from src.cognitive_branch import DeceptionXVerdictSchema, Branch3CognitiveHarness

def test_verdict_schema_validation():
    valid_data = {
        "observe_trace": [
            "T=0.45s: Forehead rPPG pulse amplitude dropped 42%",
            "T=1.20s: AU12 smile without AU06 orbicularis oculi contraction"
        ],
        "think_trace": "Subject shows high autonomic response combined with lack of perceptual details in narrative.",
        "classification": "Deceptive",
        "p_argon": 0.88,
        "forensic_summary": "Pronounced autonomic divergence and micro-expression masking indicate deception."
    }
    schema = DeceptionXVerdictSchema.model_validate(valid_data)
    assert schema.classification == "Deceptive"
    assert schema.p_argon == 0.88
    assert len(schema.observe_trace) == 2

def test_verdict_schema_invalid_range():
    invalid_data = {
        "observe_trace": ["AU12 detected"],
        "think_trace": "Analysis...",
        "classification": "Truthful",
        "p_argon": 1.25, # Invalid: must be <= 0.99
        "forensic_summary": "Summary..."
    }
    with pytest.raises(Exception):
        DeceptionXVerdictSchema.model_validate(invalid_data)

def test_tier3_emergency_fallback():
    harness = Branch3CognitiveHarness(api_key=None)
    
    # Truthful case
    result_truth = harness._tier3_emergency_fallback(p_neural=0.20, p_xgb=0.10)
    assert result_truth["tier_used"] == "LOCAL_EMERGENCY"
    assert result_truth["classification"] == "Truthful"
    expected_p = 0.58 * 0.20 + 0.42 * 0.10
    assert abs(result_truth["p_argon"] - expected_p) < 1e-4

    # Deceptive case
    result_dec = harness._tier3_emergency_fallback(p_neural=0.85, p_xgb=0.90)
    assert result_dec["classification"] == "Deceptive"
    assert result_dec["p_argon"] > 0.80

def test_evaluate_utterance_offline():
    harness = Branch3CognitiveHarness(api_key="")
    result = asyncio.run(harness.evaluate_utterance(
        payload_parts=["Sample utterance"],
        p_neural=0.75,
        p_xgb=0.65
    ))
    assert result["tier_used"] == "LOCAL_EMERGENCY"
    assert "p_argon" in result
    assert result["classification"] in ["Deceptive", "Truthful"]
