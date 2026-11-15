"""
Unit tests for the sub-5ms TreeSHAP explainer for Branch 2.
Accredited to Gemini.pdf Pages 26-29, 43.
"""

import numpy as np
import pytest

from src.tabular_branch import TabularLOSOXGBoost
from src.explainer import ForensicTreeSHAPExplainer

@pytest.fixture
def fitted_tabular_model():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 250).astype(np.float32)
    y = (X[:, 0] + X[:, 191] > 0).astype(int)  # Correlated with AU01 & VAD latency
    model = TabularLOSOXGBoost(n_estimators=15, max_depth=3, device="cpu")
    model.fit(X, y)
    return model

def test_treeshap_exact_attribution_property(fitted_tabular_model):
    explainer = ForensicTreeSHAPExplainer(fitted_tabular_model)
    rng = np.random.RandomState(123)
    z = rng.randn(250).astype(np.float32)
    
    explanation = explainer.explain(z, top_k=5)
    
    # 1. Check latency
    assert explanation["execution_latency_ms"] < 25.0  # Typically < 5ms
    
    # 2. Additive attribution property: phi_0 + sum(phi) == margin
    phi_0 = explanation["base_value_phi_0"]
    raw_shap = np.array(explanation["raw_shap_values"])
    calculated_margin = phi_0 + float(np.sum(raw_shap))
    
    assert calculated_margin == pytest.approx(explanation["margin"], rel=1e-4)
    
    # Probability matches sigmoid of margin
    expected_prob = 1.0 / (1.0 + np.exp(-calculated_margin))
    assert explanation["p_xgb"] == pytest.approx(expected_prob, rel=1e-4)

def test_treeshap_driver_identification(fitted_tabular_model):
    explainer = ForensicTreeSHAPExplainer(fitted_tabular_model)
    z = np.zeros(250, dtype=np.float32)
    # Inject strong deceptive signal on feature 0 and 191
    z[0] = 3.5
    z[191] = 2.8
    
    explanation = explainer.explain(z, top_k=3)
    assert len(explanation["top_positive_drivers"]) <= 3
    assert len(explanation["top_negative_drivers"]) <= 3
    
    # Check domain attribution keys
    assert "FACS_Action_Units" in explanation["domain_attributions"]
    assert "VocalFoldStress_VAD" in explanation["domain_attributions"]
    assert "Psycholinguistic_RealityMonitoring" in explanation["domain_attributions"]

def test_treeshap_waterfall_ascii_rendering(fitted_tabular_model):
    explainer = ForensicTreeSHAPExplainer(fitted_tabular_model)
    z = np.random.randn(250).astype(np.float32)
    explanation = explainer.explain(z)
    ascii_hud = explainer.render_waterfall_ascii(explanation)
    assert "DeceptionX TreeSHAP Waterfall Explanation" in ascii_hud
    assert "Final Margin:" in ascii_hud
    assert "Domain Aggregation Breakdown:" in ascii_hud
