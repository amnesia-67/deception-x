"""Test Branch 2 Tabular LOSO XGBoost with GPU acceleration."""
import pytest
import numpy as np
import torch
from src.tabular_branch import TabularLOSOXGBoost

def test_xgboost_gpu_fit_predict():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    np.random.seed(42)
    N = 100
    X = np.random.randn(N, 250).astype(np.float32)
    y = np.random.randint(0, 2, N)
    
    model = TabularLOSOXGBoost(n_estimators=50, device="cuda")
    model.fit(X, y)
    
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    assert len(probs) == N
    assert len(preds) == N
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert set(np.unique(preds)).issubset({0, 1})

def test_loso_evaluation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    np.random.seed(42)
    N = 72
    n_speakers = 6
    X = np.random.randn(N, 250).astype(np.float32)
    y = np.random.randint(0, 2, N)
    speakers = np.array([i % n_speakers for i in range(N)])
    
    results = TabularLOSOXGBoost.evaluate_loso(X, y, speakers, device="cuda")
    
    assert "oof_accuracy" in results
    assert "oof_auroc" in results
    assert 0.0 <= results["oof_accuracy"] <= 1.0
    assert 0.0 <= results["oof_auroc"] <= 1.0
    assert len(results["oof_preds"]) == N
