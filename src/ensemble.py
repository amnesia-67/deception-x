"""
Meta-Classifier Stacking Ensemble & Platt Calibration
Specified in Gemini.pdf Pages 35-37.
Combines Branch 1 (Neural), Branch 2 (XGBoost), and Branch 3 (Cognitive Reasoning)
with out-of-fold Platt Scaling and ridge-regularized weights.
"""

import json
import numpy as np
from typing import Tuple, Dict, Optional, Any
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, roc_auc_score

def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))

class PlattScaler:
    """Fits 1D logistic regression P_cal = 1 / (1 + exp(A*s + B)) via NLL."""
    def __init__(self, A: float = 1.0, B: float = 0.0):
        self.A = float(A)
        self.B = float(B)

    def fit(self, s: np.ndarray, y: np.ndarray):
        # Initial guess A=-1.0 (if s is log-odds, P = 1/(1+exp(-s))), B=0.0
        def nll(params):
            a, b = params
            val = np.clip(a * s + b, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(val))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()
            return loss

        res = minimize(nll, [-1.0, 0.0], method="Nelder-Mead")
        self.A = float(res.x[0])
        self.B = float(res.x[1])
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        val = np.clip(self.A * s + self.B, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(val))

    def to_dict(self) -> Dict[str, float]:
        return {"A": self.A, "B": self.B}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PlattScaler":
        return cls(A=data["A"], B=data["B"])

class MetaStackingEnsemble:
    """
    Combines calibrated branch probabilities:
    P(Deceptive) = w1 * P_tilde_1 + w2 * P_tilde_2 + w3 * P_tilde_3
    with non-negative weights summing to 1.
    """
    def __init__(self, w1: float = 0.42, w2: float = 0.33, w3: float = 0.25):
        self.weights = np.array([w1, w2, w3], dtype=np.float32)
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        self.platt_scalers = [PlattScaler() for _ in range(3)]

    def fit_scalers(self, s_branch: np.ndarray, y: np.ndarray):
        """
        s_branch: (N, 3) matrix containing:
            col 0: logit(P_neural)
            col 1: logit(P_xgb)
            col 2: P_argon
        """
        for b in range(3):
            self.platt_scalers[b].fit(s_branch[:, b], y)
        return self

    def calibrate(self, s_branch: np.ndarray) -> np.ndarray:
        p_cal = np.zeros_like(s_branch, dtype=np.float32)
        for b in range(3):
            p_cal[:, b] = self.platt_scalers[b].transform(s_branch[:, b])
        return p_cal

    def predict_proba(self, s_branch: np.ndarray) -> np.ndarray:
        p_cal = self.calibrate(s_branch)
        return np.dot(p_cal, self.weights)

    def predict(self, s_branch: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        return (self.predict_proba(s_branch) >= threshold).astype(int)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "scalers": [scaler.to_dict() for scaler in self.platt_scalers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaStackingEnsemble":
        weights = data["weights"]
        instance = cls(w1=weights[0], w2=weights[1], w3=weights[2])
        instance.platt_scalers = [PlattScaler.from_dict(s) for s in data["scalers"]]
        return instance

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MetaStackingEnsemble":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
