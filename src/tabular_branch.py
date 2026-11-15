"""
Branch 2: Identity-Agnostic Leave-One-Speaker-Out (LOSO) XGBoost
Operating on the 250-dimensional within-subject Z-score digest z in R^250.
Target Hardware: NVIDIA RTX 5070 (device="cuda").
Hyperparameters locked to Gemini.pdf Pages 26-29.
"""

import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, roc_auc_score

class TabularLOSOXGBoost:
    """
    XGBoost Classifier for Branch 2.
    Trained under Leave-One-Speaker-Out (LOSO) cross-validation
    with GPU tree_method='hist' on RTX 5070.
    """
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        colsample_bytree: float = 0.35,
        subsample: float = 0.80,
        reg_alpha: float = 1.5,  # L1 regularization
        reg_lambda: float = 5.0, # L2 regularization
        learning_rate: float = 0.05,
        device: str = "cuda"
    ):
        self.device = device
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "learning_rate": learning_rate,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": self.device,
            "random_state": 42
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit model on GPU."""
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs probabilities P(Deceptive) in [0, 1] using DMatrix to avoid host-device mismatch."""
        assert self.is_fitted, "Model is not fitted yet!"
        dmat = xgb.DMatrix(X)
        return self.model.get_booster().predict(dmat)

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """Binary predictions."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save_model(self, filepath: str):
        """Saves fitted booster to JSON file."""
        assert self.is_fitted, "Cannot save unfitted model!"
        self.model.save_model(filepath)

    def load_model(self, filepath: str):
        """Loads fitted booster from JSON file."""
        self.model.load_model(filepath)
        self.is_fitted = True

    @staticmethod
    def evaluate_loso(
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: np.ndarray,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Executes strict Leave-One-Speaker-Out (LOSO) cross validation across all speakers.
        Guarantees zero speaker leakage.
        """
        unique_speakers = np.unique(speaker_ids)
        oof_preds = np.zeros(len(y), dtype=np.float32)
        
        for spk in unique_speakers:
            val_idx = np.where(speaker_ids == spk)[0]
            train_idx = np.where(speaker_ids != spk)[0]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            fold_model = TabularLOSOXGBoost(device=device)
            fold_model.fit(X_train, y_train)
            oof_preds[val_idx] = fold_model.predict_proba(X_val)
            
        acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
        try:
            auroc = roc_auc_score(y, oof_preds)
        except ValueError:
            auroc = 0.5
            
        return {
            "oof_accuracy": float(acc),
            "oof_auroc": float(auroc),
            "oof_preds": oof_preds
        }
