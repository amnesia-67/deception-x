"""
Branch 2: Fast TreeSHAP Explainer & Glass-Box Forensic Attribution
Accredited to Gemini.pdf Specifications (Pages 26-29, 43).

Computes exact additive feature attributions phi in R^250 in < 5 ms:
    y_hat = phi_0 + sum_{j=1}^{250} phi_j
Identifies top positive (deceptive) and negative (truthful) within-subject Z-score drivers,
and formats the Glass-Box waterfall summary.
"""

import time
import numpy as np
import xgboost as xgb
from typing import Dict, List, Any, Optional, Tuple

from src.tabular_branch import TabularLOSOXGBoost
from src.feature_schema import FEATURE_NAMES, DOMAIN_SLICES, get_feature_domain


class ForensicTreeSHAPExplainer:
    """
    Sub-5ms TreeSHAP attribution engine for Branch 2 TabularLOSOXGBoost.
    """
    def __init__(self, tabular_model: TabularLOSOXGBoost, feature_names=None):
        """feature_names: names of the model's input columns (default: the PDF's 250-dim schema)."""
        self.tabular_model = tabular_model
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        self.custom_schema = feature_names is not None
        assert self.tabular_model.is_fitted, "TabularLOSOXGBoost must be fitted before explaining."
        self.booster = self.tabular_model.model.get_booster()

    def explain(
        self,
        z: np.ndarray,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Computes TreeSHAP attributions for a single sample z in R^250 or batch (N, 250).
        Returns top deceptive drivers, top truthful drivers, and domain breakdowns.
        """
        t0 = time.perf_counter()
        z_arr = np.array(z, dtype=np.float32)
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)
            
        n = len(self.feature_names)
        assert z_arr.shape[1] == n, f"Expected {n} features, got {z_arr.shape[1]}"
        
        # Native C++ TreeSHAP execution via pred_contribs=True in XGBoost
        dmat = xgb.DMatrix(z_arr)
        contribs = self.booster.predict(dmat, pred_contribs=True)
        # contribs shape: (N, 251) where last column is bias phi_0 (base margin)
        phi_0 = float(contribs[0, -1])
        phi_vec = contribs[0, :-1]
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        
        # Margin prediction and probability
        margin = phi_0 + float(np.sum(phi_vec))
        p_xgb = 1.0 / (1.0 + np.exp(-margin))
        
        # Identify top positive (deceptive) drivers
        pos_indices = np.where(phi_vec > 0)[0]
        sorted_pos = pos_indices[np.argsort(-phi_vec[pos_indices])]
        top_positive_drivers = []
        for idx in sorted_pos[:top_k]:
            top_positive_drivers.append({
                "index": int(idx),
                "name": self.feature_names[idx],
                "domain": self._domain(idx),
                "z_value": float(z_arr[0, idx]),
                "shap_impact": float(phi_vec[idx]),
            })
            
        # Identify top negative (truthful) drivers
        neg_indices = np.where(phi_vec < 0)[0]
        sorted_neg = neg_indices[np.argsort(phi_vec[neg_indices])]
        top_negative_drivers = []
        for idx in sorted_neg[:top_k]:
            top_negative_drivers.append({
                "index": int(idx),
                "name": self.feature_names[idx],
                "domain": self._domain(idx),
                "z_value": float(z_arr[0, idx]),
                "shap_impact": float(phi_vec[idx]),
            })
            
        # Domain attribution breakdown
        domain_attributions: Dict[str, float] = {}
        if self.custom_schema:
            for idx, v in enumerate(phi_vec):
                d = self._domain(idx)
                domain_attributions[d] = domain_attributions.get(d, 0.0) + float(v)
        else:
            for domain, (start, end) in DOMAIN_SLICES.items():
                domain_attributions[domain] = float(np.sum(phi_vec[start:end]))
            
        return {
            "execution_latency_ms": elapsed_ms,
            "base_value_phi_0": phi_0,
            "margin": margin,
            "p_xgb": p_xgb,
            "top_positive_drivers": top_positive_drivers,
            "top_negative_drivers": top_negative_drivers,
            "domain_attributions": domain_attributions,
            "raw_shap_values": phi_vec.tolist(),
        }

    def _domain(self, idx: int) -> str:
        if not self.custom_schema:
            return get_feature_domain(idx)
        prefix = self.feature_names[idx].split("_", 1)[0]
        return {"txt": "text_psycholinguistic", "voc": "voice_prosody", "face": "face_kinematics",
                "emo": "emotion2vec_affect", "intent": "dialogue_intent"}.get(prefix, prefix)

    def render_waterfall_ascii(self, explanation: Dict[str, Any], max_rows: int = 10) -> str:
        """
        Renders an ASCII waterfall chart of top drivers for fast terminal telemetry.
        """
        lines = []
        lines.append("=" * 70)
        lines.append(" DeceptionX TreeSHAP Waterfall Explanation (Branch 2)")
        lines.append("=" * 70)
        lines.append(f"Base Margin (phi_0): {explanation['base_value_phi_0']:+.4f}")
        lines.append(f"Final Margin:        {explanation['margin']:+.4f} (P_XGB = {explanation['p_xgb']:.3f})")
        lines.append(f"Computation Latency: {explanation['execution_latency_ms']:.2f} ms")
        lines.append("-" * 70)
        lines.append(f"{'Feature Name':<38} | {'z-score':<8} | {'SHAP Impact':<12}")
        lines.append("-" * 70)
        
        # Merge positive and negative drivers sorted by absolute impact
        all_drivers = explanation["top_positive_drivers"] + explanation["top_negative_drivers"]
        all_drivers.sort(key=lambda d: abs(d["shap_impact"]), reverse=True)
        
        for d in all_drivers[:max_rows]:
            feat_name = d["name"][:36]
            z_val = f"{d['z_value']:+.2f}σ"
            impact = f"{d['shap_impact']:+.4f}"
            bar = "▲" * int(min(abs(d["shap_impact"]) * 10, 15)) if d["shap_impact"] > 0 else "▼" * int(min(abs(d["shap_impact"]) * 10, 15))
            lines.append(f"{feat_name:<38} | {z_val:<8} | {impact:<12} {bar}")
            
        lines.append("-" * 70)
        lines.append("Domain Aggregation Breakdown:")
        for dom, val in explanation["domain_attributions"].items():
            lines.append(f"  - {dom:<34}: {val:+.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)
