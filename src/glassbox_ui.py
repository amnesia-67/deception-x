"""
Glass-Box Interactive Forensic Dashboard & Telemetry Display
Accredited to Gemini.pdf Specifications (Pages 40-43).

Renders the real-time Glass-Box forensic HUD:
- Multi-Branch Probability Dials (Branch 1, Branch 2, Branch 3, Stacking Ensemble)
- Modality Reliability Weights (alpha_v, alpha_a, alpha_l)
- Cross-Modal Discrepancy Cosine Distances (D_va, D_vl, D_al)
- TreeSHAP Waterfall Table with Top Positive & Negative Drivers
- Streaming Observe-Think-Summarize Cognitive Verdict Card
- Latency & Hardware Performance Metrics
"""

from typing import Dict, Any, Optional


def render_glassbox_hud(telemetry: Dict[str, Any]) -> str:
    """
    Renders a terminal HUD string representing the DeceptionX Glass-Box telemetry.
    telemetry: dictionary produced by JudgeInterrogationSession or DeceptionXPipeline
    """
    pipe = telemetry.get("pipeline_telemetry", telemetry)
    b_probs = pipe.get("branch_probabilities", {})
    p_ens = pipe.get("p_ensemble", 0.5)
    verdict = pipe.get("classification", "Unknown")
    discs = pipe.get("cross_modal_discrepancies", {})
    alphas = pipe.get("modality_reliability_weights", {})
    b3 = pipe.get("cognitive_verdict", {})
    shap_info = telemetry.get("shap_explanation", {})
    
    total_ms = telemetry.get("total_latency_ms", 185.0)
    ground_truth = telemetry.get("ground_truth")
    is_correct = telemetry.get("is_correct")
    
    buf = []
    buf.append("=" * 82)
    buf.append(" ☀️  DECEPTIONX: GLASS-BOX FORENSIC TELEMETRY ENGINE (RTX 5070 + GEMINI 4 ARGON)  ☀️")
    buf.append("=" * 82)
    
    # 1. Prediction & Ground Truth Summary
    verdict_badge = f" [ {verdict.upper()} ] " if verdict == "Deceptive" else f" [ {verdict.upper()} ] "
    status_str = f"Verdict: {verdict_badge:<14} | P(Deceptive): {p_ens:6.2%} | Latency: {total_ms:.1f}ms"
    if ground_truth:
        acc_badge = "✅ CORRECT" if is_correct else "❌ MISMATCH"
        status_str += f" | Ground Truth: {ground_truth} ({acc_badge})"
    buf.append(status_str)
    buf.append("-" * 82)
    
    # 2. Multi-Branch Probability Dials
    p1 = b_probs.get("branch1_neural", 0.5)
    p2 = b_probs.get("branch2_xgboost", 0.5)
    p3 = b_probs.get("branch3_cognitive", 0.5)
    
    def bar(p: float, width: int = 14) -> str:
        filled = int(p * width)
        return "█" * filled + "░" * (width - filled)
        
    buf.append("MULTI-BRANCH PROBABILISTIC CONVERGENCE:")
    buf.append(f"  Branch 1 (Neural Discrepancy): [{bar(p1)}] {p1:6.2%} (w1=0.42)")
    buf.append(f"  Branch 2 (Tabular LOSO XGB)  : [{bar(p2)}] {p2:6.2%} (w2=0.33)")
    buf.append(f"  Branch 3 (Cognitive Reasoner): [{bar(p3)}] {p3:6.2%} (w3=0.25)")
    buf.append(f"  ------------------------------------------------------------")
    buf.append(f"  Meta-Ensemble Calibrated P   : [{bar(p_ens)}] {p_ens:6.2%}")
    buf.append("-" * 82)
    
    # 3. Discrepancy & Modality Reliability
    d_va = discs.get("D_va", 0.0)
    d_vl = discs.get("D_vl", 0.0)
    d_al = discs.get("D_al", 0.0)
    a_v = alphas.get("alpha_visual", 0.33)
    a_a = alphas.get("alpha_audio", 0.33)
    a_l = alphas.get("alpha_text", 0.33)
    
    buf.append("CROSS-MODAL DISCREPANCY & DYNAMIC RELIABILITY (ReDA):")
    buf.append(f"  Discrepancy: Audio-Visual D_va={d_va:.4f} | Visual-Text D_vl={d_vl:.4f} | Audio-Text D_al={d_al:.4f}")
    buf.append(f"  Reliability: Visual alpha_v={a_v:.3f}     | Audio alpha_a={a_a:.3f}     | Text alpha_l={a_l:.3f}")
    buf.append("-" * 82)
    
    # 4. TreeSHAP Waterfall Top Drivers
    if shap_info:
        buf.append("TREESHAP FORENSIC WATERFALL DRIVERS (Branch 2):")
        pos_drivers = shap_info.get("top_positive_drivers", [])[:3]
        neg_drivers = shap_info.get("top_negative_drivers", [])[:3]
        
        buf.append("  Top Deceptive Indicators (+):")
        for d in pos_drivers:
            buf.append(f"    ▲ {d['name'][:35]:<35} | z = {d['z_value']:+5.2f}σ | impact = {d['shap_impact']:+.4f}")
            
        buf.append("  Top Truthful Indicators (-):")
        for d in neg_drivers:
            buf.append(f"    ▼ {d['name'][:35]:<35} | z = {d['z_value']:+5.2f}σ | impact = {d['shap_impact']:+.4f}")
        buf.append("-" * 82)
        
    # 5. Cognitive Observe-Think-Summarize Verdict Card
    buf.append("BRANCH 3 COGNITIVE VERDICT (Gemini 4 Argon / Gemini 3.8 Flash):")
    if "observe_trace" in b3 and b3["observe_trace"]:
        buf.append("  <observe>:")
        for obs in b3["observe_trace"][:3]:
            buf.append(f"    • {obs}")
    if "think_trace" in b3 and b3["think_trace"]:
        buf.append(f"  <think>: {b3['think_trace'][:200]}...")
    if "forensic_summary" in b3 and b3["forensic_summary"]:
        buf.append(f"  <verdict>: {b3['forensic_summary']}")
    buf.append("=" * 82)
    
    return "\n".join(buf)


def print_glassbox_hud(telemetry: Dict[str, Any]):
    """Convenience helper to print the HUD directly."""
    print(render_glassbox_hud(telemetry))
