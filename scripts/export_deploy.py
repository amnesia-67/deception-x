#!/usr/bin/env python3
"""Export the deployable Branch-2 digest model (served by the socket server) from REAL data.

Fits the same XGBoost configuration that was cross-validated in stage 2 on all 121
Real-Life Trials clips and writes:
  models/branch2_xgboost.json        the booster
  models/branch2_digest_schema.json  feature names, honest CV metrics and caveats
The numbers in the schema are the identity-grouped out-of-fold results, not training fit.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from scripts.train_stage2 import FEAT, OUT, TAB_VERSION, XGB_PARAMS, tabular_fingerprint
    from src.clip_store import ClipStore
    from src.tabular_branch import TabularLOSOXGBoost
    st = ClipStore()
    d = np.load(os.path.join(FEAT, "tabular.npz"), allow_pickle=False)
    assert list(d["ids"]) == st.ids and str(d["version"]) == TAB_VERSION, "run scripts/train_stage2.py first"
    assert str(d["fingerprint"]) == tabular_fingerprint(st), "tabular.npz is stale; run scripts/train_stage2.py"
    X, names = d["X"].astype(np.float32), [str(n) for n in d["names"]]
    y = st.labels(st.ids)
    model = TabularLOSOXGBoost(**XGB_PARAMS, device="cuda")
    model.fit(X, y)
    model.save_model(os.path.join(ROOT, "models", "branch2_xgboost.json"))
    s2 = json.load(open(os.path.join(OUT, "summary.json")))
    xm = s2["branches"]["xgb_digest"]
    schema = {
        "trained_on": "Real-Life Trials (Perez-Rosas et al. 2015), 121 real courtroom clips, 61 deceptive / 60 truthful",
        "feature_version": TAB_VERSION,
        "tabular_fingerprint": str(d["fingerprint"]),
        "feature_names": names,
        "cv_metrics": {"protocol": "5-fold identity-grouped CV, out-of-fold", "auroc": xm["auroc"],
                       "auroc_ci95": xm["auroc_ci95"], "accuracy": xm["acc"], "accuracy_ci95": xm["acc_ci95"],
                       "permutation_null": s2["honesty_checks"]["xgb_digest_permutation_null"]},
        "caveats": [
            "121 clips / 57 people: confidence intervals are wide.",
            "Recording conditions alone predict the label with AUROC "
            f"{s2['honesty_checks']['recording_conditions_probe']['auroc']:.2f} on this dataset, so part of any "
            "signal may be courtroom/channel artefacts rather than deception.",
            "Features must be computed exactly as src/features_tabular.py does (text from a transcript, voice "
            "from 16 kHz mono audio, face from 384x384 face crops + MediaPipe boxes).",
            "Not suitable for real-world decisions about whether a person is lying.",
        ],
    }
    json.dump(schema, open(os.path.join(ROOT, "models", "branch2_digest_schema.json"), "w"), indent=2)
    print(f"exported digest model with {len(names)} features; CV AUROC {xm['auroc']:.3f}")
    export_ensemble(s2)


def export_ensemble(s2):
    """Served policy: mean of the 5 LoRA-tuned dasheng folds + the digest XGBoost, Platt + convex stacker refit
    on those two branches' OOF predictions, and a balanced-accuracy threshold from the stacker's grouped CV.
    The published metrics are the FULLY NESTED evaluation of exactly this recipe (nested_deploy_eval.py)."""
    import hashlib

    from scripts.train_stage2 import TAB_VERSION
    from src.provenance import git_revision

    pol = s2["deploy_policy"]
    branches = list(pol["branches"])
    assert branches == ["tuned_dasheng", "xgb_digest"], f"deploy service does not implement {branches}"
    nested_path = os.path.join(ROOT, "runs", "nested_deploy", "summary.json")
    if not os.path.exists(nested_path):
        raise SystemExit("run scripts/nested_deploy_eval.py first: the served policy needs its nested evaluation")
    nested = json.load(open(nested_path))
    if nested.get("feature_version") != TAB_VERSION or s2.get("feature_version") != TAB_VERSION:
        raise SystemExit("stage2 / nested evaluation were made with a different feature version; re-run them")
    import torch

    from src.jepa_deploy import canonical_json, state_schema

    dst = os.path.join(ROOT, "models", "dasheng_lora")
    os.makedirs(dst, exist_ok=True)
    fold_entries = {}
    for f in range(5):
        old = torch.load(os.path.join(ROOT, "runs", "tune_dasheng", f"fold{f}.pt"), map_location="cpu",
                         weights_only=True)
        if old["config"]["backbone"] != "dasheng":
            raise ValueError("expected tuned dasheng folds")
        # Only primitive config + tensors are deployed (no test ids, labels or probabilities).
        payload = {"config": json.loads(canonical_json(old["config"])), "state": old["state"]}
        out = os.path.join(dst, f"fold{f}.pt")
        torch.save(payload, out)
        import hashlib
        data = open(out, "rb").read()
        fold_entries[f"fold{f}.pt"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                                       "config": payload["config"], "state_schema": state_schema(payload["state"])}
    sp = nested["served_policy"]
    # The nested result above re-fits the threshold inside every outer fold. For transparency, also score the
    # nested served-aggregation predictions at the FIXED exported cutoff (fitted on all clips, so not clean).
    import csv as _csv

    from src.clip_store import ClipStore
    from src.tuning import classwise

    rows = list(_csv.DictReader(open(os.path.join(ROOT, "runs", "nested_deploy", "oof.csv"))))
    y_n = np.array([int(r["label"]) for r in rows])
    p_n = np.array([float(r["policy"]) for r in rows])
    assert [r["clip_id"] for r in rows] == ClipStore().ids
    fixed = {**classwise(y_n, p_n, pol["threshold"]),
             "caveat": "fixed cutoff fitted on all 121 clips' OOF scores, then applied to the nested served-"
                       "aggregation scores: not an untouched estimate; reported because it is the cutoff in use"}
    sha = lambda rel: hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()  # noqa: E731
    meta = {
        "format": "sunhacks-ensemble-v2",
        "class_1_meaning": "deceptive (label 1 = Real-Life Trials 'Deceptive' folder)",
        "feature_version": TAB_VERSION,
        "aggregation": "tuned_dasheng = mean of the 5 fold models' probabilities; xgb_digest = one model refit "
                       "on all 121 clips; p = sum_i weights[i] * sigmoid(platt_A[i] * logit(p_i) + platt_B[i])",
        "dasheng_folds": fold_entries,
        "digest_model_sha256": sha("models/branch2_xgboost.json"),
        "digest_schema_sha256": sha("models/branch2_digest_schema.json"),
        "branches": branches,
        "platt_A": pol["A"], "platt_B": pol["B"], "weights": pol["w"],
        "threshold": pol["threshold"],
        "threshold_provenance": "prior-matched: the fraction of the stacker's identity-grouped CV predictions "
                                "(training data only) called deceptive equals the training class base rate; the "
                                "same rule was re-fitted inside every outer fold of the nested evaluation, whose "
                                "result is cv_metrics",
        "cv_metrics": {"protocol": nested["protocol"], "residual_selection": nested["residual_selection"],
                       "auroc": sp["auroc"], "auroc_ci95": sp["auroc_ci95"],
                       "at_threshold_refit_per_outer_fold": {
                           **sp["at_nested_threshold"],
                           "note": "evaluates the threshold-FITTING procedure (thresholds re-fitted inside each "
                                   "outer fold, 0.448-0.485); not measured at the fixed exported cutoff"},
                       "at_fixed_exported_cutoff_on_nested_scores": fixed,
                       "at_0.5": sp["at_0.5"],
                       "branches_as_served": {b: {"auroc": m["auroc"], "auroc_ci95": m["auroc_ci95"]}
                                              for b, m in nested["branches_as_served"].items()},
                       "seven_second_dasheng_diagnostic": nested["seven_second_diagnostic"]},
        "code_revision": git_revision(),
        "caveats": [
            "Trained on 121 courtroom clips (Real-Life Trials) with human transcripts. Webcam audio/video and "
            "Whisper transcripts are a different domain; webcam accuracy has NOT been measured.",
            f"Recording conditions alone predict the training labels with AUROC "
            f"{s2['honesty_checks']['recording_conditions_probe']['auroc']:.2f}; part of the signal may be artefacts.",
            "Probabilities are Platt-scaled on courtroom out-of-fold predictions, not calibrated for live input.",
            "Not suitable for real-world decisions about whether a person is lying.",
        ],
    }
    json.dump(meta, open(os.path.join(ROOT, "models", "ensemble_deploy.json"), "w"), indent=2)
    print(f"exported ensemble: {branches} weights {[round(w, 3) for w in meta['weights']]}, cutoff "
          f"{pol['threshold']:.3f} | nested AUROC {sp['auroc']:.3f}; at the fixed cutoff on nested scores: lies "
          f"caught {fixed['lie_sensitivity']:.0%}, truths passed {fixed['truth_specificity']:.0%}")


def export_jepa(destination=None):
    """Package EXISTING V-JEPA 2.1 ViT-L folds only (runs/tune_vjepa2_1_l); no dataset access, fit or training.

    Refuse to overwrite a published directory. Stage a complete immutable export
    and rename it into place after every fold/config/hash is ready.
    """
    import shutil
    import tempfile
    from pathlib import Path
    import torch
    from src.jepa_deploy import (BASE_RELATIVE_PATH, CONTRACT, MANIFEST_FORMAT,
        MODEL_LABEL, canonical_json, file_sha256, read_checkpoint, state_schema)

    source = Path(ROOT) / "runs" / "tune_vjepa2_1_l"
    target = Path(destination) if destination is not None else Path(ROOT) / "models" / "vjepa2_1_l_lora"
    if target.exists():
        raise FileExistsError("Choose a new export directory; do not overwrite a running deployment")
    expected = {f"fold{k}.pt" for k in range(5)}
    if {path.name for path in source.glob("fold*.pt")} != expected:
        raise ValueError("Need exactly five EXISTING V-JEPA 2.1 ViT-L folds: fold0.pt through fold4.pt")
    base_path = Path(ROOT) / BASE_RELATIVE_PATH
    base = {"path": BASE_RELATIVE_PATH, "size": base_path.stat().st_size,
            "sha256": file_sha256(base_path)}
    summary = json.loads((source / "summary.json").read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".vjepa21-export-", dir=target.parent))
    try:
        entries = {}
        for name in sorted(expected):
            old, _, _ = read_checkpoint(source / name)
            # Strip test IDs/labels/probabilities and arbitrary training info. Only
            #primitive JSON config plus tensor state belongs in deployment.
            config = json.loads(canonical_json(old["config"]))
            payload = {"config": config, "state": old["state"]}
            out_path = stage / name
            torch.save(payload, out_path)
            checked, digest, size = read_checkpoint(out_path)
            entries[name] = {"sha256": digest, "size": size, "config": config,
                             "state_schema": state_schema(checked["state"])}
        meta = {"format": MANIFEST_FORMAT, "model": MODEL_LABEL, "contract": dict(CONTRACT),
                "base_encoder": base, "folds": entries,
                "cv_metrics": {"protocol": summary["protocol"], "auroc": summary["oof"]["auroc"],
                    "auroc_ci95": summary["auroc_ci95"], "accuracy": summary["oof"]["acc"],
                    "per_fold_auroc": [fold["test"]["auroc"] for fold in summary["folds"]]},
                "fingerprint": summary.get("fingerprint"),
                "caveat": "V-JEPA 2.1 ViT-L classifier score (uncalibrated); wide identity-grouped confidence "
                          "intervals; scores are not evidence of deception."}
        (stage / "deploy_meta.json").write_text(canonical_json(meta) + "\n")
        if target.exists():
            raise FileExistsError("Export target appeared while staging; refusing overwrite")
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"exported 5 existing V-JEPA 2.1 ViT-L fold models to {target}")
    return target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export existing models")
    parser.add_argument("--jepa21-only", action="store_true",
                        help="Package existing V-JEPA 2.1 ViT-L folds; no training or dataset reads")
    parser.add_argument("--ensemble-only", action="store_true",
                        help="Re-export the stacked ensemble (tuned dasheng folds + stacker) only")
    parser.add_argument("--jepa21-output", default=None, help="Fresh output directory for the 2.1-only export")
    args = parser.parse_args()
    if args.ensemble_only:
        export_ensemble(json.load(open(os.path.join(ROOT, "runs", "stage2", "summary.json"))))
    elif args.jepa21_only:
        export_jepa(args.jepa21_output)
    else:
        if args.jepa21_output is not None:
            parser.error("--jepa21-output requires --jepa21-only")
        main()
