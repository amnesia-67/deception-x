#!/usr/bin/env python3
"""Fully nested (base + meta) evaluation of the EXACT served `analyze` policy on unseen people.

The served policy is: mean of 5 LoRA-tuned dasheng fold models + one XGBoost digest refit on all training
clips -> per-branch Platt -> convex weights -> decision threshold. Every one of those pieces is fitted from
training data only, so for each outer identity-grouped fold k this script re-runs the WHOLE recipe on the
outer-train clips and applies it to the untouched outer-test clips:

  1. inner 5-fold grouped CV on outer-train:
       - LoRA-tune dasheng per inner fold (same ladder / early stopping / refit as stage 1)
         -> inner OOF dasheng probabilities, and each inner model's prediction for the outer test clips
       - XGBoost digest inner OOF
  2. stacker (Platt + convex weights) fit on the inner OOF matrix only; decision threshold = balanced-
     matched to the outer-train lie rate on the stacker's own grouped CV (scripts.train_stage2.fit_deploy_policy)
  3. outer test clips scored exactly as served: mean of the inner dasheng models, digest refit on all of
     outer-train, then the fold's stacker and threshold.

No outer-test clip, label or identity enters any fitted quantity of its fold; every fold writes a manifest
that asserts this. Also reports a 7-second-answer diagnostic for dasheng (inherited clip labels; see report).

  bash scripts/run.sh scripts/nested_deploy_eval.py            (resumes finished inner runs)
Outputs: runs/nested_deploy/{summary.json, report.md, oof.csv, fold<k>_manifest.json, inner/*.json}
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_stage2 import (DEPLOY_BRANCHES, FEAT, TAB_VERSION, apply_stacker, fit_deploy_policy,  # noqa: E402
                                  tabular_fingerprint, xgb_model, xgb_oof)
from src.clip_store import SR, ClipStore  # noqa: E402
from src.provenance import git_revision, tuning_fingerprint  # noqa: E402
from src.tuning import (LADDER, RUNS, GPUMonitor, RunReporter, TuneConfig, TunedModel, classwise,  # noqa: E402
                        group_bootstrap_ci, limit_to_physical_vram, metrics, outer_folds, predict, tune_fold)

OUT = os.path.join(RUNS, "nested_deploy")
K_OUTER, K_INNER, SEED = 5, 5, 0
LIVE_ANSWER_S = 7.0


def inner_dasheng(store, k, j, tr_ids, va_ids, te_ids, reporter, gpu, fp):
    return inner_tune("dasheng", store, k, j, tr_ids, va_ids, te_ids, reporter, gpu, fp)


def inner_tune(bb, store, k, j, tr_ids, va_ids, te_ids, reporter, gpu, fp):
    """One inner LoRA run, cached on disk (resumable) and bound to the tuning fingerprint."""
    path = os.path.join(OUT, "inner", f"{bb}_k{k}_j{j}.json")
    if os.path.exists(path):
        rec = json.load(open(path))
        if rec["fingerprint"] == fp["all"] and rec["val_ids"] == va_ids and rec["test_ids"] == te_ids:
            return np.array(rec["p_val"]), np.array(rec["p_test"]), rec["info"]
        print(f"  inner run k{k} j{j} is stale - redoing", flush=True)
    p_val, info, model, _ = tune_fold(bb, store, tr_ids, va_ids, 100 + 10 * k + j, reporter, gpu, seed=SEED)
    p_te = predict(model, store, te_ids, 1 if model.backbone.modality == "audio" else 4)
    # Keep the learning evidence: per-attempt traces (init/returned state hashes, optimizer steps, loss,
    # grad norm, val AUROC per epoch, update L2) and the refit trace.
    info = {k_: v for k_, v in info.items() if k_ in ("config", "inner_val", "attempts", "final_model",
                                                       "refit_epochs", "refit_trace", "final_state_sha256")}
    json.dump({"fingerprint": fp["all"], "train_ids": tr_ids, "val_ids": va_ids, "test_ids": te_ids,
               "p_val": p_val.tolist(), "p_test": p_te.tolist(), "info": info}, open(path, "w"))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return p_val, p_te, info


@torch.no_grad()
def seven_second_diagnostic(store, ids, y, folds):
    """Stage-1 dasheng fold models on their own held-out clips: full answer (<= 60 s, as trained/evaluated)
    versus only the central 7 s (the live answer length). Labels are inherited from the whole clip, so this
    measures the duration shift of the model, not statement-level accuracy on 7 s answers."""
    out = {"full": np.zeros(len(ids)), "7s": np.zeros(len(ids))}
    for f, (_, te) in enumerate(folds):
        ck = torch.load(os.path.join(RUNS, "tune_dasheng", f"fold{f}.pt"), map_location="cpu", weights_only=False)
        assert list(ck["test_ids"]) == [ids[i] for i in te]
        cfg = TuneConfig(**{a: b for a, b in ck["config"].items() if a in TuneConfig.__dataclass_fields__})
        m = TunedModel(cfg).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        m.load_trainable_state(ck["state"])
        for i in te:
            a = np.array(store.audio(ids[i]), dtype=np.float32)
            for key, lim in (("full", 60.0), ("7s", LIVE_ANSWER_S)):
                n = int(lim * SR)
                x = a[(len(a) - n) // 2:(len(a) - n) // 2 + n] if len(a) > n else a
                x = np.pad(x, (0, max(0, SR - len(x))))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = float(m.head(m.backbone(torch.from_numpy(np.ascontiguousarray(x)).cuda()[None])).float())
                out[key][i] = 1 / (1 + np.exp(-lg))
        del m
        torch.cuda.empty_cache()
    durations = np.array([store.meta(c)["duration_s"] for c in ids])
    res = {"note": "inherited whole-clip labels; central window only; not a 7 s target-domain evaluation",
           "n_clips_longer_than_7s": int((durations > LIVE_ANSWER_S).sum())}
    for key in out:
        res[key] = {**metrics(y, out[key]), "classwise": classwise(y, out[key]),
                    "mean_prob": float(out[key].mean())}
    return res, out


def boot_rates(y, yhat, groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    s, t, b = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        yy, hh = y[idx], yhat[idx]
        if yy.min() == yy.max():
            continue
        se, sp = hh[yy == 1].mean(), 1 - hh[yy == 0].mean()
        s.append(se), t.append(sp), b.append((se + sp) / 2)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"lie_sensitivity_ci95": q(s), "truth_specificity_ci95": q(t), "bal_acc_ci95": q(b)}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(os.path.join(OUT, "inner"), exist_ok=True)
    t0 = time.time()
    store = ClipStore()
    ids = store.ids
    y, g = store.labels(ids), store.groups(ids)
    d = np.load(os.path.join(FEAT, "tabular.npz"), allow_pickle=False)
    if list(d["ids"]) != ids or str(d["version"]) != TAB_VERSION or str(d["fingerprint"]) != tabular_fingerprint(store):
        raise SystemExit("tabular.npz is stale; run scripts/train_stage2.py first")
    X = d["X"].astype(np.float32)
    folds = outer_folds(ids, y, g, k=K_OUTER, seed=SEED)
    fp = tuning_fingerprint(store, "dasheng", LADDER["dasheng"], SEED, K_INNER)
    reporter, gpu = RunReporter(OUT), GPUMonitor()

    print("== 7-second duration diagnostic (stage-1 dasheng folds)", flush=True)
    diag, diag_p = seven_second_diagnostic(store, ids, y, folds)
    print(f"   full: AUROC {diag['full']['auroc']:.3f}, called deceptive {diag['full']['classwise']['predicted_lie_fraction']:.0%}"
          f" | 7 s: AUROC {diag['7s']['auroc']:.3f}, called deceptive {diag['7s']['classwise']['predicted_lie_fraction']:.0%}",
          flush=True)

    p_pol, thr_row = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    p_dash, p_xgb = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    fold_rows = []
    for k, (tr, te) in enumerate(folds):
        assert not set(g[tr]) & set(g[te])
        tr_ids, te_ids = [ids[i] for i in tr], [ids[i] for i in te]
        inner = outer_folds(tr_ids, y[tr], g[tr], k=K_INNER, seed=SEED + 1)
        s_dash_in = np.full(len(tr), np.nan)
        te_by_inner = []
        print(f"== outer fold {k}: {len(tr)} train / {len(te)} test clips; {K_INNER} inner LoRA runs", flush=True)
        for j, (itr, iva) in enumerate(inner):
            assert not set(g[tr][itr]) & set(g[tr][iva])
            pv, pt, _ = inner_dasheng(store, k, j, [tr_ids[i] for i in itr], [tr_ids[i] for i in iva], te_ids,
                                      reporter, gpu, fp)
            s_dash_in[iva] = pv
            te_by_inner.append(pt)
        s_xgb_in = xgb_oof(X[tr], y[tr], g[tr], inner)
        S_in = np.column_stack([s_dash_in, s_xgb_in])
        assert DEPLOY_BRANCHES == ["tuned_dasheng", "xgb_digest"] and np.isfinite(S_in).all()
        pol = fit_deploy_policy(S_in, y[tr], inner)
        clf = xgb_model(SEED)
        clf.fit(X[tr], y[tr])
        p_xgb[te] = clf.predict_proba(X[te])[:, 1]
        p_dash[te] = np.mean(te_by_inner, 0)                      # served aggregation: mean of fold models
        S_te = np.column_stack([p_dash[te], p_xgb[te]])
        p_pol[te] = apply_stacker({k_: np.asarray(v) for k_, v in pol.items() if k_ in ("A", "B", "w")}, S_te)
        thr_row[te] = pol["threshold"]
        manifest = {"outer_fold": k, "test_ids": te_ids, "train_ids": tr_ids,
                    "test_identities": sorted(map(int, set(g[te]))),
                    "asserts": {"no_test_clip_in_training": not set(te_ids) & set(tr_ids),
                                "no_test_identity_in_training": not set(g[te]) & set(g[tr]),
                                "inner_folds_identity_disjoint": all(not set(g[tr][a]) & set(g[tr][b])
                                                                     for a, b in inner)},
                    "policy": pol, "tuning_fingerprint": fp,
                    "score_sd_single_model_inner_oof": float(np.std(s_dash_in)),
                    "score_sd_mean_of_models_outer": float(np.std(p_dash[te]))}
        assert all(manifest["asserts"].values())
        json.dump(manifest, open(os.path.join(OUT, f"fold{k}_manifest.json"), "w"), indent=1)
        fm = metrics(y[te], p_pol[te])
        fold_rows.append({"fold": k, "n_test": len(te), "auroc": fm["auroc"], "threshold": pol["threshold"],
                          "weights": pol["w"], "inner_dasheng_auroc": metrics(y[tr], s_dash_in)["auroc"],
                          "inner_xgb_auroc": metrics(y[tr], s_xgb_in)["auroc"]})
        print(f"   fold {k}: policy AUROC {fm['auroc']:.3f}, threshold {pol['threshold']:.3f}, "
              f"weights {np.round(pol['w'], 3).tolist()}", flush=True)
    gpu.stop()

    yhat_thr = (p_pol >= thr_row).astype(int)
    res = {"protocol": "fully nested: outer 5-fold identity-grouped CV; inside each outer-train set an inner "
                       "5-fold identity-grouped CV re-tunes dasheng LoRA, produces digest OOF, fits the stacker "
                       "and picks the threshold; outer test scored with the served aggregation",
           "residual_selection": "Which two branches are served was chosen earlier from the 7-branch stacker on "
                                 "all clips; that choice is not re-made inside the outer folds.",
           "n_clips": len(ids), "n_identities": int(len(set(g))), "feature_version": TAB_VERSION,
           "served_policy": {**metrics(y, p_pol), **group_bootstrap_ci(y, p_pol, g),
                             "at_0.5": classwise(y, p_pol, 0.5),
                             "at_nested_threshold": {**classwise(y, p_pol, thr_row), **boot_rates(y, yhat_thr, g)},
                             "at_0.5_ci": boot_rates(y, (p_pol >= 0.5).astype(int), g)},
           "branches_as_served": {"tuned_dasheng_mean_of_fold_models": {**metrics(y, p_dash), **group_bootstrap_ci(y, p_dash, g),
                                                                         "classwise": classwise(y, p_dash)},
                                  "xgb_digest_full_refit": {**metrics(y, p_xgb), **group_bootstrap_ci(y, p_xgb, g),
                                                            "classwise": classwise(y, p_xgb)}},
           "folds": fold_rows, "seven_second_diagnostic": diag, "minutes": (time.time() - t0) / 60,
           "code_revision": git_revision()}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    with open(os.path.join(OUT, "oof.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "group", "label", "dasheng_served", "xgb_served", "policy", "threshold",
                    "dasheng_full_stage1", "dasheng_7s_stage1"])
        for i, c in enumerate(ids):
            w.writerow([c, int(g[i]), int(y[i]), f"{p_dash[i]:.5f}", f"{p_xgb[i]:.5f}", f"{p_pol[i]:.5f}",
                        f"{thr_row[i]:.5f}", f"{diag_p['full'][i]:.5f}", f"{diag_p['7s'][i]:.5f}"])
    write_report(res)
    print(open(os.path.join(OUT, "report.md")).read())


def write_report(r):
    sp = r["served_policy"]
    a, b = sp["at_0.5"], sp["at_nested_threshold"]
    L = ["# Served `analyze` policy - fully nested evaluation on unseen people", "",
         r["protocol"] + ".", "", f"Residual limitation: {r['residual_selection']}", "",
         f"**AUROC {sp['auroc']:.3f}** (95% identity-bootstrap CI {sp['auroc_ci95'][0]:.3f}-{sp['auroc_ci95'][1]:.3f}),"
         f" n = {sp['n']} clips / {r['n_identities']} people. Class 1 = deceptive.", "",
         "| decision rule | lies caught | truths passed | balanced acc | accuracy | called deceptive | Brier | ECE |",
         "|---|---|---|---|---|---|---|---|"]
    for name, c in (("fixed 0.5", a), (f"prior-matched threshold RE-FITTED inside each outer fold "
                                       f"(mean {b['threshold']:.3f})", b)):
        L.append(f"| {name} | {c['tp_lie']}/{c['tp_lie'] + c['fn_lie']} ({c['lie_sensitivity']:.0%}) | "
                 f"{c['tn_truth']}/{c['tn_truth'] + c['fp_truth']} ({c['truth_specificity']:.0%}) | {c['bal_acc']:.3f} | "
                 f"{c['acc']:.3f} | {c['predicted_lie_fraction']:.0%} | {c['brier']:.3f} | {c['ece_10bin']:.3f} |")
    L += ["", "The re-fitted row evaluates the threshold-fitting procedure; it is not a measurement at the "
          "single cutoff exported for serving (see models/ensemble_deploy.json cv_metrics).",
          "", f"Re-fitted-threshold 95% CIs: lie sensitivity {b['lie_sensitivity_ci95'][0]:.2f}-{b['lie_sensitivity_ci95'][1]:.2f},"
          f" truth specificity {b['truth_specificity_ci95'][0]:.2f}-{b['truth_specificity_ci95'][1]:.2f}, balanced accuracy "
          f"{b['bal_acc_ci95'][0]:.2f}-{b['bal_acc_ci95'][1]:.2f}. Baselines: always-truthful accuracy "
          f"{a['baseline_always_truthful_acc']:.3f}, balanced accuracy 0.500; class-prior Brier {a['baseline_prior_brier']:.3f}.",
          "", "| branch as served | AUROC | 95% CI | lies caught @0.5 | truths passed @0.5 |", "|---|---|---|---|---|"]
    for name, m in r["branches_as_served"].items():
        c = m["classwise"]
        L.append(f"| {name} | {m['auroc']:.3f} | {m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f} | "
                 f"{c['lie_sensitivity']:.0%} | {c['truth_specificity']:.0%} |")
    L += ["", "| outer fold | n test | AUROC | threshold | weights (dasheng, digest) | inner dasheng AUROC | inner digest AUROC |",
          "|---|---|---|---|---|---|---|"]
    for f in r["folds"]:
        L.append(f"| {f['fold']} | {f['n_test']} | {f['auroc']:.3f} | {f['threshold']:.3f} | "
                 f"{', '.join(f'{w:.2f}' for w in f['weights'])} | {f['inner_dasheng_auroc']:.3f} | {f['inner_xgb_auroc']:.3f} |")
    d = r["seven_second_diagnostic"]
    L += ["", "## Duration shift: dasheng on the central 7 s vs the whole answer", "",
          f"{d['note']}; {d['n_clips_longer_than_7s']} of {sp['n']} clips are longer than 7 s.", "",
          "| audio given to the model | AUROC | lies caught @0.5 | truths passed @0.5 | called deceptive | mean prob |",
          "|---|---|---|---|---|---|"]
    for key, name in (("full", "whole answer (<= 60 s, as trained)"), ("7s", "central 7 s (live length)")):
        c = d[key]["classwise"]
        L.append(f"| {name} | {d[key]['auroc']:.3f} | {c['lie_sensitivity']:.0%} | {c['truth_specificity']:.0%} | "
                 f"{c['predicted_lie_fraction']:.0%} | {d[key]['mean_prob']:.3f} |")
    L += ["", "These are source-domain (courtroom) estimates. They are not webcam accuracy; that needs a frozen, "
          "consented, labelled 7-second target-domain test set with unseen people."]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
