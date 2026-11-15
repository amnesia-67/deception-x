#!/usr/bin/env python3
"""Does V-JEPA 2.1 ViT-L video help the verdict? A fully nested, paired ablation on unseen people.

Same outer identity-grouped folds, inner folds and recipe as scripts/nested_deploy_eval.py. Inside every
outer-train set, ViT-L is LoRA-tuned per inner fold exactly like dasheng (ladder, early stopping, refit), and
two decision rules are fitted from inner OOF predictions only:

  without video : dasheng + digest          (the served policy)
  with video    : dasheng + digest + ViT-L  (same stacker + prior-matched threshold recipe)

Both rules score the SAME untouched outer-test people with the served aggregation (mean of inner fold
models; digest refit on all outer-train). The difference is reported with a paired identity bootstrap.
Video earns a place in the served verdict only if this comparison supports it; no weight is forced.

  bash scripts/run.sh scripts/nested_video_ablation.py [--video vjepa2_1_l]
Outputs: runs/nested_video_ablation/{summary.json, report.md, oof.csv}; inner runs are cached and resumable
in runs/nested_deploy/inner/ (shared with nested_deploy_eval.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.nested_deploy_eval import K_INNER, K_OUTER, SEED, boot_rates, inner_tune  # noqa: E402
from scripts.train_stage2 import (FEAT, TAB_VERSION, apply_stacker, fit_deploy_policy,  # noqa: E402
                                  tabular_fingerprint, xgb_model, xgb_oof)
from src.clip_store import ClipStore  # noqa: E402
from src.provenance import git_revision, tuning_fingerprint  # noqa: E402
from src.tuning import (LADDER, RUNS, GPUMonitor, RunReporter, classwise, group_bootstrap_ci,  # noqa: E402
                        limit_to_physical_vram, metrics, outer_folds)

OUT = os.path.join(RUNS, "nested_video_ablation")


def paired_delta_auc(y, a, b, groups, n_boot=2000, seed=0):
    """AUROC(b) - AUROC(a) on the same clips, identity-bootstrap CI and the share of resamples where b > a."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    d = np.array(d)
    return {"delta_auroc": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            "delta_auroc_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_of_resamples_with_video_better": float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="vjepa2_1_l")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(RUNS, "nested_deploy", "inner"), exist_ok=True)
    t0 = time.time()
    store = ClipStore()
    ids = store.ids
    y, g = store.labels(ids), store.groups(ids)
    d = np.load(os.path.join(FEAT, "tabular.npz"), allow_pickle=False)
    if list(d["ids"]) != ids or str(d["version"]) != TAB_VERSION or str(d["fingerprint"]) != tabular_fingerprint(store):
        raise SystemExit("tabular.npz is stale; run scripts/train_stage2.py first")
    X = d["X"].astype(np.float32)
    folds = outer_folds(ids, y, g, k=K_OUTER, seed=SEED)
    fps = {bb: tuning_fingerprint(store, bb, LADDER[bb], SEED, K_INNER) for bb in ("dasheng", args.video)}
    reporter, gpu = RunReporter(OUT), GPUMonitor()
    p_base, p_vid, p_vid_only = (np.full(len(ids), np.nan) for _ in range(3))
    thr_base, thr_vid = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    rows = []
    for k, (tr, te) in enumerate(folds):
        tr_ids, te_ids = [ids[i] for i in tr], [ids[i] for i in te]
        inner = outer_folds(tr_ids, y[tr], g[tr], k=K_INNER, seed=SEED + 1)
        s_in = {bb: np.full(len(tr), np.nan) for bb in fps}
        s_te = {bb: [] for bb in fps}
        print(f"== outer fold {k}: inner runs for {list(fps)}", flush=True)
        for j, (itr, iva) in enumerate(inner):
            for bb in fps:
                pv, pt, _ = inner_tune(bb, store, k, j, [tr_ids[i] for i in itr], [tr_ids[i] for i in iva], te_ids,
                                       reporter, gpu, fps[bb])
                s_in[bb][iva] = pv
                s_te[bb].append(pt)
        x_in = xgb_oof(X[tr], y[tr], g[tr], inner)
        clf = xgb_model(SEED)
        clf.fit(X[tr], y[tr])
        x_te = clf.predict_proba(X[te])[:, 1]
        d_te, v_te = np.mean(s_te["dasheng"], 0), np.mean(s_te[args.video], 0)
        pol_b = fit_deploy_policy(np.column_stack([s_in["dasheng"], x_in]), y[tr], inner)
        pol_v = fit_deploy_policy(np.column_stack([s_in["dasheng"], x_in, s_in[args.video]]), y[tr], inner)
        arr = lambda pol: {a: np.asarray(pol[a]) for a in ("A", "B", "w")}  # noqa: E731
        p_base[te] = apply_stacker(arr(pol_b), np.column_stack([d_te, x_te]))
        p_vid[te] = apply_stacker(arr(pol_v), np.column_stack([d_te, x_te, v_te]))
        p_vid_only[te] = v_te
        thr_base[te], thr_vid[te] = pol_b["threshold"], pol_v["threshold"]
        rows.append({"fold": k, "auroc_without_video": metrics(y[te], p_base[te])["auroc"],
                     "auroc_with_video": metrics(y[te], p_vid[te])["auroc"],
                     "video_weight": pol_v["w"][2], "inner_video_auroc": metrics(y[tr], s_in[args.video])["auroc"]})
        print(f"   fold {k}: without video {rows[-1]['auroc_without_video']:.3f} | with video "
              f"{rows[-1]['auroc_with_video']:.3f} (video weight {pol_v['w'][2]:.3f})", flush=True)
    gpu.stop()
    res = {"video_branch": args.video, "protocol": "fully nested paired ablation (see module docstring)",
           "n_clips": len(ids), "feature_version": TAB_VERSION, "folds": rows,
           "video_alone_mean_of_fold_models": {**metrics(y, p_vid_only), **group_bootstrap_ci(y, p_vid_only, g),
                                               "classwise": classwise(y, p_vid_only)},
           "without_video": {**metrics(y, p_base), **group_bootstrap_ci(y, p_base, g),
                             "at_threshold": {**classwise(y, p_base, thr_base),
                                              **boot_rates(y, (p_base >= thr_base).astype(int), g)}},
           "with_video": {**metrics(y, p_vid), **group_bootstrap_ci(y, p_vid, g),
                          "at_threshold": {**classwise(y, p_vid, thr_vid),
                                           **boot_rates(y, (p_vid >= thr_vid).astype(int), g)}},
           "paired": paired_delta_auc(y, p_base, p_vid, g), "minutes": (time.time() - t0) / 60,
           "code_revision": git_revision()}
    dl = res["paired"]
    res["decision"] = ("include video" if dl["delta_auroc_ci95"][0] > 0 else
                       "do not include video in the verdict: no evidence it helps on unseen people")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    with open(os.path.join(OUT, "oof.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "group", "label", "video_alone", "without_video", "with_video"])
        for i, c in enumerate(ids):
            w.writerow([c, int(g[i]), int(y[i]), f"{p_vid_only[i]:.5f}", f"{p_base[i]:.5f}", f"{p_vid[i]:.5f}"])
    v, a, b = res["video_alone_mean_of_fold_models"], res["without_video"], res["with_video"]
    L = [f"# Does {args.video} help? Fully nested paired ablation on unseen people", "",
         f"**Decision: {res['decision']}.** Delta AUROC (with - without video) {dl['delta_auroc']:+.3f}, 95% identity-"
         f"bootstrap CI {dl['delta_auroc_ci95'][0]:+.3f} to {dl['delta_auroc_ci95'][1]:+.3f}; video better in "
         f"{dl['share_of_resamples_with_video_better']:.0%} of resamples.", "",
         "| rule | AUROC | 95% CI | lies caught | truths passed | called deceptive |", "|---|---|---|---|---|---|"]
    for name, m in (("video alone (mean of fold models, @0.5)", {**v, "c": v["classwise"]}),
                    ("dasheng + digest (served)", {**a, "c": a["at_threshold"]}),
                    ("dasheng + digest + video", {**b, "c": b["at_threshold"]})):
        c = m["c"]
        L.append(f"| {name} | {m['auroc']:.3f} | {m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f} | "
                 f"{c['tp_lie']}/{c['tp_lie'] + c['fn_lie']} | {c['tn_truth']}/{c['tn_truth'] + c['fp_truth']} | "
                 f"{c['predicted_lie_fraction']:.0%} |")
    L += ["", "| outer fold | without video | with video | video weight | inner video AUROC |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['fold']} | {r['auroc_without_video']:.3f} | {r['auroc_with_video']:.3f} | "
                 f"{r['video_weight']:.3f} | {r['inner_video_auroc']:.3f} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
