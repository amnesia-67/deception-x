#!/usr/bin/env python3
"""LoRA-tune each backbone on Real-Life Trials with 5-fold identity-grouped CV.

  bash scripts/run.sh scripts/tune_backbones.py --backbones vjepa2 vjepa2_1 dasheng emotion2vec qwen3

Outputs per backbone in runs/tune_<backbone>/:
  oof.csv         clip_id, group, fold, label, prob   (out-of-fold, never trained on)
  fold<k>.pt      LoRA + head weights for that fold
  metrics.jsonl   per-epoch telemetry (also mirrored live to status.json)
  summary.json    OOF metrics + group-bootstrap 95% CIs + per-fold detail
  report.md       human-readable report
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clip_store import ClipStore  # noqa: E402
from src.provenance import tuning_fingerprint  # noqa: E402
from src.tuning import (LADDER, RUNS, GPUMonitor, RunReporter, group_bootstrap_ci,  # noqa: E402
                        limit_to_physical_vram, metrics, outer_folds, tune_fold)


def check_fold_file(ck, fp, f, accept_legacy: bool) -> str:
    """'ok' when the fold was produced by exactly the current cache/code/base weights/ladder; otherwise refuse
    (stale folds must not silently survive a pipeline change) unless the caller explicitly accepts it."""
    have = (ck.get("fingerprint") or {}).get("all")
    if have == fp["all"]:
        return "ok"
    why = "has no fingerprint (made before fingerprinting)" if have is None else "fingerprint differs"
    diff = [k for k in fp if k != "all" and (ck.get("fingerprint") or {}).get(k) != fp[k]]
    if not accept_legacy:
        raise SystemExit(f"fold{f}.pt {why} {diff}: rerun with --force to retrain, or --accept-stale to reuse it "
                         "knowingly")
    print(f"  WARNING fold{f}.pt {why} {diff}; reused because --accept-stale was given", flush=True)
    return "stale-accepted"


def run_backbone(bb: str, store: ClipStore, k: int, seed: int, folds_to_run=None, force: bool = False,
                 accept_stale: bool = False):
    run_dir = os.path.join(RUNS, f"tune_{bb}")
    os.makedirs(run_dir, exist_ok=True)
    reporter = RunReporter(run_dir)
    gpu = GPUMonitor()
    ids = store.ids
    y, g = store.labels(ids), store.groups(ids)
    folds = outer_folds(ids, y, g, k=k, seed=seed)
    fp = tuning_fingerprint(store, bb, LADDER[bb], seed, k)
    oof = np.full(len(ids), np.nan)
    fold_of = np.full(len(ids), -1)
    fold_info = []
    t0 = time.time()
    print(f"\n===== {bb}: {len(ids)} real clips, {len(set(g))} identities, {k}-fold grouped CV =====", flush=True)
    for f, (tr, te) in enumerate(folds):
        if folds_to_run is not None and f not in folds_to_run:
            continue
        if os.path.exists(os.path.join(run_dir, f"fold{f}.pt")) and not force:
            ck = torch.load(os.path.join(run_dir, f"fold{f}.pt"), map_location="cpu", weights_only=False)
            check_fold_file(ck, fp, f, accept_stale)
            print(f"  fold {f}: already done with the current fingerprint - skipping; use --force to redo", flush=True)
            continue
        assert not set(g[tr]) & set(g[te]), "identity leak between train and test"
        tr_ids = [ids[i] for i in tr]
        te_ids = [ids[i] for i in te]
        p, info, model, state = tune_fold(bb, store, tr_ids, te_ids, f, reporter, gpu, seed=seed)
        oof[te] = p
        fold_of[te] = f
        fm = metrics(y[te], p)
        info.update({"fold": f, "n_train": len(tr), "n_test": len(te), "test": fm})
        fold_info.append(info)
        torch.save({"state": state, "config": info["config"], "info": info, "test_ids": te_ids,
                    "test_prob": p.tolist(), "test_labels": y[te].tolist(), "fingerprint": fp},
                   os.path.join(run_dir, f"fold{f}.pt"))
        reporter.log({"backbone": bb, "fold": f, "epoch": -1, "event": "fold_done", "tag": info["config"]["tag"],
                      "test_auroc": fm["auroc"], "test_acc": fm["acc"], "inner_val_auroc": info["inner_val"]["val_auroc"]})
        print(f"  >> fold {f} TEST (held-out identities): AUROC {fm['auroc']:.3f} acc {fm['acc']:.3f} "
              f"(n={len(te)}) | chosen {info['config']['tag']} by inner-val AUROC {info['inner_val']['val_auroc']:.3f}",
              flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  memory after fold {f}: {torch.cuda.memory_allocated()/2**20:.0f} MB allocated", flush=True)
    gpu.stop()
    # Rebuild OOF from every fold file on disk (a crash or --only-folds run never loses earlier folds).
    fold_info = []
    for f in range(k):
        fold_path = os.path.join(run_dir, f"fold{f}.pt")
        if not os.path.exists(fold_path):
            continue
        ck = torch.load(fold_path, map_location="cpu", weights_only=False)
        _, te = folds[f]
        assert list(ck["test_ids"]) == [ids[i] for i in te], f"fold {f} file was made with different folds"
        ck["info"]["provenance"] = check_fold_file(ck, fp, f, accept_stale)
        oof[te] = np.asarray(ck["test_prob"])
        fold_of[te] = f
        fold_info.append(ck["info"])
    done = fold_of >= 0
    with open(os.path.join(run_dir, "oof.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "group", "fold", "label", "prob"])
        for i in np.where(done)[0]:
            w.writerow([ids[i], int(g[i]), int(fold_of[i]), int(y[i]), float(oof[i])])
    om = metrics(y[done], oof[done])
    ci = group_bootstrap_ci(y[done], oof[done], g[done])
    per_fold = [fi["test"]["auroc"] for fi in fold_info]
    summary = {"backbone": bb, "oof": om, **ci, "folds_done": int(len(fold_info)), "fingerprint": fp,
               "per_fold_auroc_mean": float(np.mean(per_fold)), "per_fold_auroc_sd": float(np.std(per_fold)),
               "folds": fold_info, "minutes": (time.time() - t0) / 60,
               "protocol": f"{k}-fold StratifiedGroupKFold over manually reviewed identity groups; identity-balanced "
                           "inner split for early stopping / retries; chosen config refit on all outer-train clips; "
                           "no test-fold selection"}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    lines = [f"# {bb} LoRA tuning on Real-Life Trials", "",
             f"Pooled OOF AUROC **{om['auroc']:.3f}** (95% CI {ci['auroc_ci95'][0]:.3f}-{ci['auroc_ci95'][1]:.3f}); "
             f"per-fold AUROC {np.mean(per_fold):.3f} +- {np.std(per_fold):.3f}; "
             f"accuracy@0.5 **{om['acc']:.3f}** (95% CI {ci['acc_ci95'][0]:.3f}-{ci['acc_ci95'][1]:.3f}), n={om['n']}",
             "", "| fold | test AUROC | test acc | n | config | inner-val AUROC | final model | attempts |",
             "|---|---|---|---|---|---|---|---|"]
    for fi in fold_info:
        att = "; ".join(f"{a['tag']}:{a['status']}" for a in fi["attempts"])
        lines.append(f"| {fi['fold']} | {fi['test']['auroc']:.3f} | {fi['test']['acc']:.3f} | {fi['n_test']} | "
                     f"{fi['config']['tag']} | {fi['inner_val']['val_auroc']:.3f} | {fi.get('final_model')} | {att} |")
    open(os.path.join(run_dir, "report.md"), "w").write("\n".join(lines) + "\n")
    print(f"===== {bb} OOF AUROC {om['auroc']:.3f} CI {ci['auroc_ci95']} | acc {om['acc']:.3f} "
          f"| {summary['minutes']:.1f} min =====", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["qwen3", "dasheng", "emotion2vec", "vjepa2_1", "vjepa2"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="redo folds that already have a fold<k>.pt")
    ap.add_argument("--accept-stale", action="store_true",
                    help="reuse fold files whose fingerprint (cache/code/base weights/ladder) does not match")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to train on CPU")
    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    store = ClipStore()
    out = {}
    for bb in args.backbones:
        out[bb] = run_backbone(bb, store, args.folds, args.seed,
                               set(args.only_folds) if args.only_folds is not None else None, args.force,
                               args.accept_stale)["oof"]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
