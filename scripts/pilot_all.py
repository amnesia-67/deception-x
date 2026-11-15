#!/usr/bin/env python3
"""Fine-tune on ALL four pilot people (sunhacks-webcam-pilot-400-v2) and fix the per-person "always Lie" bias.

Why: the deployed webcam-pilot models score mostly WHO is on camera. Per person, their calibrated scores sit almost
entirely above or below the single global threshold (pilot ViT-L: P3 3% called Lie, P4 91%), while
within-person AUROC is near chance. A new person or session then gets the same verdict for every answer.

Two conditions per backbone, same LoRA ladder, same data:
  bce       the deployed recipe (src.tuning.train_one, BCE, early stopping on pooled validation AUROC), now trained
            on all four people's 300 primary train rows.
  pairwise  person-invariant: every micro-batch is one (lie, truth) pair from the SAME person and the loss is
            softplus(-(logit_lie - logit_truth)). A per-person offset cancels inside every pair, so the model gets no
            reward for recognising who is speaking. Early stopping on mean WITHIN-person validation AUROC.
Person grouping is used only to form training pairs and to report per-person results; it is never a model input.

Verdict rules reported on held-out clips:
  global    Platt + prior-matched threshold fitted on validation (the rule deployed today).
  centred   logit minus the median logit of that person's other clips in the same pool >= 0. Label-free; at serving
            time it becomes a per-session baseline (the session's own earlier answers).

Protocols (README of the export):
  primary   train 300 / select on validation 60 / test 40, reported for old (P1, P2) and new (P3, P4)
            recordings separately. Familiar people, NOT unseen-person performance.
  lopo      protocols/leave-one-person-out.csv, 4 folds. Each fold trains from the pre-pilot base on the other three
            people; epoch/config selection uses only those three people's primary-validation rows (never the
            held-out person); the held-out person's 94-105 clips are scored once.

  bash scripts/run.sh scripts/pilot_all.py run --backbones dasheng [vjepa2_1_l] [--conditions bce pairwise]
  bash scripts/run.sh scripts/pilot_all.py report
Writes runs/pilot_all/. Exporting/serving is a separate, explicit step.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.pilot_finetune as pf  # noqa: E402
from scripts.pilot_continue import V2, Store  # noqa: E402
from src.tuning import LADDER, RUNS, SubParRun  # noqa: E402
from src.participants import alias, real  # noqa: E402

OUT = os.path.join(RUNS, "pilot_all")
PEOPLE = ("P1", "P2", "P3", "P4")
BS = {"dasheng": 8, "vjepa2_1_l": 2}


def _rows(name):
    return list(csv.DictReader(open(os.path.join(V2, name), encoding="utf-8")))


def primary():
    return tuple([r["clip_id"] for r in _rows(f"manifests/{s}.csv")] for s in ("train", "validation", "test"))


def lopo(held):
    """(inner train, inner validation, test) for one fold; selection never sees the held-out person."""
    split = {r["clip_id"]: r["split"] for r in _rows("manifest.csv")}
    fold = [r for r in _rows("protocols/leave-one-person-out.csv") if r["fold"] == f"holdout_{real(held)}"]
    other = [r["clip_id"] for r in fold if r["split"] != "test"]
    test = [r["clip_id"] for r in fold if r["split"] == "test"]
    if {alias(r["participant_id"]) for r in fold if r["split"] == "test"} != {held} or \
            any(alias(r["participant_id"]) == held for r in fold if r["split"] != "test"):
        raise SystemExit(f"fold holdout_{held} is not a clean person hold-out")
    return [c for c in other if split[c] != "validation"], [c for c in other if split[c] == "validation"], test


# ----------------------------------------------------------------------------- metrics
def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def within_person_auroc(store, ids, prob):
    by = {}
    for c, p in zip(ids, prob):
        by.setdefault(store.by_id[c].person, []).append((store.by_id[c].label, p))
    aucs = {k: _auc([a for a, _ in v], [b for _, b in v]) for k, v in by.items()}
    aucs = {k: v for k, v in aucs.items() if np.isfinite(v)}
    return float(np.mean(list(aucs.values()))) if aucs else float("nan"), aucs


def centred(store, ids, prob):
    """Logit minus the median logit of the same person's clips in this pool (label-free per-session baseline)."""
    lg, out = _logit(prob), np.zeros(len(ids))
    person = np.array([store.by_id[c].person for c in ids])
    for p in set(person):
        m = person == p
        out[m] = lg[m] - np.median(lg[m])
    return out


def global_rule(val_ids, val_prob, train_ids, store):
    from sklearn.linear_model import LogisticRegression

    yv = store.labels(val_ids)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(val_prob)[:, None], yv)
    cal = {"A": float(lr.coef_[0, 0]), "B": float(lr.intercept_[0])}
    cv = 1 / (1 + np.exp(-(cal["A"] * _logit(val_prob) + cal["B"])))
    base = float(np.mean(store.labels([c for c in train_ids if store.by_id[c].person in PEOPLE])))
    return cal, pf._prior_matched_threshold(cv, base)


def per_person_calls(store, ids, prob, cal, thr):
    y = store.labels(ids)
    pc = 1 / (1 + np.exp(-(cal["A"] * _logit(prob) + cal["B"])))
    rules = {"global": pc >= thr, "centred": centred(store, ids, prob) >= 0}
    person = np.array([store.by_id[c].person for c in ids])
    out = {}
    for p in [q for q in PEOPLE if q in set(person)]:
        m = person == p
        row = {"n": int(m.sum()), "auroc": _auc(y[m], np.asarray(prob)[m])}
        for name, call in rules.items():
            c, yy = call[m], y[m]
            row[name] = {"called_lie": float(c.mean()), "lies_caught": float(c[yy == 1].mean()),
                         "truths_passed": float((~c[yy == 0]).mean()),
                         "bal_acc": float((c[yy == 1].mean() + (~c[yy == 0]).mean()) / 2)}
        out[p] = row
    return out


# ----------------------------------------------------------------------------- pairwise training
def train_pairwise(cfg, store, train_ids, val_ids, reporter, gpu, fold, phase="inner"):
    """Same optimiser/schedule/watchdog as src.tuning.train_one; micro-batch = one same-person (lie, truth) pair."""
    import torch
    import torch.nn.functional as F

    from src.provenance import state_hash
    from src.tuning import RunReporter, TunedModel, assert_on_gpu, forward_logits, predict

    torch.manual_seed(cfg.seed + fold)
    rng = np.random.default_rng(cfg.seed * 1000 + fold + 13)
    model = TunedModel(cfg)
    assert_on_gpu(model)
    by = {}
    for c in train_ids:
        r = store.by_id[c]
        if r.person in PEOPLE:
            by.setdefault(r.person, {0: [], 1: []})[r.label].append(c)
    n_pairs = sum(max(len(v[0]), len(v[1])) for v in by.values())
    accum = max(1, cfg.eff_batch // 2)
    params = [{"params": model.backbone.lora_parameters(), "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay},
              {"params": list(model.head.parameters()), "lr": cfg.lr_head, "weight_decay": 1e-4}]
    opt = torch.optim.AdamW(params)
    steps_per_epoch = math.ceil(n_pairs / accum)
    total = steps_per_epoch * cfg.max_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / steps_per_epoch) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    init = model.trainable_state()
    best, best_state, bad, first_loss, history = {"val_wp_auroc": -1.0, "epoch": 0}, None, 0, None, []
    print(f"  fold {fold} [{cfg.tag}] pairwise {phase}: {n_pairs} same-person pairs/epoch from "
          f"{sum(len(v[0]) + len(v[1]) for v in by.values())} clips, val {len(val_ids) if val_ids else '-'} | "
          f"micro-batch 2 (1 pair) x accum {accum} on {torch.cuda.get_device_name(0)}", flush=True)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        pairs = []
        for v in by.values():
            lies, truths = list(rng.permutation(v[1])), list(rng.permutation(v[0]))
            k = max(len(lies), len(truths))
            lies = (lies * math.ceil(k / len(lies)))[:k]
            truths = (truths * math.ceil(k / len(truths)))[:k]
            pairs += list(zip(lies, truths))
        pairs = [pairs[i] for i in rng.permutation(len(pairs))]
        losses, margins, gn = [], [], []
        opt.zero_grad(set_to_none=True)
        for i, (lie, truth) in enumerate(pairs):
            lg = forward_logits(model, store, [lie, truth], True, rng).float()
            margin = lg[0] - lg[1]
            loss = F.softplus(-margin)
            if not torch.isfinite(loss):
                raise SubParRun(f"non-finite loss at epoch {epoch} pair {i}")
            group_len = min(accum, len(pairs) - (i // accum) * accum)
            (loss / group_len).backward()
            if (i + 1) % accum == 0 or i + 1 == len(pairs):
                gn.append(float(torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            margins.append(margin.item())
        ep_loss, pair_acc = float(np.mean(losses)), float(np.mean(np.array(margins) > 0))
        first_loss = first_loss or ep_loss
        rec = {"backbone": cfg.backbone, "fold": fold, "tag": cfg.tag, "phase": f"pairwise-{phase}", "epoch": epoch,
               "train_loss": ep_loss, "train_auroc": pair_acc, "lr_lora": opt.param_groups[0]["lr"],
               "clips_per_s": 2 * len(pairs) / (time.time() - t0), "micro_batch": 2, "accum": accum}
        if val_ids:
            vp = predict(model, store, val_ids, BS[cfg.backbone])
            wp, _ = within_person_auroc(store, val_ids, vp)
            rec.update({"val_auroc": wp, "val_pooled_auroc": _auc(store.labels(val_ids), vp)})
            if wp > best["val_wp_auroc"] + 1e-6:
                best, best_state, bad = {"val_wp_auroc": wp, "epoch": epoch}, model.trainable_state(), 0
                rec["status"] = "best"
            else:
                bad += 1
                rec["status"] = f"no-improve {bad}/{cfg.patience}"
        else:
            rec["status"] = f"refit {epoch}/{cfg.max_epochs}"
        rec.update({"epoch_s": time.time() - t0, "torch_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "gpu": gpu.window(t0)})
        g, reason = rec["gpu"], None
        if g and g.get("gpu_util_mean", 100) < 5.0 and epoch >= 2:
            reason = f"GPU utilisation {g['gpu_util_mean']:.1f}% - not running on the GPU"
        elif epoch >= 3 and ep_loss > 1.5 * first_loss:
            reason = f"training loss diverged ({first_loss:.3f} -> {ep_loss:.3f})"
        elif val_ids and epoch >= cfg.min_epochs and pair_acc < 0.60 and best["val_wp_auroc"] < 0.55:
            reason = f"not learning: train pair accuracy {pair_acc:.3f}, best within-person val {best['val_wp_auroc']:.3f}"
        if reason:
            rec["status"] = "TERMINATED: " + reason
        reporter.log(rec)
        history.append(rec)
        print("   " + RunReporter.line(rec).replace("trainAUC", "pairAcc"), flush=True)
        if reason:
            raise SubParRun(reason)
        if val_ids and epoch >= cfg.min_epochs and bad >= cfg.patience:
            print(f"   early stop at epoch {epoch}; best epoch {best['epoch']} within-person val "
                  f"{best['val_wp_auroc']:.3f}", flush=True)
            break
    if not val_ids:
        best_state, best = model.trainable_state(), {"epoch": cfg.max_epochs}
    upd = float(sum((a - b).float().norm() ** 2 for k in ("lora", "head")
                    for a, b in zip([v for _, v in sorted(best_state[k].items())],
                                    [v for _, v in sorted(init[k].items())])) ** 0.5)
    if upd == 0.0:
        raise SubParRun("returned weights are identical to the initial weights (no learning)")
    best = {**best, "update_l2": upd, "returned_state_sha256": state_hash(best_state)}
    return model, best_state, best, history


# ----------------------------------------------------------------------------- driver
def fit(cond, bb, store, tr, va, reporter, gpu, fold0):
    """Ladder as before: escalate only while validation is sub-par. Returns (cfg, state, best, attempts)."""
    import torch

    from src.tuning import TunedModel, predict, train_one

    attempts, chosen = [], None
    for k, cfg in enumerate(LADDER[bb]):
        cfg = replace(cfg, seed=0)
        try:
            if cond == "bce":
                m, state, best, _ = train_one(cfg, store, tr, va, reporter, gpu, fold=fold0 + k)
            else:
                m, state, best, _ = train_pairwise(cfg, store, tr, va, reporter, gpu, fold=fold0 + k)
            del m
            gc.collect()
            torch.cuda.empty_cache()
            mm = TunedModel(cfg)
            mm.load_trainable_state(state)
            vp = predict(mm, store, va, BS[bb])
            del mm
            wp, _ = within_person_auroc(store, va, vp)
            score = wp  # the same yardstick for both conditions when choosing along the ladder
            attempts.append({"tag": cfg.tag, "epoch": int(best["epoch"]), "val_within_person_auroc": wp,
                             "val_pooled_auroc": _auc(store.labels(va), vp)})
            print(f"[{bb} {cond} f{fold0 + k}] {cfg.tag}: val within-person AUROC {wp:.3f}, pooled "
                  f"{attempts[-1]['val_pooled_auroc']:.3f}, epoch {best['epoch']}", flush=True)
            if chosen is None or score > chosen[3]:
                chosen = (cfg, state, best, score)
        except SubParRun as e:
            attempts.append({"tag": cfg.tag, "terminated": str(e)})
            print(f"[{bb} {cond} f{fold0 + k}] {cfg.tag}: terminated ({e})", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if chosen is not None and chosen[3] >= 0.55:
            break
    if chosen is None:
        return None, None, None, attempts
    return chosen[0], chosen[1], chosen[2], attempts


def run(backbones, conditions):
    import torch

    from src.tuning import GPUMonitor, RunReporter, TunedModel, limit_to_physical_vram, predict

    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(OUT, exist_ok=True)
    store = Store()
    tr, va, te = primary()
    for bb in backbones:
        for cond in conditions:
            path = os.path.join(OUT, f"{bb}_{cond}.json")
            if os.path.exists(path):
                print(f"[{bb} {cond}] done already", flush=True)
                continue
            reporter, gpu = RunReporter(os.path.join(OUT, f"{bb}_{cond}")), GPUMonitor()
            res = {"backbone": bb, "condition": cond, "data": V2}
            cfg, state, best, attempts = fit(cond, bb, store, tr, va, reporter, gpu, 500)
            res["primary"] = {"attempts": attempts}
            if cfg is not None:
                torch.save({"config": asdict(cfg), "state": state}, os.path.join(OUT, f"{bb}_{cond}_state.pt"))
                m = TunedModel(cfg)
                m.load_trainable_state(state)
                pv, pt = predict(m, store, va, BS[bb]), predict(m, store, te, BS[bb])
                del m
                cal, thr = global_rule(va, pv, tr, store)
                res["primary"].update({"chosen": cfg.tag, "epoch": int(best["epoch"]), "calibration": cal,
                                       "threshold": thr, "val": {"ids": va, "prob": pv.tolist()},
                                       "test": {"ids": te, "prob": pt.tolist()},
                                       "test_per_person": per_person_calls(store, te, pt, cal, thr)})
            res["lopo"] = {}
            for k, held in enumerate(PEOPLE):
                ltr, lva, lte = lopo(held)
                lcfg, lstate, lbest, latt = fit(cond, bb, store, ltr, lva, reporter, gpu, 600 + 10 * k)
                if lcfg is None:
                    res["lopo"][held] = {"attempts": latt}
                    continue
                m = TunedModel(lcfg)
                m.load_trainable_state(lstate)
                lv, lt = predict(m, store, lva, BS[bb]), predict(m, store, lte, BS[bb])
                del m
                gc.collect()
                torch.cuda.empty_cache()
                cal, thr = global_rule(lva, lv, ltr, store)
                res["lopo"][held] = {"attempts": latt, "chosen": lcfg.tag, "epoch": int(lbest["epoch"]),
                                     "train_clips": len(ltr), "val_clips": len(lva), "ids": lte, "prob": lt.tolist(),
                                     "per_person": per_person_calls(store, lte, lt, cal, thr)[held]}
                r = res["lopo"][held]["per_person"]
                print(f"[{bb} {cond}] unseen {held}: AUROC {r['auroc']:.3f} | global rule: called Lie "
                      f"{r['global']['called_lie']:.0%}, bal acc {r['global']['bal_acc']:.2f} | centred: bal acc "
                      f"{r['centred']['bal_acc']:.2f}", flush=True)
            gpu.stop()
            json.dump(res, open(path, "w"), indent=1)
    report()


def report():
    store = Store()
    lines = ["# All four pilot people: per-person bias and accuracy", "",
             "Primary test = 10 held-out recordings per person (familiar people). Unseen person = leave-one-person-out "
             "(trained on the other three, selection never saw that person). 'global' = deployed rule (Platt + "
             "prior-matched threshold on validation); 'centred' = per-person/session baseline (median of that "
             "person's clips in the pool). Called Lie should be near 50% for every person.", ""]
    for bb in ("dasheng", "vjepa2_1_l"):
        for cond in ("bce", "pairwise"):
            path = os.path.join(OUT, f"{bb}_{cond}.json")
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            p = d["primary"]
            lines.append(f"## {bb} — {cond} (primary: {p.get('chosen')}, epoch {p.get('epoch')})")
            lines += ["", "| person | set | n | AUROC | global: called Lie | global: bal acc | centred: bal acc |",
                      "|---|---|---|---|---|---|---|"]
            for person in PEOPLE:
                for label, row in (("primary test", (p.get("test_per_person") or {}).get(person)),
                                   ("unseen person", (d["lopo"].get(person) or {}).get("per_person"))):
                    if row:
                        lines.append(f"| {person} | {label} | {row['n']} | {row['auroc']:.3f} | "
                                     f"{row['global']['called_lie']:.0%} | {row['global']['bal_acc']:.2f} | "
                                     f"{row['centred']['bal_acc']:.2f} |")
            lines.append("")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--backbones", nargs="+", default=["dasheng"])
    ap.add_argument("--conditions", nargs="+", default=["bce", "pairwise"])
    a = ap.parse_args()
    run(a.backbones, a.conditions) if a.stage == "run" else report()
