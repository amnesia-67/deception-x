#!/usr/bin/env python3
"""Stage 2 on real data: Branch-1 fusion (ReDA+DARE+DMC), Branch-2 XGBoost digest, stacked ensemble.

Inputs: frozen features (scripts/extract_frozen_features.py) and the out-of-fold
probabilities of the LoRA-tuned backbones (runs/tune_<bb>/oof.csv).
Everything uses the SAME identity-grouped outer folds as stage 1, and every
model is fit only on the training folds of each split.

  bash scripts/run.sh scripts/train_stage2.py
Outputs in runs/stage2/: oof_all.csv, summary.json, report.md, metrics.jsonl
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clip_store import ClipStore  # noqa: E402
from src.features_tabular import (FEATURE_VERSION, INTENTS, face_features, text_features,  # noqa: E402
                                  voice_features)
from src.multimodal_fusion import Branch1MultimodalNetwork  # noqa: E402
from src.rlt_data import CACHE_DIR  # noqa: E402
from src.provenance import clip_cache_fingerprint, code_hash, sha256_file  # noqa: E402
from src.tuning import (RUNS, GPUMonitor, RunReporter, balanced_threshold, classwise,  # noqa: E402
                        group_bootstrap_ci, inner_split, limit_to_physical_vram, metrics, outer_folds,
                        prior_matched_threshold)

FEAT = os.path.join(CACHE_DIR, "features")
OUT = os.path.join(RUNS, "stage2")
BACKBONES = ["vjepa2", "vjepa2_1", "vjepa2_1_l", "dasheng", "emotion2vec", "qwen3"]  # missing OOF -> skipped
TAB_VERSION = FEATURE_VERSION
HEADLINE = "ensemble_stacked"  # pre-registered headline metric of the 7-branch table (see deploy_policy too)
# Branches of the served policy. Chosen from the earlier 7-branch stacker's weights on this same data (the
# other branches got < 2e-6), so this choice is not itself nested; everything fitted after it is.
DEPLOY_BRANCHES = ["tuned_dasheng", "xgb_digest"]
XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03, colsample_bytree=0.35, subsample=0.8,
                  reg_alpha=1.5, reg_lambda=5.0)


def tabular_fingerprint(st: ClipStore) -> str:
    """Binds tabular.npz to the clip cache, transcripts, labels/groups, feature code and frozen embeddings."""
    import hashlib

    parts = {"version": TAB_VERSION, "cache": clip_cache_fingerprint(st),
             "emotion_probs": sha256_file(os.path.join(FEAT, "emotion_probs.npy")),
             "qwen3_full": sha256_file(os.path.join(FEAT, "qwen3_full.npy"))}
    # Feature code + the build_tabular source (not all of train_stage2.py, which changes for unrelated reasons).
    import inspect
    parts["code"] = code_hash(["src/features_tabular.py"]) + hashlib.sha256(
        inspect.getsource(build_tabular).encode()).hexdigest()[:16]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# ============================================================================ tabular digest
def build_tabular(st: ClipStore, ids):
    path = os.path.join(FEAT, "tabular.npz")
    fp = tabular_fingerprint(st)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        if (list(d["ids"]) == list(ids) and str(d["version"]) == TAB_VERSION
                and "fingerprint" in d.files and str(d["fingerprint"]) == fp):
            return d["X"], list(d["names"])
        print("  tabular cache stale (ids / version / fingerprint of inputs or feature code changed) - rebuilding")
    rows = []
    emo = np.load(os.path.join(FEAT, "emotion_probs.npy"))
    q = np.load(os.path.join(FEAT, "qwen3_full.npy")).astype(np.float32)
    intent_emb = intent_embeddings()
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    sims = qn @ intent_emb.T
    intent_p = np.exp((sims - sims.max(1, keepdims=True)) / 0.05)
    intent_p /= intent_p.sum(1, keepdims=True)
    for i, cid in enumerate(ids):
        meta = st.meta(cid)
        r = {}
        r.update(text_features(st.transcript(cid), meta["duration_s"]))
        r.update(voice_features(np.asarray(st.audio(cid))))
        r.update(face_features(st.frames(cid), meta))
        for k, e in enumerate(["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised",
                               "unknown"]):
            r[f"emo_{e}"] = float(emo[i, k])
        for k in range(len(INTENTS)):
            r[f"intent_{k:02d}"] = float(intent_p[i, k])
        rows.append(r)
    names = sorted({k for r in rows for k in r})
    X = np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=np.float64)
    np.savez(path, X=X, names=np.array(names), ids=np.array(ids), version=np.array(TAB_VERSION),
             fingerprint=np.array(fp))
    return X, names


@torch.no_grad()
def intent_embeddings():
    from src.backbones import Qwen3Text

    m = Qwen3Text(lora=None).eval()
    # Qwen3-Embedding convention: queries carry the instruction, documents do not.
    batch = m.tok(INTENTS, padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e = m.model(**batch).last_hidden_state[:, -1].float()
    e = F.normalize(e, dim=-1).cpu().numpy()
    del m
    torch.cuda.empty_cache()
    return e


def xgb_model(seed=0):
    import xgboost as xgb

    # Hyperparameters fixed from Gemini.pdf (no tuning on test folds). Same settings as the exported model.
    return xgb.XGBClassifier(**XGB_PARAMS, tree_method="hist", device="cuda", eval_metric="logloss",
                             random_state=seed)


def xgb_oof(X, y, groups, folds, seed=0):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        clf = xgb_model(seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


# ============================================================================ honesty checks
def recording_conditions(st: ClipStore, ids):
    """Features that describe only HOW a clip was recorded (no behaviour). If these predict the label
    under identity-grouped CV, part of any 'deception' signal may just be courtroom/channel artefacts."""
    rows = []
    for cid in ids:
        m = st.meta(cid)
        a = np.asarray(st.audio(cid), dtype=np.float32)
        fr = st.frames(cid)
        border = np.concatenate([np.asarray(fr[::16, :24]).reshape(-1, 3), np.asarray(fr[::16, -24:]).reshape(-1, 3)])
        rows.append([m["src_wh"][0], m["src_wh"][1], m["fps"], m["duration_s"], m["face_detect_rate"],
                     float(np.sqrt(np.mean(a ** 2)) + 1e-9), float(np.percentile(np.abs(a), 10)),
                     *border.mean(0).tolist()])
    return np.array(rows, dtype=np.float64)


def permutation_null(X, y, groups, n_perm=100, seed=0):
    """OOF AUROC of the digest under the null, with EXACTLY the real procedure (same 400-tree model, folds
    rebuilt by the same grouped splitter for each permutation).

    Exchangeability unit: the clip label, permuted across clips while the identity grouping of the
    features is kept. Under H0 (features carry no label information) labels are exchangeable across clips;
    rebuilding the grouped folds per permutation keeps every person on one side of each split, as in the
    real run. Identity-block permutation is not used because identity sizes differ (1-20 clips) and five
    identities carry both labels, so whole-identity label vectors are not exchangeable."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ids = list(range(len(y)))
    aucs = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        f = outer_folds(ids, yp, groups, k=5, seed=int(rng.integers(1 << 30)))
        aucs.append(roc_auc_score(yp, xgb_oof(X, yp, groups, f)))
    return np.array(aucs)


def duplicate_audit(groups, ids, thr=0.995):
    """Near-duplicate clips ACROSS identity groups (would leak through grouped CV): cosine similarity of
    mean-pooled frozen audio (dasheng) and video (V-JEPA 2.1) embeddings."""
    out = {}
    for name in ("dasheng", "vjepa2_1"):
        E = np.load(os.path.join(FEAT, f"{name}.npy")).astype(np.float32)
        E = E.reshape(len(E), -1, E.shape[-1]).mean(1)
        E = E - E.mean(0)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        S = E @ E.T
        cross = groups[:, None] != groups[None, :]
        iu = np.triu_indices(len(E), 1)
        sims = S[iu][cross[iu]]
        pairs = [(float(S[i, j]), ids[i], ids[j]) for i, j in zip(*iu) if cross[i, j]]
        pairs.sort(reverse=True)
        same = S[iu][~cross[iu]]
        out[name] = {"n_cross_group_pairs_above": int((sims >= thr).sum()), "threshold": thr,
                     "max_cross_group_cos": float(sims.max()), "median_cross_group_cos": float(np.median(sims)),
                     "median_same_group_cos": float(np.median(same)) if len(same) else None,
                     "top_cross_group_pairs": [{"cos": round(c, 5), "a": a, "b": b} for c, a, b in pairs[:5]]}
    return out


# ============================================================================ Branch-1 fusion
class FusionNet(nn.Module):
    """Gemini.pdf Branch 1 (projectors + DARE + ReDA + DMC) with input dropout for a 121-clip regime.

    MMPDA (multi-source domain adaptation) is disabled (lambda_3 = 0): only one
    real source domain (Real-Life Trials) is available, so there is no domain to align.
    """

    def __init__(self, vdim, adim, tdim, p_drop=0.3):
        super().__init__()
        self.net = Branch1MultimodalNetwork(visual_dim=vdim, audio_dim=adim, text_dim=tdim, d=256,
                                            lambda_1=0.25, lambda_2=0.05, lambda_3=0.0)
        self.drop = nn.Dropout(p_drop)

    def forward(self, v, a, t):
        return self.net(self.drop(v), self.drop(a), self.drop(t))


def load_fusion_inputs():
    v = [np.load(os.path.join(FEAT, f"{b}.npy")).astype(np.float32) for b in ("vjepa2", "vjepa2_1")]
    nv = min(x.shape[1] for x in v)
    V = np.concatenate([x[:, :nv] for x in v], axis=-1)                      # (N, views, 32, 1792)
    A = np.concatenate([np.load(os.path.join(FEAT, f"{b}.npy")).astype(np.float32)[:, 0]
                        for b in ("dasheng", "emotion2vec")], axis=-1)       # (N, 32, 1792)
    T = np.load(os.path.join(FEAT, "qwen3.npy")).astype(np.float32)[:, 0, 0]  # (N, 128)
    return V, A, T


def standardize(train, *others):
    mu = train.reshape(-1, train.shape[-1]).mean(0)
    sd = train.reshape(-1, train.shape[-1]).std(0) + 1e-5
    return [(x - mu) / sd for x in (train, *others)]


def fusion_oof(V, A, T, y, groups, folds, reporter, gpu, seeds=(0, 1, 2, 3, 4), max_epochs=80, patience=12):
    dev = "cuda"
    oof = np.zeros(len(y))
    for f, (tr, te) in enumerate(folds):
        itr, iva = inner_split(y[tr], groups[tr], seed=f)
        a_tr, a_va = tr[itr], tr[iva]
        logits_te = np.zeros(len(te))
        for seed in seeds:
            torch.manual_seed(seed * 100 + f)
            rng = np.random.default_rng(seed * 100 + f)
            Vs = standardize(V[a_tr], V[a_va], V[te])
            As = standardize(A[a_tr], A[a_va], A[te])
            Ts = standardize(T[a_tr], T[a_va], T[te])
            Vtr, Vva, Vte = [torch.tensor(x, device=dev) for x in Vs]
            Atr, Ava, Ate = [torch.tensor(x, device=dev) for x in As]
            Ttr, Tva, Tte = [torch.tensor(x, device=dev) for x in Ts]
            ytr = torch.tensor(y[a_tr], dtype=torch.float32, device=dev)
            model = FusionNet(V.shape[-1], A.shape[-1], T.shape[-1]).to(dev)
            # PDF: AdamW lr 2e-4. Weight decay raised from the PDF's 1e-4 to 1e-2 for a 121-clip dataset.
            opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
            best, best_state, bad = (-1.0, 1e9), None, 0
            for ep in range(1, max_epochs + 1):
                model.train()
                perm = rng.permutation(len(a_tr))
                for s in range(0, len(perm), 16):
                    b = perm[s:s + 16]
                    view = torch.as_tensor(rng.integers(0, Vtr.shape[1], len(b)), device=dev)
                    vb = Vtr[torch.as_tensor(b, device=dev), view]
                    out = model(vb, Atr[b], Ttr[b])
                    losses = model.net.compute_loss(out, ytr[b])
                    opt.zero_grad(set_to_none=True)
                    losses["loss_total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(sum(model(Vva[:, k], Ava, Tva)["logit_fused"]
                                           for k in range(Vva.shape[1])) / Vva.shape[1])
                vm = metrics(y[a_va], pv.cpu().numpy())
                key = (np.nan_to_num(vm["auroc"], nan=0.0), -vm["log_loss"])
                if key > (best[0], -best[1]):
                    best, bad = (vm["auroc"], vm["log_loss"]), 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                lt = sum(model(Vte[:, k], Ate, Tte)["logit_fused"] for k in range(Vte.shape[1])) / Vte.shape[1]
            logits_te += lt.cpu().numpy()
            reporter.log({"backbone": "fusion", "fold": f, "seed": seed, "epoch": ep, "tag": "fusion",
                          "val_auroc": best[0], "val_loss": best[1], "gpu": gpu.window(time.time() - 30)})
        oof[te] = 1 / (1 + np.exp(-logits_te / len(seeds)))
        fm = metrics(y[te], oof[te])
        print(f"  fusion fold {f}: test AUROC {fm['auroc']:.3f} acc {fm['acc']:.3f}", flush=True)
    return oof


# ============================================================================ stacking
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_stacker(S, y, l2=1.0, iters=500):
    """Platt-scale each branch, then convex (non-negative, sum-to-1) weights - Gemini.pdf meta-learner."""
    from scipy.optimize import minimize

    k = S.shape[1]
    Z = logit(S)

    def nll(theta):
        A, B, w_raw = theta[:k], theta[k:2 * k], theta[2 * k:]
        w = np.exp(w_raw - w_raw.max())
        w /= w.sum()
        P = 1 / (1 + np.exp(-(Z * A + B)))
        p = np.clip(P @ w, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + l2 * 1e-2 * np.sum((A - 1) ** 2 + B ** 2)

    th0 = np.concatenate([np.ones(k), np.zeros(k), np.zeros(k)])
    res = minimize(nll, th0, method="L-BFGS-B", options={"maxiter": iters})
    th = res.x
    w = np.exp(th[2 * k:] - th[2 * k:].max())
    return {"A": th[:k], "B": th[k:2 * k], "w": w / w.sum()}


def apply_stacker(m, S):
    P = 1 / (1 + np.exp(-(logit(S) * m["A"] + m["B"])))
    return P @ m["w"]


def nested_stack_oof(S, y, folds):
    oof = np.zeros(len(y))
    weights = []
    for tr, te in folds:
        m = fit_stacker(S[tr], y[tr])
        oof[te] = apply_stacker(m, S[te])
        weights.append(m["w"])
    return oof, np.mean(weights, 0)


def fit_deploy_policy(S, y, folds):
    """The served decision rule, fitted from base-model OOF probabilities S (columns = DEPLOY_BRANCHES) of a
    training set only: Platt + convex stacker on all rows, and a prior-matched decision threshold computed on
    the stacker's own grouped-CV predictions (so the threshold never sees rows the stacker was fit on).

    Why prior-matched and not argmax balanced accuracy: on the 121-clip OOF the balanced-accuracy curve is
    flat (0.66-0.70 for thresholds 0.40-0.70), so its argmax (0.64) is noise and would call only 25% of answers
    deceptive. The prior-matched rule only corrects the documented shift (calibrated scores call ~41% of
    answers deceptive while 50% are lies). This rule was chosen after inspecting that full-data curve, i.e.
    before - not from - the nested evaluation; the argmax value is still reported for comparison."""
    m = fit_stacker(S, y)
    inner, _ = nested_stack_oof(S, y, folds)
    return {"A": m["A"].tolist(), "B": m["B"].tolist(), "w": m["w"].tolist(),
            "threshold": prior_matched_threshold(y, inner),
            "threshold_balanced_argmax_for_reference": balanced_threshold(y, inner)}


# ============================================================================ main
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    limit_to_physical_vram()
    os.makedirs(OUT, exist_ok=True)
    reporter = RunReporter(OUT)
    gpu = GPUMonitor()
    st = ClipStore()
    ids = json.load(open(os.path.join(FEAT, "ids.json")))
    assert ids == st.ids, "feature cache order mismatch"
    y, g = st.labels(ids), st.groups(ids)
    folds = outer_folds(ids, y, g, k=5, seed=0)
    branches = {}

    # ---- stage-1 tuned backbones (OOF from runs/tune_<bb>/oof.csv)
    for bb in BACKBONES:
        p = os.path.join(RUNS, f"tune_{bb}", "oof.csv")
        if not os.path.exists(p):
            print(f"  (no tuned OOF for {bb} yet)")
            continue
        rows = {r["clip_id"]: r for r in csv.DictReader(open(p))}
        if len(rows) != len(ids):
            print(f"  (partial OOF for {bb}: {len(rows)} clips) - skipped")
            continue
        assert all(int(rows[c]["fold"]) == f for f, (_, te) in enumerate(folds) for c in np.array(ids)[te]), \
            f"{bb} OOF used different folds"
        branches[f"tuned_{bb}"] = np.array([float(rows[c]["prob"]) for c in ids])

    # ---- Branch 2: real behavioural digest + within-identity z + XGBoost (GPU)
    X, names = build_tabular(st, ids)
    # Within-identity z-scores are NOT used: they are NaN for identities with < 3 clips, and identity
    # size correlates with the label in RLT (64% vs 38% deceptive), so XGBoost would learn group size.
    print(f"  tabular digest: {X.shape[1]} real features (no within-identity z)")
    branches["xgb_digest"] = xgb_oof(X, y, g, folds)

    # ---- honesty checks (not part of the ensemble)
    from sklearn.metrics import roc_auc_score
    C = recording_conditions(st, ids)
    p_cond = xgb_oof(C, y, g, folds)
    checks = {"recording_conditions_probe": {**metrics(y, p_cond), **group_bootstrap_ci(y, p_cond, g),
                                             "features": "src width/height, fps, duration, face-detect rate, audio "
                                                         "level & noise floor, crop border colour"}}
    null = permutation_null(X, y, g, n_perm=100)
    real_auc = roc_auc_score(y, branches["xgb_digest"])
    checks["xgb_digest_permutation_null"] = {
        "n_perm": int(len(null)), "null_auroc_mean": float(null.mean()), "null_auroc_p95": float(np.percentile(null, 95)),
        "real_auroc": float(real_auc), "p_value": float((1 + np.sum(null >= real_auc)) / (1 + len(null))),
        "procedure": "identical to the real digest (400 trees, same params); grouped folds rebuilt per permutation",
        "exchangeability_unit": "clip label across clips (see permutation_null docstring)"}
    checks["near_duplicate_audit"] = duplicate_audit(g, ids)
    print(f"  near-duplicate audit: {json.dumps({k: v['n_cross_group_pairs_above'] for k, v in checks['near_duplicate_audit'].items()})} "
          "cross-identity pairs above cos 0.995", flush=True)
    print(f"  recording-conditions probe AUROC {checks['recording_conditions_probe']['auroc']:.3f}; "
          f"digest permutation null mean {null.mean():.3f} (p95 {np.percentile(null, 95):.3f}) vs real {real_auc:.3f}",
          flush=True)

    # ---- Branch 1: ReDA + DARE + DMC fusion on frozen tokens
    V, A, T = load_fusion_inputs()
    branches["fusion_redadare"] = fusion_oof(V, A, T, y, g, folds, reporter, gpu)

    # ---- per-branch OOF metrics
    summary = {"n_clips": len(ids), "n_identities": int(len(set(g))), "headline": HEADLINE,
               "honesty_checks": checks, "branches": {},
               "feature_version": TAB_VERSION,
               "notes": ["The 7-branch ensemble_stacked number is NOT fully nested: each outer fold's meta-model is "
                         "fit on base-model OOF rows that were produced by base models trained on that outer test "
                         "fold. It is kept for comparison only. The served policy is evaluated with full base+meta "
                         "nesting by scripts/nested_deploy_eval.py (runs/nested_deploy/summary.json).",
                         "Accuracy uses a fixed 0.5 threshold; AUROC is threshold-free; 'classwise' gives the "
                         "confusion matrix, lie sensitivity, truth specificity and calibration at 0.5."]}
    for name, p in branches.items():
        summary["branches"][name] = {**metrics(y, p), **group_bootstrap_ci(y, p, g), "classwise": classwise(y, p)}

    # ---- stacked ensemble (nested: stacker fit on training folds only) + unfitted mean
    names_b = list(branches)
    S = np.column_stack([branches[n] for n in names_b])
    stack_oof, w = nested_stack_oof(S, y, folds)
    mean_oof = S.mean(1)
    summary["ensemble_stacked"] = {**metrics(y, stack_oof), **group_bootstrap_ci(y, stack_oof, g),
                                   "classwise": classwise(y, stack_oof),
                                   "mean_weights": dict(zip(names_b, map(float, w)))}
    summary["ensemble_mean"] = {**metrics(y, mean_oof), **group_bootstrap_ci(y, mean_oof, g),
                                "classwise": classwise(y, mean_oof)}
    final = fit_stacker(S, y)
    summary["deploy_stacker"] = {"branches": names_b, "A": final["A"].tolist(), "B": final["B"].tolist(),
                                 "w": final["w"].tolist()}
    # ---- the served policy, refit on its own two branches (evaluated fully nested by nested_deploy_eval.py)
    if all(b in branches for b in DEPLOY_BRANCHES):
        S2 = np.column_stack([branches[b] for b in DEPLOY_BRANCHES])
        summary["deploy_policy"] = {"branches": DEPLOY_BRANCHES, **fit_deploy_policy(S2, y, folds),
                                    "fit_on": "all 121 base-model OOF rows; threshold from the stacker's grouped CV"}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    with open(os.path.join(OUT, "oof_all.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["clip_id", "group", "label", *names_b, "stacked", "mean"])
        for i, c in enumerate(ids):
            wr.writerow([c, int(g[i]), int(y[i]), *[f"{branches[n][i]:.5f}" for n in names_b],
                         f"{stack_oof[i]:.5f}", f"{mean_oof[i]:.5f}"])
    lines = ["# Stage 2 - real-data results (5-fold identity-grouped CV, out-of-fold)", "",
             f"Digest feature version `{TAB_VERSION}`. Class 1 = deceptive. Rates at threshold 0.5.", "",
             "| branch | AUROC | 95% CI | accuracy | 95% CI | lies caught (TP/61) | truths passed (TN/60) "
             "| called deceptive | Brier |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in list(summary["branches"].items()) + [
            ("ENSEMBLE stacked, 7 branches (meta-level nesting only)", summary["ensemble_stacked"]),
            ("ENSEMBLE mean", summary["ensemble_mean"])]:
        c = m["classwise"]
        lines.append(f"| {name} | {m['auroc']:.3f} | {m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f} | "
                     f"{m['acc']:.3f} | {m['acc_ci95'][0]:.3f}-{m['acc_ci95'][1]:.3f} | {c['tp_lie']} "
                     f"({c['lie_sensitivity']:.0%}) | {c['tn_truth']} ({c['truth_specificity']:.0%}) | "
                     f"{c['predicted_lie_fraction']:.0%} | {c['brier']:.3f} |")
    lines += ["", f"Always-truthful baseline: accuracy {1 - y.mean():.3f}, balanced accuracy 0.500.",
              "The served policy's fully nested result is in runs/nested_deploy/report.md."]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    gpu.stop()


if __name__ == "__main__":
    main()
