#!/usr/bin/env python3
"""Final pre-presentation LoRA tuning on sunhacks-webcam-pilot-850-v3 (850 x 7 s clips, 5 declared people).

Protocol (decided before any training):
  * P5 (50 clips) is the UNSEEN-PERSON test: none of those clips is used for training, selection,
    calibration or the cutoff. P5 is scored once at the end.
  * The other four people use the export's own 75/15/10 split ("classic"): train rows to fit, validation rows to
    choose everything, test rows (80) scored once and reported as previous (40, already evaluated in earlier
    experiments) and new (40, never evaluated) separately, as the export README asks. These are new recordings
    of FAMILIAR people, not unseen-person performance.
  * The unidentified clip stays training-only. No metadata, path, split or participant field is a model input;
    person grouping only forms same-person training pairs and per-person reports.

Branches: dasheng (audio), emotion2vec_plus_large (audio), V-JEPA 2.1 ViT-L (video), each LoRA-tuned from its
pre-pilot base with the served preprocessing contract. Qwen3 (text) is not tuned: the export has no transcripts
and no ASR model is available on this machine.
Conditions per branch (scripts/pilot_all.py): bce (deployed recipe) and pairwise (same-person lie/truth pairs).
Choice of condition per branch, then of the branch combination, uses validation within-person AUROC only.
Fusion = mean of validation-standardised logits of the chosen branches (no fitted weights, little to overfit);
calibration (Platt) and the prior-matched cutoff are fitted on validation.

  bash scripts/run.sh scripts/pilot_final.py run --backbones dasheng emotion2vec [--conditions bce pairwise]
  bash scripts/run.sh scripts/pilot_final.py fuse        # selection on validation, then test + P5 once
Writes runs/pilot_final/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.pilot_all as pa  # noqa: E402
import scripts.pilot_finetune as pf  # noqa: E402
from src.tuning import RUNS  # noqa: E402
from src.participants import alias  # noqa: E402

V3 = os.path.join(os.path.dirname(pf.PILOT), "sunhacks-webcam-pilot-850-v3-2026-11-15")
OUT = os.path.join(RUNS, "pilot_final")
HELD = "P5"  # pseudonyms P1..P5: src/participants.py
PEOPLE = ("P1", "P2", "P3", "P4", HELD)
GROUP = {"P1": 100001, "P2": 100002, "P3": 100004, "P4": 100005, HELD: 100006}
BRANCHES = ("dasheng", "emotion2vec", "vjepa2_1_l")
pa.PEOPLE = PEOPLE  # pilot_all helpers read this for pairing and per-person reports
pa.BS.update({"emotion2vec": 8})


def rows(name):
    return list(csv.DictReader(open(os.path.join(V3, name), encoding="utf-8")))


class Store:
    def __init__(self):
        self.records = [SimpleNamespace(clip_id=r["clip_id"], label=int(r["label"]), person=alias(r["participant_id"]),
                                        group=GROUP.get(alias(r["participant_id"]), 100003), transcript="",
                                        batch=r["recording_batch"],
                                        captured=json.load(open(os.path.join(V3, r["metadata_path"])))["captured_utc"])
                        for r in rows("manifest.csv")]
        self.by_id = {r.clip_id: r for r in self.records}

    @property
    def ids(self):
        return [r.clip_id for r in self.records]

    def labels(self, ids):
        return np.array([self.by_id[i].label for i in ids], dtype=np.int64)

    def groups(self, ids):
        return np.array([self.by_id[i].group for i in ids], dtype=np.int64)

    def audio(self, cid):
        return np.load(os.path.join(pf.CACHE, "audio", cid + ".npy"), mmap_mode="r")

    def frames(self, cid):
        return np.load(os.path.join(pf.CACHE, "frames", cid + ".npy"), mmap_mode="r")

    def transcript(self, cid):
        return ""


def splits(store):
    sp = {s: [r["clip_id"] for r in rows(f"manifests/{s}.csv")] for s in ("train", "validation", "test")}
    tr, va, te = ([c for c in sp[s] if store.by_id[c].person != HELD] for s in ("train", "validation", "test"))
    st = [r.clip_id for r in store.records if r.person == HELD]
    assert not ({store.by_id[c].person for c in tr + va + te} & {HELD}), "held-out person leaked into train/val/test"
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te) or set(st) & set(tr + va + te))
    assert all(os.path.exists(os.path.join(pf.CACHE, "meta", c + ".json")) for c in tr + va + te + st), \
        "run preprocessing first"
    return tr, va, te, st


# ----------------------------------------------------------------------------- training
def run(backbones, conditions):
    import torch

    from src.tuning import GPUMonitor, RunReporter, TunedModel, limit_to_physical_vram, predict

    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(OUT, exist_ok=True)
    store = Store()
    tr, va, te, st = splits(store)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | {HELD} (unseen) {len(st)}", flush=True)
    for bb in backbones:
        for cond in conditions:
            path = os.path.join(OUT, f"{bb}_{cond}.json")
            if os.path.exists(path):
                print(f"[{bb} {cond}] done already", flush=True)
                continue
            reporter, gpu = RunReporter(os.path.join(OUT, f"{bb}_{cond}")), GPUMonitor()
            cfg, state, best, attempts = pa.fit(cond, bb, store, tr, va, reporter, gpu, 700)
            gpu.stop()
            res = {"backbone": bb, "condition": cond, "dataset": os.path.basename(V3), "attempts": attempts}
            if cfg is not None:
                torch.save({"config": asdict(cfg), "state": state}, os.path.join(OUT, f"{bb}_{cond}_state.pt"))
                m = TunedModel(cfg)
                m.load_trainable_state(state)
                bs = pa.BS[bb]
                res.update({"chosen": cfg.tag, "epoch": int(best["epoch"]),
                            **{k: {"ids": ids, "prob": predict(m, store, ids, bs).tolist()}
                               for k, ids in (("val", va), ("test", te), ("unseen", st))}})
                del m
                gc.collect()
                torch.cuda.empty_cache()
                wp, _ = pa.within_person_auroc(store, va, res["val"]["prob"])
                print(f"[{bb} {cond}] chosen {cfg.tag} epoch {best['epoch']}: val within-person AUROC {wp:.3f}",
                      flush=True)
            json.dump(res, open(path, "w"), indent=1)


# ----------------------------------------------------------------------------- selection, fusion, evaluation
def _z(lg, mu, sd):
    return (lg - mu) / sd


def causal_session(store, ids, p, min_answers=4):
    """src.pilot_webcam.SessionBaseline replayed per person in recording order: each answer vs the median of that
    person's answers so far (including it). Returns calls (None until min_answers) aligned with ids."""
    from src.pilot_webcam import SessionBaseline

    lg = dict(zip(ids, pa._logit(p)))
    calls = {}
    for person in {store.by_id[c].person for c in ids}:
        sb = SessionBaseline(min_answers)
        for c in sorted([c for c in ids if store.by_id[c].person == person], key=lambda c: store.by_id[c].captured):
            v = sb.add(lg[c])["verdict"]
            calls[c] = None if v is None else v == "Deceptive"
    return [calls[c] for c in ids]


def evaluate(store, ids, p, cal, thr):
    from sklearn.metrics import roc_auc_score

    p = np.asarray(p, dtype=np.float64)
    y = store.labels(ids)
    pc = 1 / (1 + np.exp(-(cal["A"] * pa._logit(p) + cal["B"])))
    two = len(set(y)) == 2
    out = {"n": len(ids), "auroc": float(roc_auc_score(y, p)) if two else float("nan"),
           "auroc_ci95": pf.boot_auc(y, p) if two else None,
           "within_person_auroc": pa.within_person_auroc(store, ids, p)[0]}
    cs = causal_session(store, ids, p)
    keep = np.array([c is not None for c in cs])
    rules = {"global": (pc >= thr, np.ones(len(ids), bool)),
             "session_baseline_transductive": (pa.centred(store, ids, p) >= 0, np.ones(len(ids), bool)),
             "session_baseline_causal": (np.array([bool(c) for c in cs]), keep)}
    for name, (c, m) in rules.items():
        cc, yy = c[m], y[m]
        out[name] = {"n_with_verdict": int(m.sum()), "acc": float((cc == yy).mean()),
                     "bal_acc": float((cc[yy == 1].mean() + (~cc[yy == 0]).mean()) / 2),
                     "lies_caught": f"{int(cc[yy == 1].sum())}/{int((yy == 1).sum())}",
                     "truths_passed": f"{int((~cc[yy == 0]).sum())}/{int((yy == 0).sum())}",
                     "called_lie": float(cc.mean())}
    return out


def named_sets(store, va, te, st):
    return {"validation (selection; optimistic)": va, "test: all 4 familiar people": te,
            "test: previous 40 (used before)": [c for c in te if store.by_id[c].batch == "previous_400"],
            "test: new 40 (clean)": [c for c in te if store.by_id[c].batch == "new_450"],
            f"{HELD}: unseen person (50)": st}


def fuse():
    store = Store()
    tr, va, te, st = splits(store)
    yv = store.labels(va)
    runs = {}
    for bb in BRANCHES:
        for cond in ("bce", "pairwise"):
            p = os.path.join(OUT, f"{bb}_{cond}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                if "val" in d:
                    assert d["val"]["ids"] == va and d["test"]["ids"] == te and d["unseen"]["ids"] == st
                    runs[(bb, cond)] = d
    # 1) condition per branch: validation within-person AUROC (ties -> bce, the deployed recipe)
    chosen = {}
    for bb in BRANCHES:
        cands = [(pa.within_person_auroc(store, va, runs[(bb, c)]["val"]["prob"])[0], c == "bce", c)
                 for c in ("bce", "pairwise") if (bb, c) in runs]
        if cands:
            chosen[bb] = max(cands)[2]
    logits, standardisation = {}, {}
    for bb, c in chosen.items():
        d = runs[(bb, c)]
        lv = pa._logit(d["val"]["prob"])
        mu, sd = float(lv.mean()), float(lv.std() + 1e-9)
        standardisation[bb] = {"logit_mean": mu, "logit_sd": sd}
        logits[bb] = {k: _z(pa._logit(d[k]["prob"]), mu, sd) for k in ("val", "test", "unseen")}
    # 2) branch combination: validation within-person AUROC (ties -> fewer branches)
    combos = []
    for n in range(1, len(logits) + 1):
        for combo in itertools.combinations(sorted(logits), n):
            fv = np.mean([logits[b]["val"] for b in combo], 0)
            combos.append((pa.within_person_auroc(store, va, 1 / (1 + np.exp(-fv)))[0], -n, combo))
    combos.sort(reverse=True)
    best_combo = combos[0][2]
    fused = {k: np.mean([logits[b][k] for b in best_combo], 0) for k in ("val", "test", "unseen")}
    prob = {k: 1 / (1 + np.exp(-v)) for k, v in fused.items()}
    cal, thr = pa.global_rule(va, prob["val"], tr, store)

    sets = named_sets(store, va, te, st)
    lookup = {k: dict(zip(ids, prob[k])) for k, ids in (("val", va), ("test", te), ("unseen", st))}
    allp = {**lookup["val"], **lookup["test"], **lookup["unseen"]}
    results = {name: evaluate(store, ids, [allp[c] for c in ids], cal, thr) for name, ids in sets.items()}
    per_person = pa.per_person_calls(store, te + st, np.array([allp[c] for c in te + st]), cal, thr)
    # single-branch reference rows (same evaluation, each branch's chosen condition alone, its own val calibration)
    branch_rows, branch_rules = {}, {}
    for bb in logits:
        pv = 1 / (1 + np.exp(-logits[bb]["val"]))
        c2, t2 = pa.global_rule(va, pv, tr, store)
        branch_rules[bb] = {"calibration_on_standardised_logit": c2, "threshold": t2}
        for k, ids in (("test", te), ("unseen", st)):
            r = evaluate(store, ids, 1 / (1 + np.exp(-logits[bb][k])), c2, t2)
            branch_rows[f"{bb} ({chosen[bb]}) {k}"] = {
                "auroc": r["auroc"], "acc_global": r["global"]["acc"],
                "acc_session_baseline_causal": r["session_baseline_causal"]["acc"]}
    summary = {"chosen_condition": chosen, "branch_combination": list(best_combo),
               "combination_val_within_person_auroc": [(round(a, 4), list(c)) for a, _, c in combos],
               "standardisation": standardisation, "calibration": cal, "threshold": thr, "results": results,
               "per_person": per_person, "single_branch_reference": branch_rows, "single_branch_rules": branch_rules}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    lines = [f"# Final pilot tuning ({os.path.basename(V3)})", "",
             f"Chosen on validation only: conditions {chosen}; combination {list(best_combo)}. Cutoff "
             f"{thr:.4f} on the Platt-calibrated fused score. 'session (causal)' = each answer vs the median of that "
             f"person's answers so far in recording order, no verdict for the first 3 (src.pilot_webcam.SessionBaseline).",
             "", "| set | n | AUROC (95% CI) | acc (global cutoff) | bal acc | lies caught | truths passed | "
             "acc (session, causal) | answers with a session verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        ci, g, s = r["auroc_ci95"], r["global"], r["session_baseline_causal"]
        lines.append(f"| {name} | {r['n']} | {r['auroc']:.3f} ({ci[0]:.2f}–{ci[1]:.2f}) | {g['acc']:.2f} | "
                     f"{g['bal_acc']:.2f} | {g['lies_caught']} | {g['truths_passed']} | {s['acc']:.2f} | "
                     f"{s['n_with_verdict']} |")
    lines += ["", "| person (test rows; P5 = all 50 unseen) | n | AUROC | called Lie (global) | bal acc (global) | "
              "bal acc (session, median of all answers) |", "|---|---|---|---|---|---|"]
    for p, r in per_person.items():
        lines.append(f"| {p} | {r['n']} | {r['auroc']:.3f} | {r['global']['called_lie']:.0%} | "
                     f"{r['global']['bal_acc']:.2f} | {r['centred']['bal_acc']:.2f} |")
    lines += ["", "| single branch | AUROC | acc (global) | acc (session, causal) |", "|---|---|---|---|"]
    for k, r in branch_rows.items():
        lines.append(f"| {k} | {r['auroc']:.3f} | {r['acc_global']:.2f} | {r['acc_session_baseline_causal']:.2f} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ----------------------------------------------------------------------------- export (replaces the 206-clip profile)
def export():
    """Write models/pilot_webcam/ from the validation-chosen runs: one manifest per branch + fusion.json.

    Each saved state is reloaded and must reproduce the validation probabilities it was selected on. The previous
    profile (206-clip models) is replaced as a whole; it stays in git history.
    """
    import hashlib
    import shutil
    import tempfile

    import torch

    from src.jepa_deploy import BASE_RELATIVE_PATH, canonical_json, file_sha256, state_schema
    from src.pilot_webcam import (FUSION_FILE, FUSION_FORMAT, LABELS, MANIFEST_FORMAT, PILOT_DIR, PROFILE,
                                  read_fusion, read_manifest, video_contract)
    from src.provenance import git_revision
    from src.tuning import TuneConfig, TunedModel, limit_to_physical_vram, predict

    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True  # the numerics run() trained and scored with
    summ = json.load(open(os.path.join(OUT, "summary.json")))
    store = Store()
    tr, va, te, st = splits(store)
    sets = named_sets(store, va, te, st)
    trained_on = {"dataset": f"{os.path.basename(V3)} (850 x 7 s webcam clips, 5 declared people; labels "
                             f"self-reported by participants)",
                  "train_clips": len(tr), "validation_clips": len(va), "test_clips": len(te),
                  "unseen_person_test": {"person": HELD, "clips": len(st)},
                  "rule": f"trained on the 4 familiar people's train rows; condition, config, epoch, branch "
                          f"combination, calibration and cutoff chosen on their validation rows only; their test rows "
                          f"and all {len(st)} {HELD} clips (never trained on or used for any choice) scored once"}
    caveats = ["Labels are participant self-reports, not verified.",
               "Familiar-people test = new recordings of the 4 training people; only the P5 rows measure an "
               "unseen person, and that is ONE person (n=50).",
               "A single global cutoff can give a new person nearly the same verdict for every answer; prefer the "
               "session baseline (src.pilot_webcam.SessionBaseline) and see metrics for both.",
               "Not suitable for any real-world decision about whether a person is lying."]
    stage = tempfile.mkdtemp(prefix=".pilot_webcam-final-", dir=os.path.dirname(PILOT_DIR))
    new = os.path.join(stage, "pilot_webcam")
    os.makedirs(new)
    try:
        for bb, cond in summ["chosen_condition"].items():
            run_d = json.load(open(os.path.join(OUT, f"{bb}_{cond}.json")))
            ck = torch.load(os.path.join(OUT, f"{bb}_{cond}_state.pt"), map_location="cpu", weights_only=True)
            config, state = json.loads(canonical_json(ck["config"])), ck["state"]
            m = TunedModel(TuneConfig(**config))
            m.load_trainable_state(state)
            pv = predict(m, store, va, pa.BS[bb])
            diff = float(np.max(np.abs(pv - np.array(run_d["val"]["prob"]))))
            if diff > 2e-3:
                raise SystemExit(f"{bb}: saved state does not reproduce its validation scores (max diff {diff})")
            p_all = {k: dict(zip(run_d[k]["ids"], run_d[k]["prob"])) for k in ("val", "test", "unseen")}
            allp = {**p_all["val"], **p_all["test"], **p_all["unseen"]}
            cal, thr = pa.global_rule(va, np.array(run_d["val"]["prob"]), tr, store)
            # The server and PilotAudioScorer run without TF32: measure what that changes (no choice depends on it)
            torch.backends.cuda.matmul.allow_tf32 = False
            serving = {"setting": "CUDA bf16 autocast with TF32 matmul OFF (as PilotAudioScorer / the server run); "
                                  "the metrics above use TF32 ON (as trained and selected)",
                       "prob_max_abs_diff": {}, "verdict_changes_at_threshold": {}}
            for k, ids in (("val", va), ("test", te), ("unseen", st)):
                ps = predict(m, store, ids, pa.BS[bb])
                pr = np.array(run_d[k]["prob"])
                vc = lambda q: 1 / (1 + np.exp(-(cal["A"] * pa._logit(q) + cal["B"]))) >= thr  # noqa: E731
                serving["prob_max_abs_diff"][k] = float(np.max(np.abs(ps - pr)))
                serving["verdict_changes_at_threshold"][k] = f"{int((vc(ps) != vc(pr)).sum())}/{len(ids)}"
            torch.backends.cuda.matmul.allow_tf32 = True
            del m
            gc.collect()
            torch.cuda.empty_cache()
            d = os.path.join(new, bb)
            os.makedirs(d)
            torch.save({"config": config, "state": state}, os.path.join(d, "fold0.pt"))
            data = open(os.path.join(d, "fold0.pt"), "rb").read()
            meta = {"format": MANIFEST_FORMAT, "profile": PROFILE, "backbone": bb, "model": LABELS[bb],
                    "folds": {"fold0.pt": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                                           "config": config, "state_schema": state_schema(state)}},
                    "trained_on": {**trained_on, "condition": cond,
                                   "condition_meaning": {"bce": "binary cross-entropy (deployed recipe)",
                                                         "pairwise": "same-person (lie, truth) pair ranking loss"}[cond]},
                    "selection": {"attempts": run_d["attempts"], "chosen_config": run_d["chosen"],
                                  "best_epoch": run_d["epoch"],
                                  "criterion": "validation within-person AUROC (condition and ladder step)"},
                    "calibration": {**cal, "method": f"Platt: sigmoid(A * logit(raw) + B), fitted on the {len(va)} "
                                                     f"validation clips (also used for selection, so optimistic)"},
                    "threshold": thr,
                    "threshold_provenance": "prior-matched on validation: share called deceptive = training base rate "
                                            "(the served rule, fitted on validation only)",
                    "metrics": {name: evaluate(store, ids, [allp[c] for c in ids], cal, thr)
                                for name, ids in sets.items()},
                    "reproduction": {"validation_prob_max_abs_diff_reloaded_vs_selected": diff,
                                     "numerics": "CUDA bf16 autocast, TF32 matmul on (as trained and selected)",
                                     "serving_numerics": serving},
                    "code_revision": git_revision(), "caveats": caveats}
            if bb == "vjepa2_1_l":
                bp = os.path.join(ROOT, BASE_RELATIVE_PATH)
                meta["contract"] = video_contract()
                meta["base_encoder"] = {"path": BASE_RELATIVE_PATH, "size": os.path.getsize(bp),
                                        "sha256": file_sha256(bp)}
            open(os.path.join(d, "deploy_meta.json"), "w", newline="\n").write(
                json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n")
            read_manifest(bb, new)  # the strict loader accepts what was written
            print(f"[export] {bb} ({cond}): reload diff {diff:.2e}, threshold {thr:.4f}; serving numerics: "
                  f"{serving['prob_max_abs_diff']} verdict changes {serving['verdict_changes_at_threshold']}",
                  flush=True)
        branches = {bb: {"manifest_sha256": hashlib.sha256(open(os.path.join(new, bb, "deploy_meta.json"), "rb")
                                                           .read()).hexdigest(), **summ["standardisation"][bb]}
                    for bb in summ["branch_combination"]}
        fusion = {"format": FUSION_FORMAT, "profile": PROFILE, "branches": branches,
                  "rule": "mean over branches of (raw logit - logit_mean) / logit_sd, then Platt, then cutoff",
                  "calibration": summ["calibration"], "threshold": summ["threshold"],
                  "selection": {"criterion": "validation within-person AUROC over all branch combinations",
                                "combinations": summ["combination_val_within_person_auroc"]},
                  "metrics": summ["results"], "per_person": summ["per_person"], "trained_on": trained_on,
                  "serving_numerics": "metrics use TF32-on scores (as selected); each branch manifest's "
                                      "reproduction.serving_numerics gives the TF32-off differences and verdict changes",
                  "session_baseline": "src.pilot_webcam.SessionBaseline(min_answers=4) on fused_logit; "
                                      "metrics[...]['session_baseline_causal'] replays it in recording order",
                  "caveats": caveats}
        open(os.path.join(new, FUSION_FILE), "w", newline="\n").write(
            json.dumps(fusion, indent=1, sort_keys=True, allow_nan=False) + "\n")
        read_fusion(new)
        old = os.path.join(stage, "previous")
        if os.path.exists(PILOT_DIR):
            os.rename(PILOT_DIR, old)
        os.rename(new, PILOT_DIR)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"[export] wrote {PILOT_DIR}: branches {list(summ['chosen_condition'])}, fusion {summ['branch_combination']}")


def fixture():
    """Expected webcam-pilot outputs on fixtures/golden_v1 for every exported branch and the fusion (Mac parity).

    Keeps the dasheng keys of the previous file (audio_cuda_bf16 / audio_cuda_fp32 / audio_suggested_tolerance_logit)
    and adds audio_branches, video_cuda_bf16 and fusion_cuda_bf16.
    """
    import torch

    from scripts.make_golden_fixture import load_payload
    from src.pilot_webcam import (AUDIO_BRANCHES, PilotAudioScorer, PilotVideoService, available, fuse,
                                  read_fusion)

    fx = os.path.join(ROOT, "fixtures", "golden_v1")
    payload, files = load_payload()
    audio = np.frombuffer(open(os.path.join(fx, "audio_f32le_16k.bin"), "rb").read(), dtype="<f4")
    out = {"fixture": "golden_v1", "input_sha256": files, "device": torch.cuda.get_device_name(0), "audio_branches": {}}
    keep = ("model", "manifest_sha256", "logit", "raw_score", "probability", "threshold", "verdict")
    have, logits = available(), {}
    for bb in AUDIO_BRANCHES:
        if not have[bb]:
            continue
        s = PilotAudioScorer(device="cuda", backbone=bb)
        r1, r2 = s.score(audio), s.score(audio)
        del s
        s = PilotAudioScorer(device="cuda", amp_dtype=None, backbone=bb)
        r3 = s.score(audio)
        del s
        gc.collect()
        torch.cuda.empty_cache()
        rep, gap = abs(r1["logit"] - r2["logit"]), abs(r1["logit"] - r3["logit"])
        out["audio_branches"][bb] = {"cuda_bf16": {k: r1[k] for k in keep}, "cuda_fp32": {k: r3[k] for k in keep},
                                     "bf16_repeatability_logit_abs": rep, "bf16_vs_fp32_logit_abs": gap,
                                     "suggested_tolerance_logit": max(3 * rep, 1.5 * gap, 1e-4)}
        logits[bb] = r1["logit"]
    if "dasheng" in out["audio_branches"]:
        a = out["audio_branches"]["dasheng"]
        out.update({"audio_cuda_bf16": a["cuda_bf16"], "audio_cuda_fp32": a["cuda_fp32"],
                    "audio_bf16_repeatability_logit_abs": a["bf16_repeatability_logit_abs"],
                    "audio_bf16_vs_fp32_logit_abs": a["bf16_vs_fp32_logit_abs"],
                    "audio_suggested_tolerance_logit": a["suggested_tolerance_logit"],
                    "audio_note": "Mac CPU FP32 / MPS: compare each branch's logit with audio_branches[b].cuda_fp32 "
                                  "within its suggested_tolerance_logit (audio_cuda_* = dasheng, kept for old clients)"})
    if have["vjepa2_1_l"]:
        vs = PilotVideoService(device="cuda")
        v1, v2 = vs.predict(dict(payload, action="jepa_video")), vs.predict(dict(payload, action="jepa_video"))
        del vs
        out["video_cuda_bf16"] = {k: v1[k] for k in ("model", "manifest_sha256", "raw_score", "calibrated_probability",
                                                     "threshold", "per_fold_probability", "temporal_attention_32")}
        out["video_bf16_repeatability_abs"] = abs(v1["raw_score"] - v2["raw_score"])
        logits["vjepa2_1_l"] = float(pa._logit(v1["raw_score"]))
    fu = read_fusion()
    out["fusion_cuda_bf16"] = {"inputs_logit": {b: logits[b] for b in fu["branches"]},
                               **{k: v for k, v in fuse({b: logits[b] for b in fu["branches"]}, fu).items()
                                  if k != "standardised_logits"}}
    open(os.path.join(fx, "pilot_webcam_expected.json"), "w", newline="\n").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("input_sha256",)}, indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "fuse", "export", "fixture"])
    ap.add_argument("--backbones", nargs="+", default=list(BRANCHES))
    ap.add_argument("--conditions", nargs="+", default=["bce", "pairwise"])
    a = ap.parse_args()
    run(a.backbones, a.conditions) if a.stage == "run" else globals()[a.stage]()
