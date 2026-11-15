#!/usr/bin/env python3
"""Length-controlled experiment: does adding Box of Lies (BoL) help on unseen people?

Every example is EXACTLY one 7 s window (the live answer length), so segment length cannot leak the label (in BoL
it does: lies are long descriptions, truths short asides; duration alone gives AUROC 0.87 there).

Windows
  RLT7  centred 7 s of each Real-Life Trials clip >= 7 s (118 of 121; 3 shorter truthful clips excluded).
        Transcript: the clip's full transcript (RLT has no word timing); the digest uses rate features only.
  BoL7  centred 7 s of each BoL veracity segment >= 7 s with describer speech; transcript = the annotated
        describer's utterances overlapping the window (utterance-level timing).
  Features come from the SERVED code (src.ensemble_service: same face crop, voice, text, emotion2vec, intents).

Design: separate 5-fold person-grouped splits per dataset (seed 0); fold k of the union = RLT fold k + BoL fold k.
For each fold, three training conditions use only the other folds: RLT-only, BoL-only, RLT+BoL. Every model is
scored on held-out people of BOTH datasets. Branches:
  digest   XGBoost on the 88 served digest features (exported hyperparameters, mean of 5 seeds)
  dasheng  LoRA-tuned dasheng via src.tuning.tune_fold (same ladder, inner split, early stopping, refit)
  mean     unweighted mean of the two branch probabilities (no fitted stacker)
Paired person-bootstrap deltas compare each condition with RLT-only on the same test windows.

  bash scripts/run.sh scripts/exp_7s_boxoflies.py extract|digest|dasheng|report|all
Writes runs/exp7s/
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_boxoflies import BOL, build_segments, encode, overlap, segment_frames, tiers, video_audio  # noqa
from src.tuning import RUNS, classwise, metrics, outer_folds  # noqa: E402

OUT = os.path.join(RUNS, "exp7s")
SR = 16000
WIN = 7.0
K = 5
CONDS = ("rlt_only", "bol_only", "rlt_plus_bol")
DATASETS = ("RLT7", "BoL7")


# ----------------------------------------------------------------------------- windows + served features
def rlt_windows():
    from src.clip_store import ClipStore

    st = ClipStore()
    out = []
    for r in st.records:
        a = np.asarray(st.audio(r.clip_id), np.float32)
        dur = len(a) / SR
        if dur < WIN:
            continue
        s = dur / 2 - WIN / 2
        out.append({"wid": f"rlt_{r.clip_id}", "dataset": "RLT7", "label": int(r.label), "group": f"rlt_{r.group}",
                    "video_path": r.video_path, "start": s, "transcript": r.transcript, "transcript_span_s": dur,
                    "_audio": a[int(s * SR): int(s * SR) + int(WIN * SR)]})
    return out


def bol_windows():
    segs, _ = build_segments()
    out, audio_cache = [], {}
    for sg in segs:
        if sg["duration"] < WIN:
            continue
        s = sg["start"] + sg["duration"] / 2 - WIN / 2
        e = s + WIN
        T = tiers(os.path.join(BOL, "annotations", f"{sg['video']}.BoL.An.Tr.eaf"))
        words, spans = [], []
        for a, b, x in sorted(T.get(f"{sg['describer']}_verbal", [])):
            if x and overlap(s, e, a, b) > 0 and x not in words:
                words.append(x)
                spans.append((a, b))
        if not words:
            continue
        span = max(b for _, b in spans) - min(a for a, _ in spans)
        path = os.path.join(BOL, "videos", f"{sg['video']:02d}_{sg['youtube_id']}.mp4")
        if path not in audio_cache:
            audio_cache.clear()
            audio_cache[path] = video_audio(path)
        a = audio_cache[path]
        out.append({"wid": f"bol_{sg['video']}_{sg['seg']}", "dataset": "BoL7", "label": sg["label"],
                    "group": f"bol_{sg['group']}", "video_path": path, "start": s, "transcript": " ".join(words),
                    "transcript_span_s": span,
                    "describer": sg["describer"], "_audio": a[int(s * SR): int(s * SR) + int(WIN * SR)].copy()})
    return out


def extract():
    import cv2
    import torch

    from src.ensemble_service import EnsembleService

    os.makedirs(os.path.join(OUT, "audio"), exist_ok=True)
    wins = rlt_windows() + bol_windows()
    for ds in DATASETS:
        sub = [w for w in wins if w["dataset"] == ds]
        print(f"{ds}: {len(sub)} windows, {sum(w['label'] for w in sub)} lies, {len(set(w['group'] for w in sub))} people",
              flush=True)
    svc = EnsembleService(device="cuda")
    names, X, t0 = None, [], time.time()
    for n, w in enumerate(wins):
        cap = cv2.VideoCapture(w["video_path"])
        frames, ts = segment_frames(cap, w["start"], w["start"] + WIN)
        cap.release()
        a = w.pop("_audio")
        np.save(os.path.join(OUT, "audio", w["wid"] + ".npy"), a)
        # duration_s only feeds txt_words_per_s: it must be the span the transcript covers, not the 7 s window
        # (RLT has whole-clip transcripts; words / 7 s would encode the whole clip's length).
        r = svc.predict({"frames": encode(frames), "timestamps": ts.tolist(), "duration_s": w["transcript_span_s"],
                         "audio_f32_b64": base64.b64encode(a.astype("<f4").tobytes()).decode(),
                         "transcript": w["transcript"], "return_intermediates": True})
        im = r["intermediates"]
        names = [k for k, _ in im["digest_88_ordered"]]
        X.append([np.nan if v is None else v for _, v in im["digest_88_ordered"]])
        w.update({"served_p": r["probability"], "served_dasheng": im["branch_raw"]["tuned_dasheng"],
                  "served_digest": im["branch_raw"]["xgb_digest"], "face_rate": r["face"]["face_detect_rate"],
                  "n_frames": len(frames)})
        if n % 25 == 0:
            print(f"  {n + 1}/{len(wins)} {w['wid']} | {(time.time() - t0) / 60:.1f} min", flush=True)
    del svc
    torch.cuda.empty_cache()
    # folds: person-grouped, stratified, per dataset
    for ds in DATASETS:
        idx = [i for i, w in enumerate(wins) if w["dataset"] == ds]
        y = np.array([wins[i]["label"] for i in idx])
        g = np.array([wins[i]["group"] for i in idx])
        for k, (_, te) in enumerate(outer_folds([wins[i]["wid"] for i in idx], y, g, k=K, seed=0)):
            for j in te:
                wins[idx[j]]["fold"] = k
    json.dump(wins, open(os.path.join(OUT, "windows.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "digest.npz"), X=np.asarray(X, np.float32), names=np.array(names))


def fix_rate():
    """Repair txt_words_per_s in an extraction made before duration_s = transcript span (the only feature that reads
    duration). Rebuilds the windows' transcript spans (no GPU), checks the stored rate is words / 7 s, and recomputes
    it as words / span with the served tokenizer. Everything else in the digest is unchanged."""
    from src.features_tabular import _tokens

    wins = json.load(open(os.path.join(OUT, "windows.json")))
    d = np.load(os.path.join(OUT, "digest.npz"))
    X, names = d["X"].copy(), list(d["names"])
    col = names.index("txt_words_per_s")
    fresh = {w["wid"]: w for w in rlt_windows() + bol_windows()}
    for i, w in enumerate(wins):
        f = fresh[w["wid"]]
        assert f["transcript"] == w["transcript"], w["wid"]
        n = len(_tokens(w["transcript"]))
        assert abs(X[i, col] - n / WIN) < 1e-3 * max(1.0, n / WIN), (w["wid"], X[i, col], n / WIN)
        w["transcript_span_s"] = f["transcript_span_s"]
        X[i, col] = n / max(f["transcript_span_s"], 1e-3)
    json.dump(wins, open(os.path.join(OUT, "windows.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "digest.npz"), X=X, names=np.array(names))
    for ds in DATASETS:
        m = np.array([w["dataset"] == ds for w in wins])
        print(f"{ds}: txt_words_per_s median {np.median(X[m, col]):.2f} (was words/7 s); "
              f"span median {np.median([w['transcript_span_s'] for w, k in zip(wins, m) if k]):.1f} s", flush=True)


def load():
    wins = json.load(open(os.path.join(OUT, "windows.json")))
    d = np.load(os.path.join(OUT, "digest.npz"))
    return wins, d["X"]


def train_mask(wins, cond, k):
    ok = {"rlt_only": ("RLT7",), "bol_only": ("BoL7",), "rlt_plus_bol": DATASETS}[cond]
    return np.array([w["dataset"] in ok and w["fold"] != k for w in wins])


# ----------------------------------------------------------------------------- digest branch
def digest():
    from scripts.train_stage2 import xgb_model

    wins, X = load()
    y = np.array([w["label"] for w in wins])
    fold = np.array([w["fold"] for w in wins])
    res = {}
    for cond in CONDS:
        p = np.full(len(wins), np.nan)
        for k in range(K):
            tr, te = train_mask(wins, cond, k), fold == k
            preds = []
            for seed in range(5):
                m = xgb_model(seed)
                m.fit(X[tr], y[tr])
                preds.append(m.predict_proba(X[te])[:, 1])
            p[te] = np.mean(preds, 0)
        res[cond] = p.tolist()
        print(f"digest {cond}: " + ", ".join(f"{ds} {metrics(y[m], p[m])['auroc']:.3f}" for ds in DATASETS
                                              for m in [np.array([w['dataset'] == ds for w in wins])]), flush=True)
    json.dump(res, open(os.path.join(OUT, "digest_oof.json"), "w"))


# ----------------------------------------------------------------------------- dasheng branch
class WindowStore:
    """The subset of ClipStore that src.tuning uses for an audio backbone, backed by the 7 s windows."""

    def __init__(self, wins):
        gid = {g: i for i, g in enumerate(sorted({w["group"] for w in wins}))}
        self.records = [SimpleNamespace(clip_id=w["wid"], label=w["label"], group=gid[w["group"]],
                                        transcript=w["transcript"]) for w in wins]
        self.by_id = {r.clip_id: r for r in self.records}

    @property
    def ids(self):
        return [r.clip_id for r in self.records]

    def labels(self, ids):
        return np.array([self.by_id[i].label for i in ids], dtype=np.int64)

    def groups(self, ids):
        return np.array([self.by_id[i].group for i in ids], dtype=np.int64)

    def audio(self, cid):
        return np.load(os.path.join(OUT, "audio", cid + ".npy"), mmap_mode="r")

    def transcript(self, cid):
        return self.by_id[cid].transcript


def dasheng():
    import gc

    import torch

    from src.tuning import GPUMonitor, RunReporter, limit_to_physical_vram, tune_fold

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    wins, _ = load()
    store = WindowStore(wins)
    fold = np.array([w["fold"] for w in wins])
    os.makedirs(os.path.join(OUT, "dasheng"), exist_ok=True)
    reporter, gpu = RunReporter(os.path.join(OUT, "dasheng")), GPUMonitor()
    for cond in CONDS:
        for k in range(K):
            path = os.path.join(OUT, "dasheng", f"{cond}_k{k}.json")
            if os.path.exists(path):
                continue
            tr_ids = [w["wid"] for w, m in zip(wins, train_mask(wins, cond, k)) if m]
            te_ids = [w["wid"] for w, m in zip(wins, fold == k) if m]
            print(f"== dasheng {cond} fold {k}: train {len(tr_ids)}, test {len(te_ids)}", flush=True)
            p, info, model, _ = tune_fold("dasheng", store, tr_ids, te_ids, 100 + 10 * CONDS.index(cond) + k,
                                          reporter, gpu, seed=0)
            json.dump({"test_ids": te_ids, "prob": [float(v) for v in p], "config": info["config"]["tag"],
                       "attempts": [(a["tag"], a["status"]) for a in info["attempts"]]}, open(path, "w"))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    gpu.stop()
    res = {}
    for cond in CONDS:
        p = {}
        for k in range(K):
            d = json.load(open(os.path.join(OUT, "dasheng", f"{cond}_k{k}.json")))
            p.update(zip(d["test_ids"], d["prob"]))
        res[cond] = [p[w["wid"]] for w in wins]
    json.dump(res, open(os.path.join(OUT, "dasheng_oof.json"), "w"))


# ----------------------------------------------------------------------------- report
def person_boot(y, ps, g, n=2000, seed=0):
    """AUROC CI per score and paired deltas (ps[i] - ps[0]) resampling people."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    by = {u: np.where(g == u)[0] for u in ug}
    aucs = []
    for _ in range(n):
        idx = np.concatenate([by[u] for u in rng.choice(ug, len(ug))])
        if len(set(y[idx])) < 2:
            continue
        aucs.append([roc_auc_score(y[idx], p[idx]) for p in ps])
    a = np.array(aucs)
    ci = [np.percentile(a[:, i], [2.5, 97.5]).tolist() for i in range(len(ps))]
    delta = [{"delta": float(np.mean(a[:, i] - a[:, 0])), "ci95": np.percentile(a[:, i] - a[:, 0], [2.5, 97.5]).tolist(),
              "p_better": float(np.mean(a[:, i] > a[:, 0]))} for i in range(1, len(ps))]
    return ci, delta


def report():
    from sklearn.metrics import roc_auc_score

    wins, _ = load()
    y = np.array([w["label"] for w in wins])
    g = np.array([w["group"] for w in wins])
    ds = np.array([w["dataset"] for w in wins])
    branches = {"digest": json.load(open(os.path.join(OUT, "digest_oof.json")))}
    if os.path.exists(os.path.join(OUT, "dasheng_oof.json")):
        branches["dasheng"] = json.load(open(os.path.join(OUT, "dasheng_oof.json")))
        branches["mean"] = {c: ((np.array(branches["digest"][c]) + np.array(branches["dasheng"][c])) / 2).tolist()
                            for c in CONDS}
    summ = {"protocol": __doc__.split("\n\n")[2].strip(), "n": {}, "results": {}, "served_zero_shot": {}}
    lines = ["# Length-controlled 7 s experiment: does Box of Lies help?", "",
             "Every example is exactly 7 s. Person-grouped 5-fold; scores are out-of-fold. Paired deltas are vs "
             "training on Real-Life Trials only, on the same test windows (person bootstrap).", ""]
    for d in DATASETS:
        m = ds == d
        summ["n"][d] = {"windows": int(m.sum()), "lies": int(y[m].sum()), "people": int(len(set(g[m])))}
        lines.append(f"- {d}: {m.sum()} windows, {y[m].sum()} lies, {len(set(g[m]))} people")
    bm = ds == "BoL7"
    sp = np.array([w["served_p"] for w in wins])
    summ["served_zero_shot"]["BoL7_auroc"] = float(roc_auc_score(y[bm], sp[bm]))
    summ["served_zero_shot"]["BoL7_at_cutoff"] = classwise(y[bm], sp[bm], 0.4238521751969923)
    lines += ["", f"Served model, unchanged, on BoL7: AUROC {summ['served_zero_shot']['BoL7_auroc']:.3f} "
                  "(RLT7 not shown: the served model was trained on those clips).", "",
              "| branch | test on | trained on | AUROC (95% person CI) | delta vs RLT-only (95% CI) | P(better) |",
              "|---|---|---|---|---|---|"]
    for b, res in branches.items():
        for d in DATASETS:
            m = ds == d
            ps = [np.array(res[c])[m] for c in CONDS]
            ci, delta = person_boot(y[m], ps, g[m])
            for i, c in enumerate(CONDS):
                auc = float(roc_auc_score(y[m], ps[i]))
                summ["results"].setdefault(b, {}).setdefault(d, {})[c] = {
                    "auroc": auc, "ci95": ci[i], **({"vs_rlt_only": delta[i - 1]} if i else {})}
                dl = "-" if i == 0 else f"{delta[i - 1]['delta']:+.3f} ({delta[i - 1]['ci95'][0]:+.2f} to {delta[i - 1]['ci95'][1]:+.2f})"
                pb = "-" if i == 0 else f"{delta[i - 1]['p_better']:.0%}"
                lines.append(f"| {b} | {d} | {c} | {auc:.3f} ({ci[i][0]:.2f}-{ci[i][1]:.2f}) | {dl} | {pb} |")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "fix_rate", "digest", "dasheng", "report", "all"])
    a = ap.parse_args()
    for st in (["extract", "digest", "dasheng", "report"] if a.stage == "all" else [a.stage]):
        globals()[st]()
