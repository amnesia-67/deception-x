#!/usr/bin/env python3
"""Zero-shot evaluation of the SERVED `analyze` policy on Box of Lies (out of domain; nothing is refit).

Box of Lies (Soldner, Perez-Rosas & Mihalcea, NAACL 2019): 25 Tonight Show / Late Night clips, utterance-level
veracity annotations. Videos are fetched separately (YouTube playlist named in the dataset README; see
~/deceptionx_cache/boxoflies/manifest.json for ids, SHA256 and the alignment check) and are not in git.

Each annotated veracity segment is one "answer":
  describer   the annotated describer (Describer_Object tier); talk time only where that tier is empty
  transcript  the describer's annotated utterances overlapping the segment (annotation transcripts, not ASR)
  audio       the segment's audio, 16 kHz mono float32 (contains the other speaker and the studio audience)
  frames      up to 128 time-uniform frames from the segment, JPEG q95, with timestamps
and is scored by src.ensemble_service.EnsembleService.predict exactly as the socket server would.

Two face variants (only the digest's face features differ; dasheng is audio-only):
  served    full frames: the served detector keeps the largest confident face, which may be either person
  describer frames cropped to the describer's side of the table (guest left, host right in every wide shot)

Label 1 = deceptive. Annotation rule (README): a segment is deceptive unless it accurately describes the object.

  bash scripts/run.sh scripts/eval_boxoflies.py [--limit N]
Writes runs/boxoflies_zero_shot/{segments.csv, summary.json, report.md}
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tuning import RUNS, classwise, group_bootstrap_ci, metrics  # noqa: E402

BOL = os.path.join(os.environ.get("DECEPTIONX_CACHE", os.path.expanduser("~/deceptionx_cache")), "boxoflies")
OUT = os.path.join(RUNS, "boxoflies_zero_shot")
SR = 16000
N_FRAMES = 128
FPS = 30.0


def tiers(path):
    r = ET.parse(path).getroot()
    ts = {t.attrib["TIME_SLOT_ID"]: int(t.attrib.get("TIME_VALUE", -1)) for t in r.iter("TIME_SLOT")}
    out = {}
    for t in r.iter("TIER"):
        out[t.attrib["TIER_ID"]] = [(ts[a.attrib["TIME_SLOT_REF1"]] / 1000.0, ts[a.attrib["TIME_SLOT_REF2"]] / 1000.0,
                                     (a.find("ANNOTATION_VALUE").text or "").strip())
                                    for a in t.iter("ALIGNABLE_ANNOTATION")]
    return out


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def guest_name(title):
    name = re.sub(r"^Box of Lies with\s+", "", title.replace("\xa0", " "))
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[-–]*\s*Part\s*\d+", "", name)
    return name.strip(" -")


def build_segments():
    index = [l.rstrip("\n").split("\t") for l in open(os.path.join(BOL, "index25.tsv"))]
    segs, skipped = [], {"no_speech": 0, "under_1s": 0}
    for idx, vid, _, title in index:
        i = int(idx)
        T = tiers(os.path.join(BOL, "annotations", f"{i}.BoL.An.Tr.eaf"))
        utt = {who: [(s, e, x) for s, e, x in T.get(f"{who}_verbal", []) if x] for who in ("Host", "Guest")}
        for k, (s, e, lab) in enumerate(T.get("Veracity_description", [])):
            if lab not in ("lie", "truthful"):
                continue
            if e - s < 1.0:
                skipped["under_1s"] += 1
                continue
            ov = {who: sum(overlap(s, e, a, b) for a, b, _ in rows) for who, rows in utt.items()}
            if max(ov.values()) == 0:
                skipped["no_speech"] += 1
                continue
            # The annotated describer (Describer_Object tier) when present; talk time is only a fallback, because
            # the listener often talks more (questions, banter) than the describer.
            dob = [(a, b, x) for a, b, x in T.get("Describer_Object", []) if x in ("Host", "Guest")
                   and overlap(s, e, a, b) > 0]
            if dob:
                who, describer_source = max(dob, key=lambda r: overlap(s, e, r[0], r[1]))[2], "Describer_Object"
            else:
                who, describer_source = max(ov, key=ov.get), "talk_time_fallback"
            if not any(overlap(s, e, a, b) > 0 for a, b, _ in utt[who]):
                skipped["no_speech"] += 1
                continue
            words, seen = [], set()
            for a, b, x in sorted(utt[who]):
                if overlap(s, e, a, b) > 0 and (a, x) not in seen:
                    seen.add((a, x))
                    words.append(x)
            segs.append({"video": i, "youtube_id": vid, "seg": k, "start": s, "end": e, "duration": e - s,
                         "label": int(lab == "lie"), "describer": who, "describer_source": describer_source,
                         "group": "Jimmy Fallon" if who == "Host" else guest_name(title),
                         "transcript": " ".join(words)})
    return segs, skipped


def video_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def segment_frames(cap, s, e):
    """Time-uniform frames in [s, e): read sequentially, keep the frame nearest each target time."""
    import cv2

    n = int(min(N_FRAMES, max(16, round((e - s) * FPS))))
    targets = s + (np.arange(n) + 0.5) * (e - s) / n
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, s - 0.5) * 1000)
    out, ts, j = [], [], 0
    prev = None
    while j < n:
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        while j < n and t >= targets[j]:
            pick = f if prev is None or abs(t - targets[j]) <= abs(prev[1] - targets[j]) else prev[0]
            out.append(pick)
            ts.append(targets[j] - s)
            j += 1
        prev = (f, t)
    return out, np.asarray(ts)


def encode(frames, side=None):
    import cv2

    enc = []
    for f in frames:
        if side is not None:
            w = f.shape[1]
            f = f[:, : w // 2] if side == "left" else f[:, w // 2:]
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])  # BGR in, as cv2 decodes
        enc.append(base64.b64encode(buf.tobytes()).decode())
    return enc


def summarise(rows, key, thr):
    y = np.array([r["label"] for r in rows])
    p = np.array([r[key] for r in rows], float)
    g = np.array([r["group"] for r in rows])
    m = metrics(y, p)
    ci = group_bootstrap_ci(y, p, g)
    cw = classwise(y, p, thr)
    return {"n": int(len(y)), "n_lie": int(y.sum()), "n_people": int(len(set(g))), "auroc": m["auroc"],
            "auroc_ci95_person_bootstrap": ci["auroc_ci95"], "at_cutoff": cw}


def length_confound(rows):
    """How much of the ranking is just segment length? Duration-only baselines and within-length-band AUROCs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    y = np.array([r["label"] for r in rows])
    d = np.array([r["duration"] for r in rows], float)
    w = np.array([r["n_words"] for r in rows], float)
    p = np.clip(np.array([r["served_p"] for r in rows], float), 1e-4, 1 - 1e-4)
    g = np.array([r["group"] for r in rows])

    def cv_auc(X):
        pp = cross_val_predict(LogisticRegression(), X, y, groups=g, cv=GroupKFold(5), method="predict_proba")[:, 1]
        return float(roc_auc_score(y, pp))

    bands = []
    q = np.quantile(d, np.linspace(0, 1, 6))
    for lo, hi in zip(q[:-1], q[1:]):
        m = (d >= lo) & (d <= hi)
        bands.append({"lo_s": float(lo), "hi_s": float(hi), "n": int(m.sum()), "lie_rate": float(y[m].mean()),
                      "auroc": float(roc_auc_score(y[m], p[m])) if len(set(y[m])) == 2 else None})
    return {"median_duration_lie_s": float(np.median(d[y == 1])), "median_duration_truth_s": float(np.median(d[y == 0])),
            "auroc_duration_alone": float(roc_auc_score(y, d)), "auroc_word_count_alone": float(roc_auc_score(y, w)),
            "corr_model_log_duration": float(np.corrcoef(p, np.log(d))[0, 1]),
            "grouped_cv_duration_only": cv_auc(np.c_[np.log(d)]),
            "grouped_cv_duration_plus_model": cv_auc(np.c_[np.log(d), np.log(p / (1 - p))]),
            "within_duration_quintiles": bands}


def main():
    import cv2
    import torch

    from src.ensemble_service import EnsembleService

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    os.makedirs(OUT, exist_ok=True)
    segs, skipped = build_segments()
    if args.limit:
        segs = segs[: args.limit]
    print(f"{len(segs)} segments ({sum(s['label'] for s in segs)} lie); skipped {skipped}", flush=True)
    svc = EnsembleService(device="cuda")
    thr = svc.threshold
    rows, t0 = [], time.time()
    cur_vid, cap, audio = None, None, None
    for n, sg in enumerate(segs):
        if sg["video"] != cur_vid:
            path = os.path.join(BOL, "videos", f"{sg['video']:02d}_{sg['youtube_id']}.mp4")
            cap, audio, cur_vid = cv2.VideoCapture(path), video_audio(path), sg["video"]
        frames, ts = segment_frames(cap, sg["start"], sg["end"])
        a = audio[int(sg["start"] * SR): int(sg["end"] * SR)]
        base = {"timestamps": ts.tolist(), "audio_f32_b64": base64.b64encode(a.astype("<f4").tobytes()).decode(),
                "transcript": sg["transcript"], "duration_s": sg["duration"], "return_intermediates": True,
                "request_id": f"bol-{sg['video']}-{sg['seg']}", "source_id": "boxoflies"}
        row = {k: v for k, v in sg.items() if k != "transcript"} | {"n_words": len(sg["transcript"].split()),
                                                                     "n_frames": len(frames)}
        for variant, side in (("served", None), ("describer", "left" if sg["describer"] == "Guest" else "right")):
            r = svc.predict({**base, "frames": encode(frames, side)})
            im = r["intermediates"]
            row[f"{variant}_p"] = r["probability"]
            row[f"{variant}_dasheng"] = im["branch_raw"]["tuned_dasheng"]
            row[f"{variant}_digest"] = im["branch_raw"]["xgb_digest"]
            row[f"{variant}_face_rate"] = r["face"]["face_detect_rate"]
        rows.append(row)
        if n % 10 == 0 or n == len(segs) - 1:
            print(f"  {n + 1}/{len(segs)} v{sg['video']} seg{sg['seg']} {sg['duration']:.1f}s label={sg['label']} "
                  f"served={row['served_p']:.3f} describer={row['describer_p']:.3f} "
                  f"| {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(os.path.join(OUT, "segments.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    subsets = {"all": rows, "guest_describing": [r for r in rows if r["describer"] == "Guest"],
               "host_describing": [r for r in rows if r["describer"] == "Host"],
               "segments_at_least_7s": [r for r in rows if r["duration"] >= 7.0]}
    summ = {"protocol": "zero-shot: served ensemble (dasheng + digest, Platt, weights, cutoff) unchanged; Box of Lies "
                        "veracity segments; person-grouped bootstrap (guests by name, Fallon as one person)",
            "cutoff": thr, "skipped": skipped, "results": {}}
    for name, sub in subsets.items():
        if len(set(r["label"] for r in sub)) < 2:
            continue
        summ["results"][name] = {
            "final_served_faces": summarise(sub, "served_p", thr),
            "final_describer_faces": summarise(sub, "describer_p", thr),
            "dasheng_only_auroc": metrics(np.array([r["label"] for r in sub]),
                                          np.array([r["served_dasheng"] for r in sub]))["auroc"],
            "digest_only_auroc_served_faces": metrics(np.array([r["label"] for r in sub]),
                                                      np.array([r["served_digest"] for r in sub]))["auroc"],
            "digest_only_auroc_describer_faces": metrics(np.array([r["label"] for r in sub]),
                                                         np.array([r["describer_digest"] for r in sub]))["auroc"]}
    summ["length_confound"] = length_confound(rows)
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    lines = ["# Zero-shot: served `analyze` policy on Box of Lies", "",
             f"Cutoff {thr:.3f} (unchanged). Label 1 = deceptive. Skipped: {skipped}.", "",
             "| subset | faces | n (lies) | people | AUROC (95% person CI) | lies caught | truths passed | bal. acc | called deceptive |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, res in summ["results"].items():
        for fv in ("served", "describer"):
            s = res[f"final_{fv}_faces"]
            c = s["at_cutoff"]
            lo, hi = s["auroc_ci95_person_bootstrap"]
            lines.append(f"| {name} | {fv} | {s['n']} ({s['n_lie']}) | {s['n_people']} | {s['auroc']:.3f} ({lo:.2f}-{hi:.2f}) | "
                         f"{c['tp_lie']}/{c['tp_lie'] + c['fn_lie']} | {c['tn_truth']}/{c['tn_truth'] + c['fp_truth']} | "
                         f"{c['bal_acc']:.3f} | {c['predicted_lie_fraction']:.0%} |")
    lines += ["", "| subset | dasheng alone | digest alone (served faces) | digest alone (describer faces) |",
              "|---|---|---|---|"]
    for name, res in summ["results"].items():
        lines.append(f"| {name} | {res['dasheng_only_auroc']:.3f} | {res['digest_only_auroc_served_faces']:.3f} | "
                     f"{res['digest_only_auroc_describer_faces']:.3f} |")
    lc = summ["length_confound"]
    lines += ["", "## Length confound", "",
              f"Lies are long descriptions and truths are often short asides (median {lc['median_duration_lie_s']:.1f} s "
              f"vs {lc['median_duration_truth_s']:.1f} s). Segment duration alone gives AUROC "
              f"**{lc['auroc_duration_alone']:.3f}** (word count {lc['auroc_word_count_alone']:.3f}); the served score "
              f"correlates {lc['corr_model_log_duration']:.2f} with log duration.", "",
              f"Person-grouped 5-fold logistic regression: duration only AUROC {lc['grouped_cv_duration_only']:.3f}; "
              f"duration + served score {lc['grouped_cv_duration_plus_model']:.3f}.", "",
              "| duration band | n | lie rate | served-score AUROC within band |", "|---|---|---|---|"]
    for b in lc["within_duration_quintiles"]:
        auc = "-" if b["auroc"] is None else f"{b['auroc']:.3f}"
        lines.append(f"| {b['lo_s']:.1f}-{b['hi_s']:.1f} s | {b['n']} | {b['lie_rate']:.2f} | {auc} |")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
