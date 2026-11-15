#!/usr/bin/env python3
"""Fine-tune the served audio (dasheng) and video (V-JEPA 2.1 ViT-L) branches on the SunHacks webcam pilot.

Pilot: sunhacks-webcam-pilot-2026-11-13 (206 x 7 s webcam clips, 103 truth / 103 lie, participant self-report,
2 known people + 1 unidentified lie clip). Rules from its README are followed:
  * train only on `train` rows, select (ladder config, epoch) on `validation`, score `test` once;
  * the primary split holds out NEW RECORDINGS OF FAMILIAR PEOPLE, not new people;
  * leave-one-person-out (protocols/leave-one-person-out.csv): the held-out person is never used for selection;
    each fold reuses the config and epoch count chosen on the primary split; the unidentified clip is excluded;
  * no transcripts exist, so text features are not used here (ASR would be needed for the digest branch);
  * no metadata, path, split or participant field is ever a model input.
Preprocessing is the SERVED contract: frames go through src.live_frames.prepare_face_crops with their recorded
capture timestamps (JPEG q95, as a client would send them); audio is the canonical 16 kHz float32 WAV.

Conditions per backbone: served (unchanged deployed folds), pilot_only, rlt_plus_pilot (all 121 RLT clips added to
training; selection and testing on pilot only).

  bash scripts/run.sh scripts/pilot_finetune.py prep|zeroshot|tune|report|export|fixture [--backbones ...]
`export` re-trains the validation-selected (condition, config) per backbone with the recorded seed and fold, checks it
reproduces the recorded validation/test predictions, fits Platt + threshold on validation only and writes
models/pilot_webcam/<backbone>/ (see src/pilot_webcam.py). `fixture` scores fixtures/golden_v1 with those exports.
Writes ~/deceptionx_cache/pilot/cache (crops, audio) and runs/pilot_finetune/. MU3D/DOLOS are not involved.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tuning import LADDER, RUNS, SubParRun, classwise, metrics  # noqa: E402
from src.participants import alias, real  # noqa: E402

PILOT = os.path.join(os.environ.get("DECEPTIONX_CACHE", os.path.expanduser("~/deceptionx_cache")), "pilot",
                     "sunhacks-webcam-pilot-2026-11-13")
CACHE = os.path.join(os.path.dirname(PILOT), "cache")
OUT = os.path.join(RUNS, "pilot_finetune")
SR = 16000
PERSON_GROUP = {"P1": 100001, "P2": 100002}
UNRESOLVED = 100003


def manifest(name="manifest.csv"):
    return list(csv.DictReader(open(os.path.join(PILOT, name), encoding="utf-8")))


# ----------------------------------------------------------------------------- preprocessing (served contract)
def _trailing_frame_missing(video_path, ts, tol_s=0.005):
    """True if the container's frame times match the metadata timestamps minus the LAST one (within tol_s)."""
    import av

    with av.open(video_path) as c:
        vs = c.streams.video[0]
        pts = np.array([float(f.pts * vs.time_base) for f in c.decode(vs)])
    t = np.asarray(ts, float)[:-1]
    return len(pts) == len(t) and float(np.max(np.abs((pts - pts[0]) - (t - t[0])))) <= tol_s


def prep():
    import cv2
    import soundfile as sf

    from src.live_frames import prepare_face_crops

    for sub in ("frames", "audio", "meta"):
        os.makedirs(os.path.join(CACHE, sub), exist_ok=True)
    rows = manifest()
    t0 = time.time()
    for n, r in enumerate(rows):
        cid = r["clip_id"]
        if os.path.exists(os.path.join(CACHE, "meta", cid + ".json")):
            continue
        md = json.load(open(os.path.join(PILOT, r["metadata_path"])))
        ts = np.asarray(md["frame_timestamps_seconds"], float)
        cap = cv2.VideoCapture(os.path.join(PILOT, r["video_path"]))
        jpgs = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            jpgs.append(base64.b64encode(buf.tobytes()).decode())
        dropped_last = False
        if len(jpgs) == len(ts) - 1 and _trailing_frame_missing(os.path.join(PILOT, r["video_path"]), ts):
            ts, dropped_last = ts[:-1], True  # encoder lost the final frame; the other timestamps align exactly
            print(f"  {cid}: video has {len(jpgs)} frames for {len(ts) + 1} timestamps; the encoded frames align with "
                  f"all but the LAST timestamp (checked), so that one is dropped", flush=True)
        if len(jpgs) != len(ts):
            raise SystemExit(f"{cid}: decoded {len(jpgs)} frames but metadata has {len(ts)} timestamps")
        crops, fmeta, duration, info = prepare_face_crops({"frames": jpgs, "timestamps": ts.tolist()})
        a, sr = sf.read(os.path.join(PILOT, r["audio_path"]), dtype="float32")
        if sr != SR or a.ndim != 1:
            raise SystemExit(f"{cid}: audio must be 16 kHz mono (got {sr} Hz, shape {a.shape})")
        np.save(os.path.join(CACHE, "frames", cid + ".npy"), crops)
        np.save(os.path.join(CACHE, "audio", cid + ".npy"), a)
        json.dump({"face": info, "duration_s": duration, "n_frames": len(ts),
                   **({"dropped_trailing_timestamp": True} if dropped_last else {})},
                  open(os.path.join(CACHE, "meta", cid + ".json"), "w"))
        if n % 25 == 0:
            print(f"  {n + 1}/{len(rows)} {cid} face rate {info['face_detect_rate']:.2f} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    rates = [json.load(open(os.path.join(CACHE, "meta", r["clip_id"] + ".json")))["face"]["face_detect_rate"]
             for r in rows]
    print(f"prep done: {len(rows)} clips; face detect rate median {np.median(rates):.2f}, min {np.min(rates):.2f}")


# ----------------------------------------------------------------------------- store
class Store:
    """What src.tuning reads (ids/labels/groups/audio/frames/transcript) for pilot clips, optionally with RLT."""

    def __init__(self, with_rlt: bool):
        self.records, self._rlt = [], None
        for r in manifest():
            g = PERSON_GROUP.get(alias(r["participant_id"]), UNRESOLVED)
            self.records.append(SimpleNamespace(clip_id=r["clip_id"], label=int(r["label"]), group=g, transcript=""))
        if with_rlt:
            from src.clip_store import ClipStore

            self._rlt = ClipStore()
            self.records += list(self._rlt.records)
        self.by_id = {r.clip_id: r for r in self.records}

    @property
    def ids(self):
        return [r.clip_id for r in self.records]

    def labels(self, ids):
        return np.array([self.by_id[i].label for i in ids], dtype=np.int64)

    def groups(self, ids):
        return np.array([self.by_id[i].group for i in ids], dtype=np.int64)

    def _is_rlt(self, cid):
        return self._rlt is not None and cid in self._rlt.by_id

    def audio(self, cid):
        return self._rlt.audio(cid) if self._is_rlt(cid) else np.load(os.path.join(CACHE, "audio", cid + ".npy"),
                                                                      mmap_mode="r")

    def frames(self, cid):
        return self._rlt.frames(cid) if self._is_rlt(cid) else np.load(os.path.join(CACHE, "frames", cid + ".npy"),
                                                                       mmap_mode="r")

    def transcript(self, cid):
        return self.by_id[cid].transcript


# ----------------------------------------------------------------------------- served (unchanged) models
def zeroshot():
    import torch

    from src.ensemble_service import EnsembleService, eval_audio
    from src.jepa_service import JEPAService

    rows = manifest()
    res = {r["clip_id"]: {} for r in rows}
    svc = EnsembleService(device="cuda")
    with torch.no_grad():
        for r in rows:
            a = torch.from_numpy(eval_audio(np.load(os.path.join(CACHE, "audio", r["clip_id"] + ".npy")))).cuda()[None]
            probs = []
            for key, state in svc.dasheng_folds:
                m = svc.dasheng_models[key]
                svc._load_strict(m, state)
                with svc._amp():
                    probs.append(float(torch.sigmoid(m.head(m.backbone(a)).float())))
            res[r["clip_id"]]["dasheng"] = float(np.mean(probs))
    del svc
    gc.collect()
    torch.cuda.empty_cache()
    vid = JEPAService(device="cuda")
    store = Store(with_rlt=False)
    for r in rows:
        crops = np.asarray(store.frames(r["clip_id"]))
        res[r["clip_id"]]["vjepa2_1_l"] = served_jepa_from_crops(vid, crops)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "zeroshot.json"), "w"), indent=1)
    print("zeroshot written", flush=True)


def served_jepa_from_crops(vid, crops):
    """Served ViT-L fold mean on already-prepared crops (same as JEPAService.predict after _prepare)."""
    import torch

    from src.jepa_deploy import load_fold_state_strict

    windows = vid._windows(crops)
    with torch.no_grad(), vid.lock:
        prefixes = None
        if vid.share_prefix:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefixes = vid._shared_prefixes(windows)
        per_fold = []
        for f in vid.folds:
            model = vid.models[f["key"]]
            load_fold_state_strict(model, f["state"])
            logits = []
            for v, win in enumerate(windows):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = vid._fold_tokens(model, f["key"], prefixes[v]) if prefixes is not None else \
                        model.backbone(win.cuda()[None])
                    logits.append(float(model.head(tokens).float()))
            per_fold.append(float(np.mean(logits)))
    return float(1 / (1 + np.exp(-np.mean(per_fold))))


# ----------------------------------------------------------------------------- fine-tuning
def split_ids(split):
    return [r["clip_id"] for r in manifest(f"manifests/{split}.csv")]


def lopo_ids(held_out):
    rows = [r for r in manifest("protocols/leave-one-person-out.csv") if r["fold"] == f"holdout_{real(held_out)}"]
    return ([r["clip_id"] for r in rows if r["split"] != "test"], [r["clip_id"] for r in rows if r["split"] == "test"])


def tune(backbones):
    import torch

    from src.tuning import GPUMonitor, RunReporter, limit_to_physical_vram, predict, train_one

    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(OUT, exist_ok=True)
    tr, va, te = split_ids("train"), split_ids("validation"), split_ids("test")
    for bb in backbones:
        for cond in ("pilot_only", "rlt_plus_pilot"):
            path = os.path.join(OUT, f"{bb}_{cond}.json")
            if os.path.exists(path):
                continue
            store = Store(with_rlt=cond == "rlt_plus_pilot")
            train = tr + ([r.clip_id for r in store.records if store._is_rlt(r.clip_id)] if cond == "rlt_plus_pilot" else [])
            reporter, gpu = RunReporter(os.path.join(OUT, f"{bb}_{cond}")), GPUMonitor()
            attempts, chosen = [], None
            for k, cfg in enumerate(LADDER[bb]):
                cfg = replace(cfg, seed=0)
                try:
                    model, state, best, _ = train_one(cfg, store, train, va, reporter, gpu, fold=200 + k)
                    attempts.append({"tag": cfg.tag, "val_auroc": best["val_auroc"], "epoch": best["epoch"]})
                    print(f"[{bb} {cond}] {cfg.tag}: val AUROC {best['val_auroc']:.3f} at epoch {best['epoch']}", flush=True)
                    if chosen is None or best["val_auroc"] > chosen[1]["val_auroc"]:
                        chosen = (cfg, best, state)
                    del model
                except SubParRun as e:
                    attempts.append({"tag": cfg.tag, "terminated": str(e)})
                    print(f"[{bb} {cond}] {cfg.tag}: terminated ({e})", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                if chosen is not None and chosen[1]["val_auroc"] >= 0.55:
                    break  # same rule as tune_fold: escalate the ladder only while validation is sub-par
            cfg, best, state = chosen
            from src.tuning import TunedModel

            torch.save({"config": asdict(cfg), "state": state},
                       os.path.join(OUT, f"{bb}_{cond}_state.pt"))  # selected primary-split model (train rows only)

            model = TunedModel(cfg)
            model.load_trainable_state(state)
            bs = 2 if bb == "vjepa2_1_l" else 8
            p_val, p_test = predict(model, store, va, bs), predict(model, store, te, bs)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            lopo = {}
            for person in ("P1", "P2"):
                ltr, lte = lopo_ids(person)
                ltr = ltr + (train[len(tr):] if cond == "rlt_plus_pilot" else [])
                rcfg = replace(cfg, max_epochs=max(1, int(best["epoch"])))
                m2, _, _, _ = train_one(rcfg, store, ltr, None, reporter, gpu, fold=300 + len(lopo), phase="refit")
                lopo[person] = {"ids": lte, "prob": predict(m2, store, lte, bs).tolist()}
                del m2
                gc.collect()
                torch.cuda.empty_cache()
            gpu.stop()
            json.dump({"backbone": bb, "condition": cond, "chosen": cfg.tag, "best": best, "attempts": attempts,
                       "val": {"ids": va, "prob": p_val.tolist()}, "test": {"ids": te, "prob": p_test.tolist()},
                       "lopo": lopo}, open(path, "w"), indent=1)


# ----------------------------------------------------------------------------- export (opt-in webcam_pilot profile)
def _recorded_micro_batch(run_dir, fold, tag):
    for line in open(os.path.join(OUT, run_dir, "metrics.jsonl")):
        r = json.loads(line)
        if r.get("fold") == fold and r.get("tag") == tag and r.get("phase") == "inner" and "micro_batch" in r:
            return r["micro_batch"]
    return None


def _prior_matched_threshold(p_cal, base_rate):
    """Share of validation clips called deceptive = training base rate (the served rule, on validation only)."""
    q = np.sort(np.asarray(p_cal, dtype=np.float64))
    k = int(round(len(q) * (1 - base_rate)))        # clips below the cutoff
    k = min(max(k, 1), len(q) - 1)
    return float((q[k - 1] + q[k]) / 2)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def export(backbones):
    """Re-create each backbone's validation-selected model, calibrate on validation, write models/pilot_webcam/."""
    import hashlib
    import shutil
    import tempfile

    import torch
    from sklearn.linear_model import LogisticRegression

    from src.jepa_deploy import canonical_json, state_schema
    from src.pilot_webcam import LABELS, MANIFEST_FORMAT, PILOT_DIR, PROFILE, video_contract
    from src.provenance import git_revision
    from src.tuning import GPUMonitor, RunReporter, TunedModel, limit_to_physical_vram, predict, train_one

    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    lab = {r["clip_id"]: int(r["label"]) for r in manifest()}
    tr, va, te = split_ids("train"), split_ids("validation"), split_ids("test")
    summary = json.load(open(os.path.join(OUT, "summary.json")))
    for bb in backbones:
        runs = {c: json.load(open(os.path.join(OUT, f"{bb}_{c}.json"))) for c in ("pilot_only", "rlt_plus_pilot")
                if os.path.exists(os.path.join(OUT, f"{bb}_{c}.json"))}
        if set(runs) != {"pilot_only", "rlt_plus_pilot"}:
            raise SystemExit(f"{bb}: both conditions must finish `tune` before selection")
        # Selection on validation AUROC only (ties -> the simpler pilot_only); test is never consulted here.
        cond = max(("pilot_only", "rlt_plus_pilot"), key=lambda c: (runs[c]["best"]["val_auroc"], c == "pilot_only"))
        d = runs[cond]
        k = [c.tag for c in LADDER[bb]].index(d["chosen"])
        cfg = replace(LADDER[bb][k], seed=0)
        store = Store(with_rlt=cond == "rlt_plus_pilot")
        train = tr + ([r.clip_id for r in store.records if store._is_rlt(r.clip_id)] if cond == "rlt_plus_pilot" else [])
        saved = os.path.join(OUT, f"{bb}_{cond}_state.pt")
        if os.path.exists(saved):
            ck = torch.load(saved, map_location="cpu", weights_only=True)
            state, best, how = ck["state"], d["best"], "state saved by `tune`"
        else:
            # Same call as `tune`: same seed, fold number, ladder config, data order and early stopping.
            reporter, gpu = RunReporter(os.path.join(OUT, f"export_{bb}")), GPUMonitor()
            _, state, best, _ = train_one(cfg, store, train, va, reporter, gpu, fold=200 + k)
            gpu.stop()
            how = "re-trained with the recorded seed, fold, config, data and early stopping (tune kept no state)"
        model = TunedModel(cfg)
        model.load_trainable_state(state)
        bs = 2 if bb == "vjepa2_1_l" else 8
        p_val, p_test = predict(model, store, va, bs), predict(model, store, te, bs)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        repro = {"how": how, "recorded_best_epoch": int(d["best"]["epoch"]), "export_best_epoch": int(best["epoch"]),
                 "recorded_val_auroc": float(d["best"]["val_auroc"]),
                 "recorded_micro_batch": _recorded_micro_batch(f"{bb}_{cond}", 200 + k, cfg.tag),
                 "export_micro_batch": _recorded_micro_batch(f"export_{bb}", 200 + k, cfg.tag)
                 if not os.path.exists(saved) else None,
                 "val_prob_max_abs_diff_vs_recorded": float(np.max(np.abs(p_val - np.array(d["val"]["prob"])))),
                 "test_prob_max_abs_diff_vs_recorded": float(np.max(np.abs(p_test - np.array(d["test"]["prob"]))))}
        print(f"[export {bb}] {cond} {cfg.tag}: {json.dumps(repro)}", flush=True)
        if repro["export_best_epoch"] != repro["recorded_best_epoch"]:
            print(f"[export {bb}] WARNING: best epoch differs from the recorded run; the manifest's metrics are the "
                  f"exported model's own", flush=True)
        # Calibration and threshold: validation clips only.
        yv, yt = np.array([lab[i] for i in va]), np.array([lab[i] for i in te])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_val)[:, None], yv)
        cal = {"A": float(lr.coef_[0, 0]), "B": float(lr.intercept_[0])}
        cal_val = _sigmoid(cal["A"] * _logit(p_val) + cal["B"])
        cal_test = _sigmoid(cal["A"] * _logit(p_test) + cal["B"])
        base_rate = float(np.mean([lab[i] for i in tr]))
        thr = _prior_matched_threshold(cal_val, base_rate)
        test_m = metrics(yt, p_test)
        lopo = {p: metrics(np.array([lab[i] for i in v["ids"]]), np.array(v["prob"]))["auroc"]
                for p, v in d["lopo"].items()}
        served = summary[f"{bb}_served"]
        config = json.loads(canonical_json(asdict(cfg)))
        target = os.path.join(PILOT_DIR, bb)
        if os.path.exists(target):
            raise SystemExit(f"{target} exists; refusing to overwrite an export (remove it deliberately first)")
        os.makedirs(PILOT_DIR, exist_ok=True)
        stage = tempfile.mkdtemp(prefix=f".{bb}-export-", dir=PILOT_DIR)
        try:
            torch.save({"config": config, "state": state}, os.path.join(stage, "fold0.pt"))
            data = open(os.path.join(stage, "fold0.pt"), "rb").read()
            meta = {
                "format": MANIFEST_FORMAT, "profile": PROFILE, "backbone": bb, "model": LABELS[bb],
                "folds": {"fold0.pt": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                                       "config": config, "state_schema": state_schema(state)}},
                "trained_on": {
                    "dataset": "sunhacks-webcam-pilot-2026-11-13 (206 x 7 s webcam clips, 2 people + 1 unidentified "
                               "clip; labels self-reported by participants)",
                    "condition": cond, "train_clips": len(train), "pilot_train_clips": len(tr),
                    "rlt_clips_added": len(train) - len(tr), "validation_clips": len(va), "test_clips": len(te),
                    "rule": "trained on train rows only; condition, config, epoch, calibration and threshold chosen "
                            "on validation only; test scored after export"},
                "selection": {"condition_val_auroc": {c: runs[c]["best"]["val_auroc"] for c in runs},
                              "ladder_attempts": d["attempts"], "chosen_config": cfg.tag,
                              "best_epoch": int(best["epoch"])},
                "calibration": {**cal, "method": "Platt: sigmoid(A * logit(raw) + B), fitted on the 31 validation "
                                                 "clips (the same clips used for selection, so optimistic)"},
                "threshold": thr,
                "threshold_provenance": f"prior-matched on validation: share called deceptive = training base rate "
                                        f"{base_rate:.3f} (the served rule, fitted on validation only)",
                "metrics": {
                    "validation_auroc_used_for_selection": metrics(yv, p_val)["auroc"],
                    "test": {"n": len(te), "auroc": test_m["auroc"], "auroc_ci95": boot_auc(yt, p_test),
                             "at_threshold": classwise(yt, cal_test, thr),
                             "what": "20 NEW recordings of the 2 FAMILIAR people (not new people)"},
                    "leave_one_person_out_auroc": {**lopo, "what": "recorded run: same config and epochs refit "
                                                                   "without that person; 2 people only"},
                    "served_courtroom_model_same_test": {"auroc": served["test"]["auroc"],
                                                         "auroc_ci95": served["test_ci"],
                                                         "leave_one_person_out_auroc": served["lopo"]}},
                "reproduction": repro,
                "code_revision": git_revision(),
                "caveats": [
                    "Not better than chance on held-out pilot clips: see metrics.test and leave_one_person_out_auroc.",
                    "Two people only; nothing here measures performance on new people.",
                    "Labels are participant self-reports, not verified.",
                    "Calibration and threshold come from 31 validation clips that were also used for selection.",
                    "Not suitable for any real-world decision about whether a person is lying.",
                ],
            }
            if bb == "vjepa2_1_l":
                from src.jepa_deploy import BASE_RELATIVE_PATH, file_sha256

                meta["contract"] = video_contract()
                bp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), BASE_RELATIVE_PATH)
                meta["base_encoder"] = {"path": BASE_RELATIVE_PATH, "size": os.path.getsize(bp),
                                        "sha256": file_sha256(bp)}
            open(os.path.join(stage, "deploy_meta.json"), "w", newline="\n").write(
                json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n")
            os.rename(stage, target)
        finally:
            if os.path.exists(stage):
                shutil.rmtree(stage)
        print(f"[export {bb}] wrote {target}: test AUROC {test_m['auroc']:.3f}, threshold {thr:.3f}", flush=True)


def fixture():
    """Expected webcam-pilot outputs on the generated golden fixture (parity reference for the Mac)."""
    import torch

    from src.pilot_webcam import PilotAudioScorer, PilotVideoService, available

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fx = os.path.join(root, "fixtures", "golden_v1")
    from scripts.make_golden_fixture import load_payload

    payload, files = load_payload()
    audio = np.frombuffer(open(os.path.join(fx, "audio_f32le_16k.bin"), "rb").read(), dtype="<f4")
    out = {"fixture": "golden_v1", "input_sha256": files, "device": torch.cuda.get_device_name(0)}
    have = available()
    if have["dasheng"]:
        keep = ("model", "manifest_sha256", "logit", "raw_score", "probability", "threshold", "verdict")
        bf = PilotAudioScorer(device="cuda")
        r1, r2 = bf.score(audio), bf.score(audio)
        del bf
        fp = PilotAudioScorer(device="cuda", amp_dtype=None)
        r3 = fp.score(audio)
        del fp
        out["audio_cuda_bf16"] = {k: r1[k] for k in keep}
        out["audio_cuda_fp32"] = {k: r3[k] for k in keep}
        out["audio_bf16_repeatability_logit_abs"] = abs(r1["logit"] - r2["logit"])
        out["audio_bf16_vs_fp32_logit_abs"] = abs(r1["logit"] - r3["logit"])
        out["audio_suggested_tolerance_logit"] = max(3 * out["audio_bf16_repeatability_logit_abs"],
                                                     1.5 * out["audio_bf16_vs_fp32_logit_abs"], 1e-4)
        out["audio_note"] = ("Mac CPU FP32 / MPS: compare the logit with audio_cuda_fp32 within "
                             "audio_suggested_tolerance_logit")
        gc.collect()
        torch.cuda.empty_cache()
    if have["vjepa2_1_l"]:
        vs = PilotVideoService(device="cuda")
        v1, v2 = vs.predict(dict(payload, action="jepa_video")), vs.predict(dict(payload, action="jepa_video"))
        out["video_cuda_bf16"] = {k: v1[k] for k in ("model", "manifest_sha256", "raw_score", "calibrated_probability",
                                                     "threshold", "per_fold_probability", "temporal_attention_32")}
        out["video_bf16_repeatability_abs"] = abs(v1["raw_score"] - v2["raw_score"])
    open(os.path.join(fx, "pilot_webcam_expected.json"), "w", newline="\n").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "input_sha256"}, indent=1)[:2000])

# ----------------------------------------------------------------------------- report
def boot_auc(y, p, n=2000, seed=0):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5]).tolist()


def report():
    lab = {r["clip_id"]: int(r["label"]) for r in manifest()}
    zs = json.load(open(os.path.join(OUT, "zeroshot.json")))
    lines = ["# Pilot fine-tuning (SunHacks webcam pilot, 2 people)", "",
             "Primary test = 20 NEW recordings of the 2 FAMILIAR people (not new people). Leave-one-person-out = the "
             "other person only, config and epochs fixed from the primary run. Clip-bootstrap CIs; with n=20 they are "
             "very wide.", "",
             "| branch | trained on | val AUROC (used for selection) | test AUROC (95% CI) | test acc @0.5 | "
             "held out P1 AUROC | held out P2 AUROC |", "|---|---|---|---|---|---|---|"]
    summ = {}
    va, te = split_ids("validation"), split_ids("test")
    for bb in ("dasheng", "vjepa2_1_l"):
        yv, yt = np.array([lab[i] for i in va]), np.array([lab[i] for i in te])
        pv = np.array([zs[i][bb] for i in va])
        pt = np.array([zs[i][bb] for i in te])
        lop = {p: (lambda ids: metrics(np.array([lab[i] for i in ids]), np.array([zs[i][bb] for i in ids]))["auroc"])(
            lopo_ids(p)[1]) for p in ("P1", "P2")}
        row = {"val": metrics(yv, pv)["auroc"], "test": metrics(yt, pt), "test_ci": boot_auc(yt, pt), "lopo": lop}
        summ[f"{bb}_served"] = row
        lines.append(f"| {bb} | served (courtroom only, unchanged) | {row['val']:.3f} | {row['test']['auroc']:.3f} "
                     f"({row['test_ci'][0]:.2f}-{row['test_ci'][1]:.2f}) | {row['test']['acc']:.2f} | "
                     f"{lop['P1']:.3f} | {lop['P2']:.3f} |")
        for cond in ("pilot_only", "rlt_plus_pilot"):
            path = os.path.join(OUT, f"{bb}_{cond}.json")
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            yv = np.array([lab[i] for i in d["val"]["ids"]])
            yt = np.array([lab[i] for i in d["test"]["ids"]])
            pt = np.array(d["test"]["prob"])
            lop = {p: metrics(np.array([lab[i] for i in v["ids"]]), np.array(v["prob"]))["auroc"]
                   for p, v in d["lopo"].items()}
            row = {"chosen": d["chosen"], "val": metrics(yv, np.array(d["val"]["prob"]))["auroc"],
                   "test": metrics(yt, pt), "test_ci": boot_auc(yt, pt), "test_at_0.5": classwise(yt, pt, 0.5),
                   "lopo": lop}
            summ[f"{bb}_{cond}"] = row
            lines.append(f"| {bb} | {cond} ({d['chosen']}) | {row['val']:.3f} | {row['test']['auroc']:.3f} "
                         f"({row['test_ci'][0]:.2f}-{row['test_ci'][1]:.2f}) | {row['test']['acc']:.2f} | "
                         f"{lop['P1']:.3f} | {lop['P2']:.3f} |")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "zeroshot", "tune", "report", "export", "fixture"])
    ap.add_argument("--backbones", nargs="+", default=["dasheng", "vjepa2_1_l"])
    a = ap.parse_args()
    if a.stage in ("tune", "export"):
        globals()[a.stage](a.backbones)
    else:
        globals()[a.stage]()
