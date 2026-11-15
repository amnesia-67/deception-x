#!/usr/bin/env python3
"""Continue training the webcam-pilot models on the NEW people of the 400-clip export (P3, P4 only).

Data: sunhacks-webcam-pilot-400-v2-2026-11-13. Its previous 206 clips (P1, P2, 1 unidentified) are byte-identical
to the first export, so they add nothing new and are excluded from training here. Only `recording_batch == new_194`
rows are used: manifests/new_train.csv (145) for fitting, new_validation.csv (29) for every choice (config, epoch, and
whether further training beats the starting model at all), new_test.csv (20) scored once for reporting.

Start point ("train further"): the exported webcam-pilot checkpoint models/pilot_webcam/<backbone>/fold0.pt, which
was trained on P1/P2 train rows only and has never seen P3 or P4. Its architecture (LoRA rank/blocks) is
kept; only learning rates differ along the continuation ladder.

Also reported, never used for any choice:
  * old_test: the 20 earlier P1/P2 test clips (already evaluated before; a forgetting check only).
  * new-person check: refit from the same start on ONE new person's train rows (config and epoch count fixed from the
    primary run, so the held-out person is never used for selection), scored on ALL of the other person's clips.
    The starting checkpoint never saw either new person, so this is a genuine unseen-person test (n = 1 person).
Preprocessing is the served contract (scripts/pilot_finetune.py prep: face crops from recorded timestamps, 16 kHz WAV).
No metadata, path, split or participant field is a model input.

  bash scripts/run.sh scripts/pilot_continue.py check|prep|run|report [--backbones dasheng vjepa2_1_l]
Writes runs/pilot_continue/ (results JSON, selected state, metrics logs). Exporting is a separate, explicit step.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.pilot_finetune as pf  # noqa: E402
from src.participants import alias  # noqa: E402
from src.tuning import RUNS, SubParRun, classwise, metrics  # noqa: E402

V2 = os.path.join(os.path.dirname(pf.PILOT), "sunhacks-webcam-pilot-400-v2-2026-11-13")
OUT = os.path.join(RUNS, "pilot_continue")
NEW_PEOPLE = ("P3", "P4")
GROUP = {"P1": 100001, "P2": 100002, "P3": 100004, "P4": 100005}
MAX_EPOCHS = {"dasheng": 16, "vjepa2_1_l": 12}
BS = {"dasheng": 8, "vjepa2_1_l": 2}


def rows(name):
    return list(csv.DictReader(open(os.path.join(V2, name), encoding="utf-8")))


def new_split(split):
    r = rows(f"manifests/new_{split}.csv")
    bad = [x["clip_id"] for x in r if alias(x["participant_id"]) not in NEW_PEOPLE or x["recording_batch"] != "new_194"
           or x["split"] != split]
    if bad:
        raise SystemExit(f"new_{split}.csv has rows outside the new P3/P4 recordings: {bad[:3]}")
    return [x["clip_id"] for x in r]


def old_test_ids():
    return [x["clip_id"] for x in rows("manifests/test_previous_recordings.csv")]


class Store:
    """src.tuning store interface over the v2 manifest (crops/audio cached by clip id by pilot_finetune prep)."""

    def __init__(self):
        self.records = [SimpleNamespace(clip_id=r["clip_id"], label=int(r["label"]),
                                        group=GROUP.get(alias(r["participant_id"]), 100003), transcript="",
                                        person=alias(r["participant_id"])) for r in rows("manifest.csv")]
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


# ----------------------------------------------------------------------------- check / prep
def check():
    old = {r["clip_id"]: r for r in pf.manifest()}
    new = {r["clip_id"]: r for r in rows("manifest.csv")}
    prev = [c for c, r in new.items() if r["recording_batch"] == "previous_206"]
    key = ("video_sha256", "audio_sha256", "split", "label", "participant_id")
    changed = [c for c in prev if c not in old or any(new[c][k] != old[c][k] for k in key)]
    if set(prev) != set(old) or changed:
        raise SystemExit(f"previous 206 clips differ from the first export: {changed[:3]}")
    tr, va, te = new_split("train"), new_split("validation"), new_split("test")
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te) or (set(tr) | set(va) | set(te)) & set(old):
        raise SystemExit("new splits overlap each other or the previous clips")
    s = Store()
    print(f"check ok: previous 206 unchanged; new train/val/test {len(tr)}/{len(va)}/{len(te)}; people "
          f"{sorted({s.by_id[i].person for i in tr + va + te})}; old test {len(old_test_ids())}", flush=True)


def prep():
    pf.PILOT = V2  # same served preprocessing; clips already cropped from the first export are skipped by clip id
    pf.prep()


# ----------------------------------------------------------------------------- baselines
def served_scores(bb, ids):
    import torch

    from src.ensemble_service import EnsembleService, eval_audio
    from src.jepa_service import JEPAService

    store, out = Store(), {}
    if bb == "dasheng":
        svc = EnsembleService(device="cuda")
        with torch.no_grad():
            for cid in ids:
                a = torch.from_numpy(eval_audio(np.array(store.audio(cid)))).cuda()[None]
                probs = []
                for key, state in svc.dasheng_folds:
                    m = svc.dasheng_models[key]
                    svc._load_strict(m, state)
                    with svc._amp():
                        probs.append(float(torch.sigmoid(m.head(m.backbone(a)).float())))
                out[cid] = float(np.mean(probs))
    else:
        svc = JEPAService(device="cuda")
        for cid in ids:
            out[cid] = pf.served_jepa_from_crops(svc, np.asarray(store.frames(cid)))
    del svc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def start_point(bb):
    from src.pilot_webcam import read_manifest
    from src.tuning import TuneConfig

    meta, sha, ck = read_manifest(bb)
    return TuneConfig(**ck["config"]), ck["state"], {"manifest_sha256": sha, "config": ck["config"],
                                                     "best_epoch": meta["selection"]["best_epoch"]}


def warm_model(cfg, state):
    from src.tuning import TunedModel

    m = TunedModel(cfg)
    m.load_trainable_state(state)
    return m


# ----------------------------------------------------------------------------- continuation
def run(backbones):
    import torch

    from src.tuning import GPUMonitor, RunReporter, limit_to_physical_vram, predict, train_one

    limit_to_physical_vram()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(OUT, exist_ok=True)
    store = Store()
    tr, va, te, ot = new_split("train"), new_split("validation"), new_split("test"), old_test_ids()
    lab = dict(zip(store.ids, store.labels(store.ids)))
    y = lambda ids: np.array([lab[i] for i in ids])  # noqa: E731
    for bb in backbones:
        path = os.path.join(OUT, f"{bb}.json")
        if os.path.exists(path):
            print(f"[{bb}] {path} exists; skipping", flush=True)
            continue
        cfg0, state0, start = start_point(bb)
        bs = BS[bb]
        res = {"backbone": bb, "start": start, "data": {"train": len(tr), "validation": len(va), "test": len(te),
                                                         "people": list(NEW_PEOPLE)}}
        # Baselines on the same clips: served courtroom model, and the starting checkpoint with no further training.
        all_new = [r.clip_id for r in store.records if r.person in NEW_PEOPLE]
        sv = served_scores(bb, all_new + ot)
        sets = (("val", va), ("test", te), ("old_test", ot), ("all_new", all_new))
        res["served"] = {s: {"ids": ids, "prob": [sv[i] for i in ids]} for s, ids in sets}
        m = warm_model(cfg0, state0)
        res["start_model"] = {s: {"ids": ids, "prob": predict(m, store, ids, bs).tolist()} for s, ids in sets}
        del m
        gc.collect()
        torch.cuda.empty_cache()
        start_val = metrics(y(va), np.array(res["start_model"]["val"]["prob"]))["auroc"]
        print(f"[{bb}] start checkpoint (no further training): new-val AUROC {start_val:.3f}", flush=True)

        ladder = [replace(cfg0, seed=0, max_epochs=MAX_EPOCHS[bb], micro_batch=0, tag="cont_lr2e-4",
                          lr_lora=2e-4, lr_head=1e-3),
                  replace(cfg0, seed=0, max_epochs=MAX_EPOCHS[bb], micro_batch=0, tag="cont_lr5e-5",
                          lr_lora=5e-5, lr_head=3e-4)]
        reporter, gpu = RunReporter(os.path.join(OUT, bb)), GPUMonitor()
        attempts, chosen = [], None
        for k, cfg in enumerate(ladder):
            try:
                _, state, best, _ = train_one(cfg, store, tr, va, reporter, gpu, fold=400 + k,
                                              model=warm_model(cfg, state0))
                attempts.append({"tag": cfg.tag, "val_auroc": best["val_auroc"], "epoch": best["epoch"]})
                print(f"[{bb}] {cfg.tag}: new-val AUROC {best['val_auroc']:.3f} at epoch {best['epoch']}", flush=True)
                if chosen is None or best["val_auroc"] > chosen[1]["val_auroc"]:
                    chosen = (cfg, best, state)
            except SubParRun as e:
                attempts.append({"tag": cfg.tag, "terminated": str(e)})
                print(f"[{bb}] {cfg.tag}: terminated ({e})", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            if chosen is not None and chosen[1]["val_auroc"] >= 0.55:
                break  # same rule as before: escalate only while validation is sub-par
        res["attempts"] = attempts
        res["start_val_auroc"] = start_val
        # Selection on new validation only: further training must beat the starting checkpoint to be kept.
        if chosen is None or chosen[1]["val_auroc"] <= start_val:
            res["selected"] = "start_model (further training did not beat it on new validation)"
            res["continued"] = None
        else:
            cfg, best, state = chosen
            torch.save({"config": asdict(cfg), "state": state}, os.path.join(OUT, f"{bb}_state.pt"))
            m = warm_model(cfg, state)
            res["selected"] = f"continued ({cfg.tag}, epoch {best['epoch']})"
            res["continued"] = {"tag": cfg.tag, "best": {k: v for k, v in best.items() if k != "trace"},
                                "trace": best["trace"],
                                **{s: {"ids": ids, "prob": predict(m, store, ids, bs).tolist()} for s, ids in
                                   (("val", va), ("test", te), ("old_test", ot))}}
            del m
            gc.collect()
            torch.cuda.empty_cache()
            # New-person check: same start, one new person's train rows, fixed config/epochs, other person scored.
            res["new_person_holdout"] = {}
            for k, held in enumerate(NEW_PEOPLE):
                ltr = [i for i in tr if store.by_id[i].person != held]
                lte = [r.clip_id for r in store.records if r.person == held]
                rcfg = replace(cfg, max_epochs=max(1, int(best["epoch"])))
                m2, _, _, _ = train_one(rcfg, store, ltr, None, reporter, gpu, fold=410 + k,
                                        model=warm_model(rcfg, state0), phase="refit")
                res["new_person_holdout"][held] = {
                    "train_clips": len(ltr), "ids": lte, "prob": predict(m2, store, lte, bs).tolist()}
                del m2
                gc.collect()
                torch.cuda.empty_cache()
        gpu.stop()
        json.dump(res, open(path, "w"), indent=1)
        print(f"[{bb}] wrote {path}: {res['selected']}", flush=True)
    report()


# ----------------------------------------------------------------------------- report
def report():
    store = Store()
    lab = dict(zip(store.ids, store.labels(store.ids)))
    lines = ["# Continued training on the new pilot people (P3, P4)", "",
             "Train 145 / select on 29 / test 20 NEW recordings of P3 and P4 (P1/P2 excluded: no new "
             "clips). 'old test' = the 20 earlier P1/P2 test clips, already evaluated before (forgetting check "
             "only). 'Unseen person' = refit on one new person, scored on all clips of the other; the start checkpoint "
             "never saw either. Clip-bootstrap 95% CIs; n is small.", "",
             "| branch | model | new val AUROC (selection) | new test AUROC (95% CI) | new test acc @0.5 | "
             "old test AUROC | unseen P3 | unseen P4 |", "|---|---|---|---|---|---|---|---|"]
    summ = {}
    for bb in ("dasheng", "vjepa2_1_l"):
        path = os.path.join(OUT, f"{bb}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        for name, block in (("served courtroom", d["served"]), ("pilot start (no further training)", d["start_model"]),
                            ("continued", d.get("continued"))):
            if not block:
                continue
            row = {}
            for s in ("val", "test", "old_test"):
                yy, pp = np.array([lab[i] for i in block[s]["ids"]]), np.array(block[s]["prob"])
                row[s] = metrics(yy, pp)
                if s == "test":
                    row["test_ci"] = pf.boot_auc(yy, pp)
                    row["test_at_0.5"] = classwise(yy, pp, 0.5)
            # Unseen person: served/start never saw P3 or P4, so all of that person's clips count; the
            # continued model is scored only by its refit that excluded that person.
            unseen = {}
            for held in NEW_PEOPLE:
                if name == "continued":
                    v = (d.get("new_person_holdout") or {}).get(held)
                    ids, pp = (v["ids"], v["prob"]) if v else (None, None)
                else:
                    keep = [k for k, i in enumerate(block["all_new"]["ids"]) if store.by_id[i].person == held]
                    ids = [block["all_new"]["ids"][k] for k in keep]
                    pp = [block["all_new"]["prob"][k] for k in keep]
                if ids:
                    unseen[held] = metrics(np.array([lab[i] for i in ids]), np.array(pp))["auroc"]
            row["unseen_person"] = unseen
            summ[f"{bb}:{name}"] = row
            u = lambda p: f"{unseen[p]:.3f}" if p in unseen else "–"  # noqa: E731
            lines.append(f"| {bb} | {name} | {row['val']['auroc']:.3f} | {row['test']['auroc']:.3f} "
                         f"({row['test_ci'][0]:.2f}–{row['test_ci'][1]:.2f}) | {row['test']['acc']:.2f} | "
                         f"{row['old_test']['auroc']:.3f} | {u('P3')} | {u('P4')} |")
        lines.append(f"| {bb} | selected on new validation | {d['selected']} | | | | | |")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "prep", "run", "report"])
    ap.add_argument("--backbones", nargs="+", default=["dasheng", "vjepa2_1_l"])
    a = ap.parse_args()
    run(a.backbones) if a.stage == "run" else globals()[a.stage]()
