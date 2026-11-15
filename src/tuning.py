"""Speaker-independent LoRA tuning of one backbone, with live reports and sub-par termination.

Protocol (no test-fold information ever influences training or model choice):
  outer: StratifiedGroupKFold(5) over identity groups -> out-of-fold (OOF) predictions
  inner: one StratifiedGroupKFold split of the outer-train part -> validation set used
         for early stopping, sub-par detection and choosing between retry configs
Reports: runs/<run>/metrics.jsonl (every epoch), status.json (live), report.md (end).
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.backbones import BACKBONES, AttnPoolHead, LoraSpec
from src.clip_store import ClipStore, audio_tensor, eval_starts, text_view, video_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


class SubParRun(Exception):
    """Raised by the watchdog to terminate a run whose metrics are sub-par."""


@dataclass
class TuneConfig:
    backbone: str
    lora_r: int = 8
    lora_blocks: int = 4
    lora_dropout: float = 0.1
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    head_dropout: float = 0.3
    label_smoothing: float = 0.05
    max_epochs: int = 12
    min_epochs: int = 4
    patience: int = 4
    eff_batch: int = 8
    micro_batch: int = 0  # 0 = choose from VRAM probe
    seed: int = 0
    tag: str = "base"


# Retry ladder per backbone: tried in order when the inner-validation watchdog flags a run sub-par.
LADDER: Dict[str, List[TuneConfig]] = {}
for _bb, _ep in (("vjepa2", 10), ("vjepa2_1", 12), ("vjepa2_1_l", 12), ("dasheng", 16), ("emotion2vec", 16),
                 ("qwen3", 16)):
    LADDER[_bb] = [
        TuneConfig(_bb, max_epochs=_ep, tag="r8_last4_lr2e-4"),
        TuneConfig(_bb, max_epochs=_ep, lr_lora=5e-5, lr_head=3e-4, head_dropout=0.5, tag="r8_last4_lr5e-5"),
        TuneConfig(_bb, max_epochs=_ep, lora_r=16, lora_blocks=8, lr_lora=3e-4, tag="r16_last8_lr3e-4"),
    ]


# ------------------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    out = {"n": int(len(y)), "acc": float(accuracy_score(y, p >= 0.5)),
           "bal_acc": float(balanced_accuracy_score(y, p >= 0.5)),
           "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))}
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan")
    return out


def classwise(y, p, thr: float = 0.5, n_bins: int = 10) -> Dict[str, float]:
    """Confusion matrix and class-wise rates at `thr` (class 1 = deceptive), plus proper scores.

    Reports what an accuracy number hides: how many lies are caught (sensitivity), how many truths are
    passed (specificity), the fraction of answers called deceptive, Brier score, log loss and the
    expected calibration error, next to the always-truthful and class-prior baselines.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    thr_arr = np.broadcast_to(np.asarray(thr, dtype=np.float64), p.shape)  # per-row thresholds allowed
    yhat = (p >= thr_arr).astype(int)
    thr = float(thr_arr.mean())
    tp, fn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 1)).sum())
    tn, fp = int(((yhat == 0) & (y == 0)).sum()), int(((yhat == 1) & (y == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                    for b in range(n_bins) if (bins == b).any()))
    prior = y.mean()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"threshold": float(thr), "tp_lie": tp, "fn_lie": fn, "tn_truth": tn, "fp_truth": fp,
            "lie_sensitivity": float(sens), "truth_specificity": float(spec), "bal_acc": float((sens + spec) / 2),
            "acc": float((tp + tn) / len(y)), "predicted_lie_fraction": float(yhat.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), "ece_10bin": ece,
            "baseline_always_truthful_acc": float(1 - prior), "baseline_always_truthful_bal_acc": 0.5,
            "baseline_prior_brier": float(prior * (1 - prior))}


def balanced_threshold(y, p) -> float:
    """Threshold maximising balanced accuracy on (y, p); ties broken towards 0.5. Fit on training data only."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    u = np.unique(p)
    cands = np.concatenate([[0.5], (u[:-1] + u[1:]) / 2]) if len(u) > 1 else np.array([0.5])
    best, best_key = 0.5, None
    for t in cands:
        yhat = p >= t
        sens = (yhat & (y == 1)).sum() / max(1, (y == 1).sum())
        spec = (~yhat & (y == 0)).sum() / max(1, (y == 0).sum())
        key = (round(float(sens + spec), 9), -abs(t - 0.5))
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    return best


def prior_matched_threshold(y, p) -> float:
    """Threshold at which the fraction of (training) predictions called deceptive equals the training
    deception rate. One low-variance degree of freedom: it removes a systematic shift of the score
    distribution (e.g. calibrated scores whose median sits below 0.5 while half the clips are lies)
    without optimising a noisy accuracy curve. Fit on training predictions only."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    return float(np.quantile(p, 1.0 - y.mean()))


def group_bootstrap_ci(y, p, groups, n_boot=2000, seed=0):
    """95% CI of AUROC / accuracy, resampling identity groups (not clips)."""
    rng = np.random.default_rng(seed)
    y, p, groups = map(np.asarray, (y, p, groups))
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    aucs, accs = [], []
    for _ in range(n_boot):
        gs = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(set(y[idx].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        accs.append(np.mean((p[idx] >= 0.5) == y[idx]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"auroc_ci95": q(aucs), "acc_ci95": q(accs), "n_boot": len(aucs)}


def outer_folds(ids: Sequence[str], y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0):
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(np.asarray(tr), np.asarray(te)) for tr, te in sgkf.split(np.zeros(len(ids)), y, groups)]


def inner_split(y: np.ndarray, groups: np.ndarray, seed: int = 0, big_frac: float = 0.10,
                max_group_share: float = 0.25):
    """Grouped inner train/val split in which no single identity dominates validation.

    Identities larger than `big_frac` of the outer-train set always stay in inner-train
    (otherwise one 20-clip person can supply every deceptive val clip, and early stopping
    would reward recognising that person). The rest is split with StratifiedGroupKFold;
    among its 5 candidate splits we take the most class-balanced one whose largest identity
    is at most `max_group_share` of val.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    ug, counts = np.unique(groups, return_counts=True)
    big = set(ug[counts > big_frac * len(y)])
    small_idx = np.array([i for i in range(len(y)) if groups[i] not in big])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 101)
    best, best_key = None, None
    for _, va_s in sgkf.split(np.zeros(len(small_idx)), y[small_idx], groups[small_idx]):
        va = small_idx[va_s]
        yv = y[va]
        if len(set(yv.tolist())) < 2:
            continue
        share = np.bincount(np.unique(groups[va], return_inverse=True)[1]).max() / len(va)
        key = (share <= max_group_share, min(yv.sum(), len(yv) - yv.sum()), -share)
        if best_key is None or key > best_key:
            best, best_key = va, key
    if best is None:
        raise RuntimeError("could not build a two-class inner validation split")
    va = np.sort(best)
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


# ------------------------------------------------------------------------- GPU monitor
class GPUMonitor:
    """Samples nvidia-smi every `period` s so reports show real utilisation, not guesses."""

    def __init__(self, period: float = 2.0, require: bool = True):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.ok = True
        self.failures = 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        if require:
            t0 = time.time()
            while not self.samples and time.time() - t0 < 10:
                time.sleep(0.2)
            if not self.samples:
                raise RuntimeError("nvidia-smi produced no samples: GPU utilisation watchdog would be blind")

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                                      "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.samples.append((time.time(), *[float(v) for v in out[:5]]))
            except Exception as e:
                self.failures += 1
                if self.failures in (1, 10, 100):
                    print(f"[GPUMonitor] WARNING nvidia-smi sample failed ({self.failures}x): {e}", flush=True)
                self.ok = False
            self._stop.wait(self.period)

    def window(self, since: float) -> Dict[str, float]:
        s = [x for x in self.samples if x[0] >= since]
        if not s:
            return {}
        a = np.array(s)[:, 1:]
        return {"gpu_util_mean": float(a[:, 0].mean()), "gpu_util_max": float(a[:, 0].max()),
                "gpu_mem_used_mb": float(a[:, 1].max()), "gpu_mem_total_mb": float(a[0, 2]),
                "gpu_temp_c": float(a[:, 3].max()), "gpu_power_w": float(a[:, 4].mean())}

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------------------- reporter
class RunReporter:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self.jsonl = open(os.path.join(run_dir, "metrics.jsonl"), "a", buffering=1)

    def log(self, rec: Dict):
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        self.jsonl.write(json.dumps(rec) + "\n")
        with open(os.path.join(self.dir, "status.json"), "w") as f:
            json.dump(rec, f, indent=2)

    @staticmethod
    def line(rec: Dict) -> str:
        g = rec.get("gpu", {})
        return (f"[{rec.get('backbone')} f{rec.get('fold')} {rec.get('tag')} ep{rec.get('epoch'):>2}] "
                f"loss {rec.get('train_loss', float('nan')):.4f} trainAUC {rec.get('train_auroc', float('nan')):.3f} | "
                f"val AUC {rec.get('val_auroc', float('nan')):.3f} acc {rec.get('val_acc', float('nan')):.3f} "
                f"loss {rec.get('val_loss', float('nan')):.3f} | {rec.get('clips_per_s', 0):.2f} clip/s | "
                f"GPU {g.get('gpu_util_mean', float('nan')):.0f}% {g.get('gpu_mem_used_mb', float('nan')):.0f}MB "
                f"peak-alloc {rec.get('torch_peak_alloc_mb', 0):.0f}MB | {rec.get('status', '')}")


# ------------------------------------------------------------------------- core model
class TunedModel(nn.Module):
    def __init__(self, cfg: TuneConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        lora = LoraSpec(r=cfg.lora_r, alpha=2 * cfg.lora_r, dropout=cfg.lora_dropout, blocks=cfg.lora_blocks)
        self.backbone = BACKBONES[cfg.backbone](lora=lora, device=device)
        self.head = AttnPoolHead(self.backbone.out_dim, dropout=cfg.head_dropout).to(device)

    def trainable_state(self):
        return {"lora": self.backbone.lora_state_dict(),
                "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()}}

    def load_trainable_state(self, st):
        self.backbone.load_lora_state_dict(st["lora"])
        self.head.load_state_dict(st["head"])


def forward_logits(model: TunedModel, store: ClipStore, ids: Sequence[str], train: bool,
                   rng: np.random.Generator, start: int = 0) -> torch.Tensor:
    bb = model.backbone
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if bb.modality == "video":
            x = video_batch(store, ids, bb.n_frames, train, rng, start=start)
            return model.head(bb(x))
        if bb.modality == "audio":
            feats = [bb(audio_tensor(store, cid, train, rng)) for cid in ids]
            return model.head(torch.cat(feats, 0))
        texts = [text_view(store.transcript(cid), train, rng) for cid in ids]
        return model.head(bb(texts))


def eval_views(model: TunedModel) -> List[int]:
    """Video: window starts covering both halves of the clip (logits averaged). Others: one view."""
    return eval_starts() if model.backbone.modality == "video" else [0]


@torch.no_grad()
def predict(model: TunedModel, store: ClipStore, ids: Sequence[str], bs: int) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(0)
    logits = np.zeros(len(ids))
    views = eval_views(model)
    for st in views:
        for i in range(0, len(ids), bs):
            chunk = list(ids[i:i + bs])
            logits[i:i + len(chunk)] += forward_logits(model, store, chunk, False, rng, st).float().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits / len(views)))


def probe_micro_batch(model: TunedModel, store: ClipStore, ids: Sequence[str], cfg: TuneConfig,
                      budget_frac: float = 0.85) -> int:
    """Largest micro-batch that divides eff_batch and whose fwd+bwd peak stays under budget (measured)."""
    if cfg.micro_batch:
        return cfg.micro_batch
    if model.backbone.modality == "audio":
        return 1  # per-clip forward (variable length); gradient accumulation gives eff_batch
    total = _VRAM_CAP_BYTES or torch.cuda.mem_get_info(0)[0] + torch.cuda.memory_reserved(0)
    rng = np.random.default_rng(0)
    if model.backbone.modality == "text":
        ids = sorted(ids, key=lambda c: -len(store.transcript(c)))  # worst-case token lengths
    # Ascending probe over divisors of eff_batch, so the effective batch is exactly eff_batch.
    # Never launch a huge batch first: on WSL/WDDM an oversized allocation can wedge the device.
    best = 1
    for mb in [d for d in (1, 2, 4, 8, 16) if cfg.eff_batch % d == 0]:
        if mb > len(ids):
            break
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            loss = forward_logits(model, store, list(ids[:mb]), True, rng).float().mean()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            print(f"    VRAM probe: micro-batch {mb} OOM -> using {best}", flush=True)
            break
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if peak >= budget_frac * total:
            print(f"    VRAM probe: micro-batch {mb} peak {peak/2**20:.0f} MB exceeds {budget_frac:.0%} budget "
                  f"-> using {best}", flush=True)
            break
        best = mb
        print(f"    VRAM probe: micro-batch {mb} peak {peak/2**20:.0f} MB / {total/2**20:.0f} MB fits", flush=True)
    return best


_VRAM_CAP_BYTES: Optional[int] = None


def limit_to_physical_vram(safety: float = 0.96) -> int:
    """Cap this process at the VRAM that is actually free (the Windows desktop holds ~1 GB of the 12 GB).

    Without the cap, WDDM lets CUDA spill into shared host memory, which shows up as
    'CUDA driver error: device not ready' instead of a clean, catchable OOM.
    """
    global _VRAM_CAP_BYTES
    free, total = torch.cuda.mem_get_info(0)
    cap = int((free + torch.cuda.memory_reserved(0)) * safety)
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    _VRAM_CAP_BYTES = cap
    print(f"VRAM cap {cap/2**20:.0f} MB of {total/2**20:.0f} MB (free at start {free/2**20:.0f} MB)", flush=True)
    return cap


def assert_on_gpu(model: TunedModel):
    bad = [n for n, p in model.named_parameters() if not p.is_cuda]
    if bad:
        raise SubParRun(f"parameters on CPU: {bad[:3]}")
    if not torch.cuda.is_available():
        raise SubParRun("CUDA not available")


def train_one(cfg: TuneConfig, store: ClipStore, train_ids: Sequence[str], val_ids: Optional[Sequence[str]],
              reporter: RunReporter, gpu: GPUMonitor, fold: int, model: Optional[TunedModel] = None,
              min_gpu_util: float = 5.0, phase: str = "inner"):
    """Train one model.

    val_ids given  -> early stopping on val AUROC (ties: val loss); returns the best-epoch state.
    val_ids None   -> refit mode: exactly cfg.max_epochs epochs, no validation, returns the final state.
    Returns (model, state, best, history).
    """
    torch.manual_seed(cfg.seed + fold)
    rng = np.random.default_rng(cfg.seed * 1000 + fold + (7 if phase == "refit" else 0))
    if model is None:
        model = TunedModel(cfg)
    assert_on_gpu(model)
    mb = probe_micro_batch(model, store, train_ids, cfg)
    accum = max(1, cfg.eff_batch // mb)
    params = [{"params": model.backbone.lora_parameters(), "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay},
              {"params": list(model.head.parameters()), "lr": cfg.lr_head, "weight_decay": 1e-4}]
    opt = torch.optim.AdamW(params)
    n_micro = math.ceil(len(train_ids) / mb)
    steps_per_epoch = math.ceil(n_micro / accum)
    total = steps_per_epoch * cfg.max_epochs
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    ytr = store.labels(train_ids)
    yva = store.labels(val_ids) if val_ids is not None else None
    best = {"val_auroc": -1.0, "val_loss": 1e9, "epoch": 0}
    best_state, history, first_loss = None, [], None
    bad_epochs = 0
    tm, ep_loss = {"auroc": float("nan")}, float("nan")
    n_trainable = sum(p.numel() for g in params for p in g["params"])
    from src.provenance import state_hash

    init_state = model.trainable_state()
    trace = {"init_state_sha256": state_hash(init_state), "optimizer_steps": 0, "epoch_train_loss": [],
             "epoch_grad_norm_mean": [], "epoch_val_auroc": []}
    print(f"  fold {fold} [{cfg.tag}] {phase}: train {len(train_ids)} "
          f"val {len(val_ids) if val_ids is not None else '-'} | micro-batch {mb} x accum {accum} = {mb * accum} "
          f"| trainable {n_trainable/1e6:.2f}M params on {torch.cuda.get_device_name(0)}", flush=True)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        order = rng.permutation(len(train_ids))
        losses, tr_p, tr_y, gnorms = [], [], [], []
        opt.zero_grad(set_to_none=True)
        for step_i, s in enumerate(range(0, len(order), mb)):
            idx = order[s:s + mb]
            ids = [train_ids[i] for i in idx]
            y = torch.tensor(ytr[idx], dtype=torch.float32, device="cuda")
            logits = forward_logits(model, store, ids, True, rng).float()
            target = y * (1 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise SubParRun(f"non-finite loss {loss.item()} at epoch {epoch} step {step_i}")
            group = step_i // accum
            group_len = min(accum, n_micro - group * accum)  # a short last group is weighted correctly
            (loss / group_len).backward()
            if (step_i + 1) % accum == 0 or step_i + 1 == n_micro:
                gn = torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                gnorms.append(float(gn))
                opt.step()
                trace["optimizer_steps"] += 1
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            tr_p.extend(torch.sigmoid(logits.detach()).cpu().tolist())
            tr_y.extend(ytr[idx].tolist())
        train_time = time.time() - t0
        tm = metrics(np.array(tr_y), np.array(tr_p))
        ep_loss = float(np.mean(losses))
        first_loss = first_loss or ep_loss
        trace["epoch_train_loss"].append(round(ep_loss, 6))
        trace["epoch_grad_norm_mean"].append(round(float(np.mean(gnorms)), 6) if gnorms else None)
        rec = {"backbone": cfg.backbone, "fold": fold, "tag": cfg.tag, "phase": phase, "epoch": epoch,
               "train_loss": ep_loss, "train_auroc": tm["auroc"], "train_acc": tm["acc"],
               "lr_lora": opt.param_groups[0]["lr"], "clips_per_s": len(train_ids) / train_time,
               "micro_batch": mb, "accum": accum}
        if val_ids is not None:
            vm = metrics(yva, predict(model, store, val_ids, max(1, mb)))
            rec.update({"val_auroc": vm["auroc"], "val_acc": vm["acc"], "val_loss": vm["log_loss"]})
            trace["epoch_val_auroc"].append(round(float(vm["auroc"]), 6))
            improved = (vm["auroc"] > best["val_auroc"] + 1e-6 or
                        (abs(vm["auroc"] - best["val_auroc"]) <= 1e-6 and vm["log_loss"] < best["val_loss"]))
            if improved:
                best = {"val_auroc": vm["auroc"], "val_loss": vm["log_loss"], "val_acc": vm["acc"], "epoch": epoch}
                best_state = model.trainable_state()
                bad_epochs = 0
                rec["status"] = "best"
            else:
                bad_epochs += 1
                rec["status"] = f"no-improve {bad_epochs}/{cfg.patience}"
        else:
            rec["status"] = f"refit {epoch}/{cfg.max_epochs}"
        rec.update({"epoch_s": time.time() - t0, "torch_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "torch_alloc_mb": torch.cuda.memory_allocated() / 2**20, "gpu": gpu.window(t0)})
        # ---------------- watchdog: terminate sub-par runs (inner-val / train signals only)
        g = rec["gpu"]
        reason = None
        if g and g.get("gpu_util_mean", 100) < min_gpu_util and epoch >= 2:
            reason = f"GPU utilisation {g['gpu_util_mean']:.1f}% - not running on the GPU"
        elif epoch >= 3 and ep_loss > 1.5 * first_loss:
            reason = f"training loss diverged ({first_loss:.3f} -> {ep_loss:.3f})"
        elif (val_ids is not None and epoch >= cfg.min_epochs and tm["auroc"] < 0.60
              and best["val_auroc"] < 0.55):
            reason = f"not learning: train AUROC {tm['auroc']:.3f}, best val AUROC {best['val_auroc']:.3f}"
        if reason:
            rec["status"] = "TERMINATED: " + reason
        reporter.log(rec)
        history.append(rec)
        print("   " + RunReporter.line(rec), flush=True)
        if reason:
            raise SubParRun(reason)
        if val_ids is not None and epoch >= cfg.min_epochs and bad_epochs >= cfg.patience:
            print(f"   early stop at epoch {epoch}; best epoch {best['epoch']} val AUROC {best['val_auroc']:.3f}",
                  flush=True)
            break
    if val_ids is None:
        best_state = model.trainable_state()
        best = {"epoch": cfg.max_epochs, "train_auroc": tm["auroc"], "train_loss": ep_loss}
    # Learning evidence beyond "GPU busy, tensors finite": did the returned weights actually move?
    trace["epochs_run"] = len(trace["epoch_train_loss"])
    trace["returned_state_sha256"] = state_hash(best_state) if best_state is not None else None
    if best_state is not None:
        def _flat(st):
            out = []
            for v in (st["lora"], st["head"]):
                out += [t.float().reshape(-1) for _, t in sorted(v.items())]
            return torch.cat(out)
        a, b = _flat(init_state), _flat(best_state)
        trace["update_l2"] = float((b - a).norm())
        trace["init_l2"] = float(a.norm())
        if trace["update_l2"] == 0.0:
            raise SubParRun("returned weights are identical to the initial weights (no learning)")
    best = {**best, "trace": trace}
    return model, best_state, best, history


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def tune_fold(backbone: str, store: ClipStore, train_ids, test_ids, fold: int, reporter: RunReporter,
              gpu: GPUMonitor, min_val_auroc: float = 0.55, seed: int = 0, refit: bool = True):
    """Inner phase: try the config ladder on an identity-balanced inner split, choose by inner-val AUROC.
    Refit phase: retrain the chosen config on ALL outer-train clips for the chosen number of epochs.
    The outer test fold is used only for the final prediction."""
    ytr, gtr = store.labels(train_ids), store.groups(train_ids)
    itr, iva = inner_split(ytr, gtr, seed=seed + fold)
    inner_train = [train_ids[i] for i in itr]
    inner_val = [train_ids[i] for i in iva]
    vg = gtr[iva]
    print(f"  fold {fold}: inner-val {len(iva)} clips, {int(ytr[iva].sum())} deceptive, {len(set(vg))} identities, "
          f"largest identity {np.bincount(np.unique(vg, return_inverse=True)[1]).max()} clips", flush=True)
    attempts = []
    chosen = None
    for cfg in LADDER[backbone]:
        cfg = replace(cfg, seed=seed)
        model = None
        try:
            model, state, best, _ = train_one(cfg, store, inner_train, inner_val, reporter, gpu, fold)
            status = "ok" if best["val_auroc"] >= min_val_auroc else "sub-par"
        except SubParRun as e:
            state, best, status = None, {"val_auroc": -1.0, "reason": str(e)}, "terminated"
            print(f"   >>> TERMINATED sub-par run fold {fold} [{cfg.tag}]: {e}", flush=True)
        att = {"tag": cfg.tag, "status": status, **{k: v for k, v in best.items() if k != "trace"},
               "trace": best.get("trace")}
        attempts.append(att)
        reporter.log({"backbone": backbone, "fold": fold, "tag": cfg.tag, "epoch": -1, "event": "attempt_done", **att})
        if state is not None and (chosen is None or best["val_auroc"] > chosen[1]["val_auroc"]):
            chosen = (cfg, best, state)  # only the small LoRA+head state is kept
        model = None
        _free()
        if status == "ok":
            break
        print(f"   attempt [{cfg.tag}] {status}; fixing by moving to the next config in the ladder", flush=True)
    if chosen is None:
        raise RuntimeError(f"{backbone} fold {fold}: every attempt was terminated")
    cfg, best, state = chosen
    used = "inner"
    model = None
    refit_trace = None
    if refit:
        rcfg = replace(cfg, max_epochs=max(2, int(best["epoch"])))
        try:
            model, rstate, rbest, _ = train_one(rcfg, store, list(train_ids), None, reporter, gpu, fold,
                                                phase="refit")
            state, used, refit_trace = rstate, "refit", rbest.get("trace")
        except SubParRun as e:
            print(f"   >>> refit terminated ({e}); falling back to the inner-train model", flush=True)
            model = None
            _free()
    if used == "inner":
        model = TunedModel(cfg)
        model.load_trainable_state(state)
    bs = 1 if model.backbone.modality == "audio" else 4
    p_test = predict(model, store, test_ids, bs)
    from src.provenance import state_hash

    info = {"config": asdict(cfg), "inner_val": {k: v for k, v in best.items() if k != "trace"},
            "attempts": attempts, "final_model": used,
            "refit_epochs": int(refit_trace["epochs_run"]) if used == "refit" else None,
            "refit_trace": refit_trace, "final_state_sha256": state_hash(state)}
    return p_test, info, model, state
