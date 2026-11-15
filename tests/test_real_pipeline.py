"""Tests for the real-data pipeline: dataset, identity folds, cache, backbones on GPU, tuning watchdog.

Run inside the CUDA env:  bash scripts/run.sh -m pytest -q tests/test_real_pipeline.py
GPU tests are skipped (not faked) when CUDA or the verified weights are missing.
"""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.rlt_data import CACHE_DIR, GROUPS_PATH, RLT_DIR, index_clips

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA GPU required")
HAVE_GROUPS = os.path.exists(GROUPS_PATH)
needs_data = pytest.mark.skipif(not (os.path.isdir(RLT_DIR) and HAVE_GROUPS),
                                reason="needs Real-Life Trials (scripts/fetch_data.sh); data/rlt_identity_groups.json is kept local (not published); build it with scripts/build_identity_groups.py")
needs_cache = pytest.mark.skipif(not (os.path.isdir(os.path.join(CACHE_DIR, "frames")) and HAVE_GROUPS),
                                 reason="run preprocess first; data/rlt_identity_groups.json is kept local (not published); build it with scripts/build_identity_groups.py")


# ----------------------------------------------------------------------------- data
@needs_data
def test_dataset_is_the_real_121_clips():
    recs = index_clips(require_groups=True)
    assert len(recs) == 121
    assert sum(r.label for r in recs) == 61 and sum(1 - r.label for r in recs) == 60
    assert all(r.transcript.strip() for r in recs), "every clip has a human transcript"
    assert all(os.path.getsize(r.video_path) > 10_000 for r in recs)
    assert {r.clip_id.split("_")[1] for r in recs} == {"lie", "truth"}
    assert all((r.label == 1) == ("_lie_" in r.clip_id) for r in recs)


@needs_data
def test_identity_groups_cover_every_clip_and_known_merges():
    recs = {r.clip_id: r for r in index_clips(require_groups=True)}
    assert all(r.group is not None for r in recs.values())
    assert 50 <= len({r.group for r in recs.values()}) <= 60
    merges_path = os.path.join(ROOT, "data", "rlt_identity_merges.local.json")  # local only, never published
    if os.path.exists(merges_path):
        merges = json.load(open(merges_path, encoding="utf-8"))
        ids = [{recs[c].group for c in members} for members in merges.values()]
        assert all(len(s) == 1 for s in ids), "each reviewed merge is one identity group"
        assert len({next(iter(s)) for s in ids}) == len(ids), "different merges stay different groups"


@needs_data
def test_outer_and_inner_folds_never_share_an_identity():
    from src.tuning import inner_split, outer_folds

    recs = index_clips(require_groups=True)
    ids = [r.clip_id for r in recs]
    y = np.array([r.label for r in recs])
    g = np.array([r.group for r in recs])
    folds = outer_folds(ids, y, g, k=5, seed=0)
    seen = np.zeros(len(ids), int)
    for f, (tr, te) in enumerate(folds):
        assert not set(g[tr]) & set(g[te])
        seen[te] += 1
        assert 0.25 < y[te].mean() < 0.75, "each test fold contains both classes in reasonable balance"
        itr, iva = inner_split(y[tr], g[tr], seed=f)  # same seed the trainer uses
        assert not set(g[tr][itr]) & set(g[tr][iva])
        yv, gv = y[tr][iva], g[tr][iva]
        assert len(set(yv.tolist())) == 2 and min(yv.sum(), len(yv) - yv.sum()) >= 4
        share = np.bincount(np.unique(gv, return_inverse=True)[1]).max() / len(gv)
        assert share <= 0.25, f"fold {f}: one identity is {share:.0%} of inner-val (early stopping would learn identity)"
    assert (seen == 1).all(), "every clip is tested exactly once"


@needs_cache
def test_preprocessed_cache_shapes():
    from src.clip_store import ClipStore

    st = ClipStore()
    for cid in ("trial_lie_001", "trial_truth_060"):
        fr = st.frames(cid)
        assert fr.shape == (128, 384, 384, 3) and fr.dtype == np.uint8
        assert fr.std() > 5, "frames are not blank"
        a = st.audio(cid)
        m = st.meta(cid)
        assert a.dtype == np.float32 and abs(len(a) / 16000 - m["duration_s"]) < 1.5
        assert np.abs(a).max() > 0.01, "audio is not silent"


# ----------------------------------------------------------------------------- helpers
def test_interp_time_and_frame_selection():
    from src.backbones import interp_time
    from src.clip_store import select_frames

    x = torch.arange(50, dtype=torch.float32).view(1, 50, 1).repeat(1, 1, 3)
    y = interp_time(x, 32)
    assert y.shape == (1, 32, 3) and torch.all(y[0, 1:, 0] > y[0, :-1, 0])
    # downsampling must average every frame (not read 2 neighbours): a single spike must survive
    spike = torch.zeros(1, 3200, 1)
    spike[0, 1234] = 1.0
    assert interp_time(spike, 32).sum().item() > 0.009  # 1/100 of the spike lands in its bin
    up = interp_time(torch.arange(16.0).view(1, 16, 1), 32)
    assert up.shape == (1, 32, 1)
    assert select_frames(128, 64, 1).tolist()[:3] == [1, 2, 3]       # V-JEPA 2: contiguous 64
    assert select_frames(128, 32, 3).tolist()[:3] == [3, 5, 7]       # V-JEPA 2.1: 32 at stride 2
    assert select_frames(128, 64, 100).max() == 127                  # clipped to the clip


@needs_cuda
def test_video_augmentation_keeps_shape_dtype():
    from src.clip_store import augment_video

    x = torch.randint(0, 255, (8, 64, 64, 3), dtype=torch.uint8, device="cuda")
    y = augment_video(x, np.random.default_rng(0))
    assert y.shape == x.shape and y.dtype == torch.uint8 and y.is_cuda


def test_metrics_and_group_bootstrap():
    from src.tuning import group_bootstrap_ci, metrics

    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.6])
    m = metrics(y, p)
    assert m["auroc"] == 1.0 and m["acc"] == 1.0
    ci = group_bootstrap_ci(y, p, np.arange(8), n_boot=200)
    assert ci["auroc_ci95"][1] <= 1.0


# ----------------------------------------------------------------------------- backbones on GPU
_VITL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "vjepa2_1-vitl-384",
                     "encoder.safetensors")
VITL = pytest.param("vjepa2_1_l", marks=pytest.mark.skipif(not os.path.exists(_VITL),
                                                          reason="V-JEPA 2.1 ViT-L weights not fetched"))
BACKBONE_CASES = ["vjepa2", "vjepa2_1", VITL, "dasheng", "emotion2vec", "qwen3"]


def _inputs(bb, n=2):
    if bb.modality == "video":
        return torch.randint(0, 255, (n, bb.n_frames, 384, 384, 3), dtype=torch.uint8, device="cuda")
    if bb.modality == "audio":
        return torch.randn(n, 16000 * 4, device="cuda") * 0.1
    return ["I did not take the money, I was at home that night.", "um I think maybe I saw him"] [:n]


@needs_cuda
@pytest.mark.parametrize("name", BACKBONE_CASES)
def test_backbone_forward_backward_on_gpu(name):
    from src.backbones import BACKBONES, LoraSpec

    bb = BACKBONES[name](lora=LoraSpec(r=4, blocks=2))
    assert all(p.is_cuda for p in bb.parameters()), "every weight lives on the GPU"
    trainable = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable), "only LoRA adapters are trainable"
    x = _inputs(bb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = bb(x)
    expect = (2, 32, bb.out_dim) if bb.modality != "text" else (2, 128)
    assert tuple(out.shape) == expect and out.is_cuda and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    grads = [p.grad for p in bb.lora_parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "gradients reach the LoRA adapters"
    assert all(p.grad is None for n, p in bb.named_parameters() if not p.requires_grad)
    del bb
    torch.cuda.empty_cache()


@needs_cuda
def test_frozen_trunk_stays_in_eval_mode_during_training():
    from src.backbones import Emotion2vecAudio, LoraSpec

    bb = Emotion2vecAudio(lora=LoraSpec(r=4, blocks=2)).train()
    frozen_training = [n for n, m in bb.model.named_modules() if m.training and "lora_dropout" not in n
                       and not any(k in n for k in ("lora_A", "lora_B"))]
    lora_drop = [m.training for n, m in bb.model.named_modules() if n.endswith("lora_dropout.default")]
    assert lora_drop and all(lora_drop), "LoRA dropout follows train mode"
    assert not [n for n in frozen_training if n and "lora" not in n], frozen_training[:5]
    del bb
    torch.cuda.empty_cache()


@needs_cuda
@pytest.mark.parametrize("name", ["vjepa2_1", VITL, "dasheng", "qwen3"])
def test_lora_is_identity_at_init(name):
    """peft zero-initialises lora_B, so a freshly adapted model must equal the frozen model."""
    from src.backbones import BACKBONES, LoraSpec

    torch.manual_seed(0)
    a = BACKBONES[name](lora=None).eval()
    b = BACKBONES[name](lora=LoraSpec(r=4, blocks=2)).eval()
    x = _inputs(a)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ya, yb = a(x).float(), b(x).float()
    assert torch.allclose(ya, yb, atol=2e-2, rtol=2e-2)
    del a, b
    torch.cuda.empty_cache()


@needs_cuda
def test_vjepa2_predictor_is_removed():
    from src.backbones import VJEPA2Video

    bb = VJEPA2Video(lora=None)
    assert bb.model.predictor is None
    assert sum(p.numel() for p in bb.parameters()) < 310e6  # encoder only (~304M), no predictor


@needs_cuda
@needs_cache
def test_emotion2vec_on_real_speech_gives_a_distribution():
    from src.backbones import Emotion2vecAudio
    from src.clip_store import ClipStore, audio_tensor

    st = ClipStore()
    bb = Emotion2vecAudio(lora=None).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = bb.emotion_probs(audio_tensor(st, "trial_lie_001", False, np.random.default_rng(0)))
    assert p.shape == (1, 9) and abs(p.sum().item() - 1) < 1e-3
    assert p.max().item() > 0.2, "a trained emotion head is not uniform on real speech"


# ----------------------------------------------------------------------------- watchdog
class _TinyText(nn.Module):
    """Minimal stand-in backbone (text modality) so watchdog paths run in seconds."""
    name, modality, out_dim = "tiny", "text", 4

    def __init__(self, mode="learn", **_):
        super().__init__()
        self.mode = mode
        self.lora_A = nn.Linear(3, 4).cuda()

    def lora_parameters(self):
        return list(self.lora_A.parameters())

    def lora_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_lora_state_dict(self, sd):
        self.load_state_dict(sd, strict=False)

    def forward(self, texts):
        f = torch.tensor([[len(t) / 500.0, t.lower().count(" not ") / 3.0, 1.0] for t in texts], device="cuda")
        out = self.lora_A(f)
        if self.mode == "nan":
            out = out * float("nan")
        if self.mode == "const":
            out = out * 0
        return out


def _run_tiny(mode, monkeypatch, tmp_path, epochs=6, min_epochs=3):
    from src import tuning
    from src.clip_store import ClipStore

    monkeypatch.setitem(tuning.BACKBONES, "tiny", lambda lora=None, device="cuda": _TinyText(mode))
    st = ClipStore() if os.path.isdir(os.path.join(CACHE_DIR, "frames")) and HAVE_GROUPS else None
    if st is None:
        pytest.skip("cache or data/rlt_identity_groups.json missing")
    ids = st.ids
    cfg = tuning.TuneConfig("tiny", max_epochs=epochs, min_epochs=min_epochs, patience=10, lora_blocks=0,
                            micro_batch=8)
    rep = tuning.RunReporter(str(tmp_path))
    gpu = tuning.GPUMonitor(period=0.5)
    try:
        return tuning.train_one(cfg, st, ids[::2], ids[1::2], rep, gpu, fold=0, min_gpu_util=0.0)
    finally:
        gpu.stop()


@needs_cuda
def test_watchdog_terminates_on_nan_loss(monkeypatch, tmp_path):
    from src.tuning import SubParRun

    with pytest.raises(SubParRun, match="non-finite"):
        _run_tiny("nan", monkeypatch, tmp_path)


@needs_cuda
def test_watchdog_terminates_run_that_cannot_learn(monkeypatch, tmp_path):
    from src.tuning import SubParRun

    with pytest.raises(SubParRun, match="not learning"):
        _run_tiny("const", monkeypatch, tmp_path)


@needs_cuda
def test_reports_written_every_epoch(monkeypatch, tmp_path):
    import json

    _, state, best, hist = _run_tiny("learn", monkeypatch, tmp_path, epochs=3, min_epochs=99)
    lines = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    assert len(lines) == 3 and all("val_auroc" in l and "gpu" in l for l in lines)
    assert (tmp_path / "status.json").exists() and state is not None


@pytest.mark.skipif(not os.path.exists(_VITL), reason="V-JEPA 2.1 ViT-L weights not fetched")
def test_vitl_weights_match_their_fetch_manifest_and_architecture():
    import hashlib
    import json

    from safetensors import safe_open

    man = json.load(open(os.path.join(os.path.dirname(_VITL), "manifest.json")))
    assert man.get("official_source_url", man["source_url"]).endswith("/vjepa2_1_vitl_dist_vitG_384.pt")
    assert man["checkpoint_key"] == "ema_encoder"
    h = hashlib.sha256()
    with open(_VITL, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    assert h.hexdigest() == man["output_sha256"]
    with safe_open(_VITL, "pt") as f:
        keys = list(f.keys())
        assert sum(k.startswith("blocks.") and k.endswith(".attn.qkv.weight") for k in keys) == 24
        assert tuple(f.get_slice("blocks.0.attn.qkv.weight").get_shape()) == (3072, 1024)
