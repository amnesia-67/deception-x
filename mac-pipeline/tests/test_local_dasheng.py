import copy
import hashlib
import json
import math
import threading

import numpy as np
import pytest
import torch
from torch import nn

from sunhacks.local_dasheng import (
    AttnPoolHead, LocalDashengBackbone, LocalDashengUnavailable, TunedDashengEnsemble,
    combine_exported_scores, eval_audio, interp_time, load_fold_state,
    tensor_schema,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_scale = nn.Parameter(torch.zeros(1))

    def forward(self, wav):
        return self.lora_scale.expand(1, 32, 768).clone()


class TinyFold(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = AttnPoolHead()
        nn.init.zeros_(self.head.out.weight)


def exported_fixture(tmp_path):
    config = {"backbone": "dasheng", "lora_r": 8, "lora_blocks": 4,
              "lora_dropout": .1, "head_dropout": .3}
    meta = {"format": "sunhacks-ensemble-v1", "class_1_meaning": "deceptive",
            "branches": ["tuned_dasheng", "xgb_digest"],
            "weights": [.6, .4], "platt_A": [1.2, 1.4], "platt_B": [.2, .4],
            "dasheng_folds": {}}
    for i in range(5):
        m = TinyFold()
        with torch.no_grad():
            m.backbone.lora_scale.fill_(i)
            m.head.out.bias.fill_(i)
        state = {"lora": m.backbone.state_dict(), "head": m.head.state_dict()}
        name = f"fold{i}.pt"
        path = tmp_path / name
        torch.save({"config": config, "state": state}, path)
        raw = path.read_bytes()
        meta["dasheng_folds"][name] = {"config": config, "sha256": hashlib.sha256(raw).hexdigest(),
                                        "size": len(raw), "state_schema": tensor_schema(state)}
    manifest = tmp_path / "manifest.json"
    raw = json.dumps(meta).encode()
    manifest.write_bytes(raw)
    return manifest, hashlib.sha256(raw).hexdigest(), meta


def make_ensemble(tmp_path):
    manifest, digest, meta = exported_fixture(tmp_path)
    calls = []
    def factory(config):
        calls.append(config)
        return TinyFold()
    model = TunedDashengEnsemble(manifest, tmp_path, model_factory=factory,
                                expected_manifest_sha256=digest)
    return model, calls, meta


def test_actual_fold_swapping_averages_probabilities_and_features(tmp_path):
    model, calls, _ = make_ensemble(tmp_path)
    r = model.infer(np.zeros(16000, np.float32), source_window_id=52)
    expected = [1/(1+math.exp(-v)) for v in range(5)]
    assert len(calls) == 1  # one shared trunk, not five duplicate base models
    assert r["per_fold_scores"] == pytest.approx(expected)
    assert r["raw_score"] == pytest.approx(np.mean(expected))
    assert r["raw_score"] != pytest.approx(1/(1+math.exp(-2)))
    assert np.all(r["_sequence_embedding"] == 2)
    assert r["_embedding"].shape == (768,)
    assert r["source_window_id"] == 52 and not r["calibrated"]
    assert not r["gpu_bf16_parity_verified"]
    assert model.prepare()["weights_loaded"]
    assert not any(p.requires_grad for p in model.models[next(iter(model.models))].parameters())


def test_missing_head_state_never_mutates_backbone():
    m = TinyFold()
    state = {"lora": {"lora_scale": torch.ones(1)}, "head": copy.deepcopy(m.head.state_dict())}
    state["head"].pop("out.bias")
    with pytest.raises(LocalDashengUnavailable, match="Incomplete"):
        load_fold_state(m, state)
    assert torch.equal(m.backbone.lora_scale, torch.zeros(1))


@pytest.mark.parametrize("mutation", ["missing_lora", "unknown_lora", "wrong_shape", "wrong_dtype", "nan"])
def test_invalid_fold_is_rejected_before_mutation(mutation):
    m = TinyFold()
    state = {"lora": {"lora_scale": torch.ones(1)}, "head": copy.deepcopy(m.head.state_dict())}
    if mutation == "missing_lora": state["lora"] = {"different_lora": torch.ones(1)}
    if mutation == "unknown_lora": state["lora"]["extra_lora"] = torch.ones(1)
    if mutation == "wrong_shape": state["lora"]["lora_scale"] = torch.ones(2)
    if mutation == "wrong_dtype": state["lora"]["lora_scale"] = torch.ones(1, dtype=torch.float64)
    if mutation == "nan": state["lora"]["lora_scale"][0] = float("nan")
    with pytest.raises(LocalDashengUnavailable): load_fold_state(m, state)
    assert torch.equal(m.backbone.lora_scale, torch.zeros(1))


@pytest.mark.parametrize("mutation", ["hash", "extra_fold", "missing_fold"])
def test_corrupted_or_incomplete_export_fails(tmp_path, mutation):
    manifest, digest, _ = exported_fixture(tmp_path)
    if mutation == "hash": (tmp_path / "fold0.pt").write_bytes(b"bad bytes")
    if mutation == "extra_fold": (tmp_path / "fold5.pt").write_bytes(b"extra")
    if mutation == "missing_fold": (tmp_path / "fold3.pt").unlink()
    with pytest.raises(LocalDashengUnavailable):
        TunedDashengEnsemble(manifest, tmp_path, model_factory=lambda c: TinyFold(),
                            expected_manifest_sha256=digest)


def test_export_manifest_must_be_reviewed(tmp_path):
    manifest, _, _ = exported_fixture(tmp_path)
    with pytest.raises(LocalDashengUnavailable, match="reviewed"):
        TunedDashengEnsemble(manifest, tmp_path, model_factory=lambda c: TinyFold())


def test_eval_audio_keeps_center60_and_pads_to_one_second():
    a = np.linspace(-1, 1, 62*16000, dtype=np.float32)
    np.testing.assert_array_equal(eval_audio(a), a[16000:61*16000])
    short = np.full(8000, .25, np.float32)
    padded = eval_audio(short)
    assert np.all(padded[:8000] == .25) and np.all(padded[8000:] == 0)


@pytest.mark.parametrize("audio,sr", [
    (np.zeros((2,16000), np.float32),16000), (np.zeros(16000,np.int16),16000),
    (np.full(16000,np.nan,np.float32),16000), (np.ones(16000,np.float32)*2,16000),
    (np.zeros(16000,np.float32),48000), (np.zeros(7999,np.float32),16000),
])
def test_invalid_audio_fails(audio, sr):
    with pytest.raises(LocalDashengUnavailable): eval_audio(audio, sr)


def test_temporal_downsample_reads_all_frames():
    tokens = torch.arange(64, dtype=torch.float32).view(1,64,1)
    expected = torch.arange(32, dtype=torch.float32)*2+.5
    assert torch.equal(interp_time(tokens)[0,:,0], expected)


def test_actual_peft_injection_targets_last_four_blocks_and_bn_runs_once():
    class CountingBN(nn.Module):
        def __init__(self): super().__init__(); self.calls = 0
        def forward(self, x): self.calls += 1; return x
    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.qkv=nn.Linear(2,6); self.proj=nn.Linear(2,2)
    class Block(nn.Module):
        def __init__(self): super().__init__(); self.attn=Attention()
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks=nn.ModuleList([Block() for _ in range(12)])
            self.init_bn=CountingBN()
        def forward(self,x):
            assert self.init_bn is None  # wrapper already applied BN in float32
            return x.mean((1,2)).unsqueeze(-1).expand(-1,-1,768).contiguous()
    base=nn.Module(); base.encoder=Encoder()
    backbone=LocalDashengBackbone(base,{"lora_r":8,"lora_blocks":4,"lora_dropout":.1})
    lora=[n for n in backbone.state_dict() if "lora_" in n]
    assert len(lora)==16
    assert all(any(f"blocks.{i}." in n for i in range(8,12)) for n in lora)
    bn=backbone.model.encoder.init_bn
    out=backbone(torch.zeros(1,16000))
    assert out.shape==(1,32,768) and torch.isfinite(out).all()
    assert bn.calls==1 and backbone.model.encoder.init_bn is bn


def test_stop_or_close_does_not_emit_a_stale_result(tmp_path):
    m, _, _ = make_ensemble(tmp_path)
    stopped = threading.Event(); stopped.set()
    with pytest.raises(LocalDashengUnavailable, match="cancelled"):
        m.infer(np.zeros(16000,np.float32),stop_event=stopped)
    m.close()
    assert m.prepare()["status"] == "stopped"
    with pytest.raises(LocalDashengUnavailable, match="cancelled"):
        m.infer(np.zeros(16000,np.float32))


def test_exported_platt_stack_formula():
    manifest = {"branches":["tuned_dasheng","xgb_digest"], "weights":[.6,.4],
                "platt_A":[1.2,1.4], "platt_B":[.2,.4]}
    scores = [.2,.7]
    cal = [1/(1+math.exp(-(math.log(p/(1-p))*a+b)))
           for p,a,b in zip(scores,manifest["platt_A"],manifest["platt_B"])]
    result = combine_exported_scores(*scores,manifest)
    assert result["raw_score"] == pytest.approx(.6*cal[0]+.4*cal[1])
    assert not result["calibrated"]
    with pytest.raises(LocalDashengUnavailable): combine_exported_scores(float("nan"),.7,manifest)
