"""Opt-in webcam_pilot profile (models/pilot_webcam/): strict loading, calibration, fusion and fixture parity.

The served default must not change: those paths are covered by test_golden_fixture.py / test_v2_socket.py.
Skipped (not faked) when an export or CUDA is missing.
"""

import json
import os
import shutil

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(ROOT, "models", "pilot_webcam")
FIX = os.path.join(ROOT, "fixtures", "golden_v1")
EXPECTED = os.path.join(FIX, "pilot_webcam_expected.json")
BRANCHES = ["dasheng", "emotion2vec", "vjepa2_1_l"]


def _have(bb):
    return os.path.exists(os.path.join(PILOT, bb, "deploy_meta.json"))


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch


def test_calibration_is_monotone_platt():
    from src.pilot_webcam import calibrate

    cal = {"A": 0.7, "B": -0.1}
    xs = [0.05, 0.3, 0.5, 0.7, 0.95]
    ys = [calibrate(x, cal) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))
    assert abs(calibrate(0.5, cal) - 1 / (1 + np.exp(0.1))) < 1e-12


def test_session_baseline_waits_then_compares_with_the_session_median():
    from src.pilot_webcam import SessionBaseline

    sb = SessionBaseline(min_answers=4)
    assert [sb.add(x)["verdict"] for x in (0.1, 0.2, 0.3)] == [None, None, None]
    assert sb.add(0.9)["verdict"] == "Deceptive"   # above the median of (0.1, 0.2, 0.3, 0.9)
    assert sb.add(-5.0)["verdict"] == "Truthful"
    # a constant per-session offset does not change any verdict
    a, b = SessionBaseline(4), SessionBaseline(4)
    xs = [0.3, -0.2, 0.8, 0.1, -0.4, 0.6]
    assert [a.add(x)["verdict"] for x in xs] == [b.add(x + 7.0)["verdict"] for x in xs]


@pytest.mark.parametrize("bb", BRANCHES)
def test_manifest_records_validation_only_selection_and_heldout_metrics(bb):
    if not _have(bb):
        pytest.skip(f"no {bb} webcam-pilot export")
    meta = json.load(open(os.path.join(PILOT, bb, "deploy_meta.json")))
    assert meta["profile"] == "webcam_pilot" and meta["backbone"] == bb
    assert 0 < meta["threshold"] < 1 and np.isfinite([meta["calibration"]["A"], meta["calibration"]["B"]]).all()
    t = meta["trained_on"]
    assert t["train_clips"] > 0 and t["validation_clips"] > 0 and t["test_clips"] > 0
    assert "validation" in meta["calibration"]["method"]
    unseen = t.get("unseen_person_test")
    if unseen:  # final (850-clip) export: the unseen person never entered training or selection
        assert any(unseen["person"] in k for k in meta["metrics"])
        assert meta["reproduction"]["validation_prob_max_abs_diff_reloaded_vs_selected"] <= 2e-3


@pytest.mark.parametrize("bb", BRANCHES)
def test_tampered_checkpoint_is_rejected(bb, tmp_path):
    pytest.importorskip("torch")
    if not _have(bb):
        pytest.skip(f"no {bb} webcam-pilot export")
    from src.pilot_webcam import read_manifest

    dst = tmp_path / bb
    shutil.copytree(os.path.join(PILOT, bb), dst)
    read_manifest(bb, str(tmp_path))  # the untouched copy loads
    data = bytearray((dst / "fold0.pt").read_bytes())
    data[-100] ^= 1
    (dst / "fold0.pt").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA256"):
        read_manifest(bb, str(tmp_path))


def test_served_vitl_manifest_is_not_accepted_as_webcam_pilot(tmp_path):
    pytest.importorskip("torch")
    served = os.path.join(ROOT, "models", "vjepa2_1_l_lora")
    if not os.path.exists(os.path.join(served, "deploy_meta.json")):
        pytest.skip("no served ViT-L export")
    from src.pilot_webcam import read_manifest

    os.makedirs(tmp_path / "vjepa2_1_l")
    shutil.copy(os.path.join(served, "deploy_meta.json"), tmp_path / "vjepa2_1_l" / "deploy_meta.json")
    with pytest.raises(ValueError, match="not a sunhacks-pilot-webcam-v1"):
        read_manifest("vjepa2_1_l", str(tmp_path))


def test_fusion_is_bound_to_the_branch_manifests(tmp_path):
    if not os.path.exists(os.path.join(PILOT, "fusion.json")):
        pytest.skip("no fusion manifest")
    from src.pilot_webcam import fuse, read_fusion

    fu = read_fusion()
    zero = {b: v["logit_mean"] for b, v in fu["branches"].items()}  # every branch at its validation mean
    r = fuse(zero, fu)
    assert abs(r["fused_logit"]) < 1e-12 and r["verdict"] in ("Deceptive", "Truthful")
    with pytest.raises(ValueError, match="needs logits"):
        fuse({}, fu)
    shutil.copytree(PILOT, tmp_path / "p")
    b = next(iter(fu["branches"]))
    meta = tmp_path / "p" / b / "deploy_meta.json"
    meta.write_bytes(meta.read_bytes().replace(b'"threshold"', b'"threshold" ', 1))
    with pytest.raises(ValueError, match="manifest"):
        read_fusion(str(tmp_path / "p"))


def _expected():
    if not os.path.exists(EXPECTED):
        pytest.skip("no webcam-pilot fixture")
    return json.load(open(EXPECTED))


@pytest.mark.parametrize("bb", ["dasheng", "emotion2vec"])
def test_pilot_audio_reproduces_fixture_bf16_and_fp32(bb):
    if not _have(bb):
        pytest.skip(f"no {bb} export")
    _cuda()
    from src.pilot_webcam import PilotAudioScorer

    exp = _expected()
    e = (exp.get("audio_branches") or {}).get(bb) or ({"cuda_bf16": exp["audio_cuda_bf16"],
                                                       "cuda_fp32": exp["audio_cuda_fp32"],
                                                       "suggested_tolerance_logit": exp["audio_suggested_tolerance_logit"]}
                                                      if bb == "dasheng" and "audio_cuda_bf16" in exp else None)
    if e is None:
        pytest.skip(f"fixture has no {bb} section")
    audio = np.frombuffer(open(os.path.join(FIX, "audio_f32le_16k.bin"), "rb").read(), dtype="<f4")
    for key, amp in (("cuda_bf16", "auto"), ("cuda_fp32", None)):
        s = PilotAudioScorer(device="cuda", amp_dtype=amp, backbone=bb)
        if s.manifest_sha256 != e[key]["manifest_sha256"]:
            pytest.skip("fixture was generated for a different export; run `pilot_final.py fixture`")
        r = s.score(audio)
        assert abs(r["logit"] - e[key]["logit"]) <= e["suggested_tolerance_logit"]
        assert r["verdict"] == e[key]["verdict"] and r["threshold"] == e[key]["threshold"]
        del s


def test_pilot_video_and_fusion_reproduce_fixture():
    if not _have("vjepa2_1_l"):
        pytest.skip("no ViT-L export")
    _cuda()
    from scripts.make_golden_fixture import load_payload
    from src.pilot_webcam import PilotVideoService, fuse, read_fusion

    exp = _expected()
    if "video_cuda_bf16" not in exp:
        pytest.skip("fixture has no webcam-pilot video section")
    svc = PilotVideoService(device="cuda")
    e = exp["video_cuda_bf16"]
    if svc.svc.manifest_sha256 != e["manifest_sha256"]:
        pytest.skip("fixture was generated for a different export; run `pilot_final.py fixture`")
    payload, _ = load_payload()
    r = svc.predict({k: payload[k] for k in ("frames", "timestamps")})
    assert r["model"] == e["model"] and r["profile"] == "webcam_pilot" and r["threshold"] == e["threshold"]
    assert abs(r["raw_score"] - e["raw_score"]) <= max(1e-5, 3 * exp["video_bf16_repeatability_abs"])
    if "fusion_cuda_bf16" in exp and os.path.exists(os.path.join(PILOT, "fusion.json")):
        f = exp["fusion_cuda_bf16"]
        fu = read_fusion()
        assert fu["fusion_sha256"] == f["fusion_sha256"]
        got = fuse(f["inputs_logit"], fu)
        assert abs(got["fused_logit"] - f["fused_logit"]) < 1e-9 and got["verdict"] == f["verdict"]
