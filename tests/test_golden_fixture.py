"""The served code must reproduce the committed golden fixture (generated, no person) within measured tolerances.

Same check the Mac runs against its own implementation (compare to `cuda_fp32` with `suggested_tolerances`).
"""

import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures", "golden_v1")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(FIX, "expected.json")), reason="no fixture")


def test_fixture_inputs_are_the_recorded_bytes():
    import hashlib

    exp = json.load(open(os.path.join(FIX, "expected.json")))
    for rel, h in exp["input_sha256"].items():
        data = open(os.path.join(FIX, rel), "rb").read()
        if rel == "transcript.txt":
            data = data.decode("utf-8").encode()
        assert hashlib.sha256(data).hexdigest() == h, rel


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "models", "ensemble_deploy.json")), reason="no export")
def test_served_ensemble_reproduces_golden_fixture():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from scripts.make_golden_fixture import load_payload
    from src.ensemble_service import EnsembleService
    from src.v2_socket_server import server_revision

    exp = json.load(open(os.path.join(FIX, "expected.json")))
    if server_revision()["artifacts_sha256"].get("models/ensemble_deploy.json") != \
            exp["artifacts"]["artifacts_sha256"]["models/ensemble_deploy.json"]:
        pytest.skip("fixture was generated for different exported artifacts; regenerate it")
    payload, _ = load_payload()
    r = EnsembleService(device="cuda").predict(payload)
    e = exp["cuda_bf16"]
    tol = exp["suggested_tolerances"]
    i, ei = r["intermediates"], e["intermediates"]
    assert i["selected_frame_indices"] == ei["selected_frame_indices"]
    assert i["crops_uint8_sha256"] == ei["crops_uint8_sha256"]
    assert r["face"]["centre_crop_fallback"] == e["face"]["centre_crop_fallback"]
    assert abs(i["final_score"] - ei["final_score"]) <= max(tol["final_score"], 1e-6)
    assert np.max(np.abs(np.subtract(i["dasheng_fold_logits"], ei["dasheng_fold_logits"]))) <= max(tol["dasheng_fold_logits"], 1e-6)
    got = [v for _, v in i["digest_88_ordered"]]
    want = [v for _, v in ei["digest_88_ordered"]]
    assert [n for n, _ in i["digest_88_ordered"]] == [n for n, _ in ei["digest_88_ordered"]]
    assert [g is None for g in got] == [w is None for w in want]
    assert max(abs(g - w) for g, w in zip(got, want) if g is not None) <= max(tol["digest_88"], 1e-6)
    assert r["verdict"] == e["verdict"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "models", "vjepa2_1_l_lora", "deploy_meta.json")),
                    reason="no ViT-L export")
def test_served_vitl_reproduces_golden_fixture():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from scripts.make_golden_fixture import load_payload
    from src.jepa_service import JEPAService

    exp = json.load(open(os.path.join(FIX, "expected.json")))
    if "video_cuda_bf16" not in exp:
        pytest.skip("fixture has no ViT-L section; run make_golden_fixture.py --video")
    svc = JEPAService(device="cuda")
    e = exp["video_cuda_bf16"]
    if svc.manifest_sha256 != e["manifest_sha256"]:
        pytest.skip("fixture was generated for a different ViT-L export; regenerate it")
    payload, _ = load_payload()
    r = svc.predict({k: payload[k] for k in ("frames", "timestamps")})
    assert r["model"] == e["model"] and r["model_id"] == e["model_id"]
    assert r["face"]["unique_frames_used"] == e["face"]["unique_frames_used"]
    assert r["face"]["centre_crop_fallback"] == e["face"]["centre_crop_fallback"]
    assert np.max(np.abs(np.subtract(r["per_fold_probability"], e["per_fold_probability"]))) <= 1e-5


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "models", "vjepa2_1_l_lora", "deploy_meta.json")),
                    reason="no ViT-L export")
def test_vitl_shared_frozen_prefix_equals_full_per_fold_forward():
    """The served speed path (frozen base blocks run once, shared by all folds) must give exactly the same
    per-fold scores and attention as running the full encoder separately for every fold."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from scripts.make_golden_fixture import load_payload
    from src.jepa_service import JEPAService

    svc = JEPAService(device="cuda")
    assert svc.share_prefix and sorted(set(svc.first_adapted.values())) == [16, 20]
    payload, _ = load_payload()
    vp = {k: payload[k] for k in ("frames", "timestamps")}
    fast = svc.predict(dict(vp))
    svc.share_prefix = False
    full = svc.predict(dict(vp))
    assert fast["shared_frozen_prefix"] and not full["shared_frozen_prefix"]
    assert fast["per_fold_probability"] == full["per_fold_probability"]
    assert fast["temporal_attention_32"] == full["temporal_attention_32"]
    assert fast["raw_score"] == full["raw_score"]
