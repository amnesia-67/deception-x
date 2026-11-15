"""Socket server tests against the deployed REAL-data digest model (models/branch2_digest_schema.json).

Inputs are real clips' digest features from the stage-2 cache, never random vectors.
Skipped (not faked) when the exported model or the feature cache is missing.
"""

import json
import os
import socket
import time

import numpy as np
import pytest

from src.rlt_data import CACHE_DIR, GROUPS_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "models", "branch2_digest_schema.json")
TAB = os.path.join(CACHE_DIR, "features", "tabular.npz")
pytestmark = pytest.mark.skipif(not (os.path.exists(SCHEMA) and os.path.exists(TAB) and os.path.exists(GROUPS_PATH)),
                                reason="run `python main.py stage2` and `python main.py export` first; needs data/rlt_identity_groups.json (kept local, not published)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    from src.v2_socket_server import V2ModelServer

    port = _free_port()
    srv = V2ModelServer(host="127.0.0.1", port=port, device="cuda")
    srv.start(block=False)
    time.sleep(0.5)
    yield srv, port
    srv.stop()


@pytest.fixture(scope="module")
def real():
    d = np.load(TAB, allow_pickle=False)
    return list(d["ids"]), [str(n) for n in d["names"]], d["X"].astype(np.float32)


def _ask(port, payload: bytes):
    with socket.create_connection(("127.0.0.1", port), timeout=20) as s:
        s.sendall(payload + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf)


def test_ping_reports_real_model(server, real):
    _, port = server
    r = _ask(port, b'{"action": "ping"}')
    assert r["status"] == "ok" and r["features_expected"] == len(real[1])
    assert "Real-Life Trials" in r["trained_on"]
    rev = r["server_revision"]
    assert rev["code"]["commit"] and len(rev["code"]["source_tree_sha256"]) == 64
    assert isinstance(rev["code"]["dirty"], bool) and "untracked_source_files" in rev["code"]
    assert set(rev["artifacts_sha256"]) and "CRLF->LF" in rev["artifacts_hash"]


def test_schema_lists_feature_names_and_honest_metrics(server, real):
    _, port = server
    r = _ask(port, b'{"action": "schema"}')
    assert r["feature_names"] == real[1]
    assert 0.0 <= r["cv_metrics"]["auroc"] <= 1.0 and r["caveats"]


def test_named_feature_query_returns_probability_and_shap(server, real):
    _, port = server
    ids, names, X = real
    x = X[ids.index("trial_truth_001")]
    payload = {"features": {n: float(v) for n, v in zip(names, x) if np.isfinite(v)}}
    r = _ask(port, json.dumps(payload).encode())
    assert r["status"] == "success" and 0.0 <= r["probability"] <= 1.0
    assert r["verdict"] in ("Deceptive", "Truthful")
    drivers = r.get("top_deceptive_drivers", []) + r.get("top_truthful_drivers", [])
    assert drivers and all(d["name"] in names for d in drivers)


def test_vector_and_binary_payloads_agree(server, real):
    _, port = server
    ids, names, X = real
    x = np.nan_to_num(X[ids.index("trial_lie_010")], nan=0.0).astype(np.float32)
    r_json = _ask(port, json.dumps({"features": x.tolist()}).encode())
    with socket.create_connection(("127.0.0.1", port), timeout=20) as s:
        s.sendall(x.tobytes())
        s.shutdown(socket.SHUT_WR)
        r_bin = json.loads(s.makefile().readline())
    assert abs(r_json["probability"] - r_bin["probability"]) < 1e-5


def test_batch_query(server, real):
    _, port = server
    ids, _, X = real
    batch = np.nan_to_num(X[:4], nan=0.0).tolist()
    r = _ask(port, json.dumps({"batch": batch}).encode())
    assert r["status"] == "success" and len(r["results"]) == 4


def test_wrong_dimension_and_unknown_names_are_rejected(server, real):
    _, port = server
    r = _ask(port, b'{"features": [1.0, 2.0, 3.0]}')
    assert r["status"] == "error" and str(len(real[1])) in r["error"]
    r = _ask(port, b'{"features": {"not_a_feature": 1.0}}')
    assert r["status"] == "error" and "Unknown feature" in r["error"]


def test_branch1_requires_real_inputs(server):
    srv, _ = server
    if srv.branch1_model is None:
        pytest.skip("no Branch-1 checkpoint deployed")
    with pytest.raises(ValueError, match="random placeholder"):
        srv.process_jepa(None, None, None)


JEPA_DIR = os.path.join(ROOT, "models", "vjepa2_1_l_lora")


@pytest.mark.skipif(not os.path.isdir(JEPA_DIR), reason="run `python main.py export` first")
def test_jepa_video_reproduces_cross_validation_prediction(server):
    """Raw video + timestamps -> the served V-JEPA 2.1 ViT-L fold that held the clip out reproduces that clip's
    stored OOF probability (time-based selection + face crop on the server match training)."""
    import csv

    from scripts.preprocess_rlt import decode_video_frames
    from src.clip_store import ClipStore
    from src.v2_socket_client import V2ModelSocketClient

    srv, port = server
    if srv.jepa is None:
        pytest.skip("V-JEPA 2.1 ViT-L not loaded")
    assert srv.jepa.info()["model"] == "vjepa2_1_vitl_lora_5fold" and srv.jepa.info()["contract"]["hidden_width"] == 1024
    oof = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "runs", "tune_vjepa2_1_l", "oof.csv")))}
    cid = "trial_truth_020"
    frames, ts, _, _ = decode_video_frames(ClipStore().by_id[cid].video_path, 128)
    r = V2ModelSocketClient("127.0.0.1", port).predict_video(list(frames), timestamps=ts, quality=95,
                                                             request_id="v-1")
    assert r["status"] == "success" and r["request_id"] == "v-1" and len(r["per_fold_probability"]) == 5
    k = int(oof[cid]["fold"])
    assert abs(r["per_fold_probability"][k] - float(oof[cid]["prob"])) < 0.03
    assert len(r["temporal_attention_32"]) == 32 and abs(sum(r["temporal_attention_32"]) - 1) < 1e-2
    assert r["cv_auroc"] is not None and "not evidence of deception" in r["caveat"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "models", "ensemble_deploy.json")),
                    reason="run `python main.py export` first")
def test_analyze_from_raw_media_reproduces_training(server, real):
    """Raw video + float32 audio + transcript -> same digest features as training, and the tuned dasheng
    fold that held the clip out reproduces its stored out-of-fold probability (fold-swap/autocast safety).
    The reply echoes the request/source ids, names the served revision and applies the exported threshold."""
    import csv

    from scripts.preprocess_rlt import decode_video_frames
    from src.clip_store import ClipStore
    from src.v2_socket_client import V2ModelSocketClient

    srv, port = server
    if srv.ensemble is None:
        pytest.skip("ensemble not loaded")
    ids, names, X = real
    st = ClipStore()
    oof = {r["clip_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "runs", "tune_dasheng", "oof.csv")))}
    cid = "trial_truth_053"
    rec = st.by_id[cid]
    frames, ts, _, _ = decode_video_frames(rec.video_path, 128)
    r = V2ModelSocketClient("127.0.0.1", port).analyze(list(frames), np.asarray(st.audio(cid)), rec.transcript,
                                                       timestamps=ts, quality=95, request_id="req-1",
                                                       source_id="fixture-053")
    assert r["status"] == "success" and 0.0 <= r["probability"] <= 1.0
    assert r["request_id"] == "req-1" and r["source_id"] == "fixture-053"
    assert r["server_revision"]["artifacts_sha256"]["models/ensemble_deploy.json"]
    assert r["verdict"] == ("Deceptive" if r["probability"] >= r["threshold"] else "Truthful")
    assert r["feature_version"] == srv.ensemble.meta["feature_version"]
    ref = X[ids.index(cid)]
    for n, b in zip(names, ref):
        a = r["digest_features"][n]
        if n.startswith(("txt_", "voc_", "emo_", "intent_")) and a is not None and np.isfinite(b):
            assert abs(a - b) <= 0.02 * (abs(b) + 1e-3), (n, a, b)
        if n.startswith("face_") and n != "face_detect_rate" and a is not None and np.isfinite(b):
            assert abs(a - b) <= 0.25 * (abs(b) + 1e-3), (n, a, b)  # JPEG + re-detection; measured ~5-7%
    k = int(oof[cid]["fold"])
    assert abs(r["branches"]["tuned_dasheng"]["per_fold"][k] - float(oof[cid]["prob"])) < 0.02
    assert set(r["branches"]) == {"tuned_dasheng", "xgb_digest"} and r["caveats"]


def test_webcam_pilot_profile_is_opt_in(server):
    """Without --webcam-pilot the profile is refused by name; unknown profiles are refused; default is unchanged."""
    _, port = server
    r = _ask(port, json.dumps({"action": "ping"}).encode())
    assert r["profiles"]["served"] == "default" and isinstance(r["profiles"]["webcam_pilot"], str)
    for profile, msg in (("webcam_pilot", "webcam_pilot profile unavailable"), ("other", "profile must be")):
        r = _ask(port, json.dumps({"action": "jepa_video", "profile": profile, "frames": [], "fps": 30,
                                   "request_id": "p1"}).encode())
        assert r["status"] == "error" and msg in r["error"] and r["request_id"] == "p1"
