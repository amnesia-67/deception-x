"""CPU tests for the repairs requested in MODEL_PIPELINE_REPAIR_HANDOFF.md.

Formula checks use small constructed inputs (a box moving at a known speed); nothing here trains a model.
"""

import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- duration-invariant face motion
def _moving_box_clip(duration_s, n=128, speed_side_per_s=0.5, side=200.0):
    ts = (np.arange(n) + 0.5) * duration_s / n
    cx = 400 + speed_side_per_s * side * ts
    boxes = np.stack([cx, np.full(n, 300.0), np.full(n, side)], 1)
    frames = np.zeros((n, 64, 64, 3), np.uint8)
    return frames, {"boxes_cxcys": boxes.tolist(), "timestamps": ts.tolist(), "face_detect_rate": 1.0}


def test_head_speed_is_independent_of_answer_length():
    from src.features_tabular import face_features

    short = face_features(*_moving_box_clip(7.0))
    long = face_features(*_moving_box_clip(60.0))
    for f in (short, long):
        assert abs(f["face_head_speed_mean"] - 0.5) < 1e-6
    assert abs(short["face_head_speed_p95"] - long["face_head_speed_p95"]) < 1e-6


def test_jitter_is_not_amplified_by_dense_sampling():
    """Detector jitter divided by a 55 ms adjacent-frame gap used to explode on 7 s answers."""
    from src.features_tabular import face_features

    rng = np.random.default_rng(0)
    out = {}
    for dur in (7.0, 27.0):
        frames, meta = _moving_box_clip(dur, speed_side_per_s=0.0)
        b = np.asarray(meta["boxes_cxcys"])
        b[:, 0] += rng.normal(0, 2.0, len(b))  # 1% of face side
        meta["boxes_cxcys"] = b.tolist()
        out[dur] = face_features(frames, meta)["face_head_speed_mean"]
    assert out[7.0] / out[27.0] < 1.5, out


def test_lag_pairs_hit_the_target_lag():
    from src.features_tabular import MOTION_LAG_S, lag_pairs

    for dur in (4.5, 7.0, 27.0, 81.0):
        ts = (np.arange(128) + 0.5) * dur / 128
        i, j, dt = lag_pairs(ts)
        assert (j > i).all() and len(i) >= 3
        assert np.all(np.abs(dt - MOTION_LAG_S) <= 0.5 * MOTION_LAG_S)


def test_text_features_have_no_raw_counts():
    from src.features_tabular import text_features

    f = text_features("Did I? No. Um, I did not, I did not do it?", 5.0)
    assert "txt_word_count" not in f and "txt_question_marks" not in f
    assert f["txt_question_marks_per100"] == pytest.approx(100 * 2 / 12)


# ---------------------------------------------------------------- classwise reporting / threshold
def test_classwise_confusion_and_rates():
    from src.tuning import classwise

    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.2, 0.1, 0.6, 0.3, 0.2])
    c = classwise(y, p, 0.5)
    assert (c["tp_lie"], c["fn_lie"], c["tn_truth"], c["fp_truth"]) == (1, 2, 3, 1)
    assert c["lie_sensitivity"] == pytest.approx(1 / 3) and c["truth_specificity"] == pytest.approx(0.75)
    assert c["predicted_lie_fraction"] == pytest.approx(2 / 7)
    c2 = classwise(y, p, np.array([0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5]))  # per-row thresholds
    assert c2["tp_lie"] == 2


def test_balanced_threshold_prefers_balance_and_0_5_on_ties():
    from src.tuning import balanced_threshold

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.45, 0.42, 0.40, 0.7, 0.2, 0.1, 0.3, 0.35])  # lies mostly scored below 0.5
    t = balanced_threshold(y, p)
    assert 0.35 < t <= 0.40
    assert balanced_threshold(np.array([1, 0]), np.array([0.9, 0.1])) == 0.5


def test_prior_matched_threshold_calls_the_base_rate_deceptive():
    from src.tuning import prior_matched_threshold

    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 50)
    p = np.clip(rng.normal(0.42, 0.1, 100) + 0.05 * y, 0, 1)   # scores shifted below 0.5
    t = prior_matched_threshold(y, p)
    assert t < 0.5 and abs((p >= t).mean() - 0.5) <= 0.02


# ---------------------------------------------------------------- provenance
def test_state_hash_is_order_independent_and_value_sensitive():
    torch = pytest.importorskip("torch")
    from src.provenance import state_hash

    a = {"lora": {"x": torch.ones(2), "y": torch.zeros(3)}, "head": {"w": torch.arange(4.0)}}
    b = {"head": {"w": torch.arange(4.0)}, "lora": {"y": torch.zeros(3), "x": torch.ones(2)}}
    assert state_hash(a) == state_hash(b)
    b["lora"]["x"] = torch.ones(2) * 2
    assert state_hash(a) != state_hash(b)


def test_stale_fold_files_are_refused():
    from scripts.tune_backbones import check_fold_file

    fp = {"cache": "c", "code": "k", "all": "A"}
    assert check_fold_file({"fingerprint": fp}, fp, 0, False) == "ok"
    with pytest.raises(SystemExit):
        check_fold_file({}, fp, 0, False)                                       # legacy, no fingerprint
    with pytest.raises(SystemExit):
        check_fold_file({"fingerprint": {**fp, "code": "x", "all": "B"}}, fp, 0, False)
    assert check_fold_file({}, fp, 0, True) == "stale-accepted"


def test_code_hash_ignores_line_endings(tmp_path, monkeypatch):
    from src import provenance

    monkeypatch.setattr(provenance, "ROOT", str(tmp_path))
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    crlf = provenance.code_hash(["a.py"])
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    assert provenance.code_hash(["a.py"]) == crlf
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 3\n")
    assert provenance.code_hash(["a.py"]) != crlf


def _git_repo(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def g(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *a],
                       cwd=tmp_path, check=True, capture_output=True)

    g("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "src" / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "run_v2_socket.py").write_bytes(b"import src.a\n")
    weights = b"\x00\x01binary weights\x02" * 100
    import hashlib

    (tmp_path / "models" / "w.bin").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + hashlib.sha256(weights).hexdigest().encode()
        + b"\nsize %d\n" % len(weights))
    g("add", "-A")
    g("commit", "-q", "-m", "c")
    (tmp_path / "models" / "w.bin").write_bytes(weights)  # what a git-lfs smudge leaves on disk
    return g


def test_git_revision_is_content_based(tmp_path, monkeypatch):
    """CRLF checkouts and smudged git-lfs objects are not 'dirty'; untracked files are listed but not hashed;
    real edits are dirty and change the tree hash."""
    from src import provenance

    _git_repo(tmp_path)
    monkeypatch.setattr(provenance, "ROOT", str(tmp_path))
    clean = provenance.git_revision()
    assert clean["dirty"] is False and clean["modified_tracked_files"] == [] and clean["untracked_source_files"] == {}

    (tmp_path / "src" / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")           # Windows checkout
    (tmp_path / "src" / "draft.py").write_bytes(b"print('not committed')\r\n")  # untracked
    r = provenance.git_revision()
    assert r["dirty"] is False
    assert r["source_tree_sha256"] == clean["source_tree_sha256"]
    assert list(r["untracked_source_files"]) == ["src/draft.py"]
    assert r["untracked_source_files"]["src/draft.py"] == provenance.sha256_bytes(b"print('not committed')\n")

    (tmp_path / "src" / "a.py").write_bytes(b"x = 1\r\ny = 3\r\n")           # a real change
    r = provenance.git_revision()
    assert r["dirty"] is True and r["modified_tracked_files"] == ["src/a.py"]
    assert r["source_tree_sha256"] != clean["source_tree_sha256"]

    (tmp_path / "src" / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "models" / "w.bin").write_bytes(b"corrupted")                  # LFS object that is not the commit's
    assert provenance.git_revision()["modified_tracked_files"] == ["models/w.bin"]


def test_artifact_hash_is_line_ending_neutral(tmp_path, monkeypatch):
    from src import provenance

    monkeypatch.setattr(provenance, "ROOT", str(tmp_path))
    (tmp_path / "m.json").write_bytes(b'{\r\n "a": 1\r\n}\r\n')
    crlf = provenance.sha256_lf_file("m.json")
    (tmp_path / "m.json").write_bytes(b'{\n "a": 1\n}\n')
    assert provenance.sha256_lf_file("m.json") == crlf == provenance.sha256_bytes(b'{\n "a": 1\n}\n')


# ---------------------------------------------------------------- server echo / revision
def test_echo_fields_validates_and_copies():
    from src.v2_socket_server import echo_fields

    assert echo_fields({"request_id": "r1", "source_id": 7, "frames": []}) == {"request_id": "r1", "source_id": 7}
    for bad in ({"request_id": "x" * 129}, {"request_id": {"a": 1}}, {"source_id": True}):
        with pytest.raises(ValueError):
            echo_fields(bad)


def test_server_revision_hashes_served_artifacts():
    from src.v2_socket_server import server_revision

    r = server_revision()
    assert "commit" in r["code"]
    for rel, h in r["artifacts_sha256"].items():
        assert os.path.exists(os.path.join(ROOT, rel)) and len(h) == 64


# ---------------------------------------------------------------- exported manifest contract
MANIFEST = os.path.join(ROOT, "models", "ensemble_deploy.json")


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="run `python main.py export` first")
def test_manifest_publishes_nested_classwise_results_and_threshold():
    from src.features_tabular import FEATURE_VERSION

    m = json.load(open(MANIFEST))
    assert m["format"] == "sunhacks-ensemble-v2" and m["feature_version"] == FEATURE_VERSION
    assert m["branches"] == ["tuned_dasheng", "xgb_digest"] and abs(sum(m["weights"]) - 1) < 1e-6
    assert 0 < m["threshold"] < 1 and "class base rate" in m["threshold_provenance"]
    cv = m["cv_metrics"]
    assert cv["protocol"].startswith("fully nested")
    for k in ("tp_lie", "fn_lie", "tn_truth", "fp_truth", "lie_sensitivity", "truth_specificity", "brier"):
        assert k in cv["at_threshold_refit_per_outer_fold"] and k in cv["at_0.5"]
        assert k in cv["at_fixed_exported_cutoff_on_nested_scores"]
    fx = cv["at_fixed_exported_cutoff_on_nested_scores"]
    assert fx["tp_lie"] + fx["fn_lie"] == 61 and abs(fx["threshold"] - m["threshold"]) < 1e-12
    schema = json.load(open(os.path.join(ROOT, "models", "branch2_digest_schema.json")))
    assert schema["feature_version"] == FEATURE_VERSION
