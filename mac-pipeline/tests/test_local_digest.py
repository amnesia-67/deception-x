import numpy as np
import pytest

from sunhacks.local_digest import (
    DigestInputError, build_digest_row, combine_ensemble_scores, load_digest_schema,
    prepare_digest_faces, select_timestamp_frames,
)


@pytest.fixture
def inputs():
    ts = np.arange(128) / 128
    boxes = np.column_stack((100 + ts, np.full(128, 100), np.full(128, 100)))
    audio_t = np.arange(16000) / 16000
    return {
        "transcript": "Um I saw the red car outside, then I heard a loud noise.",
        "audio": ((0.2 + 0.1 * np.sin(2 * np.pi * 3 * audio_t)) *
                  np.sin(2 * np.pi * 220 * audio_t)).astype(np.float32),
        "duration_s": 1.0,
        "face_crops": np.broadcast_to(np.zeros((384, 384, 3), np.uint8), (128, 384, 384, 3)),
        "face_metadata": {"boxes_cxcys": boxes, "timestamps": ts, "face_detect_rate": 1.0},
        "emotion_probabilities": np.eye(9)[4],
        "qwen_embedding": np.eye(1, 1024)[0],
        "intent_embeddings": np.eye(33, 1024),
    }


def test_schema_is_exact_89_field_companion():
    schema = load_digest_schema()
    assert schema["feature_version"] == "v2-no-within-z"
    assert len(schema["feature_names"]) == 89
    assert schema["feature_names"] == sorted(schema["feature_names"])


def test_irregular_capture_selects_by_time_and_exposes_repeats():
    ts = np.array([0, .01, .04, .09, .2, .4, .7, 1, 1.1, 1.4, 1.8, 2.4, 3.0, 3.9, 4.3, 5.0])
    indices, selected = select_timestamp_frames(ts)
    targets = (np.arange(128) + .5) * 5 / 128
    expected = np.argmin(abs(targets[:, None] - ts[None, :]), axis=1)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(selected, ts[expected])
    assert len(np.unique(indices)) <= 16 < len(indices)


@pytest.mark.parametrize("timestamps", [np.arange(15), np.zeros(16),
    np.arange(16)[::-1], np.r_[np.arange(15), np.nan]])
def test_invalid_capture_times_are_rejected(timestamps):
    with pytest.raises(DigestInputError):
        select_timestamp_frames(timestamps)


def test_actual_input_semantics_and_timing_are_preserved(inputs):
    result = build_digest_row(**inputs)
    values = result.public_features()
    assert result.values.shape == (1, 89) and result.values.dtype == np.float32
    assert tuple(values) == result.feature_names
    assert values["emo_neutral"] == 1
    assert values["txt_word_count"] == 13
    assert values["txt_disfluency_per100"] == pytest.approx(100 / 13)
    assert values["face_head_speed_mean"] == pytest.approx(.01, rel=1e-5)
    assert values["face_motion_mean"] == 0
    assert values["intent_00"] > .999
    assert sum(values[f"intent_{i:02d}"] for i in range(33)) == pytest.approx(1, abs=1e-6)


def test_silence_keeps_undefined_voice_as_nan_not_zero(inputs):
    inputs["audio"] = np.zeros(16000, np.float32)
    result = build_digest_row(**inputs)
    voice_indices = [i for i, n in enumerate(result.feature_names) if n.startswith("voc_")]
    assert len(voice_indices) == 16
    assert np.isnan(result.values[0, voice_indices]).all()
    assert set(result.missing_features) == {result.feature_names[i] for i in voice_indices}
    assert all(result.public_features()[result.feature_names[i]] is None for i in voice_indices)


@pytest.mark.parametrize("field,value", [
    ("emotion_probabilities", None), ("emotion_probabilities", np.ones(9)),
    ("qwen_embedding", np.zeros(1024)), ("qwen_embedding", np.ones(128)),
    ("intent_embeddings", np.ones((32, 1024))), ("transcript", ""),
    ("audio", np.full(16000, np.nan)), ("sample_rate", 48000),
])
def test_missing_or_incompatible_model_outputs_fail_closed(inputs, field, value):
    inputs[field] = value
    with pytest.raises(DigestInputError):
        build_digest_row(**inputs)


def test_face_callback_receives_real_selected_timestamps(inputs):
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(16)]
    ts = np.arange(16) * .1
    expected, selected_ts = select_timestamp_frames(ts)
    def crop(selected, times):
        assert [int(f[0, 0, 0]) for f in selected] == expected.tolist()
        np.testing.assert_array_equal(times, selected_ts)
        meta = dict(inputs["face_metadata"], timestamps=times)
        return inputs["face_crops"], meta
    crops, meta = prepare_digest_faces(frames, ts, crop)
    assert crops is inputs["face_crops"]
    np.testing.assert_array_equal(meta["timestamps"], selected_ts)


def test_face_adapter_cannot_replace_capture_clock(inputs):
    frames = [np.zeros((4, 4, 3), np.uint8)] * 16
    def bad_crop(selected, times):
        return inputs["face_crops"], dict(inputs["face_metadata"], timestamps=times + 1)
    with pytest.raises(DigestInputError, match="changed"):
        prepare_digest_faces(frames, np.arange(16) * .1, bad_crop)


def test_shared_selected_cache_preserves_irregular_timestamps_and_frame_ids(inputs):
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(128)]
    ts = np.arange(128, dtype=np.float64) ** 2 / 10000
    ts[70] = ts[69]  # A repeated selection is legitimate and must stay explicit.
    def crop(selected, times):
        assert [int(f[0, 0, 0]) for f in selected] == list(range(128))
        assert all(frame is original for frame, original in zip(selected, frames))
        np.testing.assert_array_equal(times, ts)
        return inputs["face_crops"], dict(inputs["face_metadata"], timestamps=times)
    crops, meta = prepare_digest_faces(frames, ts, crop, already_selected=True)
    assert crops is inputs["face_crops"]
    np.testing.assert_array_equal(meta["timestamps"], ts)


@pytest.mark.parametrize("n_frames", [127, 129])
def test_shared_cache_rejects_wrong_count_before_preprocessor(n_frames):
    def should_not_run(*args):
        pytest.fail("Invalid cache reached face preprocessor")
    with pytest.raises(DigestInputError):
        prepare_digest_faces([np.zeros((4, 4, 3), np.uint8)] * n_frames,
                             np.arange(n_frames), should_not_run, already_selected=True)


@pytest.mark.parametrize("ts", [np.zeros(128), np.arange(128)[::-1], np.r_[np.arange(127), np.nan]])
def test_shared_cache_rejects_invalid_timestamp_span(ts):
    with pytest.raises(DigestInputError):
        prepare_digest_faces([np.zeros((4, 4, 3), np.uint8)] * 128,
                             ts, lambda *_: pytest.fail("Invalid timestamps reached preprocessor"),
                             already_selected=True)


def test_stack_respects_branch_order_and_exact_platt_formula():
    meta = {"format": "sunhacks-ensemble-v1", "branches": ["xgb_digest", "tuned_dasheng"],
            "weights": [.4, .6], "platt_A": [1, 1], "platt_B": [0, 0]}
    assert combine_ensemble_scores(.8, .2, meta) == pytest.approx(.56)
    meta["platt_A"] = [2, .5]
    meta["platt_B"] = [-.1, .3]
    expected = .4 / (1 + np.exp(-(2 * np.log(.2 / .8) - .1)))
    expected += .6 / (1 + np.exp(-(.5 * np.log(.8 / .2) + .3)))
    assert combine_ensemble_scores(.8, .2, meta) == pytest.approx(expected)


@pytest.mark.parametrize("change", [{"weights": [.6, .5]}, {"branches": ["vision", "xgb_digest"]},
    {"platt_A": [np.nan, 1]}])
def test_stack_rejects_unexported_or_invalid_semantics(change):
    meta = {"format": "sunhacks-ensemble-v1", "branches": ["tuned_dasheng", "xgb_digest"],
            "weights": [.6, .4], "platt_A": [1, 1], "platt_B": [0, 0]}
    meta.update(change)
    with pytest.raises(DigestInputError):
        combine_ensemble_scores(.5, .5, meta)
