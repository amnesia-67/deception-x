import numpy as np
import pytest

from sunhacks.features import audio_diagnostics, video_diagnostics


def test_silence_is_finite_and_missing_is_distinct():
    silence = audio_diagnostics(np.zeros(1600, dtype=np.float32))
    assert silence["status"] == "ok"
    assert silence["rms_dbfs"] == -160
    assert silence["activity_fraction_proxy"] == 0
    assert audio_diagnostics(None)["rms_dbfs"] is None


def test_audio_reports_clipping_and_energy_without_claiming_voice():
    result = audio_diagnostics(np.ones(320, dtype=np.float32))
    assert result["clipping_fraction"] == 1
    assert result["rms_dbfs"] == 0
    assert "not voice activity detection" in result["activity_method"]
    with pytest.raises(ValueError):
        audio_diagnostics(np.array([np.nan]))


def test_rgb_motion_is_quality_diagnostic():
    result = video_diagnostics([np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 255, dtype=np.uint8)])
    assert result["brightness_mean"] == .5
    assert result["mean_absolute_frame_difference"] == 1
    assert video_diagnostics([])["status"] == "missing"
