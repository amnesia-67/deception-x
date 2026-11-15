"""
Unit tests for the peak anomaly keyframe extractor.
Accredited to Gemini.pdf Pages 2, 30, 40.
"""

import numpy as np
import pytest
from PIL import Image

from src.keyframe_extractor import PeakAnomalyKeyframeExtractor

def test_keyframe_extractor_synthetic_generation():
    extractor = PeakAnomalyKeyframeExtractor(target_size=(256, 256), min_frames=4, max_frames=6)
    
    keyframes = extractor.extract_keyframes(k=6)
    assert len(keyframes) == 6
    for kf in keyframes:
        assert "frame_index" in kf
        assert "timestamp_sec" in kf
        assert "anomaly_score" in kf
        assert isinstance(kf["image"], Image.Image)
        assert kf["image"].size == (256, 256)
        assert kf["mime_type"] == "image/jpeg"
        assert len(kf["jpeg_bytes"]) > 100  # Valid JPEG bytes

def test_keyframe_extractor_temporal_nms():
    extractor = PeakAnomalyKeyframeExtractor(nms_window_frames=10)
    # Create anomaly score with two very close peaks
    scores = np.zeros(100, dtype=np.float32)
    scores[20] = 5.0
    scores[22] = 4.8  # Should be suppressed by NMS
    scores[50] = 4.0
    
    peaks = extractor.find_peak_timestamps(scores, k=2)
    assert 20 in peaks
    assert 22 not in peaks
    assert 50 in peaks

def test_keyframe_extractor_with_real_frames():
    extractor = PeakAnomalyKeyframeExtractor()
    # 60 mock RGB frames
    frames = [np.full((480, 640, 3), fill_value=i, dtype=np.uint8) for i in range(60)]
    scores = np.linspace(0, 10, 60)
    
    extracted = extractor.extract_keyframes(frames=frames, anomaly_scores=scores, k=4)
    assert len(extracted) == 4
    for item in extracted:
        assert item["image"].size == (256, 256)
        assert isinstance(item["jpeg_bytes"], bytes)
