# Bundled rPPG weights

- `facephys_rlap.onnx` + `facephys_rlap_state.json.gz`: FacePhys (RLAP-trained), 721,388 params, MIT + Privacy Protection Addendum (run on-device only). Source: https://github.com/KegangWangCCNU/FacePhys-Release @ 07a1402.
- `blaze_face_short_range.tflite`: MediaPipe BlazeFace short-range face detector (Apache-2.0).

`sunhacks/rppg.py` loads these first (sha256-checked against `sunhacks/models_manifest.json`); the download cache is only a fallback.
