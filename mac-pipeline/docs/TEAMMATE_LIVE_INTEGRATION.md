# Reviewed teammate fusion integration

> Current deployment: the teammate withdrew the previous artifacts and turned the socket off while retraining. Old artifacts are quarantined. See [REPLACEMENT_HANDOFF.md](REPLACEMENT_HANDOFF.md) and [FAST_RUNTIME_STATUS.md](FAST_RUNTIME_STATUS.md). The details below describe earlier/reference interfaces, not authorization to activate the withdrawn weights.

Copy `sunhacks/teammate_fusion.py` and `sunhacks/vendor/` into the installed package. Do not overwrite its existing top-level `sunhacks/__init__.py` with the empty test staging initializer. Copy `tests/test_teammate_fusion.py` into the project tests.

The vendored model is a byte-exact copy of `amnesia-67/sunhacks-2026/src/multimodal_fusion.py` at commit `6a16734c1af7d4a5ff24400b06849241b285d5fb`. Its SHA256 is `c58c6e53d8f87e80ade698c705d7fde0aafa95cd57c7c849e41e80c9b0738124`. Only reviewed pure model definitions are imported. The friend's server, dataset generator, trainer, tabular classifier and calibration pipeline are never started.

Call `run_teammate_fusion(branches, config)` for fusion architecture `friend-reda-dare-synthetic-v1`. Configure:

```json
{
  "models": {
    "fusion": {
      "architecture": "friend-reda-dare-synthetic-v1",
      "source_revision": "6a16734c1af7d4a5ff24400b06849241b285d5fb",
      "path": "/absolute/path/to/branch1_fusion.pt",
      "sha256": "cdeeb2f19c51aa9c614665e9c36554e9b1efe03fca367842ee94beec95820e78"
    }
  },
  "runtime": {"text_scope": "video_window"}
}
```

Pass original real `base` or `trained` branch results with exact model IDs. Vision requires a private `_sequence_embedding` of `[32,1024]`: spatial mean of the native V-JEPA2.1 `[1,32*24*24,1024]` token grid, retaining its32 temporal tubelets. Dasheng requires private raw hidden states `[variable_time,768]` under `_sequence_embedding`, before any pooling or trained adapter. Text requires its native1024-dimensional private `_embedding`.

The wrapper applies the exported768-to768 dense audio adapter (`Linear`, `LayerNorm`, `GELU`) to native time steps, then linearly interpolates to32 steps using `align_corners=False`. It slices the first128 Qwen dimensions and renormalizes. These are explicitly documented live-demo transformations, not an assertion that synthetic training supplied a real-video or Qwen extraction contract. Audio/video timestamps still belong to the parent window; this interpolation is not independently calibrated hardware synchronization. Report the actual text scope if it is manual session context rather than clip-aligned ASR.

It strict-loads **both** `network_state_dict` and `adapter_state_dict` from the exact inspected checkpoint with `weights_only=True`, `mmap=True`, CPU FP32. It verifies the complete file SHA256 each call and keeps one small reviewed model cached. CPU conversion preserves each exported BF16 value. Missing, mock, zero-padded, malformed and nonfinite inputs are rejected instead of synthesized. The wrapper never downloads or trains anything.

Success status is `synthetic_trained`, deliberately distinct from `trained`. The fields are `raw_score`, `raw_logit`, `modality_gates`, `feature_distances`, provenance, shape metadata and limitations. There are no verdict, confidence, calibrated or class-probability fields. The score is an unvalidated synthetic target-class sigmoid, not a probability of deception. Emotion2vec is explicitly unused by this three-modality export. Existing complete-input stacking must reject this status; the synthetic/leaked calibration parameters are not used.

Validation:20 CPU tests passed, including the actual uploaded checkpoint's strict load, exact forward parity with the reviewed source and exported audio-adapter values; tamper/revision/identity checks; and missing/mock/zero/nonfinite/shape rejection. Tests use transient generated feature fixtures only. No training, dataset creation, sensor use, GPU inference or API calls occurred in this subtask.
