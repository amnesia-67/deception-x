# Offline teammate handoff

> Current deployment: the teammate withdrew the previous artifacts and turned the socket off while retraining. Old artifacts are quarantined. See [REPLACEMENT_HANDOFF.md](REPLACEMENT_HANDOFF.md) and [FAST_RUNTIME_STATUS.md](FAST_RUNTIME_STATUS.md). The details below describe earlier/reference interfaces, not authorization to activate the withdrawn weights.

Inference uses local artifacts. The Mac never contacts the teammate's NVIDIA computer. Transfer directories by any agreed offline means, then set their paths in the local config. No training, tuning, dataset download or dataset creation is part of this repository's build.

LoRA adapters change selected encoder weights. They do not by themselves provide a fusion network, tabular model, calibration parameters, or a validated task label. The reference heads below make the artifact interface concrete; they are **not** an implementation or checkpoint of the plan's unspecified ReDA/DARE/DMC/MMPDA methods. If the teammate chooses a different fusion architecture, implement and verify a new named contract before loading its weights.

## Encoder contract

Use `sunhacks-input-v1` preprocessing exactly as documented in [MODEL_CONTRACTS.md](MODEL_CONTRACTS.md), including the 64-frame video sampling policy, RGB spatial transform, 16 kHz mono audio, missing-input rejection, transcript timing scope, and pooling. The fused vector is ordered:

| Branch | Exact model ID | Pooled width | Pooling |
|---|---|---:|---|
| vision | `facebookresearch/vjepa2:vjepa2_1_vit_large_384` | 1024 | `mean_all_tokens` |
| dasheng | `mispeech/dasheng-base` | 768 | `mean_hidden_states` |
| emotion | `emotion2vec/emotion2vec_plus_large` | 1024 | `utterance_mean` |
| text | `Qwen/Qwen3-Embedding-0.6B` | 1024 | `last_token_l2` |

Each branch must have the exact base revision recorded. Each encoder LoRA directory needs the manifest and checksums described in MODEL_CONTRACTS.md. Preserve the architecture's original dimensions and export `adapter_model.safetensors` and `adapter_config.json`; no arbitrary Python or pickle exports are accepted by the adapter path.

The contract version alone is insufficient: training and inference must also have identical `preprocessing_settings`. Generate that complete dictionary with `sunhacks.models.input_settings(actual_runtime_config)` and retain it in each encoder-adapter and fusion manifest. The loader compares it exactly. For example, an adapter or fusion head trained at 16 sampled fps is rejected by a 24 fps runtime even though both use 64 video frames and the same embedding dimensions. Text token limits, crop/audio semantics and other settings included by the helper must also match. Do not copy a hand-written partial dictionary.

## Reference fusion artifact

`sunhacks-fusion-mlp-v1` is the exact CPU PyTorch module returned by `sunhacks.heads.fusion_module()`:

```text
Sequential(
  0: LayerNorm(3840),
  1: Linear(3840, 256),
  2: GELU(),
  3: Dropout(0.0),
  4: Linear(256, 2)
)
```

Export that module's state dictionary to `model.safetensors`; loading is strict. The two class names and their order must be supplied explicitly. The pipeline never assigns task semantics to an unlabeled index. Softmax output is exposed as experimental and uncalibrated. Its predictive quality has not been established here.

Place beside it `handoff.json` with this structure, replacing every placeholder and adding all four ordered `inputs` entries using the table above:

```json
{
  "input_contract": "sunhacks-input-v1",
  "architecture": "sunhacks-fusion-mlp-v1",
  "preprocessing_settings": {},
  "class_names": ["class_0_name", "class_1_name"],
  "text_scope": "video_window",
  "inputs": [
    {
      "branch": "vision",
      "model_id": "facebookresearch/vjepa2:vjepa2_1_vit_large_384",
      "base_revision": "EXACT_RUNTIME_REVISION",
      "output_dim": 1024,
      "pooling": "mean_all_tokens",
      "adapter_sha256": null
    }
  ],
  "sha256": {"model.safetensors": "SHA256_OF_FILE"}
}
```

`adapter_sha256` is `null` for a base encoder, or the SHA256 of the exact local encoder `adapter_model.safetensors` used for fusion training. A fusion model trained with different adapter weights is rejected even if vector dimensions happen to match. Missing, error, and mock encoder outputs are rejected; they are never converted to zeros.

The empty `preprocessing_settings` object in the schema illustration is a placeholder and will be rejected. Populate it programmatically before export:

```python
from sunhacks.models import input_settings
manifest["preprocessing_settings"] = input_settings(actual_runtime_config)
```

By default the text must be scoped to the same video window (`config.runtime.text_scope="video_window"`). Manually entered session context is not word-aligned. If the teammate deliberately trains with session context, set both the artifact's `text_scope` to `session_context` and `models.fusion.allow_session_context=true`; the runtime scope must also match. This is an explicit training/inference decision.

Configure:

```json
{"models": {"fusion": {"path": "/absolute/path/to/fusion"}}}
```

## Reference tabular artifact

`sunhacks-xgboost-diagnostics-v1` accepts a native XGBoost `model.json` with objective `binary:logistic`. It loads no Python pickle. Its embedded `feature_names` must exactly match the manifest's ordered names. Select only these implemented scalar diagnostics:

- `audio.rms_dbfs`
- `audio.peak_dbfs`
- `audio.clipping_fraction`
- `audio.activity_fraction_proxy`
- `audio.duration_seconds`
- `video.brightness_mean`
- `video.mean_absolute_frame_difference`

These are image/audio quality diagnostics. They are not the plan's 250 physiological or behavioral features. In particular, RMS threshold activity is not Silero VAD, frame difference is not a facial action unit, and neither is a deception indicator. Missing input is rejected; no imputation, zero filling, z-score baseline, or fabricated feature expansion occurs.

The artifact directory contains `model.json` and:

```json
{
  "input_contract": "sunhacks-input-v1",
  "architecture": "sunhacks-xgboost-diagnostics-v1",
  "class_names": ["class_0_name", "class_1_name"],
  "feature_names": ["audio.rms_dbfs", "video.brightness_mean"],
  "missingness_policy": "reject",
  "sha256": {"model.json": "SHA256_OF_FILE"}
}
```

Configure `models.tabular.path` to the directory. XGBoost is an optional local dependency. Until the teammate provides a model trained and validated for these exact fields, the branch remains unavailable. There is no SHAP waterfall without the corresponding trained model and explanation integration.

## Optional complete-input stack

`run_stack` accepts `sunhacks-stack-logistic-v1`: a local `parameters.json` plus hash-checked `handoff.json`. It requires every named input to have status `trained`, two finite probabilities summing to one, and identical explicit class order. Missing branches never trigger weight renormalization or an invented fallback probability.

`parameters.json`:

```json
{
  "inputs": ["fusion", "tabular"],
  "input_semantics": "class_1_probability",
  "coefficients": [0.0, 0.0],
  "intercept": 0.0,
  "platt_a": 1.0,
  "platt_b": 0.0
}
```

Numbers above are schema examples, **not** usable trained parameters. The function computes `raw = dot(coefficients, class_1_probabilities) + intercept`, then `sigmoid(platt_a * raw + platt_b)`. Preserve this sign convention during export. The manifest records architecture, input contract, class names, `missingness_policy="reject"`, and SHA256 for `parameters.json`; configure `models.stack.path`. Even supplied Platt parameters are marked `parameters_provided_validation_unverified` and `calibrated=false`, because code cannot establish the quality or provenance of their validation. The manager's prose is not a probability and cannot be substituted into this stack.

## Acceptance evidence to send with artifacts

Provide exact package/source revisions, licenses, checksums, layer targets and LoRA rank, temporal sampling/crop/audio normalization, class ordering, missingness policy, feature ordering, and a model card explaining the task and limits. Provide speaker-disjoint validation and calibration evidence separately. The pipeline checks interface compatibility; it does not establish reliable deception detection. Encoder outputs, diagnostic signals, or arousal alone cannot establish honesty, intent or deception.
