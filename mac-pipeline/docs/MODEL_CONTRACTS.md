# Model contracts and teammate handoff

Contract identifier: `sunhacks-input-v1`. These are researched interfaces, not claims of successful Mac inference. Actual installed versions and measured timings belong in the benchmark report. The original shared plan is an architectural proposal rather than evidence of model capability. All feature inference runs on the Mac; only an explicitly enabled manager API calls a remote service. There is no connection to the teammate's NVIDIA machine.

## Video: V-JEPA 2.1 ViT-L/16

Use the official distilled checkpoint `vjepa2_1_vitl_dist_vitG_384.pt`, encoder key `ema_encoder`, with the V-JEPA **2.1** architecture from the reviewed local Meta checkout. The selected source revision is `204698b45b3712590f06245fbfba32d3be539812`. Do not substitute `facebook/vjepa2-vitl-fpc64-256`: that is V-JEPA2, a different architecture/configuration. Meta publishes the model as roughly300M parameters at384px. [Official model table](https://github.com/facebookresearch/vjepa2#models)

Capture delivers64 timestamp-selected RGB `uint8` frames `[64,H,W,3]`. The adapter reproduces official NumPy single-view evaluation: bilinear OpenCV resize to short side438, center crop384 using rounded offsets, convert to float `/255`, normalize mean`[.485,.456,.406]` and std`[.229,.224,.225]`. Tensor layout is `[1,3,64,384,384]`. Face crops are **not** substituted silently. They would define a new preprocessing contract. [Meta evaluation transforms](https://github.com/facebookresearch/vjepa2/blob/main/evals/video_classification_frozen/utils.py) and [resize/crop implementation](https://github.com/facebookresearch/vjepa2/blob/main/src/datasets/utils/video/functional.py)

The ViT-L hidden width is1024, spatial patch16 and temporal tubelet2. Normal inference returns final normalized patch tokens `[1,18432,1024]`. Pipeline v1 explicitly mean-pools across all tokens to `[1024]`. This pooling removes temporal/spatial localization and must match teammate training. Future sequence fusion may need unpooled/time-grouped features and a new versioned interface. [Encoder implementation](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/vision_transformer.py)

The current upstream Hub loader contains a localhost checkpoint URL; the pipeline avoids the Hub path entirely. It loads local weights with `torch.load(weights_only=True,mmap=True)`, strips known wrapper prefixes and requires a strict encoder state match. The published checkpoint is about5.15GB because it includes training components; encoder-only inference memory is a separate measurement. [Official loaders](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py)

### Three separate rates

The intended capture request is30fps; actual timestamps determine achieved capture fps. Target model sampling is24fps, giving64 distinct frames over a nominal2.667-second window (first-to-last timestamp span2.625seconds). A32-sample hop is1.333seconds with50% overlap.24fps is a configurable design target, not a measured camera or model speed. If the source cannot provide sufficiently dense unique frames, report the gap or use an explicit lower sampling configuration; do not duplicate frames and claim24fps.

Model throughput is completed windows per second, measured after preprocessing with accelerator synchronization. Sustaining a1.333-second hop requires end-to-end service time at or below1.333seconds, averaged with sufficient headroom; slow inference must not block capture. Bounded/latest-window scheduling drops stale pending windows and reports how many were skipped. It does not convert a slow model into realtime processing.

On the16GB Mac, benchmark one model at a time in a dedicated process with a conservative MPS allocation limit.18432 tokens imply a large attention workload; parameter count alone is insufficient to establish feasibility. Record model load time, warm/cold latency, peak process RSS, PyTorch MPS allocations, failures and exact dtype. Keep64frames384px as the selected contract unless a deliberate measured alternative is documented. Do not describe lower resolution or fewer frames as the selected default model interface.

## Audio: Dasheng base

Input is finite mono float32 waveform at16000Hz. Resample actual hardware audio before the model; changing the reported sample-rate number is not resampling. Use the pinned official Dasheng extractor:64mel bands,512-point FFT/window,160-sample hop, centered STFT,0–8000Hz, power-to-decibel conversion with top_db120. Native input is `[B,64,mel_time]`. The released base encoder has width768, patch`[64,4]`, stride`[64,4]`, and target_length1008mel frames. [Extractor source](https://huggingface.co/mispeech/dasheng-base/blob/main/feature_extraction_dasheng.py), [base config](https://huggingface.co/mispeech/dasheng-base/blob/main/config.json)

Pipeline output is mean of raw `hidden_states` over time-frequency patches: `[768]`. Do not treat768 as a chosen audio input length. Output time length varies with waveform duration. The native `logits` field applies sigmoid even without a classifier, so it is not interchangeable with the raw mean embedding. The adapter explicitly uses `hidden_states.mean(dim=1)`. Custom modeling/extractor Python must be inspected locally, pinned, and enabled with `allow_reviewed_code:true`. [Model implementation](https://huggingface.co/mispeech/dasheng-base/blob/main/modeling_dasheng.py)

The base checkpoint is about342MB/85.4M parameters; the model card's introductory1.2B description refers to the larger family. License is Apache-2.0. [Official card](https://huggingface.co/mispeech/dasheng-base)

## Audio: emotion2vec plus large

Use `emotion2vec/emotion2vec_plus_large` (identical published parameters to`iic/emotion2vec_plus_large`). Input is16000Hz mono speech. The large configuration has `embed_dim:1024` and waveform normalization enabled. Original emotion2vec base examples showing768dimensions do not describe this checkpoint. [Large config](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/main/config.yaml)

The FunASR built-in architecture normalizes the waveform with layer normalization, extracts frame features, and averages them for utterance granularity. Pipeline v1 requests utterance features `[1024]`. Native frame mode has variable length near50Hz, not a hard-coded time axis. The9-class emotion scores are not deception probabilities and are not currently exposed as task predictions by this adapter. CPU is the conservative initial backend; MPS compatibility must be measured. [FunASR inference code](https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/model.py), [official model card](https://huggingface.co/emotion2vec/emotion2vec_plus_large)

Weights use the card's custom `model-license` link to FunASR; toolkit code is separately MIT. Retain the applicable [FunASR model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) and model/source attribution with the local snapshot. The adapter requires `license_reviewed:true` as a provenance flag, not an additional user permission step.

Local verification uses FunASR1.4.24 with the official YAML passed directly to its built-in `Emotion2vec` architecture; this bypasses Hub resolution and permissive generic weight loading. The released checkpoint excludes its training-only decoder. The adapter removes that unused module and requires a strict match for every inference weight using `weights_only=True,mmap=True`. Actual loaded inference parameters total162,140,185, despite the card's rough300M description. Offline load passed in3.203seconds with peak processRSS1,771,634,688bytes; this is a load-only measurement, not inference latency. See `EMOTION_LOAD_CHECK.json` for evidence. No model forward was run in this check.

## Text: Qwen3-Embedding-0.6B

Only transcript text enters this encoder. Default v1 input limit is512tokens to bound local work; the model supports32768positions. Its tokenizer, attention mask, special-token processing and last-valid-token pooling must match training. The pipeline uses normalized1024-dimensional final embeddings (`last_token_l2`). The model supports32–1024dimension truncation, but v1 retains1024. Truncating the output vector does not reduce base weight memory. Requires Transformers>=4.51; use a tested pinned version rather than an unconstrained latest install. Apache-2.0. [Official card and code example](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [configuration](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config.json)

Empty transcripts remain unavailable. Manual text is supported while local ASR artifacts are absent. Optional faster-whisper uses a configured local CTranslate2 model onCPU; this build does not rely on experimental MPS support. Recognition errors remain visible and should not be converted into confidence about behavior.

## Missing data and downstream models

Audio windows align to the video's monotonic timestamp interval. Gaps, absent devices, silent valid audio, and denied permission are distinct conditions. An unavailable modality is not a zero embedding. Demo mode emits explicitly labeled zero embeddings only for present inputs, to exercise shape plumbing. Neither base nor demo features imply trained task performance.

Fusion and tabular branches remain `unavailable` until the teammate supplies an exact trained architecture, weights, ordered named feature schema, normalization/imputation policy, masks, output labels and calibration. ReDA/DARE/DMC/MMPDA method names do not identify one ready-to-load fusion checkpoint. A named250-field vector is not valid until every field, unit and aggregation is defined. A Platt/calibration stack needs its own compatible artifact and independent validation; no percentages are fabricated.

## LoRA and offline handoff

LoRA modifies selected linear weights. It does not change a camera into a text input, choose an audio sampling rate, expand a positional budget automatically, or redefine output dimensions. V-JEPA, Dasheng and emotion2vec implementations contain fused `attn.qkv` and `attn.proj` linear candidates; Qwen uses separate `q_proj/k_proj/v_proj/o_proj`. These are inspectable candidates rather than a claim that every PEFT task wrapper works unmodified. [Custom-model PEFT guidance](https://huggingface.co/docs/peft/en/developer_guides/custom_models), [V-JEPA attention](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/utils/modules.py), [emotion2vec attention](https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/modules.py)

The teammate should send a local directory with `adapter_config.json`, `adapter_model.safetensors` and `handoff.json`; no live network link is needed. The manifest must include exact model ID, base revision, contract, output dimension, pooling, file hashes and **`preprocessing_settings`**. Generate the last field using `sunhacks.models.input_settings(config)` for the agreed runtime configuration. For example, after exporting a text adapter:

```python
import hashlib
import json
from pathlib import Path
from sunhacks.models import CONTRACT, DIMS, MODEL_IDS, POOLING, input_settings

config = json.loads(Path("config.base.local.json").read_text())
branch = "text"
export = Path("teammate-text-adapter")
manifest = {
    "model_id": MODEL_IDS[branch],
    "base_revision": config["models"][branch]["revision"],
    "input_contract": CONTRACT,
    "output_dim": DIMS[branch],
    "pooling": POOLING[branch],
    "preprocessing_settings": input_settings(config),
    "sha256": {
        name: hashlib.sha256((export / name).read_bytes()).hexdigest()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    },
}
(export / "handoff.json").write_text(json.dumps(manifest, indent=2))
```

`input_settings` records64video frames, requested capture and model sampling rates, window hop, the fixed640px capture cap, spatial preprocessing, aligned16kHzmono audio/missingness, Dasheng mel parameters, emotion waveform normalization, text truncation/prompt policy and embedding pooling. A24fps training export is rejected at16fps runtime even when all vector widths match. Changing the text token limit also rejects the old artifact. Missing legacy settings are rejected before PEFT code loads. Select and share the intended profile before training; matching shapes alone is insufficient. CUDA training and Mac inference may use different devices/dtypes, so those deployment choices are excluded from the preprocessing identity.

For other branches set `branch` to the matching name; IDs, dimensions and pooling are also listed in `sunhacks/models_manifest.json`. Configure the same `revision` and `adapter_path` locally. Adapter injection uses PEFT directly into the native module so its forward input is preserved. Missing or mismatched metadata, hashes and adapter keys fail visibly. The teammate should also include base checkpoint hashes, exact source/package versions, LoRA rank/alpha/targets, task head definitions, output semantics and held-out validation. The status `trained` means an adapter was loaded, not that accuracy was validated. Cached final embeddings alone cannot be used to train the encoder producing them. No datasets or training are performed by this build.
