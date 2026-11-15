# Local Whisper large-v3-turbo adapter

Official model snapshot downloaded to `/tmp/sunhacks-asr-model`, revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, after checking MIT metadata and159GiB free space. About1.622GB of model/processor files were downloaded, plus the retained official license. No datasets or recorded audio were downloaded or created. Exact file hashes are in ASR_DOWNLOAD.json.

This is `openai/whisper-large-v3-turbo`, with32encoder layers,4decoder layers,128mel features and16kHz input. Its native receptive field is30seconds. [Official card](https://huggingface.co/openai/whisper-large-v3-turbo), [processor config](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json), [MIT license](https://github.com/openai/whisper/blob/main/LICENSE)

## Root integration

Copy `/tmp/sunhacks-asr-stage/asr.py` to `sunhacks/asr.py` and `test_asr_for_repo.py` to `tests/test_asr.py`. Copy the downloaded model folder to a durable `.models/whisper-large-v3-turbo` folder and retain the download report/license.

At the start of `ModelRegistry.transcribe(audio)`, after resolving the `models.asr` config, route `backend == "transformers"` to:

```python
from .asr import transcribe_local
return transcribe_local(audio, cfg)
```

Retain the current faster-whisper backend as a separate option. Use:

```json
{
  "models": {
    "asr": {
      "backend": "transformers",
      "path": "/absolute/path/to/.models/whisper-large-v3-turbo",
      "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
      "device": "mps",
      "dtype": "float16",
      "language": "english",
      "max_new_tokens": 128,
      "keep_loaded": false
    }
  }
}
```

Language may be omitted for detection. The adapter returns language from generated language tokens when present, otherwise the supplied hint. It returns text, status, model/revision, elapsed/load/inference timing, source duration and explicit `timestamp_granularity:"window_only"`. Root must attach the actual audio/video source timestamps. It does not produce word timestamps or guarantee verbatim disfluency preservation. No manually invented transcript is substituted on failure or silence.

## Bounds and memory

Input must be nonempty finite mono float audio at16kHz, at most30seconds. Longer inputs are rejected rather than silently truncated or concatenated. An RMS floor avoids transcribing near-silence; this is not VAD. The processor receives an attention mask; feature tensors use the configured floating dtype while masks preserve integer dtype. Generation uses transcription mode, no sampling, one beam and bounded new tokens. A token-bound hit is marked `partial`/`truncated`; root should keep partial text visibly incomplete instead of treating it as a complete verbatim answer.

Local safetensors-only loading disables remote code and auto-downloads, checks the selected turbo architecture and rejects missing/unexpected/mismatched inference weights. MPS is explicit; unsupported device/dtype does not silently fall back to CPU. CPU is float32. CUDA is supported structurally but untested here. MPS/CUDA calls synchronize around timing. By default the model and its tensors are released and the accelerator cache cleared after each call; loading therefore contributes to live latency. If explicitly retaining models, call `clear_asr_cache()` on stop or configuration change. Keep the parent's conservative MPS memory cap enabled.

## Validation performed

Eighteen local-stub tests cover argument/data bounds, silence, matching architecture, offline/safetensors/strict loader flags, attention-mask preservation, generation options, language, cache lifecycle, failing generation and truncated output. An actual local processor-only check verifies feature and mask shapes using an in-memory sine-wave fixture. No real encoder/decoder model was instantiated and no actual GPU forward or live transcription has run yet. Root must schedule that separately while the GPU is not busy with V-JEPA; ASR speed, memory and recognition quality are still unmeasured.
