# Proposed GPU encoder contract: `sunhacks-video-encoder-v1`

**Prepared interface only. No compatible running service or real GPU inference has been verified.** The friend is correcting/retraining the models and shutting down the old socket. Keep remote video disabled and all visual results unavailable until the checks below pass. No media has been sent while preparing this client.

The committed `amnesia-67/sunhacks-2026` revision `6a16734c1af7d4a5ff24400b06849241b285d5fb` does not implement this contract. Its `jepa` action accepts visual/audio/text embeddings and substitutes random tensors when absent. Its `jepa_loaded` flag describes a small fusion head. Do not route this new request into that action or interpret a millisecond head timing as backbone inference.

## Identity and deployment gates

The Mac client deliberately distinguishes:

| Exact identity | Family | Model spatial output |
| --- | --- | --- |
| `facebook/vjepa2-vitl-fpc64-256` | V-JEPA **2** ViT-L | 256 × 256 |
| `facebookresearch/vjepa2:vjepa2_1_vit_large_384` | V-JEPA **2.1** distilled ViT-L | 384 × 384 |

No aliases or silent substitutions are accepted. The 2/256 config declares 64 frames, hidden width 1024, patch size 16 and temporal tubelet size 2. Its official Transformers feature extraction uses encoder `last_hidden_state`, and supports skipping the unused predictor. Use the exact pinned model revision and official processor configuration, with frame sampling disabled because the Mac already selects the 64 frames. [Official config](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/config.json), [official usage and processor reference](https://huggingface.co/docs/transformers/model_doc/vjepa2).

The reviewed local 2.1 source revision is `204698b45b3712590f06245fbfba32d3be539812`; its selected checkpoint is `vjepa2_1_vitl_dist_vitG_384.pt` / `ema_encoder`. Preserve its 384-pixel single-view transform and distinct loader. Its source is exposed by Meta's `vjepa2_1_vit_large_384` Hub entry. [Pinned Hub source](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/hubconf.py).

Before setting `protocol_reviewed=true`, the Mac reviewer needs the pushed GPU server source, exact source/model revision and checkpoint hash, the canonical preprocessing config and SHA256, and an explicitly synthetic **pixel fixture run through the real encoder**. Synthetic fixture pixels must not mean random/synthetic output embeddings. Compare a repeat of the same fixture for deterministic/near-deterministic output, a changed temporal fixture for changed features, and verify the actual CUDA encoder path/source. A self-reported `real_encoder` flag alone cannot establish execution.

The client also requires one matching metadata probe, exact revision and preprocessing SHA256 pins before opening a video socket. Keep the explicit enable flag false when the peer is offline or its retrained artifacts are not yet reviewed. A default metadata-only probe is safe; it sends no participant data.

## Transport and bounds

Use one plain TCP connection per request, exactly one UTF-8 JSON object followed by `\n`, and exactly one newline-delimited response. No binary autodetection, alternate legacy parsing or implicit `jepa` dispatch. Client total deadline defaults to 8 seconds including JPEG packing, connection, send, reads and validation; slow native image calls can only be checked before/after the call. Each receive uses the remaining deadline, so slow chunks cannot reset it. There is no automatic retry or local model fallback.

Request maximum is **16 MiB including newline**, response maximum **1 MiB including newline**, frame count **exactly 64**. Enforce those limits on the server before JSON/image decode. JPEGs must decode to RGB, positive dimensions, maximum edge 640 and consistent dimensions; limit decoded pixels and reject malformed/base64-invalid data, excessive image dimensions, duplicate JSON keys and nonfinite numbers. Do not deserialize pickle or model objects from network requests.

The reviewed Mac API requires a resolved peer IP literal and explicit port. There is no guessed/discovered target in the client. Use only the user's verified LAN/VPN endpoint. The optional `auth_token` comes only from the separate `SUNHACKS_SOCKET_TOKEN` environment variable, never the DeepSeek API key. Compare server tokens in constant time and do not log request bodies, image contents or tokens. Plain TCP does not encrypt this token or media; if transport protection is needed, place the service behind a reviewed authenticated VPN/tunnel rather than inventing a URL/header configuration.

GPU service should limit concurrent work and queued windows, reject requests that cannot meet their remaining budget, and return a bounded busy/expired error. Do not create unlimited per-client GPU work. A client timeout closes its socket but cannot preempt a running GPU kernel. Reject queued expired work before starting it; do not apply a completed response to a newer window.

## Metadata-only health

The existing exact ping stays `{"action":"ping"}\n`. The new server advertises this structure (the revision/hash placeholders must be replaced with actual reviewed values):

```json
{
  "status": "ok",
  "message": "pong",
  "protocols": ["sunhacks-video-encoder-v1"],
  "auth_required": false,
  "encoder": {
    "loaded": true,
    "real_encoder": true,
    "synthetic": false,
    "model_id": "facebookresearch/vjepa2:vjepa2_1_vit_large_384",
    "revision": "EXACT_REVIEWED_MODEL_AND_SOURCE_REVISION",
    "device": "cuda:0",
    "input_frames": 64,
    "output_sequence_shape": [32, 1024],
    "pooling": "spatial_mean_per_tubelet",
    "preprocessing": {
      "id": "vjepa2_1-vit-large-384-official-v1",
      "input_stage": "transport_rgb_uint8",
      "applied_once": true,
      "spatial_size": [384, 384],
      "config_sha256": "64_LOWERCASE_HEX_CHARACTERS_OF_CANONICAL_CONFIG"
    }
  }
}
```

For 2/256 use its exact model ID, `spatial_size: [256,256]` and preprocessing ID `vjepa2-vitl-fpc64-256-official-v1`. `config_sha256` is SHA256 of a UTF-8 canonical JSON object with sorted keys and compact separators. Record actual library versions, resize/interpolation/crop/rescale/normalization settings, frame order/layout and temporal pooling in that config; hand the full config to the Mac reviewer. Client/server must not independently guess a digest from a short name.

A valid declaration still yields `encoder_protocol_unverified`; no health reply is a verified inference result. Old head-only replies produce `needs_encoder_protocol`.

## `encode_video` request

The Mac accepts a NumPy RGB `uint8` array `[64,H,W,3]` and 64 strictly increasing capture timestamps, plus nonnegative integer `window_id` and finite positive `sample_fps`. It packs JPEGs at quality 90, no chroma subsampling (`subsampling=0`), without cropping or ImageNet normalization. If the source's maximum edge exceeds 640, it preserves aspect ratio with a rounded LANCZOS resize. This transport resize is separate from the **one** model preprocessing pass on the GPU. Training/validation must account for this JPEG transport; it is a distribution change from lossless local capture.

Request fields:

| Field | Required value |
| --- | --- |
| `action` | `"encode_video"` |
| `protocol` | `"sunhacks-video-encoder-v1"` |
| `request_id` | Fresh UUID string, echoed exactly |
| `window_id` | Integer in `0..2^53-1`, echoed exactly |
| `model_id`, `revision` | Exact reviewed selected identity and revision |
| `sample_fps` | Requested temporal sampling rate, >0 and ≤240 |
| `timestamp_basis` | `"client_monotonic_seconds"`; if timestamps omitted, explicitly `"relative_clip_seconds"` |
| `max_server_ms` | Remaining client budget at packing time, positive integer; use for queue/deadline admission |
| `input_stage` | `"transport_rgb_uint8"` |
| `frame_count` | `64` |
| `transport` | Object `{format:"jpeg",color:"RGB",max_edge:640,quality:90,subsampling:0,size:[width,height]}` |
| `frames` | 64 objects `{timestamp: finite_seconds, jpeg_base64: ASCII_base64}` in chronological order |
| `auth_token` | Optional separate socket token; required if health advertises `auth_required:true` |

Capture timestamps are identifiers in the Mac clock domain. Do not compare them to the GPU machine's clock. Echo them unchanged; use per-machine monotonic timers for latency. With omitted timestamps the client labels the relative clip timeline; it does not invent capture timing. Reject missing, duplicate, nonfinite or nonincreasing timestamps and frames. Do not pad or repeat incomplete windows.

The GPU decodes all 64 JPEGs into RGB uint8, applies the exact selected base model's reviewed preprocessing **once**, then runs the actual encoder in eval/inference mode. Preserve frame order; disable processor resampling/padding and any predictor/classifier path not needed for features. Do not accept normalized float images under the transport RGB stage, otherwise double normalization can silently change inputs.

## `encode_video` success response

Required fields:

| Field | Required value |
| --- | --- |
| `status` | `"ok"` |
| `action`, `protocol` | Exact request values |
| `request_id`, `window_id` | Exact source request identity |
| `model_id`, `revision` | Exact requested reviewed pins; never relabel 2 as 2.1 |
| `device` | `"cuda"` or `"cuda:N"` |
| `real_encoder`, `synthetic` | Literal JSON `true`, literal JSON `false` |
| `sample_fps`, `timestamp_basis` | Exact request values |
| `frame_timestamps` | All 64 original timestamps, same order and numeric values |
| `sequence_shape` | `[32,1024]` |
| `sequence` | 32 rows ×1024 finite numbers representable in float32 |
| `pooling` | `"spatial_mean_per_tubelet"` |
| `preprocessing` | Same exact object and pinned config digest declared in health |
| `timings` | Finite nonnegative `decode_ms`, `preprocess_ms`, `encoder_ms`, `pooling_ms`, `total_ms`; encoder time positive and ≤total |

Use the encoder's final normalized patch-token sequence. For 2/256 expect `[1,32*16*16,1024]`; for 2.1/384 expect `[1,32*24*24,1024]`. Confirm each loader's token order in its pinned source, reshape to `[1,32,spatial_tokens,1024]`, and average **only spatial tokens**, producing `[32,1024]`. Do not average the temporal dimension before returning. The Mac computes its public pooled `[1024]` embedding from that sequence; both private arrays feed only local fusion.

Time actual GPU computation with CUDA synchronization or proper CUDA events. Wall time without synchronization can report enqueue time. Report decode/preprocess/encoder/pooling and server total separately. The Mac reports client JPEG packing and complete request latency. Capture fps, selected sample fps, unique frames per second, fresh window updates and result age are different measurements. This contract establishes no expected throughput or accuracy number.

A failure response uses `status:"error"`, protocol/request identity when available and a short machine-readable code such as `busy`, `expired`, `invalid_input`, `model_unavailable` or `unauthorized`. No successful substitute embeddings, synthetic fallback or partial features. The Mac discards any failed/malformed/mismatched/stale response.

## Mac integration and artifact replacement

```python
client = RemoteVisionClient(
    verified_peer_ip, 5002,
    expected_model_id=exact_reviewed_id,
    expected_revision=exact_reviewed_revision,
    expected_preprocessing_sha256=reviewed_config_sha256,
    protocol_reviewed=False,
)
# Explicit metadata probe only. Keep false until source/fixture checks complete.
health = client.probe()
runner = make_vision_runner(client, sample_fps=24,
                            timestamps_provider=window_capture_timestamps)
# runner(clip, window_id) returns one branch. With review disabled it returns
# unavailable and opens no video socket.
```

`encode()` raises a controlled `RemoteVisionError` on failure. `make_vision_runner()` converts controlled failure to an `unavailable` vision branch with no private features. It never downloads or invokes a local backbone. Success preserves the actual model ID and returns base-encoder features; this protocol does not claim that a tuned backbone/adapter was loaded. A trained encoder export needs an explicitly reviewed manifest and contract extension, not a renamed status.

**Do not overwrite checkpoints while a process reads them.** Friend should stage each new artifact directory, finish writing it, record hashes/base revision/architecture/preprocessing/training lineage and held-out results, then select that complete version. Mac should stop inference for the affected branch, inspect the new manifest and strictly verify all state keys/shapes/hashes before enabling it. Clear old cached features and pending windows on a version change; reject old responses. Do not silently hot-swap a model or change the reviewed revision/digest on a live client. Construct a new client for a new reviewed version and rerun health, fixture and end-to-end parity checks.

Current fusion code requires the 2.1 visual ID. A genuine 2/256 sequence remains incompatible with that identity gate even though it has the same width. Agree on the real visual family and matching trained head/adapter provenance, then explicitly version the fusion contract if appropriate. Missing retrained files remain unavailable. The existing synthetic-trained head cannot establish performance on real people; no deception conclusion should be produced from these engineering checks.

## Prepared validation

The staged tests use mocked sockets and synthetic pixel arrays only, with no sensors, API keys, peer connections, downloaded datasets, training or actual model invocation. They cover review and health gates, exact source identity, JPEG RGB/aspect transport, temporal echoes, finite sequence shape, NDJSON/duplicate-key/NaN handling, byte caps, whole deadlines, token isolation and unavailable adapter behavior. They validate the client implementation, not GPU throughput, model quality or compatibility of artifacts that have not arrived.
