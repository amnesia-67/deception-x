# Mac (Node 1) client contract

## Which mode to use

The Mac team's privacy split keeps raw audio, ASR, the Qwen/emotion feature extraction and the learned
audio/behavioural models **on the Mac**. Only answer video goes to Node 2. That split uses:

| what | where | artifact / endpoint |
|---|---|---|
| tuned dasheng (5 folds), 88-feature digest XGBoost, stacker, cutoff | Mac (local) | `models/ensemble_deploy.json` (`sunhacks-ensemble-v2`), `models/dasheng_lora/fold*.pt`, `models/branch2_xgboost.json`, `models/branch2_digest_schema.json` |
| V-JEPA 2.1 **ViT-L** video score | Node 2 GPU | `jepa_video` (below), manifest `models/vjepa2_1_l_lora/deploy_meta.json` |
| parity check for the local implementation | Mac | `fixtures/golden_v1/` (generated, no person). Compare against `cuda_fp32` with `suggested_tolerances` |

`analyze` (all models on Node 2, raw audio + transcript sent) remains available. Use it only if the user
explicitly changes the privacy decision. Video does **not** affect the learned verdict: the nested ablation
(`reports/nested_video_ablation/`) found no gain on unseen people (ΔAUROC −0.005, CI −0.019 to +0.010). Show
`jepa_video` only as a separate, uncalibrated score. Never add a video weight on the Mac.

## `jepa_video` (V-JEPA 2.1 ViT-L only)

Request: `{"action": "jepa_video", "frames": [<base64 JPEG>...], "timestamps": [...] | "fps": 30,
"request_id": "...", "source_id": "..."}`. Send raw camera frames: Node 2 picks the frame nearest each of 128
uniform times and crops the face exactly as in training, so timestamps or fps are required. Reply:
`model` = `vjepa2_1_vitl_lora_5fold`, `model_id` = `facebookresearch/vjepa2:vjepa2_1_vit_large_384`, `contract`
(64 frames/view at stride 1, 2 views, 32 native tubelets, width 1024), `manifest_sha256`, `raw_score`
(uncalibrated, **no verdict**), `per_fold_probability`, `temporal_attention_32`, echoed ids and
`server_revision`. When the ViT-L export is missing, `jepa_video` returns an error and `ping` reports
`jepa_unavailable_reason`. There is no ViT-B or V-JEPA 2 fallback.

Expect about **4.2 s** per 7 s answer on Node 2's RTX 5070 (warm; the first request is about 5.3 s), and
1.1 / 3.2 / 6.4 MB request bodies at 5 / 15 / 30 fps capture. Send one request per finished answer, not a
stream. Above about 18 fps for a 7 s answer, extra frames are not used (128 time-uniform picks).
`fixtures/golden_v1/expected.json` → `video_cuda_bf16` holds the expected per-fold scores for the fixture.

Optional `"profile"`: `"served"` (default, the above) or `"webcam_pilot"`. The second one is the ViT-L LoRA-tuned on
the 850-clip webcam pilot, available only when Node 2 runs with `--webcam-pilot` (`ping` → `profiles`). Its reply adds
`profile`, `calibrated_probability` (Platt fitted on pilot validation), `threshold`, `metrics` and `caveats`. Test
AUROC 0.654 on new recordings of the four training people and 0.651 on the unseen person (P5), whose 50
answers the global cutoff all called Truthful. It is not part of the webcam-pilot fused verdict, which validation
assigned to the Mac-side dasheng model. See [MAC_MODEL_HANDOFF.md](MAC_MODEL_HANDOFF.md) for the results, the fusion
and the session baseline.

## `analyze` — all models on Node 2 (only with explicit user consent to send audio and transcript)

The Mac captures one spoken answer and sends **raw media** to Node 2 (`192.168.0.224:5002`).
No model files are needed on the Mac. Requirements: Python 3.10+, `numpy`, `opencv-python`, and a
speech-to-text engine for the transcript (the PDF specifies `faster-whisper` large-v3-turbo on Node 1).
`src/v2_socket_client.py` from this repo implements the protocol below.

## Request (one JSON object, terminated by `\n`)

```json
{
  "action": "analyze",
  "frames": ["<base64 JPEG>", "..."],
  "timestamps": [0.000, 0.033, "..."],
  "audio_f32_b64": "<base64 of little-endian float32 PCM, mono, 16 kHz>",
  "transcript": "verbatim text of the answer",
  "duration_s": 27.4,
  "request_id": "a1b2c3",
  "source_id": "capture-0042"
}
```

| field | required | notes |
|---|---|---|
| `frames` | yes, ≥ 16 | Raw camera frames of the whole answer (the server face-crops). JPEG quality ≥ 90. Sending every captured frame is fine; the server picks 128 by time. |
| `timestamps` or `fps` | one of them | Seconds per frame from the capture clock (preferred) or a constant fps. |
| `audio_f32_b64` | yes | Same time span as the frames. **Float32, not int16**: quantisation changes the silence floor the models were trained on. (`audio_pcm16_b64` is accepted but not recommended.) |
| `transcript` | yes | Verbatim, including fillers ("um", "uh") — disfluency features use them. |
| `duration_s` | no | Defaults to the frame time span. |
| `request_id`, `source_id`, `session_id`, `client_revision`, `capture_started_at` | no | String/number, ≤ 128 chars. Echoed back verbatim (also on errors). Check them before showing a result: a reply is otherwise bound only to its TCP connection. |

One request = one complete answer (training clips were whole answers, 4.5–81 s, median 27 s).
Do not send overlapping sliding windows.

## Response (one JSON line)

```json
{"status": "success", "request_id": "a1b2c3", "source_id": "capture-0042", "model": "stacked_ensemble",
 "probability": 0.44, "threshold": 0.424, "verdict": "Deceptive",
 "threshold_provenance": "prior-matched: ...", "feature_version": "v3-fixed-lag-motion",
 "server_revision": {"code": {"commit": "<git sha>", "dirty": false, "modified_tracked_files": [],
                              "source_tree_sha256": "<sha256>", "untracked_source_files": {"<path>": "<sha256>"}},
                     "artifacts_sha256": {"models/ensemble_deploy.json": "<sha256 = git blob>", "...": "..."},
                     "artifacts_hash": "sha256 of file bytes with CRLF->LF (equals the git blob for these files)",
                     "recorded_at_unix": 1794537263.5},
 "received_at_unix": 1762880000.1,
 "branches": {"tuned_dasheng": {"probability": 0.54, "per_fold": [5 values], "weight": 0.6},
              "xgb_digest":   {"probability": 0.21, "weight": 0.4,
                               "top_deceptive_drivers": [{"name": "voc_f0_st_mean", "shap_impact": 0.31, ...}],
                               "top_truthful_drivers":  [...]}},
 "emotion2vec": {"angry": 0.01, "...": 0.0},
 "face": {"face_detect_rate": 1.0, "centre_crop_fallback": false},
 "digest_features": {"<88 names>": "value or null"}, "missing_features": [],
 "timings": {"audio_decode_ms": 3, "frame_decode_face_crop_ms": 450, "handcrafted_features_ms": 90,
             "gpu_queue_wait_ms": 0, "gpu_models_ms": 120}, "latency_ms": 680,
 "cv_metrics": {"protocol": "fully nested: ...", "auroc": 0.778, "auroc_ci95": [0.68, 0.90],
                "at_threshold_refit_per_outer_fold": {"tp_lie": 53, "tn_truth": 37, "...": "..."},
                "at_fixed_exported_cutoff_on_nested_scores": {"tp_lie": 54, "tn_truth": 30, "...": "..."}, "...": "..."},
 "caveats": ["..."]}
```

`server_revision` is taken once at server start and is also returned by `ping`. It is content-based. `dirty` means a
tracked file differs from the commit; CRLF checkouts and git-lfs objects don't count as changes. `source_tree_sha256`
covers tracked source only, and untracked files are listed separately. Artifact hashes equal the git blobs on any
machine. [SERVED_SOURCE_5002.md](SERVED_SOURCE_5002.md) records the running server, and
`python scripts/verify_served_revision.py` reproduces every reported hash from git objects.

Errors come back as `{"status": "error", "error": "<reason>"}` (e.g. missing transcript, < 16 frames).

## Python (Mac)

```python
from src.v2_socket_client import V2ModelSocketClient   # copy this one file to the Mac

c = V2ModelSocketClient("192.168.0.224", 5002)
print(c.ping()["models"])            # expect ['v2_tabular_xgboost', 'stacked_ensemble', 'vjepa2_1_vitb_lora_5fold']
r = c.analyze(frames_rgb_uint8_list, audio_float32_16k_mono, transcript, timestamps=frame_times_s)
print(r["probability"], r["verdict"], r["branches"]["xgb_digest"]["top_deceptive_drivers"][:3])
```

`verdict` is `probability >= threshold`. The threshold is NOT 0.5: it is fitted on training data only so that
the share of answers called deceptive matches the training base rate (calibrated courtroom scores sit below 0.5
for many lies). Always compare against the `threshold` in the reply, never a hard-coded 0.5.

## Interpreting the output

`probability` is the stacked ensemble's score, Platt-scaled on courtroom clips. `cv_metrics` in every reply
holds the FULLY NESTED result of exactly this served policy on unseen people (AUROC, lies caught, truths
passed with per-fold re-fitted thresholds, at the fixed exported cutoff (reference only) and at 0.5). Webcam audio/video with
Whisper transcripts and 7-second answers is a different domain; webcam accuracy has not been measured.
Recording conditions alone predict the training labels with AUROC 0.669. Display the score as an
experimental model output with its interval, never as a finding that someone is lying.
