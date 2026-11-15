# SunHacks multimodal workbench

For a fresh team checkout, start with [TEAM_SETUP.md](TEAM_SETUP.md) and the
portable `config.split-answer.example.json`. In the shared repository this app
lives in `mac-pipeline/` alongside the existing GPU service. Participant recordings
and local capture reports are distributed separately from source code.

**Reviewed V-JEPA 2.1 ViT-L is wired in with runtime socket validation.** Commit `488a818` supplies genuine 1024-wide L folds and two 64-frame views. The Mac client sends original video timestamps once and verifies the L manifest, source fingerprint and echoed source IDs. No B fallback. Video remains separate from the learned verdict because the submitted nested ablation showed no improvement.

The five audit tracks found source-integrity and input-quality defects on the Mac, alongside weak lie sensitivity, validation/calibration issues and courtroom-to-webcam differences in training. The Mac repairs address engineering correctness; they do not establish accurate deception detection. [Published repair handoff](https://github.com/amnesia-67/sunhacks-2026/blob/deploy/ensemble-and-vjepa21/MODEL_PIPELINE_REPAIR_HANDOFF.md).

## Run

```sh
./scripts/run.sh
```

Open **http://127.0.0.1:8765/** for the guided showcase or **http://127.0.0.1:8765/dashboard** for technical diagnostics. When back on the same LAN as `192.168.0.224:5002`, click **Prepare models**. This loads local models, checks the L service and runs generated video before enabling capture. Startup alone opens no sensors or remote connection. Live capture remains unavailable if verification fails.

After integration, the guided test records one seven-second answer and stops on a server-owned timer. Optional 15-second practice guides two truths and a harmless lie and checks capture; it does not calibrate the participant. Cancel/reset discard late outputs. See [calibration research](docs/CALIBRATION_DECISION.md).

## Decisions and privacy

The historical learned classifier combines tuned Dasheng and an XGBoost behavioral digest. Its video score is separate and unused in the verdict. The new teammate export changes the digest to `v3-fixed-lag-motion` (88 fields), refits the two-branch stack and supplies its own threshold. These artifacts must be adopted together with their reviewed schema, hashes and decision rule. Do not copy a new threshold onto old model weights or invent a video fusion weight.

The contradiction check uses **one Drex Max decision request through OpenRouter**. With the disclosed checkbox selected (on by default), the same answer's recognized transcript and two selected camera photos go to OpenRouter's DeepInfra provider. Raw audio, embeddings, behavioral model scores and practice recordings are not sent. The checkbox can be turned off before capture, and that choice persists across answers. Legacy internal `visual_claim_check`/`visual_authorized` flags now authorize this complete transcript-and-photo request, not only an image check.

The model chooses `contradiction`, `no_contradiction`, or `insufficient_evidence` using the images, internal logical consistency and basic arithmetic. There is no separate local arithmetic or garment-rule pass, no generated explanation, no retry and no model/provider fallback. The request starts from the complete same-answer ASR callback and overlaps the other model work. A five-second request deadline bounds failures; it is not a latency target or guarantee. Missing input or invalid provider output cannot establish a contradiction.

The trained fusion classifier supplies the binary model decision. A supported same-answer contradiction can produce the game label **Lie**, with its model decision and original classifier output retained in the technical dashboard. No-contradiction and insufficient evidence leave the classifier's prediction unchanged; neither confirms truth. A cancelled or stale result cannot affect another answer. This policy has not been validated as an improvement in deception accuracy.

The technical dashboard has a **Review Drex inputs** button for the latest completed check. It shows the exact recognized transcript and two selected JPEGs, bound to the outbound observations by a SHA-256 fingerprint. This is local request correspondence, not proof of model correctness. The preview is held only in memory, omitted from polled state, fetched only on demand, and cleared on a new answer, stop or participant reset. It makes no extra model call. A correct arithmetic statement can still receive **Lie** from the behavioral classifier under the contradiction-only rule.

The explanation manager remains removed: no summary endpoint, automatic explanation call or generic manager keyframe buffer. Valid showcase results display **Truth** or **Lie**; failed recordings/missing models require another recording. Detailed diagnostics stay in the technical dashboard.

The configured OpenRouter route is `nace-ai/drex-max`, pinned to its observed `nace-ai/drex-max-20261016` response identity and DeepInfra provider. The requested 1.1 designation is the user's chosen assumption; the provider's reported identifier is preserved without claiming independent version verification. `OPENROUTER_API_KEY` is loaded server-side from the explicitly configured env file; values are never printed or returned.

When live inference is enabled, only answer video goes to the reviewed GPU service. Audio, transcription, Qwen, emotion and the learned audio/behavioral scoring run on this Mac. Practice stays local. Camera/microphone start only through explicit live controls. The server binds to loopback and credentials stay server-side.

## Repairs and verification

Repairs reject stale/mismatched source IDs, nonfinite classifier outputs and unusable audio/video/face diagnostics. The technical dashboard retains recognized words, the exact contributing scores, exported cutoff and fold score ranges. Fold ranges are not confidence intervals. A numerical distance from the cutoff is not confidence.

Video transport preserves max-edge 960 frames using bounded JPEG quality 90, 4:4:4 chroma and optimized lossless Huffman coding, subject to the unchanged 16 MiB request cap. Oversized requests are rejected. Sampling and model throughput must be measured again for the actual L temporal profile; earlier B timings and frame-rate calculations do not establish L performance.

See [repair status and evidence](docs/REPAIR_STATUS_2026-11-11.md), [required teammate corrections](docs/MODEL_PIPELINE_REPAIR_HANDOFF.md) and [historical split runtime](docs/split-answer-runtime.md). Earlier benchmark files describe their pinned historical profiles only. The teammate reports roughly 4.2 seconds for L video on the RTX 5070; Mac-to-GPU latency remains unmeasured. Local CPU FP32 reproduces the trainer's BF16-rounded Dasheng/emotion base values before applying FP32 adapters. The generated golden fixture passes the six supplied numerical tolerances with the same label; this does not establish webcam accuracy. The previous hash-bound local ensemble is retained because the new manifest hashes CRLF schema bytes while Git contains LF; its scoring parameters and model weights are unchanged.

Python 3.12 and the existing `.venv` are tested. For a fresh environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[capture,models,adapters,emotion,tabular,faces,rppg,test]'
.venv/bin/python -m pytest -q
node tests/test_showcase_ui.cjs
node tests/test_answer_dashboard.cjs
```

The configured profile requires hash-bound local base weights, matching trained fold exports, a detector asset and machine-specific paths. Preparation performs no downloads or training. `requirements-macos.lock.txt` records the installed environment. XGBoost runs in an owned local subprocess because combining its native runtime with Torch/MediaPipe previously crashed.

Scores are experimental courtroom-derived model outputs, not calibrated webcam honesty probabilities. Reliable live deception detection has not been demonstrated. The Mac runtime and tests are published in the team repository; model weights, credentials, machine-specific configuration and participant recordings stay outside this code handoff.

GPU video is optional. The showcase checkbox “Use V-JEPA 2.1 ViT-L on the GPU socket” defaults off. Local model preparation and seven-second tests then make no GPU connection and do not require a video classifier score. The camera still supplies local behavioral features. The OpenRouter contradiction checkbox remains independent. Enable GPU video while idle, then check the model connection before testing; enabled mode requires the reviewed L service and never silently falls back.


The ViT-L socket validator accepts exact reconstructed source fingerprints:
reviewed `488a818` alone (`e9eb35d8…`) and that commit plus the unused evaluation
draft (`296402ed…`) published in handoff `587035e`, plus the provenance-only
repair `70cfe69` (`40cf1046…`). Model preprocessing and classification are unchanged. The draft's exact bytes reproduce the live hash; Windows/WSL CRLF/LFS
differences explain the dirty flag. Unknown fingerprints, incompatible model
contracts, wrong L manifests and stale request identities remain rejected.
This checks server-reported correspondence, not independent remote attestation.


## Heart rate (rPPG)

When the camera is on, the technical dashboard (`/dashboard`, **Capture devices**
card, under **Camera**) shows a live heart-rate estimate: BPM, a pulse sparkline,
a signal-quality badge and an *Acquiring* state for the first ~6 seconds. It is
hidden when the camera is off. **It is display only: BPM is never an input to the
fusion, stack, digest, showcase verdict or any other lie/truth score.** It is not a
medical measurement.

How it works (`sunhacks/rppg.py`):

- `LiveCapture` hands each frame it already captured to a frame tap (a bounded,
  non-blocking append). No second camera is opened, and the same tap covers the
  dashboard session, one-answer capture, the showcase and dataset recording.
- One background thread (`rppg-heart-rate`) finds the face with the same MediaPipe
  BlazeFace short-range model the app already uses (OpenCV Haar fallback), keeps a
  stable crop and runs **FacePhys** (Wang et al. 2025, arXiv:2512.06275,
  ~0.72M parameters, ONNX on CPU, a few ms per frame) on a 36×36 face crop per frame.
- A rolling 10 s window is resampled to 30 Hz, band-passed to 0.7–3.5 Hz
  (42–210 BPM) and the spectral peak is reported about once per second. Quality is
  the in-band SNR (fundamental + harmonic vs. the rest) mapped to 0–1. The displayed
  BPM is the median of the last five confident estimates.
- If the FacePhys weights cannot be loaded (offline first run, `onnxruntime`
  missing, hash mismatch), the unsupervised **POS** algorithm (Wang et al. 2017)
  runs on the face-ROI mean colour instead, and the dashboard says
  "POS fallback".
- State is published in `GET /api/state` as `heart_rate` (and `GET /api/heart-rate`)
  through the same 1 Hz polling the dashboard already uses.

Run it on a Mac (Apple Silicon, macOS 14+ for the pinned `onnxruntime` wheel):

```sh
cd mac-pipeline
.venv/bin/python -m pip install -e '.[capture,faces,rppg]'   # or the full extras list above
.venv/bin/python -m sunhacks fetch-rppg                       # optional: pre-cache weights (needs internet once)
./scripts/run.sh                                              # open http://127.0.0.1:8765/dashboard and start the camera
```

The weights (~3.5 MB) are bundled in `mac-pipeline/sunhacks/rppg_weights/` and loaded first,
checked against the sha256 values in `sunhacks/models_manifest.json` (`rppg` section,
pinned to FacePhys-Release commit `07a1402`), so no download is needed and `fetch-rppg`
is optional. If they are missing, they are downloaded into `~/Library/Caches/sunhacks/rppg/`
(override with `SUNHACKS_RPPG_CACHE`); with no weights at all the readout falls back to POS.

Optional launch-config block (all keys optional):

```json
"heart_rate": {"enabled": true, "method": "facephys", "auto_download": true}
```

`method: "pos"` skips the neural model; `enabled: false` removes the tap and the
widget. For best readings sit still facing the camera in even, steady light;
motion, low light, flicker and heavy compression all lower the quality badge.

Licence: FacePhys weights are MIT with a Privacy Protection Addendum. Inference
must stay on the local device, and derived heart-rate values must not be
transmitted off-device without the person's explicit consent. This app computes
BPM locally and does not send it anywhere; keep it that way.

## Collect labeled webcam recordings

Open `/dashboard` and use **Record labeled data**. Keep the same participant
code for the same person across sessions; **New participant** creates a new
pseudonymous code. Click **Add new data**, wait for the recording timer, and say
one statement during the seven-second capture. Capture stops automatically.
Review the video, optionally enter the exact statement/prompt, then choose
**Save as Truth** or **Save as Lie**. **Cancel / discard** removes an unfinished
or unlabeled take. This collection flow works without the socket or prepared
models and sends nothing to the GPU or an API. It never labels from predictions.

By default, recordings live under `datasets/webcam-seven-second-v1/`:

```text
dataset.json                    # random seed and stable participant assignments
train/{truth,lie}/<take-id>/
validation/{truth,lie}/<take-id>/
test/{truth,lie}/<take-id>/
  video.mp4                     # seven-second H.264/AAC review video, VFR
  audio.wav                     # lossless float32, mono, 16 kHz; training audio
  metadata.json                 # label, participant, prompt, original frame timestamps,
                                # device/quality details and media SHA-256 hashes
.pending/<take-id>/             # complete but unlabelled; excluded from every split
```

The folder can be changed with `dataset_root` in the launch config. `ffmpeg`
with libx264 is required. Every captured camera frame is retained, with original
relative timestamps in the metadata; MP4 timestamps have millisecond source
precision. Video is CRF18 encoded without face cropping. Microphone sample times
are aligned to the same seven-second interval. Device clock uncertainty is
recorded; this is not hardware synchronization. Missing audio samples, camera
gaps over 250 ms, dropped capture queues and overflows reject a take rather than
silently pad it. Encoding is bounded and cancellation stops collection/encoding.

The split target is **75/15/10 by participant** using stable seeded random
assignment. A person's truths and lies always remain together. Proportions are
approximate with small groups; a single participant belongs to only one split.
Keep the same code when someone returns and collect similar numbers of each
label per person. Reusing a person under a new code undermines the split. Labels
are participant self-reports. A dataset of a few people is a setup-specific pilot,
not an independent accuracy benchmark. Do not tune against the test folder.

The complete `datasets/` directory is gitignored. Unlabelled takes are not
training examples; normal discard/shutdown removes the current pending take.
A hard process kill can leave `.pending` folders for manual review/removal;
saved labelled takes remain intact. No datasets are uploaded or trained by
this feature.


If a USB microphone is reconnected while the app is running, PortAudio may keep
an obsolete CoreAudio handle and fail with `-9986` / `!obj`. In the dataset card,
use **Reconnect microphone** while capture is idle. This refreshes the audio
backend and counts a brief local microphone check without retaining samples,
changing the selected device, contacting a service or reloading model weights.
It cannot run during capture or while an unlabeled take is pending.


## Selected webcam-pilot audio model

The local launch config selects `classifier_profile: webcam_pilot`, per the selected
request. This uses the single Dasheng LoRA checkpoint from the final 850-clip
webcam run (`scripts/pilot_final.py`), with its own validation-fit Platt scaling
and threshold, both read from the manifest. It replaces the courtroom ensemble as
the model verdict source. XGBoost, Qwen/emotion features and GPU video remain
separate diagnostics; they are not blended into this audio score. The
contradiction checker retains its existing contradiction-only override.

The pinned manifest is `5381b13f…1cd2`, checkpoint `c7577bcd…7afa`. Results
(`reports/pilot_final/report.md`): test AUROC 0.802 and accuracy 73.8% on 80 new
recordings of the four training participants, and AUROC 0.574 with accuracy 50.0%
on a held-out person (P5). Parity: `fixtures/golden_v1/pilot_webcam_expected.json`,
`audio_branches.dasheng.cuda_fp32`.

Only pilot audio was exported in this commit. The webcam-tuned ViT-L export is
available in this release. Courtroom ViT-L can remain a separately
identified video output when the GPU socket is reachable. GPU-off local testing
uses the same selected audio model; it does not fall back to courtroom scoring.

## Integrated Mac launch (clean-history release)

Use `../Start DeceptionX.command` on a configured Mac. The app includes Mochi's integrated countdown/pulse panel, retained Recorded BPM after capture, and adaptive JPEG packing under the socket's 16 MiB wire limit. All selected temporal frames and timestamps are preserved. Pulse stays local and never enters the verdict. The exact 850-clip fusion chooses Dasheng; tuned emotion2vec and V-JEPA 2.1 ViT-L remain diagnostic.

To fetch the pinned adapters/calibration from this clean history without using the private archive, run `.venv/bin/python scripts/pull_pilot_models.py --gpu-host YOUR_TAILSCALE_IPV4`. Base encoders remain separately installed. The local config preserves existing base-cache paths, dataset location and API-key file; it must stay ignored. Tailscale changes only the video endpoint. Keep both peers and the remote model service on. No LAN fallback or relaxed model identity checks are used.

The served root `a7cad71` was independently reconstructed from 65 source files: SHA256 `1440d8435a07666eac67ea4e5ca330b2e1d043815229eeb6b877245c11431f8c`. Checkpoint bytes match the previously tested 850-clip export; sanitized manifests/fusion have new, explicit pins. The old CPU fixture check found a Dasheng logit difference of 0.01519 versus GPU FP32 (above its supplied 0.00423 tolerance), about 0.0012 in score with the same fixture decision; exact CPU/GPU parity is not claimed.

This integration carries source changes onto the new root only. It must not merge or force-push the old repository history. Local recordings, identity annotations, caches and configurations stay in the preserved original checkout. No repository visibility change is part of setup.
