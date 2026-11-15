# Mac pipeline team setup

This directory is the complete Mac-side application. The existing repository-root
`src/`, `models/`, and socket entry points remain the GPU/training application.

## Quick start

From the repository root after checking out this branch:

```sh
cd mac-pipeline
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[capture,models,adapters,emotion,tabular,faces,test]'
cp config.split-answer.example.json config.split-answer.local.json
cp .env.example .env
./scripts/run.sh
```

Open `http://127.0.0.1:8765/` for the seven-second Truth/Lie showcase, or
`http://127.0.0.1:8765/dashboard` for model diagnostics and labeled recording.
Startup does not open capture devices. Dataset recording works without model
weights or a GPU connection. Install FFmpeg/FFprobe on PATH to save recordings.
The configured camera and microphone require macOS authorization; if requested,
run the explicit main-thread authorization commands before starting capture:

```sh
.venv/bin/python -m sunhacks authorize-camera --config config.split-answer.local.json
.venv/bin/python -m sunhacks authorize-audio --config config.split-answer.local.json
```

## Configure inference assets

The portable example points to the shared `../models/` directory. Fetch/install
the reviewed assets through the existing training repository's model workflow,
then adjust paths in your ignored `config.split-answer.local.json` if your caches
use another layout. Required assets are Dasheng base/source, pilot Dasheng
`fold0.pt` plus `deploy_meta.json`, the digest XGBoost/schema, BlazeFace, Qwen3,
emotion2vec+, and Whisper large-v3-turbo. Their pins and contracts are in the code,
example config and existing root model handoff. This code push does not copy model
binaries. `sunhacks/vendor/ensemble_v3_compat.json` is the reviewed Mac-compatible
manifest for the existing diagnostic digest; it keeps the published model
parameters with the LF schema binding expected by the Mac. It is not new weights.

The selected verdict source is the explicit webcam-pilot **single Dasheng model**,
with its own calibration and **0.5436123431421811** threshold. The original
courtroom stack is available by selecting `classifier_profile: courtroom`; do not
apply its threshold to the pilot model. Model readiness is checked on demand;
press **Prepare local models** after assets are configured. Missing assets cause
an explicit failure, never mock inference. The model tests that need optional
local legacy checkpoints skip when those files are unavailable.

GPU video defaults off. To enable it, set the GPU computer's reachable LAN IP/port
in local config and use the showcase toggle. The client requires the reviewed
V-JEPA 2.1 ViT-L model/source/manifest contract. The pilot video export is still
pending in the current handoff; the available courtroom video is a separately
identified diagnostic and does not enter the selected audio verdict.

## Optional contradiction check

Put `OPENROUTER_API_KEY` in your local `.env` to use Drex Max. When selected in the
showcase, that check sends the same answer's recognized transcript and two photos
to OpenRouter/DeepInfra. Only a supported contradiction overrides the model label
to Lie. No contradiction does not force Truth. Uncheck the sharing option to run
without the API. Raw audio stays local; the separate GPU option sends video only.
Do not commit `.env`, local config, recordings, participant reports or weights.

## Tests

```sh
.venv/bin/python -m pytest -q
node tests/test_answer_dashboard.cjs
node tests/test_showcase_ui.cjs
node tests/test_dataset_ui.cjs
node tests/test_claim_input_ui.cjs
node tests/test_pilot_audio_ui.cjs
```

The tests use generated/fake input and no live devices, third-party APIs or training.
Optional checkpoint parity tests run only when the exact local artifact exists.
The application is an experimental hackathon system; numerical and engineering
checks do not establish reliable deception detection.

## Data handoff

The recorder stores labeled MP4, lossless float32 WAV and timestamp metadata under
`datasets/webcam-seven-second-v1/`. Preserve participant codes across sessions.
The collector's default splits are by participant. The separately requested pilot
ZIP exports use recording-stratified splits and include explicit independent
person-held-out protocols; follow the manifest for the chosen experiment.
Recordings and ZIPs are shared separately, not through this code commit. Existing
local historical capture reports are intentionally excluded from source control.
