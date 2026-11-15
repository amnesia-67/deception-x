"""DeceptionX command line: the real-data pipeline from Gemini.pdf, end to end.

Run from Windows (`python main.py <cmd>`): every command is forwarded into the WSL
CUDA environment through scripts/run.sh, so it always executes on the RTX 5070.

  fetch-models   download + verify V-JEPA 2, V-JEPA 2.1, dasheng, emotion2vec+, Qwen3-Embedding, BlazeFace
  preprocess     decode Real-Life Trials clips into face crops + 16 kHz audio (data/raw must exist)
  groups         rebuild data/rlt_identity_groups.json
  tune           LoRA-tune the five backbones (5-fold identity-grouped CV, live reports in runs/)
  features       frozen backbone features for stage 2
  stage2         fusion network + XGBoost digest + honesty checks + stacked ensemble
  nested         fully nested evaluation of the served analyze policy (needed before export)
  export         fit the deployable digest model on all real clips (served by the socket)
  socket         start the model socket server (default 0.0.0.0:5002)
  test           run the test suite on the GPU
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WSL_ROOT = "/mnt/c/Users/anish/Documents/GitHub/sunhacks-2026"

COMMANDS = {
    "fetch-models": ["scripts/fetch_models.py"],
    "preprocess": ["scripts/preprocess_rlt.py"],
    "groups": ["scripts/build_identity_groups.py"],
    "tune": ["scripts/tune_backbones.py"],
    "features": ["scripts/extract_frozen_features.py"],
    "stage2": ["scripts/train_stage2.py"],
    "nested": ["scripts/nested_deploy_eval.py"],
    "export": ["scripts/export_deploy.py"],
    "socket": ["run_v2_socket.py"],
    "test": ["-m", "pytest", "-q", "tests"],
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) < 2 else 2)
    cmd, extra = sys.argv[1], sys.argv[2:]
    args = COMMANDS[cmd] + extra
    if sys.platform == "win32":
        full = ["wsl", "-d", "Ubuntu", "--cd", WSL_ROOT, "--", "bash", "scripts/run.sh", *args]
    else:
        full = ["bash", os.path.join(ROOT, "scripts", "run.sh"), *args]
    sys.exit(subprocess.call(full))


if __name__ == "__main__":
    main()
