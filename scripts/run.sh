#!/bin/bash
# Run a project script inside the WSL CUDA environment with consistent settings.
#   From Windows:  wsl -d Ubuntu --cd /mnt/c/Users/anish/Documents/GitHub/sunhacks-2026 -- bash scripts/run.sh scripts/x.py [args]
set -e
cd "$(dirname "$0")/.."
PY="${DECEPTIONX_PYTHON:-/home/amnesia/torch-env/bin/python3}"
# MediaPipe needs libGLESv2; unpacked without root from Ubuntu's libgles2 .deb (apt download).
export LD_LIBRARY_PATH="$HOME/deceptionx_libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export DECEPTIONX_CACHE="${DECEPTIONX_CACHE:-$HOME/deceptionx_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export GLOG_minloglevel=2
export PYTHONPATH="$PWD:${PYTHONPATH}"
exec "$PY" -u "$@"
