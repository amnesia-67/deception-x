#!/bin/zsh
set -eu
root="$(cd "$(dirname "$0")" && pwd)"
exec "$root/mac-pipeline/.venv/bin/python" "$root/mac-pipeline/scripts/start_mac.py" "$@"
