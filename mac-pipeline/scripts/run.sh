#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "Create the Python3.12 environment as described in README.md first." >&2
  exit 1
fi
if [ "$#" -eq 0 ] && [ -f config.split-answer.local.json ]; then
  set -- --config config.split-answer.local.json
elif [ "$#" -eq 0 ] && [ -f config.friend-fast.local.json ]; then
  set -- --config config.friend-fast.local.json
elif [ "$#" -eq 0 ] && [ -f config.friend-live.local.json ]; then
  set -- --config config.friend-live.local.json
fi
exec .venv/bin/python -m sunhacks serve "$@"
