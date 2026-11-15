#!/bin/bash
# Start a long job fully detached from the calling terminal (survives the wsl.exe / IDE session).
#   bash scripts/launch_bg.sh <logfile> <script.py> [args...]
set -e
cd "$(dirname "$0")/.."
LOG="$1"; shift
mkdir -p "$(dirname "$LOG")"
setsid nohup bash scripts/run.sh "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG.pid"
disown "$PID" 2>/dev/null || true
echo "started pid $PID -> $LOG"
