#!/bin/bash
# Keep copies off the machine on a timer, for as long as the box lasts.
#
#   export HF_REPO=<your-username>/indic-tts-bench
#   setsid nohup bash scripts/offload_loop.sh 1200 > /workspace/runs/offload.log 2>&1 &
#
# The interval is seconds, default 1200. Checkpoints are written every 5,000
# steps, which at the contended rate is roughly an hour, so twenty minutes is
# comfortably often enough to never lose more than one checkpoint.
set -uo pipefail
EVERY=${1:-1200}
REPO=${REPO:-/workspace/indic-tts-bench}
PY=${PY:-/workspace/venv/bin/python}
while true; do
  echo "== $(date -u +%H:%M:%S) UTC"
  "$PY" "$REPO/scripts/offload.py" || echo "   offload returned $?"
  sleep "$EVERY"
done
