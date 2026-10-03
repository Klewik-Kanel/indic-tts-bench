#!/bin/bash
# One screen of live training status. Written for `watch`:
#
#     watch -n 30 -t bash scripts/status.sh
#
# Read-only and cheap by design, because this runs every 30 seconds for days:
# three nvidia-smi queries, one pgrep, and one line tailed per running log.
# Nothing imports torch and nothing touches a checkpoint.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
RUNS=${RUNS:-/workspace/runs}
MAX_STEPS=${MAX_STEPS:-100000}
export MAX_STEPS

printf '%s  ' "$(date -u +'%m-%d %H:%M:%SZ')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
  --format=csv,noheader 2>/dev/null | paste -sd' ' - || echo "no nvidia-smi"

# Every process on the card, not only ours. A foreign tenant is the commonest
# reason the step rate changes and it is invisible from our own logs.
echo "--- on the card ---"
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
if [ -z "$apps" ]; then echo "  none"; else echo "$apps" | sed 's/^/  /'; fi

echo "--- our runs ---"
mapfile -t ours < <(pgrep -af 'src\.train\.launch' 2>/dev/null \
  | grep -oE 'configs/r[0-9]+\.yaml' | grep -oE 'r[0-9]+' | sort -u)
if [ "${#ours[@]}" -eq 0 ]; then
  echo "  nothing training"
else
  for r in "${ours[@]}"; do
    log="$RUNS/$r/train_log.jsonl"
    if [ -s "$log" ]; then
      tail -1 "$log" | RID="$r" python3 scripts/_status_line.py
    else
      printf '  %-5s no log yet\n' "$r"
    fi
  done
fi

echo "--- queue ---"
pid=$(pgrep -f 'bash scripts/queue_all\.sh' 2>/dev/null | head -1)
if [ -n "$pid" ]; then
  echo "  chain alive (pid $pid)"
else
  echo "  CHAIN NOT RUNNING"
fi
tail -3 "$RUNS/queue.log" 2>/dev/null | sed 's/^/  /'
