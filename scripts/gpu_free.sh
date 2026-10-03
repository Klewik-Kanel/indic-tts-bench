#!/bin/bash
# Is the card free for us to launch on?
#
# The PIDs nvidia-smi reports inside this container are from another PID
# namespace, so they cannot be matched against our own with pgrep: every
# process shows as [Not Found] and ps sees only ours. So ownership is counted,
# not matched. We know how many training processes we started; anything else
# holding memory on the card belongs to someone else.
#
#   bash scripts/gpu_free.sh            check once
#   bash scripts/gpu_free.sh --wait     poll until nobody else is on the card
#   bash scripts/gpu_free.sh --wait 60 720   poll every 60 s, up to 720 times
#
# Exit status: 0 when no other tenant holds memory, 1 when one does, so it can
# gate a launch:  bash scripts/gpu_free.sh && ./launch_next.sh
set -uo pipefail

check() {
  local ours oncard others total used free util temp rows
  ours=$(pgrep -cf "src\.train\.launch" 2>/dev/null || true)
  rows=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  oncard=$(printf '%s\n' "$rows" | grep -c . || true)
  others=$(( oncard - ours ))
  [ "$others" -lt 0 ] && others=0

  read -r total used util temp <<<"$(nvidia-smi \
    --query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits | tr -d ',')"
  free=$(( total - used ))

  echo "== $(date -u +%H:%M:%S) UTC"
  echo "   our training processes : $ours"
  echo "   processes on the card  : $oncard"
  echo "   not ours               : $others"
  echo "   memory                 : ${used} MiB used, ${free} MiB free of ${total} MiB"
  echo "   utilisation            : ${util}%, ${temp} C"
  echo "   host load              : $(cut -d' ' -f1-3 /proc/loadavg)"
  if [ -n "$rows" ]; then
    echo "   per process:"
    printf '%s\n' "$rows" | sed 's/^/     /'
  fi

  # Throttling, so a slow run is not blamed on a tenant that left.
  local ev
  ev=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>/dev/null)
  [ -n "$ev" ] && [ "$ev" != "0x0000000000000000" ] && \
    echo "   NOTE: clocks are being held back, reasons $ev"

  if [ "$oncard" -eq 0 ]; then
    echo "   VERDICT: the card is completely idle."
    return 0
  elif [ "$others" -eq 0 ]; then
    echo "   VERDICT: only our $ours run(s) on the card. No other tenant."
    return 0
  else
    echo "   VERDICT: $others process(es) not ours are still holding memory."
    return 1
  fi
}

if [ "${1:-}" = "--wait" ]; then
  every=${2:-120}
  tries=${3:-720}
  for i in $(seq 1 "$tries"); do
    if check; then
      echo "   free after $(( (i - 1) * every )) s of waiting"
      exit 0
    fi
    sleep "$every"
  done
  echo "GIVING UP: still shared after $(( tries * every )) s"
  exit 1
fi

check
