#!/usr/bin/env bash
# Push, poll and fetch Kaggle jobs. Run in YOUR OWN macOS Terminal: Claude's
# environments are denied kaggle.com by the session egress policy.
#
#   bash scripts/05_kaggle.sh push   jobs/smoke
#   bash scripts/05_kaggle.sh status jobs/smoke
#   bash scripts/05_kaggle.sh fetch  jobs/smoke
#   bash scripts/05_kaggle.sh wait   jobs/smoke
#
# No trailing "# comments" on these: interactive zsh passes them as arguments.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CMD="${1:-}"; DIR="${2:-}"
[ -n "$CMD" ] && [ -n "$DIR" ] || { sed -n '2,12p' "$0"; exit 2; }
[ -f "$DIR/kernel-metadata.json" ] || { echo "no kernel-metadata.json in $DIR"; exit 2; }

KID=$(python3 -c "import json,sys; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")
source .venv/bin/activate 2>/dev/null || true

case "$CMD" in
  push)
    echo "pushing $KID"
    kaggle kernels push -p "$DIR" 2>&1 | tee -a "logs/kaggle_$(basename "$DIR").log"
    echo
    echo "Running. Watch it at https://www.kaggle.com/code/$KID"
    echo "If this is the first push, attach the HF_TOKEN secret now:"
    echo "  open the kernel, Add-ons -> Secrets -> Attach, label HF_TOKEN,"
    echo "  then run: bash scripts/05_kaggle.sh push $DIR"
    ;;
  status)
    kaggle kernels status "$KID"
    ;;
  wait)
    echo "polling $KID every 30 s"
    for _ in $(seq 1 120); do
      S=$(kaggle kernels status "$KID" 2>&1)
      echo "$(date +%H:%M:%S)  $S"
      case "$S" in
        *complete*|*COMPLETE*) echo "done"; exit 0 ;;
        *error*|*ERROR*|*cancel*) echo "failed"; exit 1 ;;
      esac
      sleep 30
    done
    echo "gave up waiting"; exit 1
    ;;
  fetch)
    OUT="$DIR/output"
    mkdir -p "$OUT"
    kaggle kernels output "$KID" -p "$OUT"
    echo
    echo "fetched into $OUT:"
    ls -la "$OUT"
    [ -f "$OUT/smoke_result.json" ] && { echo; echo "--- result ---"; cat "$OUT/smoke_result.json"; }
    echo
    echo "Tell Claude the output is in $OUT and it will read it."
    ;;
  *)
    echo "unknown command $CMD"; exit 2 ;;
esac
