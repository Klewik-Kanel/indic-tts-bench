#!/usr/bin/env bash
# Runs the corpus export and captures everything to logs/download.log.
# Run in YOUR OWN macOS Terminal.
#
#   bash scripts/02_run_download.sh --limit 50    # trial run, ~1 minute
#   bash scripts/02_run_download.sh               # full export
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Python 3.9 is the macOS system interpreter and is end-of-life. Prefer a
# newer one if Homebrew has installed it.
PY=python3
for c in python3.13 python3.12 python3.11; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

{
  echo "=== $(date) ==="
  echo "--- interpreter ---"
  echo "using: $PY"; "$PY" -V
  if [ ! -d .venv ] || ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null; then
    echo "--- rebuilding venv on $PY ---"
    rm -rf .venv
    "$PY" -m venv .venv
  fi
  source .venv/bin/activate
  python -V
  echo "--- deps ---"
  pip install -q -U pip
  pip install -q datasets soundfile numpy huggingface_hub
  pip list 2>/dev/null | grep -Ei '^(datasets|huggingface-hub|soundfile|numpy) '
  echo "--- hf auth ---"
  hf auth whoami 2>&1 || echo "NOT LOGGED IN"
  echo "--- export ---"
  python scripts/02_download_data.py "$@"
  echo "--- exit: $? ---"
} 2>&1 | tee logs/download.log

echo
echo "Log written to logs/download.log. Tell Claude it is there."
