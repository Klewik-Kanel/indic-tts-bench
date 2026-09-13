#!/usr/bin/env bash
# Runs the download and captures everything to logs/download.log so Claude can
# read the failure instead of guessing. Run in YOUR OWN macOS Terminal.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

{
  echo "=== $(date) ==="
  echo "--- python ---"
  python3 -V
  echo "--- venv ---"
  if [ ! -d .venv ]; then python3 -m venv .venv; fi
  source .venv/bin/activate
  python -V
  echo "--- deps ---"
  pip install -q -U pip
  pip install -q "datasets[audio]" soundfile librosa numpy huggingface_hub
  pip list 2>/dev/null | grep -Ei '^(datasets|huggingface-hub|soundfile|librosa|numpy) '
  echo "--- hf auth ---"
  hf auth whoami 2>&1 || echo "NOT LOGGED IN (public datasets may still work)"
  echo "--- reachability ---"
  curl -s -o /dev/null -w 'huggingface.co -> %{http_code}\n' --max-time 20 https://huggingface.co/
  echo "--- download ---"
  python scripts/02_download_data.py
  echo "--- exit: $? ---"
} 2>&1 | tee logs/download.log

echo
echo "Log written to logs/download.log. Tell Claude it is there."
