#!/usr/bin/env bash
# Corpus export, with everything captured to logs/download.log.
# Run in YOUR OWN macOS Terminal.
#
#   bash scripts/02_run_download.sh
#   bash scripts/02_run_download.sh --only marathi
#   bash scripts/02_run_download.sh --limit 50
#
# Do NOT append a trailing "# comment" to these commands. The macOS default
# shell is zsh, and interactive zsh does not treat "#" as a comment unless
# INTERACTIVE_COMMENTS is set, so the comment words arrive as arguments and
# argparse rejects the run.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

PY=python3
for c in python3.13 python3.12 python3.11; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done

{
  echo "=== $(date) ==="
  echo "--- args ---"
  echo "argv: $*"
  echo "--- interpreter ---"
  echo "using: $PY"; "$PY" -V
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || cat <<'WARN'

  NOTE: no Python 3.11+ found, falling back to the system interpreter.
  The project code runs on 3.9, so this is not fatal. But 3.9 is end of life
  and the alignment and training tooling later in the project expects newer.
  Worth installing at some point:  brew install python@3.12

WARN
  # Build the venv only if it is missing or broken. Rebuilding on every run
  # when no newer interpreter exists just wastes minutes reinstalling wheels.
  if [ ! -d .venv ] || [ ! -x .venv/bin/python ]; then
    echo "--- creating venv on $PY ---"
    "$PY" -m venv .venv
  fi
  source .venv/bin/activate
  echo "venv: $(python -V)"
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
