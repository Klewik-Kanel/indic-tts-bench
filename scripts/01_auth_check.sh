#!/usr/bin/env bash
# Run this in YOUR OWN macOS Terminal, not inside Claude's workspace.
# Verifies that the Kaggle and Hugging Face credentials are in place and live.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec > >(tee logs/auth.log) 2>&1
echo "=== $(date) ==="

echo "== python =="
python3 -V

echo "== venv =="
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q kaggle "huggingface_hub[cli]"

echo "== kaggle credentials =="
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "MISSING: ~/.kaggle/kaggle.json"
  echo "  kaggle.com -> your avatar -> Settings -> API -> Create New Token"
  echo "  then:  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
PERMS=$(stat -f '%Lp' "$HOME/.kaggle/kaggle.json")
[ "$PERMS" = "600" ] || { echo "fixing permissions on kaggle.json"; chmod 600 "$HOME/.kaggle/kaggle.json"; }
echo "kaggle.json present, mode $(stat -f '%Lp' "$HOME/.kaggle/kaggle.json")"

echo "== kaggle API live test =="
kaggle kernels list --mine -v 2>&1 | head -5 || {
  echo "Kaggle API rejected the token. Regenerate it (Create New Token expires the old one)."
  exit 1
}

echo "== kaggle GPU quota =="
echo "  Not exposed by the API. Check manually at kaggle.com/settings -> Accelerators."

echo "== hugging face credentials =="
if ! hf auth whoami >/dev/null 2>&1; then
  echo "NOT LOGGED IN."
  echo "  huggingface.co -> Settings -> Access Tokens -> Create new token -> type: Write"
  echo "  then:  hf auth login   (paste the token; answer y to the git-credential question)"
  exit 1
fi
hf auth whoami

echo
echo "ALL GOOD. Credentials live. Next: scripts/02_download_data.py"
