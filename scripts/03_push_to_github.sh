#!/usr/bin/env bash
# Create the GitHub repository and push. Run in YOUR OWN macOS Terminal.
# Claude cannot do this: its GitHub token is a proxy placeholder with no
# linked account.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_NAME="${1:-indic-tts-bench}"
VISIBILITY="${2:-private}"     # private | public

command -v gh >/dev/null || { echo "Install the GitHub CLI first:  brew install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run:  gh auth login"; exit 1; }

if git remote get-url origin >/dev/null 2>&1; then
  echo "origin already set to $(git remote get-url origin); pushing"
else
  gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin \
     --description "Does explicit Devanagari G2P still matter for neural TTS? Hindi testbed, Marathi control."
fi

git branch -M main 2>/dev/null || true
git push -u origin main
echo
echo "Pushed. Repository: $(gh repo view --json url -q .url)"
echo "Tell Claude the remote is live and it will keep committing here."
