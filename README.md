# indic-tts-bench

Does explicit Devanagari grapheme-to-phoneme conversion still matter for neural
text-to-speech, or do modern architectures absorb Hindi schwa deletion from
data alone, and does the answer depend on the architecture and on how much data
you have?

Hindi is the testbed. Marathi is the control: same script, same converter, same
phone inventory, no medial schwa deletion. A data ladder from 10 hours down to
10 minutes varies resource level while holding speaker, domain and recording
chain fixed.

M.Tech dissertation work, NSUT New Delhi. Plan documents live in the attached
Claude project.

## Status

Phase 0 complete. The G2P front end is built and tested. Nothing has been
trained yet.

## Layout

    src/g2p/        the front end this study measures
    src/data/       download, cleaning, splits, forced alignment
    src/train/      one thin wrapper per architecture, shared config schema
    src/kaggle/     headless job control
    src/eval/       MCD, F0, ASR-WER, predicted MOS, RTF
    src/analysis/   tables, plots, statistics
    configs/        one YAML per run; the run is the unit of reproducibility
    stresstests/    the contested-schwa word list with gold pronunciations
    results/raw/    per-utterance metric rows, append only
    scripts/        the steps that need network, run from a native terminal

## Standing rules

Every run carries a config file and a git commit hash. Every metric is written
per utterance and never pre-aggregated, so aggregation choices stay revisable.
`RESULTS.md` is appended to, never rewritten. No number reaches the paper that
cannot be regenerated from a config file. No difference smaller than the
measured seed variance is reported as a finding.

## Running the tests

    python3 -m venv .venv && source .venv/bin/activate
    pip install pytest
    python -m pytest tests/ -q

## The network split

Claude's environments sit behind an egress allowlist that denies
`huggingface.co` and `kaggle.com`. Anything touching those runs from a native
terminal via `scripts/`; everything else Claude does directly in this folder.

## Credentials

Never in this repository and never in this folder. `~/.kaggle/kaggle.json`
(mode 600) and `~/.cache/huggingface/token`, both outside the shared tree.
