#!/usr/bin/env python3
"""Copy weights and results off the DGX, because the GPU access is temporary.

Phase 1 installed a Hugging Face write token and plan v3's phase 3 lists
"checkpoint push to Hugging Face every N steps" as a task. This is that task,
done as a pull rather than a push: it walks the run directories and uploads
what is worth keeping, so it can run on a timer and after every pair without
the training loop knowing anything about it.

**What it uploads, and what it leaves.** Everything small, every time: the run
config, the vocabulary, the training log, the stopping record, the run matrix
and RESULTS.md. Those are what make a checkpoint interpretable later, and a
checkpoint without them is a blob of numbers. Then, per run, the final
checkpoint if it exists and otherwise the most recent one. Not every
intermediate: a 5,000-step cadence over 18 runs would upload the same
architecture dozens of times for no benefit, and the point is to survive losing
the machine, not to keep a complete history.

**It does not re-upload what has not changed.** A local state file records the
size and modification time of everything sent, so running this every twenty
minutes costs almost nothing after the first pass.

**The GUI needs these.** A front end consumes the fp32 export bundles from
src/export/bundle.py rather than raw checkpoints, so bundles are uploaded too
once they exist.

    python scripts/offload.py                  # Klewik/indic-tts-bench by default
    HF_REPO=other/repo python scripts/offload.py
    python scripts/offload.py --dry-run        # list what would go, send nothing
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

STEP_RE = re.compile(r"^step_(\d+)$")

REPO = pathlib.Path(os.environ.get("REPO", "/workspace/indic-tts-bench"))
RUNS = pathlib.Path(os.environ.get("RUNS", "/workspace/runs"))
STATE = RUNS / ".offloaded.json"

# Per run, uploaded every time. Small, and without them a checkpoint cannot be
# interpreted or regenerated.
SMALL = ("config.json", "vocab.json", "train_log.jsonl", "stopped.json")


def completed_steps(run: pathlib.Path) -> list[int]:
    ck = run / "checkpoints"
    if not ck.is_dir():
        return []
    out = []
    for p in ck.iterdir():
        m = STEP_RE.match(p.name)
        if m and p.is_dir() and (p / "COMPLETE").exists():
            out.append(int(m.group(1)))
    return sorted(out)


def plan() -> list[tuple[pathlib.Path, str]]:
    """(local path, path in the repository) for everything worth keeping."""
    jobs: list[tuple[pathlib.Path, str]] = []

    for name in ("RESULTS.md", "README.md"):
        p = REPO / name
        if p.exists():
            jobs.append((p, name))
    for cfg in sorted((REPO / "configs").glob("*.yaml")):
        jobs.append((cfg, f"configs/{cfg.name}"))

    # A missing runs directory is the normal state on a machine that has not
    # trained anything, not an error: this script also runs from the Mac clone.
    runs = sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.is_dir() else []
    for run in runs:
        rid = run.name
        if rid.startswith("."):
            continue
        for name in SMALL:
            p = run / name
            if p.exists():
                jobs.append((p, f"runs/{rid}/{name}"))
        steps = completed_steps(run)
        if not steps:
            continue
        # The final checkpoint if the run reached it, otherwise the newest.
        keep = 100_000 if 100_000 in steps else steps[-1]
        blob = run / "checkpoints" / f"step_{keep}" / "state.pt"
        if blob.exists():
            jobs.append((blob, f"runs/{rid}/checkpoints/step_{keep}/state.pt"))
        meta = run / "checkpoints" / f"step_{keep}" / "meta.json"
        if meta.exists():
            jobs.append((meta, f"runs/{rid}/checkpoints/step_{keep}/meta.json"))

    exports = REPO / "exports"
    if exports.is_dir():
        for p in sorted(exports.rglob("*")):
            if p.is_file():
                jobs.append((p, f"exports/{p.relative_to(exports)}"))
    return jobs


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    # Defaulted so a restarted loop cannot silently back up nothing because
    # somebody forgot an export. HF_REPO still overrides it.
    ap.add_argument("--repo-id",
                    default=os.environ.get("HF_REPO", "Klewik/indic-tts-bench"))
    ap.add_argument("--private", action="store_true", default=True)
    a = ap.parse_args(argv)

    jobs = plan()
    state = load_state()
    todo = []
    for local, remote in jobs:
        st = local.stat()
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        if state.get(remote) != sig:
            todo.append((local, remote, sig, st.st_size))

    total = sum(n for *_x, n in todo)
    print(f"{len(jobs)} files tracked, {len(todo)} changed, "
          f"{total / 1e6:.1f} MB to send")
    for local, remote, _sig, n in todo:
        print(f"  {n / 1e6:9.1f} MB  {remote}")
    if a.dry_run:
        return 0
    if not todo:
        print("nothing to do")
        return 0

    if not a.repo_id:
        print("HF_REPO is not set, so there is nowhere to send this. Set it to "
              "<your-username>/indic-tts-bench and run again. Until then "
              "nothing is backed up off this machine, which is the one risk "
              "this script exists to remove.", file=sys.stderr)
        return 2
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed in this environment: "
              "pip install huggingface_hub", file=sys.stderr)
        return 3

    api = HfApi()
    api.create_repo(repo_id=a.repo_id, repo_type="model", exist_ok=True,
                    private=bool(a.private))
    sent = 0
    for local, remote, sig, n in todo:
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                            repo_id=a.repo_id, repo_type="model")
        except Exception as exc:                             # noqa: BLE001
            print(f"  FAILED {remote}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        # Recorded only after the upload returns, so a failure is retried next
        # time rather than being marked as done.
        state[remote] = sig
        sent += 1
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"uploaded {sent} of {len(todo)} to {a.repo_id}")
    return 0 if sent == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
