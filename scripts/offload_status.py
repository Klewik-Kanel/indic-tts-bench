#!/usr/bin/env python3
"""What has actually reached Hugging Face, asked of the Hub rather than guessed.

The upload prints progress bars, which are unreadable once they are in a log and
tell you nothing about the files already done. This compares the plan against
the repository's own file listing, so the answer comes from the Hub:

  - which planned files are present, and whether their sizes agree
  - how many bytes are still to go
  - whether the loop is alive, and what rate the log last reported
  - an ETA from that rate, labelled as an estimate because it is one

Sizes from the Hub are the authority. The local state file only records what an
upload call returned, so a file could be marked sent and still be absent if
something went wrong afterwards; this notices that, which is the point.

    python scripts/offload_status.py
    python scripts/offload_status.py --missing     # just what is left
    watch -n 60 python scripts/offload_status.py
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from offload import RUNS, load_state, plan                # noqa: E402

LOG = RUNS / "offload.log"
RATE_RE = re.compile(r"([\d.]+)\s*(kB|MB|GB)/s")


def hub_files(repo_id: str) -> tuple[dict[str, int], str]:
    """{path: size} for everything in the repo, or ({}, why not)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {}, "huggingface_hub is not installed here"
    api = HfApi()
    try:
        tree = api.list_repo_tree(repo_id=repo_id, repo_type="model",
                                  recursive=True)
    except Exception as exc:                              # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"
    out = {}
    for item in tree:
        size = getattr(item, "size", None)
        lfs = getattr(item, "lfs", None)
        if lfs is not None and getattr(lfs, "size", None):
            size = lfs.size
        if size is not None:
            out[item.path] = int(size)
    return out, ""


def last_rate() -> tuple[float, str]:
    """Bytes per second from the newest rate the log mentions."""
    if not LOG.exists():
        return 0.0, ""
    try:
        tail = LOG.read_bytes()[-200_000:].decode("utf-8", "replace")
    except Exception:                                     # noqa: BLE001
        return 0.0, ""
    hits = RATE_RE.findall(tail.replace("\r", "\n"))
    if not hits:
        return 0.0, ""
    value, unit = hits[-1]
    mult = {"kB": 1e3, "MB": 1e6, "GB": 1e9}[unit]
    return float(value) * mult, f"{value} {unit}/s"


def alive(script: str) -> int:
    """Processes genuinely running `script`, read from /proc argv.

    Not `pgrep -f`, which matches the entire command line as one string: a shell
    invoked with a long -c argument that happens to mention "offload_loop.sh"
    then counts as the loop running, and a status tool that says a backup is
    running when it is not is worse than one that says nothing. Here each
    process's argv is split properly, and a match needs an interpreter in argv[0]
    and the script as an argument of its own.
    """
    me = {os.getpid()}
    pid = os.getpid()
    for _ in range(12):                 # walk up: the shell that ran us is not it
        try:
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:                                 # noqa: BLE001
            break
        if pid <= 1:
            break
        me.add(pid)

    n = 0
    proc = pathlib.Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in me:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except Exception:                                 # noqa: BLE001
            continue
        args = [a.decode("utf-8", "replace") for a in argv if a]
        if not args:
            continue
        exe = pathlib.PurePosixPath(args[0]).name
        if not re.match(r"^(ba|da|z|k)?sh$|^python[\d.]*$|^env$", exe):
            continue
        if any(pathlib.PurePosixPath(arg).name == script for arg in args[1:]):
            n += 1
    return n


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1000
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id",
                    default=os.environ.get("HF_REPO", "Klewik/indic-tts-bench"))
    ap.add_argument("--missing", action="store_true",
                    help="list only what is not on the Hub yet")
    a = ap.parse_args(argv)

    jobs = plan()
    state = load_state()
    remote, why = hub_files(a.repo_id)

    print(f"repo        {a.repo_id}")
    print(f"loop        {'running' if alive('offload_loop.sh') else 'NOT RUNNING'}"
          f"   upload {'in flight' if alive('offload.py') else 'idle'}")
    if why:
        print(f"hub         could not be read: {why}")
    print()

    done, wrong, missing = [], [], []
    for local, path in jobs:
        size = local.stat().st_size
        there = remote.get(path)
        if there is None:
            missing.append((path, size))
        elif there != size:
            wrong.append((path, size, there))
        else:
            done.append((path, size))

    if not a.missing:
        print(f"{'state':9} {'size':>12}  path")
        print("-" * 72)
        for path, size in done:
            print(f"{'on hub':9} {human(size):>12}  {path}")
        for path, size, there in wrong:
            print(f"{'SIZE≠':9} {human(size):>12}  {path}  (hub has {human(there)})")
    for path, size in missing:
        print(f"{'missing':9} {human(size):>12}  {path}"
              f"{'   [marked sent locally]' if path in state else ''}")

    left = sum(s for _p, s in missing) + sum(s for _p, s, _t in wrong)
    total = sum(s for _l, _p in jobs for s in [_l.stat().st_size])
    print()
    print(f"on the hub  {len(done)}/{len(jobs)} files, {human(total - left)} of {human(total)}")
    if wrong:
        print(f"size mismatch on {len(wrong)} file(s): re-uploaded next cycle only if "
              "the local file changed, so delete them from the repo to force it")
    liars = [p for p, _s in missing if p in state]
    if liars:
        print(f"{len(liars)} file(s) are marked sent locally but are not on the Hub. "
              f"Delete {RUNS / '.offloaded.json'} to resend them.")

    rate, shown = last_rate()
    if left and rate:
        secs = left / rate
        print(f"last rate   {shown}  ->  {human(left)} left is about "
              f"{secs/3600:.1f} h at that rate [estimate from the log]")
        if rate < 5e6:
            print()
            print("That is slow. Two things usually help, in this order:")
            print("  1. pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1")
            print("     then restart the loop; it uploads in parallel chunks.")
            print("  2. back up less often: only every 25,000 steps plus the final")
            print("     checkpoint, which is a weaker recovery point but a fifth")
            print("     of the traffic.")
    elif not left:
        print("everything planned is on the Hub")
    return 0 if not (missing or wrong) else 1


if __name__ == "__main__":
    raise SystemExit(main())
