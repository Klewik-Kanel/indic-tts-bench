#!/usr/bin/env python3
"""Is every run accounted for, and can each one that has not run actually start?

Three questions, in order of how badly a wrong answer costs:

**Is anything orphaned?** A run that is in the matrix but in no pair of
`queue_all.sh`, and is neither finished nor running, will simply never happen.
Nobody notices until the results table has a hole in it. This is the check that
exists for that: the matrix comes from `config.plan_runs`, the queue comes from
parsing the script, and the difference is reported rather than assumed empty.

**Would a queued run fail at startup?** A missing manifest, audio that was
never staged, a vocoder whose `init_from` names nothing: each of those turns a
scheduled pair into a dry-run failure hours from now, when the queue reaches it.
They are all checkable today.

**Does the config on disk still match the matrix?** `config_hash` is recomputed
from `plan_runs` and compared against the YAML. A hand-edited config file is how
a run quietly stops being the run the paper describes.

It also reports where each run stands and whether its latest checkpoint has made
it off the machine, since a finished run that exists only on a temporary GPU box
is not really finished.

Exit status is 0 only when every unfinished run could start right now.

    python scripts/verify_queue.py
    python scripts/verify_queue.py --sample 20     # check more audio files per run
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.train import batching                           # noqa: E402
from src.train.config import plan_runs                   # noqa: E402
from src.train.runner import load_config                 # noqa: E402

RUNS = pathlib.Path(os.environ.get("RUNS", "/workspace/runs"))
INTERIM = HERE / "data" / "interim"
PROCESSED = HERE / "data" / "processed"
QUEUE = HERE / "scripts" / "queue_all.sh"


def queue_pairs() -> list[list[str]]:
    """The pairs the queue will actually run, read from the script itself."""
    if not QUEUE.exists():
        return []
    text = QUEUE.read_text(encoding="utf-8")
    m = re.search(r"^PAIRS=\((.*?)^\)", text, re.S | re.M)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        for quoted in re.findall(r'"([^"]+)"', line):
            ids = quoted.split()
            if ids:
                out.append(ids)
    return out


def running_ids() -> set[str]:
    try:
        ps = subprocess.run(["pgrep", "-af", "src.train.launch"],
                            capture_output=True, text=True, timeout=10).stdout
    except Exception:                                    # noqa: BLE001
        return set()
    return set(re.findall(r"configs/(r\d\d)\.yaml", ps))


def checkpoint_steps(rid: str) -> list[int]:
    d = RUNS / rid / "checkpoints"
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        m = re.match(r"^step_(\d+)$", p.name)
        if m and p.is_dir() and (p / "COMPLETE").exists():
            out.append(int(m.group(1)))
    return sorted(out)


def offloaded() -> dict:
    try:
        return json.loads((RUNS / ".offloaded.json").read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def manifest_for(cfg: dict) -> pathlib.Path:
    """The same path runner.train derives, so this checks what the run will read."""
    lang = cfg["language"]
    return (PROCESSED / lang / "train.tsv" if cfg["data"] == "train"
            else PROCESSED / lang / "ladder" / f"{cfg['data']}.tsv")


def check_run(rid: str, expected_hash: str, sample: int) -> tuple[list[str], dict]:
    """Problems that would stop this run starting, plus what we know about it."""
    problems: list[str] = []
    info: dict = {"steps": checkpoint_steps(rid)}

    cfg_path = HERE / "configs" / f"{rid}.yaml"
    if not cfg_path.exists():
        return [f"no configs/{rid}.yaml"], info
    cfg = load_config(cfg_path)
    info["arch"] = cfg.get("architecture", "?")
    info["lang"] = cfg.get("language", "?")
    info["data"] = cfg.get("data", "?")
    info["sr"] = cfg.get("sample_rate", 0)

    if expected_hash and cfg.get("config_hash") != expected_hash:
        problems.append(f"config_hash {cfg.get('config_hash')} does not match "
                        f"the matrix's {expected_hash}: the file was edited by hand")

    mpath = manifest_for(cfg)
    if not mpath.exists():
        problems.append(f"manifest missing: {mpath.relative_to(HERE)}")
        return problems, info

    utts = batching.load_manifest(mpath, int(cfg["sample_rate"]))
    info["utts"] = len(utts)
    if not utts:
        problems.append(f"manifest is empty: {mpath.relative_to(HERE)}")
        return problems, info

    # The wav column load_manifest picked, which is the 16 kHz one for VITS. This
    # is the failure r02 hit once: the manifest was right, the files were not there.
    missing = [u.uid for u in utts[:sample] if not (INTERIM / cfg["language"] / u.wav).exists()]
    info["checked"] = min(sample, len(utts))
    info["missing"] = len(missing)
    if missing:
        problems.append(f"{len(missing)} of {info['checked']} sampled audio files "
                        f"absent, e.g. {INTERIM.name}/{cfg['language']}/{utts[0].wav} "
                        f"({missing[0]}): staging incomplete for this rate")

    if cfg.get("architecture") == "hifigan":
        init = str(cfg.get("init_from") or "")
        if not init:
            problems.append("init_from is empty, so it would train from scratch "
                            "instead of fine-tuning, which is the whole saving")
        elif not pathlib.Path(init).expanduser().exists():
            problems.append(f"init_from {init!r} is not a checkpoint on this machine")

    return problems, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=5,
                    help="audio files to check per run (default 5)")
    a = ap.parse_args(argv)

    matrix = {r.run_id: r for r in plan_runs()}
    pairs = queue_pairs()
    queued = [rid for pair in pairs for rid in pair]
    live = running_ids()
    sent = offloaded()

    print(f"matrix: {len(matrix)} runs   queue: {len(pairs)} pairs, "
          f"{len(queued)} runs   running now: {sorted(live) or 'none'}")
    print(f"runs dir: {RUNS}{'' if RUNS.is_dir() else '   (absent on this machine)'}")
    print()

    hdr = (f"{'run':4} {'arch':12} {'lang':8} {'data':6} {'state':9} "
           f"{'step':>7} {'queue':>6} {'backed up':10} notes")
    print(hdr); print("-" * len(hdr))

    blocked, orphans = [], []
    for rid in sorted(matrix):
        expected = matrix[rid].config_hash()      # a method, not a property
        problems, info = check_run(rid, expected, a.sample)
        steps = info.get("steps", [])
        step = steps[-1] if steps else 0
        done = 100_000 in steps or (RUNS / rid / "stopped.json").exists()

        if rid in live:
            state = "RUNNING"
        elif done:
            state = "done"
        elif rid in queued:
            state = "queued"
        elif not RUNS.is_dir():
            # Without the runs directory we cannot tell a finished run from an
            # orphaned one, and calling a finished run an orphan would be worse
            # than saying nothing. This is the normal case on the Mac clone.
            state = "unknown"
        else:
            state = "ORPHAN"
            orphans.append(rid)

        where = ""
        for i, pair in enumerate(pairs, 1):
            if rid in pair:
                where = f"{i}/{len(pairs)}"
        up = "-"
        if step:
            key = f"runs/{rid}/checkpoints/step_{step}/state.pt"
            up = "yes" if key in sent else "NO"

        note = problems[0] if problems else ""
        if problems and not done:
            blocked.append((rid, problems))
        print(f"{rid:4} {info.get('arch','?'):12} {info.get('lang','?'):8} "
              f"{info.get('data','?'):6} {state:9} {step:>7} {where:>6} "
              f"{up:10} {note[:60]}")

    print()
    if not RUNS.is_dir():
        print(f"{RUNS} is not here, so finished runs cannot be told from orphans: "
              "run this on the machine that trains.")
    if orphans:
        print(f"ORPHANED, in the matrix but in no pair and not finished: {orphans}")
        print("  These will never run. Add them to PAIRS in scripts/queue_all.sh.")
    else:
        print("every run is finished, running, or in a queued pair")

    stray = [rid for rid in queued if rid not in matrix]
    if stray:
        print(f"IN THE QUEUE BUT NOT IN THE MATRIX: {stray}")
        print("  The queue would launch a run the matrix does not contain.")

    dupes = [rid for rid in set(queued) if queued.count(rid) > 1]
    if dupes:
        print(f"QUEUED MORE THAN ONCE: {dupes}")
        print("  Two pairs would write the same output directory.")

    if blocked:
        print()
        print("would fail at startup:")
        for rid, problems in blocked:
            for p in problems:
                print(f"  {rid}: {p}")
    else:
        print("nothing unfinished is blocked")

    return 0 if not (orphans or stray or dupes or blocked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
