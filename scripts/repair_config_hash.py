#!/usr/bin/env python3
"""Repair provenance that the old config loader corrupted.

`load_config` used to try int() then float() on every unquoted value. A 12-hex
config hash that happens to be a valid float literal therefore became a number:
r08's `52e245223208` became inf. Every run with such a hash wrote
`config_hash: Infinity` into its `config.json` and into each checkpoint's
`meta.json`, so the one field that ties weights back to the config they came
from was gone, and the file is not even valid strict JSON.

The loader is fixed. This repairs what was already written, for the runs that
had already trained. It rewrites nothing except a `config_hash` that is not a
string, and it takes the replacement from `configs/<run>.yaml`, which is the
authority. Tensors are never touched.

    python scripts/repair_config_hash.py            # report only
    python scripts/repair_config_hash.py --apply    # write the corrections
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
RUNS = pathlib.Path(os.environ.get("RUNS", "/workspace/runs"))
HASH_RE = re.compile(r"^config_hash:\s*(\S+)\s*$", re.M)


def hash_from_yaml(rid: str) -> str | None:
    p = HERE / "configs" / f"{rid}.yaml"
    if not p.exists():
        return None
    m = HASH_RE.search(p.read_text(encoding="utf-8"))
    return m.group(1).strip("'\"") if m else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    if not RUNS.is_dir():
        print(f"{RUNS} is not here; run this where the runs live")
        return 2

    fixed = broken = 0
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        rid = run.name
        want = hash_from_yaml(rid)
        targets = [run / "config.json"]
        ck = run / "checkpoints"
        if ck.is_dir():
            targets += sorted(ck.glob("step_*/meta.json"))
        for t in targets:
            if not t.exists():
                continue
            try:
                blob = json.loads(t.read_text(encoding="utf-8"))
            except Exception as exc:                      # noqa: BLE001
                print(f"  UNREADABLE {t}: {type(exc).__name__}")
                continue
            got = blob.get("config_hash")
            if isinstance(got, str) or got is None:
                continue
            broken += 1
            rel = t.relative_to(RUNS)
            if want is None:
                print(f"  {rel}: config_hash is {got!r} and configs/{rid}.yaml "
                      "is missing, so there is nothing to restore it from")
                continue
            print(f"  {rel}: {got!r} -> {want!r}")
            if a.apply:
                blob["config_hash"] = want
                # allow_nan=False so a second corrupted field cannot be written
                # back out as Infinity and quietly stay invalid JSON.
                t.write_text(json.dumps(blob, indent=1, sort_keys=True,
                                        allow_nan=False) + "\n",
                             encoding="utf-8")
                fixed += 1

    if not broken:
        print("no corrupted config_hash found")
        return 0
    if a.apply:
        print(f"repaired {fixed} of {broken}")
        return 0 if fixed == broken else 1
    print(f"{broken} file(s) would be repaired. Re-run with --apply.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
