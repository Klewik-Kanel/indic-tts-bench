#!/usr/bin/env python3
"""Freeze the evaluation splits and build the nested data ladder.

Writes, per language:

    data/processed/<lang>/test.tsv        held out, never trained on
    data/processed/<lang>/dev.tsv
    data/processed/<lang>/train.tsv
    data/processed/<lang>/ladder/<rung>.tsv
    data/processed/<lang>/SPLITS.lock     checksums of all of the above

Three properties this file exists to guarantee.

**Deterministic assignment.** Membership comes from a SHA-256 of the utterance
id with a fixed salt, not from shuffling a list with a seeded RNG. A seeded
shuffle depends on the order the manifest happened to be written in, so
regenerating after re-running preparation, or on another machine, can silently
produce a different test set. Hashing the id makes membership a property of the
utterance alone.

**The ladder is nested.** The 10 minute rung is a subset of the 30 minute rung,
which is a subset of the 1 hour rung, and so on up. Independent samples at each
size would let sampling variance masquerade as a data-quantity effect, which is
precisely the effect the ladder is built to measure. Nesting is achieved by
ordering the training set once by hash and taking prefixes.

**The split is frozen.** Once SPLITS.lock exists the script refuses to
overwrite it. A test set that moves after results exist is not a test set.
Deliberate changes need --force, which rewrites the lock and leaves a dated
note in the file, so the change is visible in the diff rather than silent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[2]
INTERIM = HERE / "data" / "interim"
PROCESSED = HERE / "data" / "processed"

# Changing this salt reshuffles every split. It is written into SPLITS.lock so
# a mismatch is detectable rather than mysterious.
SALT = "indic-tts-bench/v1"

N_TEST = 300
N_DEV = 100

# Hours per rung, largest first.
#
# The top rung is 9 h, not the 10 h that plan v2 assumed. The profile measured
# 10.89 h and 10.80 h of untrimmed audio for the two chosen speakers, but
# silence trimming removed 5.5% of Hindi and 8.4% of Marathi, and the frozen
# test and dev sets take another 0.68 h. That leaves 9.53 h and 9.16 h of
# training audio. 9 h is the largest round figure both languages clear, and
# matching the rungs across languages matters more than reaching a round 10:
# an unmatched top rung would put a data-quantity difference inside the control
# comparison, which is the one place it must not be.
LADDER = [("9h", 9.0), ("5h", 5.0), ("1h", 1.0), ("30min", 0.5),
          ("10min", 10.0 / 60.0)]


def bucket(uid: str) -> float:
    """A stable float in [0, 1) derived from the utterance id alone."""
    h = hashlib.sha256(f"{SALT}:{uid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def read_manifest(lang: str) -> list[dict[str, str]]:
    path = INTERIM / lang / "manifest.tsv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run src.data.prepare first")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: pathlib.Path, rows: list[dict[str, str]],
              fields: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build(lang: str, force: bool) -> dict:
    rows = read_manifest(lang)
    out = PROCESSED / lang
    lock = out / "SPLITS.lock"
    if lock.exists() and not force:
        raise SystemExit(
            f"{lock} exists: splits are frozen. A test set that moves after "
            "results exist is not a test set. Pass --force only deliberately."
        )

    fields = list(rows[0].keys())

    # Order once by hash. Everything below is a slice of this order, which is
    # what makes the ladder nested and the whole thing reproducible.
    ordered = sorted(rows, key=lambda r: bucket(r["id"]))

    test = ordered[:N_TEST]
    dev = ordered[N_TEST:N_TEST + N_DEV]
    train = ordered[N_TEST + N_DEV:]

    files: dict[str, str] = {}
    files["test.tsv"] = write_tsv(out / "test.tsv", test, fields)
    files["dev.tsv"] = write_tsv(out / "dev.tsv", dev, fields)
    files["train.tsv"] = write_tsv(out / "train.tsv", train, fields)

    rungs: dict[str, dict] = {}
    for name, hours in LADDER:
        budget = hours * 3600.0
        acc, chosen = 0.0, []
        for r in train:                      # already in hash order: nested
            if acc >= budget:
                break
            chosen.append(r)
            acc += float(r["seconds"])
        digest = write_tsv(out / "ladder" / f"{name}.tsv", chosen, fields)
        files[f"ladder/{name}.tsv"] = digest
        rungs[name] = {
            "target_hours": hours,
            "actual_hours": round(acc / 3600, 4),
            "n_utterances": len(chosen),
            "short_of_target": acc < budget * 0.995,
        }

    def hrs(rs): return round(sum(float(r["seconds"]) for r in rs) / 3600, 4)

    info = {
        "language": lang,
        "salt": SALT,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_total": len(rows),
        "test": {"n": len(test), "hours": hrs(test)},
        "dev": {"n": len(dev), "hours": hrs(dev)},
        "train": {"n": len(train), "hours": hrs(train)},
        "ladder": rungs,
        "checksums": files,
    }
    lock.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def verify(lang: str) -> int:
    """Re-hash every split file and compare against the lock."""
    out = PROCESSED / lang
    lock = out / "SPLITS.lock"
    if not lock.exists():
        print(f"{lang}: no lock file")
        return 1
    info = json.loads(lock.read_text(encoding="utf-8"))
    bad = []
    for rel, digest in info["checksums"].items():
        p = out / rel
        if not p.exists():
            bad.append(f"{rel}: missing")
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if actual != digest:
            bad.append(f"{rel}: {actual} != {digest}")
    if bad:
        print(f"{lang}: SPLITS CHANGED SINCE FREEZE")
        for b in bad:
            print("  " + b)
        return 1
    print(f"{lang}: all {len(info['checksums'])} split files match the lock")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=["hindi", "marathi"], action="append")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    langs = args.lang or ["hindi", "marathi"]

    if args.verify:
        return max(verify(l) for l in langs)

    rc = 0
    for lang in langs:
        info = build(lang, args.force)
        print(f"{lang}: {info['n_total']} utts -> "
              f"test {info['test']['n']} ({info['test']['hours']} h), "
              f"dev {info['dev']['n']}, "
              f"train {info['train']['n']} ({info['train']['hours']} h)")
        for name, r in info["ladder"].items():
            flag = "  SHORT OF TARGET" if r["short_of_target"] else ""
            print(f"    ladder {name:<6} {r['actual_hours']:>7.3f} h  "
                  f"{r['n_utterances']:>5} utts{flag}")
            if r["short_of_target"]:
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
