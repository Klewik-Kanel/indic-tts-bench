#!/usr/bin/env python3
"""Choose the sentences the demo and the listening test use, on a stated rule.

Two properties are needed at once, and they pull in opposite directions. The
sentences must be from the held-out dev split, which no run trained on, so
nobody can say the demo was picked to flatter a model. And they must actually
exercise schwa deletion, because that is the phenomenon the ablation is about
and a set that happens to avoid it would demonstrate nothing.

Writing sentences by hand would get the second and lose the first. So the
selection comes from the dev split, ordered by how much medial schwa deletion
each sentence contains, stratified so the set spans none-to-several rather than
being all the hardest cases, and deterministic given the split: sorted by
utterance id inside each stratum, round-robin across strata. Re-running this
on the same split returns the same sentences, and the criterion, the strata and
the per-sentence counts are all written out so a reader can check the choice
instead of taking it on trust.

Word-final deletion is counted but not selected on: it is near-categorical in
Hindi and both arms learn it. Medial deletion is the conditioned case where the
arms should diverge.

    python scripts/build_demo_set.py --lang hindi --count 12
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.analysis.schwa_sites import sentence_stats                # noqa: E402
from src.g2p import G2P                                            # noqa: E402

SPLITS = HERE / "data" / "processed"
OUT = HERE / "results" / "tables"

# A demo sentence a listener will hear many times, against a reference and
# against the other arm. Too short carries no prosody; too long is tiring and
# makes an A/B judgement harder to hold in memory.
MIN_SECONDS = 1.5
MAX_SECONDS = 6.0


def load_dev(lang: str) -> list[dict]:
    path = SPLITS / lang / "dev.tsv"
    if not path.exists():
        raise SystemExit(f"{path}: no dev split; run src.data.splits first")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def stratum(medial: int) -> str:
    """Coarse bands, because exact medial counts are sparse in 100 utterances."""
    if medial == 0:
        return "none"
    if medial == 1:
        return "one"
    if medial <= 3:
        return "few"
    return "many"


ORDER = ("many", "few", "one", "none")


def select(rows: list[dict], lang: str, count: int,
           min_s: float, max_s: float) -> dict:
    g2p = G2P.for_language(lang)
    scored = []
    for r in rows:
        secs = float(r["seconds"])
        st = sentence_stats(g2p, r["text"])
        scored.append({
            "id": r["id"],
            "text": r["text"],
            "seconds": secs,
            "words": st["words"],
            "final": st["final"],
            "medial": st["medial"],
            "medial_words": st["medial_words"],
            "stratum": stratum(st["medial"]),
            "in_window": min_s <= secs <= max_s,
            "wav22": r.get("wav22", ""),
            "wav16": r.get("wav16", ""),
        })

    pool: dict[str, list[dict]] = {k: [] for k in ORDER}
    for s in scored:
        if s["in_window"]:
            pool[s["stratum"]].append(s)
    for k in pool:
        pool[k].sort(key=lambda s: s["id"])        # deterministic, not random

    chosen: list[dict] = []
    while len(chosen) < count and any(pool.values()):
        for k in ORDER:                            # richest stratum first
            if pool[k] and len(chosen) < count:
                chosen.append(pool[k].pop(0))
    chosen.sort(key=lambda s: (-s["medial"], s["id"]))

    return {
        "language": lang,
        "split": "dev",
        "criterion": (
            "Held-out dev split only, so no run trained on these. Filtered to "
            f"{min_s}-{max_s} s. Banded by medial schwa deletion count "
            "(none / one / few=2-3 / many=4+), taken round-robin from the "
            "richest band down, sorted by utterance id inside each band. "
            "Deterministic given the split. Word-final deletion is reported "
            "but not selected on, being near-categorical in Hindi."),
        "window_seconds": [min_s, max_s],
        "requested": count,
        "available": len(scored),
        "in_window": sum(1 for s in scored if s["in_window"]),
        "band_sizes": {k: sum(1 for s in scored
                              if s["in_window"] and s["stratum"] == k)
                       for k in ORDER},
        "chosen": chosen,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)

    result = select(load_dev(a.lang), a.lang, a.count,
                    a.min_seconds, a.max_seconds)
    out = a.out or OUT / f"demo_set_{a.lang}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                   encoding="utf-8")

    print(f"{a.lang}: {result['available']} dev utterances, "
          f"{result['in_window']} within "
          f"{a.min_seconds}-{a.max_seconds} s")
    print(f"  bands: {result['band_sizes']}")
    print(f"  chose {len(result['chosen'])} of {a.count} requested")
    print(f"  {'id':<12} {'s':>5} {'med':>4} {'fin':>4}  text")
    for s in result["chosen"]:
        print(f"  {s['id']:<12} {s['seconds']:>5.2f} {s['medial']:>4} "
              f"{s['final']:>4}  {s['text'][:58]}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
