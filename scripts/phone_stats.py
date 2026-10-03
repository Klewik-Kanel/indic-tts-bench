#!/usr/bin/env python3
"""How often does each phone occur, and in what position?

Written to answer a listening impression with a measurement. Two questions it
settles:

**Is a phone the model mishandles simply rare?** A phone at one per thousand in
nine hours has a few hundred examples, and its embedding is trained accordingly.
That is a resource claim, checkable here, and it is the kind of thing a
fixed-budget study should report rather than discover in a viva.

**Does the phone sit in a position the front end invented?** Devanagari writes a
word-final consonant with an inherent schwa, so in the phonemic arm no word ends
in a consonant unless the schwa rule deletes it. Every word-final consonant is
therefore a context the rule created, and if the rule disagrees with the speaker
the text and the audio disagree with it. That makes word-final consonants the
place where the phonemic arm is most exposed, and the last two columns measure
how much of the corpus sits there.

Counts come from G2P.deletion_sites, which is the path that fed training.

    python scripts/phone_stats.py --lang hindi --data 9h
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.g2p import G2P                                        # noqa: E402
from src.g2p.phoneset import WORD_BOUNDARY                     # noqa: E402

SPLITS = HERE / "data" / "processed"
OUT = HERE / "results" / "tables"


def split_path(lang: str, data: str) -> pathlib.Path:
    direct = SPLITS / lang / f"{data}.tsv"
    return direct if direct.exists() else SPLITS / lang / "ladder" / f"{data}.tsv"


def collect(lang: str, data: str) -> dict:
    path = split_path(lang, data)
    if not path.exists():
        raise SystemExit(f"{path}: no such split")
    g2p = G2P.for_language(lang)
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))

    total: collections.Counter[str] = collections.Counter()
    final: collections.Counter[str] = collections.Counter()
    final_by_deletion: collections.Counter[str] = collections.Counter()
    words = 0

    for row in rows:
        for word in row["text"].split():
            phones, sites = g2p._phonemize_word_with_sites(word)
            core = [p for p in phones if p != WORD_BOUNDARY]
            if not core:
                continue
            words += 1
            total.update(core)
            final[core[-1]] += 1
            # The word-final schwa is the last pre-deletion segment, so if that
            # index was deleted, this word's last consonant exists because of it.
            if sites and (len(g2p.segments(word)) - 1) in sites:
                final_by_deletion[core[-1]] += 1

    n = sum(total.values())
    ranked = total.most_common()
    return {
        "language": lang, "data": data, "split": str(path.relative_to(HERE)),
        "utterances": len(rows), "words": words, "phone_tokens": n,
        "distinct_phones": len(total),
        "words_ending_in_a_deletion_derived_consonant": sum(final_by_deletion.values()),
        "phones": [
            {"phone": p, "rank": i + 1, "count": c,
             "per_1k": round(1000 * c / n, 3),
             "word_final": final.get(p, 0),
             "word_final_by_deletion": final_by_deletion.get(p, 0)}
            for i, (p, c) in enumerate(ranked)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--data", default="9h", help="a split or ladder rung name")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args(argv)

    r = collect(a.lang, a.data)
    out = a.out or OUT / f"phone_stats_{a.lang}_{a.data}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")

    pct = 100 * r["words_ending_in_a_deletion_derived_consonant"] / r["words"]
    print(f"{a.lang} {a.data}: {r['utterances']} utterances, {r['words']} words, "
          f"{r['phone_tokens']} phone tokens, {r['distinct_phones']} distinct phones")
    print(f"words ending in a consonant because a final schwa was deleted: "
          f"{r['words_ending_in_a_deletion_derived_consonant']} ({pct:.1f}%)")
    print(f"\n{'phone':<7}{'rank':>5}{'count':>8}{'per 1k':>8}{'final':>7}{'by del':>8}")
    for e in r["phones"][:a.show]:
        print(f"{e['phone']:<7}{e['rank']:>5}{e['count']:>8}{e['per_1k']:>8.2f}"
              f"{e['word_final']:>7}{e['word_final_by_deletion']:>8}")
    print(f"\nrarest in use:")
    for e in r["phones"][-8:]:
        print(f"  {e['phone']:<7}{e['count']:>6}  {e['per_1k']:.3f} per 1k")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
