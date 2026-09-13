#!/usr/bin/env python3
"""Run the G2P over every transcript and report what it cannot handle.

The front end was built and tested on 49 hand-picked words. This runs it over
all 11,044 real transcripts and asks what breaks. Three questions:

1. Which characters survive normalisation but have no phone mapping? Those
   become literal symbols in the phone sequence and end up as entries in the
   model's embedding table, where they are effectively noise.
2. Which phones in the declared inventory never actually occur? A phone with
   no training examples is an embedding row that gets a random vector and
   whatever gradient noise reaches it.
3. Do the two languages share enough of the inventory for the control
   comparison to sit in one symbol space?

Writes results/tables/g2p_coverage.json and prints a summary.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = HERE / "data" / "processed"
OUT = HERE / "results" / "tables" / "g2p_coverage.json"

sys.path.insert(0, str(HERE))
from src.g2p import G2P, build_inventory                      # noqa: E402
from src.g2p.normalize import normalize                       # noqa: E402
from src.g2p.phoneset import SPECIALS                         # noqa: E402

LANGS = {"hindi": "hindi", "marathi": "marathi"}
SPLITS = ("train", "dev", "test")


def read_split(lang: str, split: str) -> list[dict[str, str]]:
    path = PROCESSED / lang / f"{split}.tsv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def describe(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "UNNAMED"
    return f"U+{ord(ch):04X} {name}"


def analyse(lang: str) -> dict:
    g2p = G2P.for_language(lang)
    # SPECIALS carries the word boundary, silence, unknown and the four
    # punctuation marks. Those are declared symbols with embedding rows, so
    # they belong in the inventory being checked, not subtracted from it.
    inventory = set(build_inventory())

    phone_counts: collections.Counter[str] = collections.Counter()
    unknown_chars: collections.Counter[str] = collections.Counter()
    unknown_examples: dict[str, list[str]] = collections.defaultdict(list)
    empty_words: list[str] = []
    n_utts = 0
    n_words = 0

    for split in SPLITS:
        for row in read_split(lang, split):
            n_utts += 1
            # Score the whole utterance, which is what the model is fed.
            # Scoring word by word would skip number expansion, since one
            # digit run can expand into several words.
            text = normalize(row["text"])
            n_words += len(text.split())
            for word in text.split():
                if not g2p.phonemize_word(word) and len(empty_words) < 40:
                    empty_words.append(word)
            for p in g2p.phonemize(row["text"]):
                phone_counts[p] += 1
                if p not in inventory:
                    unknown_chars[p] += 1
                    if len(unknown_examples[p]) < 5:
                        unknown_examples[p].append(p)

    used = set(phone_counts) & inventory
    unused = sorted(inventory - used)

    return {
        "language": lang,
        "n_utterances": n_utts,
        "n_word_tokens": n_words,
        "n_phone_tokens": sum(phone_counts.values()),
        "inventory_declared": len(inventory),
        "inventory_used": len(used),
        "inventory_unused": unused,
        "unknown_symbols": [
            {
                "symbol": s,
                "count": c,
                "unicode": describe(s) if len(s) == 1 else "multi-char",
                "examples": unknown_examples[s],
            }
            for s, c in unknown_chars.most_common(40)
        ],
        "n_unknown_tokens": sum(unknown_chars.values()),
        "empty_word_examples": empty_words,
        "unexpanded_numbers": sorted(set(g2p.unexpanded_numbers)),
        "top_phones": phone_counts.most_common(12),
        "rare_phones": [
            (p, c) for p, c in phone_counts.most_common()
            if p in inventory and c < 50
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", action="append", choices=sorted(LANGS))
    args = ap.parse_args(argv)
    langs = args.lang or sorted(LANGS)

    report = {}
    for lang in langs:
        info = analyse(lang)
        report[lang] = info
        pct = 100.0 * info["n_unknown_tokens"] / max(info["n_phone_tokens"], 1)
        print(f"\n=== {lang} ===")
        print(f"  {info['n_utterances']} utts, {info['n_word_tokens']} words, "
              f"{info['n_phone_tokens']} phones")
        print(f"  inventory: {info['inventory_used']}/{info['inventory_declared']} used")
        print(f"  unknown symbols: {info['n_unknown_tokens']} tokens ({pct:.3f}%)")
        for u in info["unknown_symbols"][:12]:
            print(f"    {u['symbol']!r:>8}  x{u['count']:<7} {u['unicode']}"
                  f"   e.g. {', '.join(u['examples'][:3])}")
        if info["inventory_unused"]:
            print(f"  never used: {' '.join(info['inventory_unused'])}")
        if info["rare_phones"]:
            print(f"  rare (<50 occurrences): "
                  + ", ".join(f"{p} x{c}" for p, c in info["rare_phones"]))

    # Shared symbol space is what lets Hindi and Marathi results be compared.
    if len(report) == 2:
        a, b = (set(dict(report[l]["top_phones"])) for l in langs)
        ua = set(report[langs[0]]["inventory_unused"])
        ub = set(report[langs[1]]["inventory_unused"])
        only_a = ub - ua
        only_b = ua - ub
        print(f"\n=== shared symbol space ===")
        print(f"  phones present in {langs[0]} but not {langs[1]}: "
              f"{' '.join(sorted(only_a)) or 'none'}")
        print(f"  phones present in {langs[1]} but not {langs[0]}: "
              f"{' '.join(sorted(only_b)) or 'none'}")
        report["shared"] = {
            f"only_{langs[0]}": sorted(only_a),
            f"only_{langs[1]}": sorted(only_b),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
