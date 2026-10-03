#!/usr/bin/env python3
"""Is every symbol the front end emits actually in the vocabulary it trains?

`Vocab.encode` raises on an unknown symbol rather than substituting <unk>,
deliberately, so that a front end drifting away from its inventory is loud.
The cost of that choice is where it goes loud: inside `collate`, on the
prefetch thread, at step 1, after the pair slot has already been spent and the
queue has moved on. r18 died that way, four hours of a pair gone for a
KeyError that was knowable in seconds.

This is the seconds version. Coverage is a property of the pair (language,
input_repr) and nothing else, so four checks cover all eighteen runs. Run it
before a launch, or let `queue_all.sh` run it ahead of the dry run.

Exit status is 0 only when every combination is clean.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.data.splits import read_manifest                 # noqa: E402
from src.g2p import G2P                                   # noqa: E402
from src.train.text import Vocab                          # noqa: E402

LANGUAGES = ("hindi", "marathi")
REPRS = ("phoneme", "grapheme")
PAIRS = [(lang, rep) for lang in LANGUAGES for rep in REPRS]


def check(language: str, input_repr: str, show: int = 5) -> int:
    """Report OOV symbols for one combination. Returns 1 if any were found."""
    vocab = set(Vocab.build(input_repr).symbols)
    g2p = G2P.for_language(language)
    tokenize = g2p.phonemize if input_repr == "phoneme" else g2p.graphemes

    rows = read_manifest(language)
    oov: collections.Counter[str] = collections.Counter()
    examples: list[tuple[str, str, list[str]]] = []

    for row in rows:
        try:
            unknown = set(tokenize(row["text"])) - vocab
        except Exception as exc:                       # a raise is a failure too
            unknown = {f"<raise:{type(exc).__name__}:{exc}>"}
        if unknown:
            oov.update(unknown)
            if len(examples) < show:
                examples.append((row.get("id", "?"), row["text"][:88],
                                 sorted(unknown)))

    label = f"{language}/{input_repr}"
    if not oov:
        print(f"  OK   {label:17s} {len(rows):5d} utterances, "
              f"{len(vocab):3d}-symbol vocabulary, 0 OOV")
        return 0

    # Utterances, not occurrences: the counter is fed a per-utterance set.
    print(f"  FAIL {label:17s} {len(rows):5d} utterances, "
          f"{len(oov)} distinct OOV symbol(s)")
    for symbol, count in oov.most_common(15):
        print(f"         {symbol!r} in {count} utterance(s)")
    for uid, text, unknown in examples:
        print(f"         {uid}  {unknown}  {text}")
    if g2p.unexpanded_numbers:
        missed = sorted(set(g2p.unexpanded_numbers))
        print(f"         numbers the expander refused: {missed[:20]}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=LANGUAGES,
                    help="check one language instead of all")
    ap.add_argument("--input-repr", choices=REPRS,
                    help="check one representation instead of both")
    args = ap.parse_args()

    pairs = [(lang, rep) for lang, rep in PAIRS
             if (args.lang is None or lang == args.lang)
             and (args.input_repr is None or rep == args.input_repr)]

    print(f"text coverage over {len(pairs)} (language, input_repr) combination(s)")
    failed = sum(check(lang, rep) for lang, rep in pairs)
    print("all combinations clean" if not failed
          else f"{failed} combination(s) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
