"""Score the rule-based front end against the stress-test gold forms.

Reports the front end's own accuracy before any TTS model exists. Without this
number the downstream phoneme-versus-grapheme comparison is uninterpretable: a
small effect could mean the architecture absorbs schwa deletion, or it could
mean the phoneme arm was fed bad phonemes.

Usage:
    python -m src.g2p.score stresstests/hindi_schwa_set.tsv
    python -m src.g2p.score stresstests/hindi_schwa_set.tsv --include-unvalidated
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

from . import G2P


def load(path: pathlib.Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#") and ln.strip()]
    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        if row.get("word"):
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def score(rows: list[dict[str, str]], g2p: G2P) -> dict:
    per_category: dict[str, list[bool]] = collections.defaultdict(list)
    failures: list[tuple[str, str, str, str]] = []

    for row in rows:
        word, gold, cat = row["word"], row["gold_ipa"], row.get("category", "?")
        pred = " ".join(g2p.phonemize_word(word))
        ok = pred == gold
        per_category[cat].append(ok)
        if not ok:
            failures.append((word, gold, pred, cat))

    total = sum(len(v) for v in per_category.values())
    correct = sum(sum(v) for v in per_category.values())
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_category": {
            k: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
            for k, v in sorted(per_category.items())
        },
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=pathlib.Path)
    ap.add_argument("--language", default="hindi")
    ap.add_argument(
        "--include-unvalidated",
        action="store_true",
        help="score items whose gold form has not been checked by a speaker",
    )
    args = ap.parse_args(argv)

    rows = load(args.path)
    validated = [r for r in rows if r.get("validated") == "yes"]
    scored = rows if args.include_unvalidated else validated

    if not scored:
        print(
            f"{len(rows)} items loaded, {len(validated)} validated.\n"
            "Nothing to score. Gold forms must be checked by a native speaker "
            "before this number means anything.\n"
            "Pass --include-unvalidated for a provisional figure that must not "
            "be reported as a result.",
            file=sys.stderr,
        )
        return 2

    g2p = G2P.for_language(args.language)
    res = score(scored, g2p)

    label = "PROVISIONAL, gold forms unvalidated" if args.include_unvalidated else "validated subset"
    print(f"front-end accuracy ({label})")
    print(f"  {res['correct']}/{res['n']} = {res['accuracy']:.1%}\n")
    print("  by category")
    for cat, s in res["per_category"].items():
        print(f"    {cat:<10} {s['correct']:>3}/{s['n']:<3} {s['accuracy']:>6.1%}")

    if res["failures"]:
        print(f"\n  mismatches ({len(res['failures'])})")
        for word, gold, pred, cat in res["failures"]:
            print(f"    {word:<12} [{cat}]")
            print(f"      gold {gold}")
            print(f"      pred {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
