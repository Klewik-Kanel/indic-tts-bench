"""Build a blind elicitation sheet, and merge filled sheets into gold forms.

The problem this solves: the rule and the gold pronunciations were written by
the same author, so the front-end accuracy figure measures self-consistency.
Fixing that needs judgements from speakers who have never seen the rule output.

Two design decisions make the elicitation reliable.

**Speakers judge deletion, not IPA.** Asking a native speaker to write IPA
introduces transcription noise that has nothing to do with the question. The
only variable that matters here is whether an unwritten schwa is pronounced at
a given position, which is a yes/no a speaker can answer confidently by reading
the word aloud. The IPA is then generated mechanically from their yes/no
answers through the same converter, so the symbols stay consistent with the
inventory and only the linguistic judgement is human.

**The sheet never shows the rule's answer.** It shows the word and numbered
empty slots. If the predicted pronunciation were visible, agreement with it
would stop being evidence.

    python -m src.g2p.elicit sheet stresstests/hindi_schwa_set.tsv
    python -m src.g2p.elicit merge stresstests/hindi_schwa_set.tsv \\
        stresstests/filled/*.tsv
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import html
import pathlib
import sys

from .devanagari import Segment, word_to_segments
from .normalize import normalize
from .phoneset import NASALISATION
from .schwa import SchwaConfig, delete_schwas

# Reading aid only. Approximate, deliberately not IPA, so the sheet stays
# readable by a speaker with no phonetics training.
ROMAN: dict[str, str] = {
    "k": "k", "kʰ": "kh", "ɡ": "g", "ɡʱ": "gh", "ŋ": "ng",
    "tʃ": "ch", "tʃʰ": "chh", "dʒ": "j", "dʒʱ": "jh", "ɲ": "ny",
    "ʈ": "T", "ʈʰ": "Th", "ɖ": "D", "ɖʱ": "Dh", "ɳ": "N",
    "t̪": "t", "t̪ʰ": "th", "d̪": "d", "d̪ʱ": "dh", "n": "n",
    "p": "p", "pʰ": "ph", "b": "b", "bʱ": "bh", "m": "m",
    "j": "y", "r": "r", "l": "l", "ʋ": "v", "ɭ": "L",
    "ʃ": "sh", "ʂ": "Sh", "s": "s", "ɦ": "h",
    "q": "q", "x": "kh", "ɣ": "gh", "z": "z", "ɽ": "R", "ɽʱ": "Rh", "f": "f",
    "ə": "a", "aː": "aa", "ɪ": "i", "iː": "ii", "ʊ": "u", "uː": "uu",
    "ri": "ri", "eː": "e", "ɛː": "ai", "oː": "o", "ɔː": "au",
    "ɒ": "o", "æ": "ae", "h": "h",
}


def romanise(phone: str) -> str:
    nasal = phone.endswith(NASALISATION)
    base = phone.rstrip(NASALISATION)
    return ROMAN.get(base, base) + ("~" if nasal else "")


def slots(word: str) -> tuple[list[int], str]:
    """Return (segment indices of inherent schwas, a display string).

    The display string writes every written sound and puts a numbered blank
    where an unwritten schwa might or might not be pronounced, for example
    n[1]mkiin. The speaker fills each numbered blank with yes or no.
    """
    segs: list[Segment] = word_to_segments(normalize(word))
    idxs: list[int] = []
    parts: list[str] = []
    for i, s in enumerate(segs):
        if s.inherent and s.phone == "ə":
            idxs.append(i)
            parts.append(f"[{len(idxs)}]")
        else:
            parts.append(romanise(s.phone))
    return idxs, "".join(parts)


def load_set(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#") and ln.strip()]
    return [
        {k: (v or "").strip() for k, v in row.items()}
        for row in csv.DictReader(lines, delimiter="\t")
        if row.get("word")
    ]


# --- sheet generation ------------------------------------------------------

INSTRUCTIONS = (
    "Read each word aloud the way you normally say it, at a normal "
    "conversational pace. Each numbered blank marks a place where Devanagari "
    "writes a consonant with no vowel sign, so the word may or may not have a "
    "short 'a' sound there. For every numbered blank write Y if you pronounce "
    "a vowel there and N if you do not. If you genuinely say it both ways, "
    "write V for variable. Do not consult a dictionary and do not discuss the "
    "words with anyone else who is filling in this sheet."
)


def write_sheet(rows: list[dict[str, str]], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "elicitation_sheet.tsv"
    htm = out_dir / "elicitation_sheet.html"
    max_slots = 0
    prepared = []
    for r in rows:
        idxs, display = slots(r["word"])
        max_slots = max(max_slots, len(idxs))
        prepared.append((r["word"], display, len(idxs)))

    with tsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["speaker_id", "", "", ""] + [""] * max_slots)
        w.writerow(["word", "reading_aid", "n_slots"]
                   + [f"slot{i + 1}" for i in range(max_slots)])
        for word, display, n in prepared:
            w.writerow([word, display, n] + [""] * n + ["-"] * (max_slots - n))

    body = "\n".join(
        f"<tr><td class=dev>{html.escape(w_)}</td><td class=aid>{html.escape(d)}</td>"
        + "".join(f'<td><input name="{html.escape(w_)}_{i + 1}" '
                  f'maxlength="1" autocomplete="off"></td>' for i in range(n))
        + "<td></td>" * (max_slots - n) + "</tr>"
        for w_, d, n in prepared
    )
    htm.write_text(
        "<meta charset=utf-8><title>Schwa elicitation sheet</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:820px;margin:24px auto;"
        "padding:0 16px}table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}"
        ".dev{font-size:21px}.aid{font-family:ui-monospace,monospace;color:#555}"
        "input{width:34px;text-align:center;font-size:15px}"
        "p{background:#f4f6f8;padding:12px 14px;border-left:3px solid #17494d}</style>"
        f"<h1>Schwa elicitation sheet</h1><p>{html.escape(INSTRUCTIONS)}</p>"
        "<p>Speaker id: ____________  Date: ____________</p>"
        f"<table><thead><tr><th>Word</th><th>Reading aid</th>"
        + "".join(f"<th>{i + 1}</th>" for i in range(max_slots))
        + f"</tr></thead><tbody>{body}</tbody></table>",
        encoding="utf-8",
    )
    print(f"wrote {tsv}")
    print(f"wrote {htm}")
    print(f"{len(prepared)} words, up to {max_slots} slots each")
    print("Neither file contains the rule's predicted pronunciation.")


# --- merge -----------------------------------------------------------------

def read_filled(path: pathlib.Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header_at = next(i for i, r in enumerate(rows) if r and r[0] == "word")
    out: dict[str, list[str]] = {}
    for r in rows[header_at + 1:]:
        if not r or not r[0]:
            continue
        n = int(r[2]) if len(r) > 2 and r[2].isdigit() else 0
        out[r[0]] = [c.strip().upper() for c in r[3:3 + n]]
    return out


def merge(set_path: pathlib.Path, filled_paths: list[pathlib.Path]) -> int:
    rows = load_set(set_path)
    sheets = {p.name: read_filled(p) for p in filled_paths}
    if len(sheets) < 2:
        print("WARNING: fewer than two speakers. Agreement cannot be measured, "
              "and a single speaker's idiolect is not a gold standard.",
              file=sys.stderr)

    resolved, variable, missing, disagreed = 0, 0, 0, 0
    out_rows: list[list[str]] = []

    for r in rows:
        word = r["word"]
        idxs, _ = slots(word)
        answers = [s.get(word, []) for s in sheets.values()]
        answers = [a for a in answers if len(a) == len(idxs)]
        if not idxs:
            # No unwritten schwa, so there is no deletion judgement to make.
            # The form follows from the orthography and the inventory alone.
            missing += 1
            out_rows.append([word, r.get("gold_ipa", ""), r.get("category", ""),
                             "no", "no inherent schwa; nothing to elicit"])
            continue
        if not answers:
            missing += 1
            out_rows.append([word, r.get("gold_ipa", ""), r.get("category", ""),
                             "no", "not elicited: absent from every sheet, or "
                                   "slot count mismatched"])
            continue

        keep: list[bool] = []
        note = []
        item_variable = False
        for pos in range(len(idxs)):
            votes = collections.Counter(a[pos] for a in answers)
            top, count = votes.most_common(1)[0]
            if top == "V" or count <= len(answers) / 2:
                item_variable = True
                note.append(f"slot{pos + 1} split {dict(votes)}")
            if len(votes) > 1:
                disagreed += 1
            keep.append(top != "Y")      # Y means pronounced, so not deleted

        if item_variable:
            variable += 1
            out_rows.append([word, r.get("gold_ipa", ""), r.get("category", ""),
                             "no", "variable: " + "; ".join(note)])
            continue

        # Apply the elicited decisions through the converter, so the symbols
        # come from the inventory and only the judgement is human.
        segs = word_to_segments(normalize(word))
        forced = {idxs[p] for p, k in enumerate(keep) if k}
        survivors = [s for i, s in enumerate(segs) if i not in forced]
        gold = " ".join(s.phone for s in survivors if s.phone)
        resolved += 1
        out_rows.append([word, gold, r.get("category", ""), "yes",
                         f"elicited from {len(answers)} speakers"])

    out = set_path.with_name(set_path.stem + "_validated.tsv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["word", "gold_ipa", "category", "validated", "note"])
        w.writerows(out_rows)

    print(f"speakers: {len(sheets)}")
    print(f"resolved: {resolved}   variable: {variable}   not elicited: {missing}")
    print(f"slot-level disagreements: {disagreed}")
    print(f"wrote {out}")
    print("\nItems marked variable stay validated=no on purpose. They are a "
          "finding about Hindi, not a failure: report them as a variation rate "
          "and keep them out of the accuracy denominator.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet", help="generate the blind elicitation sheet")
    s.add_argument("set_path", type=pathlib.Path)
    s.add_argument("--out", type=pathlib.Path, default=pathlib.Path("stresstests"))
    m = sub.add_parser("merge", help="merge filled sheets into gold forms")
    m.add_argument("set_path", type=pathlib.Path)
    m.add_argument("filled", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "sheet":
        write_sheet(load_set(args.set_path), args.out)
        return 0
    paths = [pathlib.Path(p) for pat in args.filled for p in glob.glob(pat)]
    if not paths:
        print("no filled sheets matched", file=sys.stderr)
        return 2
    return merge(args.set_path, paths)


if __name__ == "__main__":
    raise SystemExit(main())
