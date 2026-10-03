#!/usr/bin/env python3
"""Where the recogniser's errors fall, relative to the schwa deletion sites.

A corpus character error rate says how intelligible a system is. It does not
say whether the errors are the ones this project predicts. The graphemic arm is
handed no deletion rule, so if explicit grapheme-to-phoneme conversion buys
anything, the arm without it should err MORE at the places the rule fires and
no more elsewhere. A global rate cannot show that; a partitioned one can.

**The partition is by word, not by character.** `deletion_sites` indexes the
phone sequence, and this project has no grapheme-to-phone correspondence that
would map a phone index back to a Devanagari character offset. Inventing one
would put a guess underneath a headline number. A word either carries a
rule-predicted deletion or it does not, which is a fact the front end already
supplies, so the reference word sequence is aligned with the hypothesis word
sequence and the character errors inside each aligned pair are attributed to
that word's class.

**Three classes, because two would hide something.** A word with a final
deletion, a word with a medial deletion, and a word with neither. The medial
case is the harder one: a schwa dropped inside a word changes the syllable
count, and the duration measure in duration_bias.py cannot tell a medial site
from a final one. Keeping them apart here is what lets the two measures be
compared.

**Insertions are reported, never attributed.** A hypothesis word with no
reference partner belongs to no class. Spreading it across the classes, or
dropping it, would both move the contrast; it is counted separately and printed.

The rate within a class is pooled: total character edits over total reference
characters in that class. A mean of per-word rates would weight a two-character
word like a twelve-character one, and short words are exactly where a deleted
final schwa is proportionally largest.
"""

from __future__ import annotations

import dataclasses

FINAL = "final"
MEDIAL = "medial"
NEITHER = "neither"
CLASSES = (FINAL, MEDIAL, NEITHER)


def classify(stats: dict) -> str:
    """The class of one word, from `schwa_sites.word_stats`.

    Final wins over medial when a word has both, because the final site is the
    one the duration measure can see and the two measures have to agree on what
    they are counting.
    """
    if stats.get("final"):
        return FINAL
    if stats.get("medial"):
        return MEDIAL
    return NEITHER


@dataclasses.dataclass(frozen=True)
class ClassRate:
    """Pooled character error rate for one word class."""

    name: str
    edits: int
    chars: int
    words: int

    @property
    def cer(self) -> float:
        if not self.chars:
            raise ValueError(
                f"class {self.name!r} has no reference characters: no rate exists")
        return self.edits / self.chars

    @property
    def has_rate(self) -> bool:
        return self.chars > 0


def align_words(ref_words: list[str], hyp_words: list[str]):
    """Levenshtein alignment over words, returning the operations.

    Yields ("match"|"sub", ref_word, hyp_word), ("del", ref_word, None) or
    ("ins", None, hyp_word), in reference order. The path is recovered from a
    full matrix rather than two rows, because the path is the point here; the
    sentences are tens of words long, so the memory is irrelevant.

    Ties break towards substitution, then deletion, then insertion. The order
    is fixed here rather than left to whichever comparison happens to come
    first, because a tie-break that moves between versions moves the headline.
    """
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("match" if cost == 0 else "sub",
                            ref_words[i - 1], hyp_words[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref_words[i - 1], None))
            i -= 1
            continue
        ops.append(("ins", None, hyp_words[j - 1]))
        j -= 1
    ops.reverse()
    return ops


def partition(g2p, reference: str, hypothesis: str) -> dict:
    """Character errors split by the deletion class of the reference word.

    `reference` is the text that was fed to the synthesiser and `hypothesis` is
    what the recogniser returned. Both go through the project's own
    normalisation inside `src.eval.asr`, which is also where the character
    edit distance lives, so this module cannot drift from the headline rate.
    """
    from src.analysis.schwa_sites import word_stats
    from src.eval.asr import edit_distance, normalise_for_scoring

    # The SAME text form the corpus rate uses, punctuation removed. A trailing
    # comma is one more segment to `word_stats`, which moves a final site off
    # the end of the word and reclassifies it as medial: see
    # src/eval/asr.strip_punctuation for the measurement.
    ref_words = normalise_for_scoring(reference).split()
    hyp_words = normalise_for_scoring(hypothesis).split()

    edits = {c: 0 for c in CLASSES}
    chars = {c: 0 for c in CLASSES}
    words = {c: 0 for c in CLASSES}
    insertions = 0
    inserted_chars = 0

    for op, rw, hw in align_words(ref_words, hyp_words):
        if op == "ins":
            insertions += 1
            inserted_chars += len(hw or "")
            continue
        cls = classify(word_stats(g2p, rw))
        words[cls] += 1
        chars[cls] += len(rw)
        # A deleted reference word is wrong in every character of it.
        edits[cls] += len(rw) if op == "del" else edit_distance(rw, hw)

    return {
        "rates": {c: ClassRate(c, edits[c], chars[c], words[c]) for c in CLASSES},
        "insertions": insertions,
        "inserted_chars": inserted_chars,
    }


def accumulate(parts: list[dict]) -> dict:
    """Pool several utterances' partitions into one set of class rates."""
    edits = {c: 0 for c in CLASSES}
    chars = {c: 0 for c in CLASSES}
    words = {c: 0 for c in CLASSES}
    insertions = inserted_chars = 0
    for p in parts:
        for c in CLASSES:
            r = p["rates"][c]
            edits[c] += r.edits
            chars[c] += r.chars
            words[c] += r.words
        insertions += p["insertions"]
        inserted_chars += p["inserted_chars"]
    return {
        "rates": {c: ClassRate(c, edits[c], chars[c], words[c]) for c in CLASSES},
        "insertions": insertions,
        "inserted_chars": inserted_chars,
    }


def contrast(part: dict) -> dict:
    """The number the dissertation asks for, with its own denominator stated.

    `excess` is the deletion-site rate minus the rate on words with no site,
    in characters per reference character. Positive means errors concentrate
    where the rule fires. It is reported with both class sizes, because an
    excess computed over 40 characters is not the same claim as one computed
    over 4000, and a ratio alone hides which it was.

    Returns `None` for a comparison whose either side has no characters, rather
    than a zero that would read as no effect.
    """
    out = {}
    neither = part["rates"][NEITHER]
    for cls in (FINAL, MEDIAL):
        site = part["rates"][cls]
        if not site.has_rate or not neither.has_rate:
            out[cls] = None
            continue
        out[cls] = {
            "site_cer": site.cer,
            "neither_cer": neither.cer,
            "excess": site.cer - neither.cer,
            "site_chars": site.chars,
            "neither_chars": neither.chars,
            "site_words": site.words,
            "neither_words": neither.words,
        }
    return out
