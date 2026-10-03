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


# --- resampling ------------------------------------------------------------
#
# The class denominators are small. On the first ten Hindi test utterances the
# medial class held 96 characters in 21 word tokens, and the counts behind the
# headline were 16 edits against 4. A difference of two pooled rates over
# denominators that size has no analytic standard error worth quoting, because
# the quantity is a ratio of sums over a handful of utterances and the
# utterances are the sampling unit. Resampling them is the honest interval.

def _rates_from_rows(rows: list[dict], cls: str, idx) -> float | None:
    """Pooled rate for one class over the selected rows, or None if empty.

    `rows` are per-utterance records carrying `<cls>_edits` and `<cls>_chars`,
    which is what scripts/score_intelligibility.py already writes, so the
    bootstrap reads the same numbers the table prints rather than recomputing
    them by a second route.
    """
    e = c = 0
    for i in idx:
        r = rows[i]
        e += int(r.get(f"{cls}_edits", 0))
        c += int(r.get(f"{cls}_chars", 0))
    return None if c == 0 else e / c


def excess_from_rows(rows: list[dict], cls: str, idx=None) -> float | None:
    """The site class's rate minus the no-site class's, over selected rows."""
    if idx is None:
        idx = range(len(rows))
    idx = list(idx)
    site = _rates_from_rows(rows, cls, idx)
    other = _rates_from_rows(rows, NEITHER, idx)
    if site is None or other is None:
        return None
    return site - other


def bootstrap_difference(rows_a: list[dict], rows_b: list[dict], cls: str,
                         n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> dict:
    """A percentile interval on excess(b) minus excess(a), resampling utterances.

    PAIRED: one set of utterance indices is drawn and applied to both arms, so
    the interval is on the difference between two systems scored on the same
    material rather than on two independent samples. Unpaired resampling would
    widen it by the between-utterance variance that the pairing removes, which
    is most of it.

    `rows_a` and `rows_b` must be the same utterances in the same order; the
    function refuses rather than zipping two different sets, because a silent
    misalignment here produces a confident interval around the wrong quantity.

    Returns the point estimate, the interval, the fraction of resamples whose
    difference has the same sign as the point estimate, and the number of
    resamples that had to be discarded because a class was empty in them.
    """
    import numpy as np

    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired bootstrap needs the same utterances: {len(rows_a)} "
            f"against {len(rows_b)}")
    ids_a = [r.get("id") for r in rows_a]
    ids_b = [r.get("id") for r in rows_b]
    if ids_a != ids_b:
        raise ValueError("paired bootstrap needs the same utterance order")
    n = len(rows_a)
    if n < 3:
        raise ValueError(f"{n} utterances is too few to resample")

    point_a = excess_from_rows(rows_a, cls)
    point_b = excess_from_rows(rows_b, cls)
    if point_a is None or point_b is None:
        return {"n": n, "point": None,
                "reason": f"class {cls!r} or {NEITHER!r} has no characters"}
    point = point_b - point_a

    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        ea = excess_from_rows(rows_a, cls, idx)
        eb = excess_from_rows(rows_b, cls, idx)
        if ea is None or eb is None:
            skipped += 1
            continue
        draws.append(eb - ea)
    if len(draws) < 100:
        return {"n": n, "point": point, "n_boot": len(draws),
                "skipped": skipped,
                "reason": "too few usable resamples for an interval"}
    arr = np.sort(np.asarray(draws, dtype=float))
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    same_sign = float(np.mean(arr > 0) if point > 0 else np.mean(arr < 0))
    return {"n": n, "point": point, "lo": lo, "hi": hi,
            "n_boot": len(draws), "skipped": skipped,
            "alpha": alpha, "same_sign": same_sign,
            "excludes_zero": (lo > 0.0) or (hi < 0.0)}


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
