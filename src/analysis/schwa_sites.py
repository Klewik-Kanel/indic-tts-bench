#!/usr/bin/env python3
"""Where schwa deletion actually happens, per word and per sentence.

The ablation's claim is about schwa deletion, so a demo or a listening test
built on sentences that happen to contain almost none of it would not be
testing the claim. This counts the sites so a sentence set can be selected on
a stated, reproducible criterion rather than on what sounded interesting.

Counts come from `G2P.deletion_sites`, which is the same code path that fed
training; nothing here reimplements a schwa rule.

Final against medial matters. Word-final schwa deletion in Hindi is close to
categorical and a model picks it up from almost any data, including the
graphemic arm. Medial deletion is the conditioned, harder case, and it is
where the two arms should come apart if the phonemic front end is earning its
place. So the medial count is the one worth selecting on.
"""

from __future__ import annotations


def word_stats(g2p, word: str) -> dict:
    """Deletion sites for one orthographic word, split final from medial.

    A site is final when it is the last pre-deletion segment of the word.
    A lexicon entry reports zero sites because no rule ran, which is the true
    answer rather than a gap.
    """
    segs = g2p.segments(word)
    sites = g2p.deletion_sites(word)
    last = len(segs) - 1
    final = [i for i in sites if i == last]
    medial = [i for i in sites if i != last]
    return {
        "word": word,
        "segments": len(segs),
        "sites": sites,
        "final": len(final),
        "medial": len(medial),
    }


def sentence_stats(g2p, text: str) -> dict:
    """Totals over a sentence, with the per-word detail kept for auditing."""
    words = [w for w in text.split() if w.strip()]
    per_word = [word_stats(g2p, w) for w in words]
    return {
        "words": len(words),
        "final": sum(w["final"] for w in per_word),
        "medial": sum(w["medial"] for w in per_word),
        "sites": sum(len(w["sites"]) for w in per_word),
        "medial_words": [w["word"] for w in per_word if w["medial"]],
        "per_word": per_word,
    }
