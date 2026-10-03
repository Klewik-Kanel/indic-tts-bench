#!/usr/bin/env python3
"""Deletion sites, and the sentence statistics the demo set is selected on.

`delete_schwas` has always returned which segment indices it deleted; the only
reason that was unavailable was that `phonemize_word` discarded it. The
refactor that exposes it must not change a single phoneme, because six finished
runs trained through that function, so the first tests here are about
equivalence rather than about the new feature.

Final against medial is the distinction the selection rule rests on: word-final
deletion is near-categorical in Hindi and both arms learn it, so a demo set
chosen on total sites would be chosen on the easy case.
"""
from __future__ import annotations

import pytest

from src.analysis.schwa_sites import sentence_stats, word_stats
from src.g2p import G2P

HINDI = G2P.for_language("hindi")
MARATHI = G2P.for_language("marathi")
LANGS = [HINDI, MARATHI]


# -- the refactor must be invisible -----------------------------------------

@pytest.mark.parametrize("word", ["नमक", "कमल", "जल्दबाज़ी", "में", "सिलसिला",
                                 "राजनीतिक", "हो", "और"])
def test_asking_for_sites_does_not_change_the_phonemes(word):
    before = HINDI.phonemize_word(word)
    HINDI.deletion_sites(word)
    assert HINDI.phonemize_word(word) == before


@pytest.mark.parametrize("word", ["नमक", "कमल", "राजनीतिक"])
def test_sites_and_phonemes_come_from_one_call(word):
    """The two public methods must not be able to disagree."""
    phones, sites = HINDI._phonemize_word_with_sites(word)
    assert phones == HINDI.phonemize_word(word)
    assert sites == HINDI.deletion_sites(word)


def test_a_deleted_site_removes_a_phone():
    """नमक is /nəmək/: six segments, the last schwa gone, five phones."""
    segs = HINDI.segments("नमक")
    phones = HINDI.phonemize_word("नमक")
    sites = HINDI.deletion_sites("नमक")
    assert len(sites) == 1
    assert len(phones) == len(segs) - len(sites)


# -- final against medial ---------------------------------------------------

def test_the_word_final_schwa_is_classified_final():
    st = word_stats(HINDI, "नमक")
    assert st["final"] == 1
    assert st["medial"] == 0
    assert st["sites"] == [st["segments"] - 1]


def test_a_word_with_no_deletable_schwa_reports_nothing():
    st = word_stats(HINDI, "में")
    assert st["sites"] == []
    assert st["final"] == st["medial"] == 0


@pytest.mark.parametrize("g2p", LANGS)
def test_a_lexicon_hit_reports_no_sites(g2p):
    """No rule ran, so zero is the true answer and not a missing one."""
    import dataclasses
    with_lex = dataclasses.replace(g2p, lexicon={"नमक": ["n", "ə", "m", "ə", "k"]})
    assert with_lex.deletion_sites("नमक") == []
    assert with_lex.phonemize_word("नमक") == ["n", "ə", "m", "ə", "k"]


def test_punctuation_does_not_shift_the_site_indices():
    """Sites index the core word's segments, so a trailing comma cannot move
    them. phonemize_word's own docstring is about exactly this hazard."""
    assert HINDI.deletion_sites("नमक") == HINDI.deletion_sites("नमक,")
    assert HINDI.deletion_sites("नमक") == HINDI.deletion_sites("“नमक”")


# -- sentence level ---------------------------------------------------------

def test_sentence_totals_sum_the_words():
    text = "नमक कमल"
    st = sentence_stats(HINDI, text)
    assert st["words"] == 2
    assert st["final"] == sum(w["final"] for w in st["per_word"])
    assert st["medial"] == sum(w["medial"] for w in st["per_word"])
    assert st["sites"] == st["final"] + st["medial"]


def test_medial_words_lists_only_words_with_medial_deletion():
    st = sentence_stats(HINDI, "नमक कमल में")
    assert st["medial_words"] == [w["word"] for w in st["per_word"] if w["medial"]]
    assert all(w in "नमक कमल में".split() for w in st["medial_words"])


def test_an_empty_sentence_is_zero_and_not_an_error():
    st = sentence_stats(HINDI, "   ")
    assert st == {"words": 0, "final": 0, "medial": 0, "sites": 0,
                  "medial_words": [], "per_word": []}


def test_a_real_dev_sentence_has_the_counts_the_demo_set_recorded():
    """hi_004738, chosen for the demo set with 3 medial and 2 final sites.
    If the schwa rules change, this is where it shows up."""
    text = "पहले, राजनीतिक कारणोँ की बात की जाये"
    st = sentence_stats(HINDI, text)
    assert st["medial"] == 3
    assert st["final"] == 2
