#!/usr/bin/env python3
"""Errors have to be attributed to the right word class, or the contrast lies.

The alignment and the bookkeeping are tested against a fake front end, so the
assertions do not depend on what the Hindi rule happens to do to any particular
word. A real-G2P check is included and skips when the front end is absent.
"""

import pytest

from src.analysis import position_errors as pe


class FakeG2P:
    """A front end whose deletion classes are declared, not derived.

    `final` and `medial` are sets of orthographic words. Everything else is
    NEITHER. This is what lets the alignment be tested without entangling it
    with the Hindi rule.
    """

    def __init__(self, final=(), medial=()):
        self.final = set(final)
        self.medial = set(medial)

    def segments(self, word):
        return list(word)

    def deletion_sites(self, word):
        if word in self.final:
            return [len(word) - 1]          # the last segment: a final site
        if word in self.medial:
            return [0]                      # not the last segment: medial
        return []


# --- classify --------------------------------------------------------------

def test_a_word_with_a_final_site_is_final():
    assert pe.classify({"final": 1, "medial": 0}) == pe.FINAL


def test_a_word_with_only_a_medial_site_is_medial():
    assert pe.classify({"final": 0, "medial": 2}) == pe.MEDIAL


def test_a_word_with_no_site_is_neither():
    assert pe.classify({"final": 0, "medial": 0}) == pe.NEITHER


def test_final_wins_when_a_word_has_both():
    """The duration measure can only see final sites, so the two measures must
    agree on which class such a word belongs to."""
    assert pe.classify({"final": 1, "medial": 1}) == pe.FINAL


# --- word alignment -------------------------------------------------------

def test_identical_sequences_are_all_matches():
    ops = pe.align_words(["a", "b", "c"], ["a", "b", "c"])
    assert [o[0] for o in ops] == ["match", "match", "match"]


def test_one_changed_word_is_a_substitution():
    ops = pe.align_words(["a", "b", "c"], ["a", "x", "c"])
    assert [o[0] for o in ops] == ["match", "sub", "match"]
    assert ops[1][1:] == ("b", "x")


def test_a_dropped_word_is_a_deletion():
    ops = pe.align_words(["a", "b", "c"], ["a", "c"])
    assert [o[0] for o in ops] == ["match", "del", "match"]
    assert ops[1][1] == "b"


def test_an_extra_word_is_an_insertion():
    ops = pe.align_words(["a", "c"], ["a", "b", "c"])
    assert [o[0] for o in ops] == ["match", "ins", "match"]
    assert ops[1][2] == "b"


def test_ops_come_back_in_reference_order():
    ops = pe.align_words(["a", "b", "c", "d"], ["a", "x", "d"])
    refs = [o[1] for o in ops if o[1] is not None]
    assert refs == ["a", "b", "c", "d"]


def test_an_empty_hypothesis_deletes_every_reference_word():
    ops = pe.align_words(["a", "b"], [])
    assert [o[0] for o in ops] == ["del", "del"]


def test_an_empty_reference_inserts_every_hypothesis_word():
    ops = pe.align_words([], ["a", "b"])
    assert [o[0] for o in ops] == ["ins", "ins"]


# --- attribution -----------------------------------------------------------

def test_a_perfect_transcript_has_no_edits_in_any_class():
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "कमल खिला")
    for c in pe.CLASSES:
        assert p["rates"][c].edits == 0
    assert p["rates"][pe.FINAL].words == 1
    assert p["rates"][pe.NEITHER].words == 1


def test_an_error_lands_in_the_class_of_the_reference_word():
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "कमर खिला")
    assert p["rates"][pe.FINAL].edits == 1
    assert p["rates"][pe.NEITHER].edits == 0


def test_an_error_on_a_siteless_word_does_not_land_on_the_site_class():
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "कमल खिली")
    assert p["rates"][pe.FINAL].edits == 0
    assert p["rates"][pe.NEITHER].edits == 1


def test_a_deleted_word_is_wrong_in_every_character():
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "खिला")
    assert p["rates"][pe.FINAL].edits == len("कमल")
    assert p["rates"][pe.FINAL].chars == len("कमल")
    assert p["rates"][pe.FINAL].cer == 1.0


def test_insertions_are_counted_but_never_attributed():
    """A hypothesis word with no reference partner belongs to no class.

    Spreading it over the classes or dropping it would both move the contrast.
    """
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल", "कमल खिला")
    assert p["insertions"] == 1
    assert p["inserted_chars"] == len("खिला")
    assert sum(p["rates"][c].edits for c in pe.CLASSES) == 0


def test_medial_and_final_are_kept_apart():
    g = FakeG2P(final=["कमल"], medial=["खिला"])
    p = pe.partition(g, "कमल खिला बड़ा", "कमर खिली बड़ा")
    assert p["rates"][pe.FINAL].edits == 1
    assert p["rates"][pe.MEDIAL].edits == 1
    assert p["rates"][pe.NEITHER].edits == 0
    assert p["rates"][pe.NEITHER].words == 1


# --- pooling and rates ------------------------------------------------------

def test_the_class_rate_is_pooled_not_averaged():
    """Short words are where a dropped final schwa is proportionally largest,
    so they must not be allowed to dominate by being averaged."""
    g = FakeG2P(final=["अब", "कमलकमलकम"])
    p = pe.partition(g, "अब कमलकमलकम", "अज कमलकमलकम")
    r = p["rates"][pe.FINAL]
    short, long = "अब", "कमलकमलकम"
    assert r.edits == 1
    # From len(), not from counting by eye: the literal 9 was wrong here once.
    assert r.chars == len(short) + len(long)
    assert r.cer == pytest.approx(1 / (len(short) + len(long)))
    # The point of pooling: the long word's correctness dilutes the short
    # word's error, instead of the two being averaged to 0.25.
    per_word_mean = (1 / len(short) + 0.0) / 2
    assert r.cer < per_word_mean


def test_accumulate_sums_across_utterances():
    g = FakeG2P(final=["कमल"])
    a = pe.partition(g, "कमल खिला", "कमर खिला")
    b = pe.partition(g, "कमल खिला", "कमर खिला")
    tot = pe.accumulate([a, b])
    assert tot["rates"][pe.FINAL].edits == 2
    assert tot["rates"][pe.FINAL].chars == 2 * len("कमल")
    assert tot["rates"][pe.FINAL].cer == pytest.approx(a["rates"][pe.FINAL].cer)


def test_accumulate_pools_insertions_too():
    g = FakeG2P()
    a = pe.partition(g, "कमल", "कमल खिला")
    tot = pe.accumulate([a, a])
    assert tot["insertions"] == 2


def test_a_class_with_no_characters_has_no_rate():
    g = FakeG2P()
    p = pe.partition(g, "कमल", "कमल")
    with pytest.raises(ValueError):
        _ = p["rates"][pe.FINAL].cer
    assert p["rates"][pe.FINAL].has_rate is False


# --- the contrast ----------------------------------------------------------

def test_the_contrast_is_positive_when_errors_concentrate_at_sites():
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "कमर खिला")
    c = pe.contrast(p)[pe.FINAL]
    assert c["excess"] > 0
    assert c["neither_cer"] == 0.0


def test_the_contrast_is_negative_when_errors_avoid_the_sites():
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "कमल खिली")
    c = pe.contrast(p)[pe.FINAL]
    assert c["excess"] < 0


def test_the_contrast_reports_both_denominators():
    """An excess over 40 characters is not the claim an excess over 4000 is."""
    g = FakeG2P(final=["कमल"])
    p = pe.partition(g, "कमल खिला", "कमर खिला")
    c = pe.contrast(p)[pe.FINAL]
    assert c["site_chars"] == len("कमल")
    assert c["neither_chars"] == len("खिला")
    assert c["site_words"] == 1 and c["neither_words"] == 1


def test_a_missing_class_gives_none_rather_than_a_zero_effect():
    g = FakeG2P()
    p = pe.partition(g, "कमल", "कमल")
    assert pe.contrast(p)[pe.FINAL] is None
    assert pe.contrast(p)[pe.MEDIAL] is None


# --- against the real front end --------------------------------------------

def test_the_real_hindi_front_end_produces_all_three_classes():
    """Not a rule check. Only that a real sentence is not all one class, which
    would make the contrast undefined on real data."""
    try:
        from src.g2p import G2P
        g2p = G2P.for_language("hindi")
    except Exception:                                       # noqa: BLE001
        pytest.skip("no Hindi front end available here")
    text = "ठण्ड गहराने लगी थी, उसने मां की जैकेट निकाल कर पेहन ली"
    p = pe.partition(g2p, text, text)
    seen = [c for c in pe.CLASSES if p["rates"][c].words]
    assert len(seen) >= 2, f"only one class present: {seen}"
    assert sum(p["rates"][c].edits for c in pe.CLASSES) == 0
