"""G2P unit tests.

Every expected pronunciation here is a claim about Hindi phonology, not about
the code. They are written from the standard descriptions (Ohala 1999 for the
inventory, Narasimhan et al. 2004 for the deletion contexts). If one of these
is wrong, the front-end accuracy number in the paper is wrong, so they are
worth arguing about.
"""

from __future__ import annotations

import pytest

from src.g2p import G2P
from src.g2p.normalize import devanagari_ratio, normalize

HI = G2P.for_language("hindi")
MR = G2P.for_language("marathi")


def phones(word: str, g2p: G2P = HI) -> str:
    return " ".join(g2p.phonemize_word(word))


# --- medial schwa deletion, the V C _ C V rule ----------------------------

@pytest.mark.parametrize(
    "word,expected",
    [
        ("नमकीन", "n ə m k iː n"),        # deletes, VC_CV satisfied
        ("रचना", "r ə tʃ n aː"),
        ("लड़का", "l ə ɽ k aː"),
        ("समझना", "s ə m ə dʒʱ n aː"),     # only the second schwa goes
        ("बचपन", "b ə tʃ p ə n"),
        ("पहलवान", "p ə ɦ ə l ʋ aː n"),
        ("मनमोहन", "m ə n m oː ɦ ə n"),
    ],
)
def test_medial_deletion(word: str, expected: str) -> None:
    assert phones(word) == expected


# --- schwa retained where the context fails -------------------------------

@pytest.mark.parametrize(
    "word,expected",
    [
        ("नमक", "n ə m ə k"),      # no vowel after the final consonant
        ("कमल", "k ə m ə l"),
    ],
)
def test_medial_retention(word: str, expected: str) -> None:
    assert phones(word) == expected


# --- word-final schwa ------------------------------------------------------

@pytest.mark.parametrize(
    "word,expected",
    [
        ("घर", "ɡʱ ə r"),
        ("शब्द", "ʃ ə b d̪"),       # explicit virama cluster survives
    ],
)
def test_final_deletion(word: str, expected: str) -> None:
    assert phones(word) == expected


def test_monosyllable_keeps_its_only_vowel() -> None:
    # Deleting here would leave a word with no vowel at all.
    assert phones("न") == "n ə"


# --- nasality --------------------------------------------------------------

def test_anusvara_becomes_homorganic_nasal_before_a_stop() -> None:
    assert phones("हिंदी") == "ɦ ɪ n d̪ iː"     # dental stop -> dental nasal
    assert phones("अंक") == "ə ŋ k"             # velar stop -> velar nasal


def test_chandrabindu_nasalises_the_vowel() -> None:
    assert phones("माँ") == "m aː̃"


# --- the Marathi control ---------------------------------------------------

def test_marathi_keeps_medial_schwa_but_drops_the_final_one() -> None:
    """The control that isolates schwa deletion from the script.

    Same orthography, same converter, same inventory. Only the medial rule is
    off. If the phoneme-vs-grapheme effect tracks this switch, the effect is
    attributable to schwa deletion rather than to Devanagari generally.
    """
    assert phones("नमकीन", MR) == "n ə m ə k iː n"
    assert phones("घर", MR) == "ɡʱ ə r"


def test_hindi_and_marathi_differ_only_where_expected() -> None:
    # A word with no deletable medial schwa must be identical in both.
    assert phones("रात") == phones("रात", MR)


# --- sentence level --------------------------------------------------------

def test_word_boundaries_are_explicit() -> None:
    out = HI.phonemize("नमकीन कमल")
    assert "|" in out
    assert out[: out.index("|")] == ["n", "ə", "m", "k", "iː", "n"]


def test_grapheme_arm_uses_the_same_normalisation() -> None:
    text = "नमकीन  कमल"
    assert HI.graphemes(text).count("|") == 1
    assert "".join(HI.graphemes(text)).replace("|", "") == "नमकीनकमल"


# --- lexicon override ------------------------------------------------------

def test_lexicon_beats_the_rule() -> None:
    g = G2P.for_language("hindi", lexicon={"नमकीन": ["X"]})
    assert g.phonemize_word("नमकीन") == ["X"]


# --- normalisation ---------------------------------------------------------

def test_nukta_spellings_collapse() -> None:
    precomposed = "ज़"
    decomposed = "ज" + "़"
    assert normalize(precomposed) == normalize(decomposed)
    assert phones(precomposed) == phones(decomposed)


def test_devanagari_digits_become_ascii() -> None:
    assert normalize("२०२६") == "2026"


def test_danda_becomes_a_full_stop() -> None:
    assert normalize("यह घर है।").endswith(".")


def test_devanagari_ratio_flags_code_switching() -> None:
    assert devanagari_ratio("नमकीन") == 1.0
    assert devanagari_ratio("hello world") == 0.0
    assert 0.0 < devanagari_ratio("नमक salt") < 1.0


# --- inventory stability ---------------------------------------------------

def test_inventory_is_deterministic_and_starts_with_specials() -> None:
    a = HI.inventory()
    b = HI.inventory()
    assert a == b
    assert a[:3] == ["|", "sil", "<unk>"]
    assert len(set(a)) == len(a), "duplicate symbols would corrupt the embedding table"


# --- regressions for two bugs found while building this ---------------------

def test_combining_marks_survive_normalisation() -> None:
    """Regression: filtering on str.isalnum() deleted every matra and virama.

    Devanagari matras, the virama, anusvara and chandrabindu are Unicode
    combining marks (Mn/Mc) and isalnum() is False for all of them. The bug
    turned every word into a bare consonant skeleton and was invisible until
    the phone output was checked.
    """
    assert normalize("नमकीन") == "नमकीन"
    assert normalize("शब्द") == "शब्द"
    assert chr(0x093F) in normalize("हिंदी")     # the i-matra
    assert chr(0x094D) in normalize("शब्द")      # the virama


def test_nukta_letters_survive_composition_exclusion() -> None:
    """Regression: U+0958..U+095F are Unicode composition exclusions.

    NFC does not recompose base-plus-nukta into them, and a precomposed
    literal pasted into source is often already decomposed, so a mapping
    written with literals silently does nothing.
    """
    decomposed = chr(0x0921) + chr(0x093C)     # base letter plus combining nukta
    precomposed = chr(0x095C)                  # the single-codepoint form
    assert decomposed != precomposed, "the two spellings must start out distinct"
    assert normalize(decomposed) == normalize(precomposed) == precomposed
    assert phones("ल" + decomposed + "का") == "l ə ɽ k aː"
    assert phones("ल" + precomposed + "का") == "l ə ɽ k aː"
