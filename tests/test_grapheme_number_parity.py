#!/usr/bin/env python3
"""Both arms of the ablation must be handed the same string.

`normalize` runs `devanagari_digits_to_ascii`, and `grapheme_inventory` is the
U+0900 block plus punctuation and the word boundary, so an ASCII digit is
outside the grapheme vocabulary by construction. `phonemize` called
`_expand_numbers` after normalising; `graphemes` did not. Two separate
failures came out of that one missing call:

* r18 (fastspeech2 / marathi / grapheme) raised KeyError on '8' inside
  `collate`, on the prefetch thread, at step 1.
* for every utterance containing a numeral the phoneme arm received a spoken
  number and the grapheme arm received a digit glyph, so the phoneme-versus-
  grapheme contrast absorbed a front-end difference.

Measured over the full manifests at the time of the fix: hindi had 0 digit
runs and 0 OOV symbols in either arm across 5,485 utterances, marathi had 19
digit runs over at most 27 of 5,559. So these tests also stand as the record
that the Hindi runs r01, r04, r07 and r08 could not have been affected.
"""
from __future__ import annotations

import pytest

from src.g2p import G2P
from src.g2p.normalize import normalize
from src.g2p.numbers import expand as expand_numbers
from src.g2p.phoneset import WORD_BOUNDARY
from src.train.text import grapheme_inventory, phoneme_inventory

LANGUAGES = ["hindi", "marathi"]

# The first three are the real transcripts the coverage scan tripped on.
DIGIT_TEXTS = [
    "दोन संन्यासी, म्हणजे गुरु-शिष्य ८ महिन्यांनी पावसाळ्यात परत आले.",
    "खोलीचे दार केवळ ४ फूट उंचीचे होते.",
    "कमी होत-होत १० टक्क्यांवर आली.",
    "आज 8 वाजले",
    "१८ तारखेला",
]

CLEAN_TEXT = "नमकीन कमल आणि फायबर ग्लास."


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("text", DIGIT_TEXTS)
def test_graphemes_emit_no_digits(language, text):
    g2p = G2P.for_language(language)
    assert [t for t in g2p.graphemes(text) if t.isdigit()] == []


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("text", DIGIT_TEXTS)
def test_graphemes_stay_inside_the_inventory(language, text):
    g2p = G2P.for_language(language)
    assert set(g2p.graphemes(text)) <= set(grapheme_inventory())


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("text", DIGIT_TEXTS)
def test_phonemes_stay_inside_the_inventory(language, text):
    g2p = G2P.for_language(language)
    assert set(g2p.phonemize(text)) <= set(phoneme_inventory())


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("text", DIGIT_TEXTS)
def test_both_arms_receive_the_same_string(language, text):
    """Stronger than both-in-inventory: the same characters, same order.

    In-inventory on its own would still pass if one arm silently dropped the
    number, which is the failure mode that would be hardest to notice later.
    """
    g2p = G2P.for_language(language)
    expanded, _ = expand_numbers(normalize(text), language)
    seen = "".join(t for t in g2p.graphemes(text) if t != WORD_BOUNDARY)
    assert seen == expanded.replace(" ", "")


@pytest.mark.parametrize("language", LANGUAGES)
def test_grapheme_arm_records_what_it_could_not_expand(language):
    """A refused number must be counted, not passed through in silence.

    The expander handles the corpora as they stand but not every integer;
    '1998' is one it declines. `unexpanded_numbers` is how a coverage run
    finds out, so the grapheme arm has to append to it too.
    """
    g2p = G2P.for_language(language)
    g2p.graphemes("सन 1998 मध्ये")
    assert "1998" in g2p.unexpanded_numbers


@pytest.mark.parametrize("language", LANGUAGES)
def test_digitless_text_is_untouched(language):
    """The change must be a no-op wherever no numeral occurs.

    This is what makes the phoneme path byte-identical and leaves the finished
    runs alone, since Hindi contains no digit runs at all.
    """
    g2p = G2P.for_language(language)
    expanded, unexpanded = expand_numbers(normalize(CLEAN_TEXT), language)
    assert expanded == normalize(CLEAN_TEXT)
    assert unexpanded == []
    seen = "".join(t for t in g2p.graphemes(CLEAN_TEXT) if t != WORD_BOUNDARY)
    assert seen == normalize(CLEAN_TEXT).replace(" ", "")


def test_grapheme_vocabulary_size_is_unchanged():
    """136 symbols, so r18 requeues with the table size r04 already trained."""
    from src.train.text import Vocab
    assert len(Vocab.build("grapheme")) == 136


# -- the other thing that left the grapheme inventory -----------------------
#
# ma_007005 carried a Latin z inside a Devanagari word. Unlike the digits this
# is a transcript defect rather than a code path, and it was settled by
# listening to the recording: the speaker nasalises, so the character stands
# where anusvara should be. The repair is scoped to the Devanagari-flanked
# case, because a bare z -> anusvara mapping would rewrite any Latin token
# that ever reaches the chain.

MA_007005 = ("सध्या या सर्व गोष्टीzबरोबरच फायबर ग्लासचाही "
             "मोठ्या प्रमाणावर उपयोग केला जातो.")


def test_flanked_z_becomes_anusvara():
    assert normalize("गोष्टीzबरोबरच") == "गोष्टींबरोबरच"


def test_unflanked_z_is_left_alone():
    """Not a blanket rewrite: a z that is not between Devanagari stays.

    If one ever appears in a corpus it should fail the coverage check loudly
    rather than be silently turned into a nasal.
    """
    for text in ("zebra", "क z ब", "zकब", "कबz"):
        assert "z" in normalize(text), text


@pytest.mark.parametrize("language", LANGUAGES)
def test_ma_007005_is_inside_both_inventories(language):
    g2p = G2P.for_language(language)
    assert set(g2p.graphemes(MA_007005)) <= set(grapheme_inventory())
    assert set(g2p.phonemize(MA_007005)) <= set(phoneme_inventory())


def test_the_repair_reaches_the_phoneme_arm_too():
    """The nasal must be heard, not just made representable.

    Before the repair the phonemiser mapped the character to /z/, which is in
    the phone inventory, so the phoneme arm never raised and the defect was
    invisible on that side. Anusvara before the labial /b/ assimilates, so the
    phone at that position is now /m/.
    """
    phones = G2P.for_language("marathi").phonemize(MA_007005)
    assert "z" not in phones
    assert "".join(phones).count("m") >= 1
    window = phones[15:24]
    assert "m" in window, window
