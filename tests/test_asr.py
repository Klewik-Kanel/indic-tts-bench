#!/usr/bin/env python3
"""The intelligibility proxy's arithmetic and its pins.

Nothing here downloads a model. What is tested is the part that can silently
be wrong without anyone noticing: the error rates, the normalisation applied to
both sides of the comparison, the language codes, and the revision pins that
make a published number reproducible.
"""

import pytest

from src.eval import asr


# --- the pins --------------------------------------------------------------

def test_both_backends_are_pinned_to_a_revision():
    for name in ("indicconformer", "mms"):
        rev = asr.model_card(name)["revision"]
        assert len(rev) == 40, f"{name} revision is not a full sha: {rev!r}"
        assert all(c in "0123456789abcdef" for c in rev)


def test_the_licences_are_recorded_because_they_differ():
    assert asr.model_card("indicconformer")["licence"] == "MIT"
    assert asr.model_card("mms")["licence"] == "CC-BY-NC-4.0"


def test_mms_uses_the_three_letter_code():
    """`hi` loads a different adapter or none, and fails as a wrong-language
    transcript rather than as an exception."""
    assert asr.lang_code("mms", "hindi") == "hin"
    assert asr.lang_code("mms", "marathi") == "mar"


def test_indicconformer_uses_the_two_letter_code():
    assert asr.lang_code("indicconformer", "hindi") == "hi"
    assert asr.lang_code("indicconformer", "marathi") == "mr"


def test_an_unknown_backend_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        asr.model_card("whisper")
    with pytest.raises(KeyError):
        asr.backend("whisper", "hindi")


def test_an_unknown_language_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        asr.lang_code("mms", "sanskrit")


def test_model_card_is_a_copy_so_a_caller_cannot_edit_the_pin():
    card = asr.model_card("mms")
    card["revision"] = "deadbeef"
    assert asr.model_card("mms")["revision"] != "deadbeef"


def test_describe_carries_everything_the_table_needs():
    d = asr.MmsCtc("hindi").describe()
    for k in ("backend", "repo", "revision", "licence", "role",
              "lang_arg", "decoding", "target_sr", "resampler"):
        assert k in d, k
    assert d["decoding"] == "ctc-greedy"
    assert d["target_sr"] == 16000


# --- edit distance ---------------------------------------------------------

def test_edit_distance_of_identical_sequences_is_zero():
    assert asr.edit_distance("कमल", "कमल") == 0


def test_edit_distance_counts_one_substitution():
    assert asr.edit_distance("कमल", "कमर") == 1


def test_edit_distance_counts_one_deletion():
    assert asr.edit_distance("कमल", "कल") == 1


def test_edit_distance_against_empty_is_the_length():
    assert asr.edit_distance("कमल", "") == 3
    assert asr.edit_distance("", "कमल") == 3


def test_edit_distance_is_symmetric():
    a, b = "नमस्ते", "नमसते"
    assert asr.edit_distance(a, b) == asr.edit_distance(b, a)


# --- character error rate --------------------------------------------------

def test_a_perfect_transcript_scores_zero():
    assert asr.cer("कमल खिला", "कमल खिला") == 0.0


def test_cer_is_edits_over_reference_length():
    # 'कमल' is 3 characters; one substitution.
    assert asr.cer("कमल", "कमर") == pytest.approx(1 / 3)


def test_spaces_are_dropped_by_default_so_segmentation_is_not_measured():
    """The same characters, segmented differently, is not an error by default.

    Devanagari word boundaries interact with the phenomena under test, so a
    rate that counts spaces partly measures this project's own tokenisation.
    """
    assert asr.cer("कमल खिला", "कमलखिला") == 0.0
    assert asr.cer("कमल खिला", "कमलखिला", drop_spaces=False) > 0.0


def test_normalisation_is_applied_to_both_sides():
    """A decomposed nukta and a composed one are the same string to a listener.

    If only one side were normalised, every transcript would carry a constant
    error that has nothing to do with the audio.
    """
    composed = "ड़"                       # DDDHA, composed
    decomposed = "ड़"               # DDA + nukta
    assert asr.cer(composed, decomposed) == 0.0
    assert asr.cer(decomposed, composed) == 0.0


def test_devanagari_digits_normalise_on_both_sides():
    assert asr.cer("१२", "12") == 0.0


def test_an_empty_reference_raises_rather_than_scoring_perfect():
    """0.0 would read as a perfect transcription of nothing."""
    with pytest.raises(ValueError):
        asr.cer("", "कमल")
    with pytest.raises(ValueError):
        asr.cer("   ", "कमल")


def test_a_wholly_wrong_transcript_scores_at_least_one():
    assert asr.cer("कमल", "खबरदार") >= 1.0


def test_an_empty_hypothesis_scores_one():
    assert asr.cer("कमल", "") == 1.0


# --- word error rate -------------------------------------------------------

def test_wer_counts_whole_words():
    assert asr.wer("कमल खिला है", "कमल खिला है") == 0.0
    assert asr.wer("कमल खिला है", "कमल खिली है") == pytest.approx(1 / 3)


def test_one_slurred_character_costs_a_whole_word_in_wer_but_not_in_cer():
    """Why CER leads. A single missed schwa is 1/3 of the WER and 1/11 of the
    CER, so WER exaggerates by a factor that depends on word length."""
    ref, hyp = "कमल खिला है", "कमर खिला है"
    assert asr.wer(ref, hyp) > asr.cer(ref, hyp)


def test_wer_on_an_empty_reference_raises():
    with pytest.raises(ValueError):
        asr.wer("", "कमल")


# --- pooling ---------------------------------------------------------------

def test_corpus_cer_pools_rather_than_averaging_per_utterance():
    """A 3-character utterance must not weigh as much as a 30-character one.

    Per-utterance mean here would be (1/3 + 0)/2 = 0.1667. Pooled is
    1 edit over 3+30 = 33 characters = 0.0303.
    """
    short = ("कमल", "कमर")                       # 1 edit over 3 chars
    long_ref = "कमल" * 10                         # 30 chars, perfect
    pooled = asr.corpus_cer([short, (long_ref, long_ref)])
    assert pooled == pytest.approx(1 / 33)
    per_utt = (asr.cer(*short) + 0.0) / 2
    assert per_utt > pooled


def test_corpus_cer_of_a_perfect_set_is_zero():
    assert asr.corpus_cer([("कमल", "कमल"), ("खिला", "खिला")]) == 0.0


def test_corpus_cer_raises_on_an_empty_reference_in_the_set():
    with pytest.raises(ValueError):
        asr.corpus_cer([("कमल", "कमल"), ("", "खिला")])


# --- resampling ------------------------------------------------------------

def test_audio_already_at_the_target_rate_is_not_filtered():
    """A 16 kHz run must not be pushed through a resampler the 22.05 kHz runs
    needed. Identity, not a round trip."""
    np = pytest.importorskip("numpy")
    y = np.linspace(-1, 1, 400, dtype="float32")
    out = asr.to_target_sr(y, 16000)
    assert out is y or np.array_equal(out, y)


def test_stereo_is_mixed_to_mono():
    np = pytest.importorskip("numpy")
    y = np.stack([np.zeros(300, dtype="float32"),
                  np.ones(300, dtype="float32")])
    out = asr.to_target_sr(y, 16000)
    assert out.ndim == 1
    assert out.shape == (300,)
    assert np.allclose(out, 0.5, atol=1e-6)


def test_the_resampler_is_named_so_the_table_can_state_it():
    assert asr.RESAMPLER == "soxr_hq"
    assert asr.TARGET_SR == 16000


# --- punctuation ------------------------------------------------------------

def test_punctuation_is_removed_because_no_ctc_recogniser_emits_it():
    """Measured: 41 and 48 per cent of the character error on the first two
    ground-truth utterances was commas and a full stop."""
    assert asr.cer("कमल, खिला.", "कमल खिला") == 0.0
    assert asr.cer("कमल; खिला!", "कमल खिला") == 0.0


def test_the_danda_is_removed_too():
    assert asr.cer("कमल खिला।", "कमल खिला") == 0.0
    assert asr.cer("कमल॥", "कमल") == 0.0


def test_combining_marks_are_not_punctuation_and_must_survive():
    """The nukta, anusvara and visarga are Mn or Mc, and they are the
    contrasts this corpus is about. Stripping them would silently erase the
    distinctions being measured."""
    for mark in ("़", "ं", "ः"):
        assert mark in asr.normalise_for_scoring("क" + mark), repr(mark)
    assert asr.normalise_for_scoring("कर्ज़") == "कर्ज़"


def test_a_nukta_difference_is_still_an_error():
    """The stripping must not be so eager that it hides a real contrast."""
    assert asr.cer("कर्ज़", "कर्ज") > 0.0


def test_strip_punctuation_leaves_the_letters_alone():
    assert asr.strip_punctuation("कमल, खिला.") == "कमल खिला"
    assert asr.strip_punctuation("कमल") == "कमल"


def test_wer_also_sees_punctuation_stripped():
    """Otherwise 'खिला.' and 'खिला' would be two different words."""
    assert asr.wer("कमल खिला.", "कमल खिला") == 0.0


def test_normalise_for_scoring_collapses_the_gap_punctuation_leaves():
    """Removing a comma between two words must not leave a double space, or
    the word split produces an empty token."""
    out = asr.normalise_for_scoring("कमल , खिला")
    assert "  " not in out
    assert out.split() == ["कमल", "खिला"]


def test_corpus_cer_is_punctuation_free_as_well():
    assert asr.corpus_cer([("कमल, खिला.", "कमल खिला")]) == 0.0
