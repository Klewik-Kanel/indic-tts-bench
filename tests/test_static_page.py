#!/usr/bin/env python3
"""The listening page's claims about how each clip was made.

A Static Space serves files and runs nothing, so this page IS the deliverable
rather than a view of one. What it says about provenance is therefore a claim
on a public URL, and the claim this file exists to guard is the one a listener
cannot check by ear: which model produced the waveform they are hearing.

Until 5 October the page said the vocoder was "still training" in three
places. r06 and r17 finished on 4 October, so that had become false, and a
vocoded FastSpeech 2 clip would have been shown with no indication that its
mel and its waveform came from different models.

Read as text, not rendered: these assertions are about what the file says.
"""

# --- the trained vocoder's label --------------------------------------------

def test_the_page_labels_a_hifigan_clip_with_which_vocoder():
    """A vocoded mel is not end to end: the mel came from one model and the
    waveform from another, and that is the one thing a listener cannot hear."""
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert 'vocoder.startsWith("hifigan:")' in s
    assert "waveform from HiFi-GAN" in s


def test_the_page_no_longer_claims_the_vocoder_is_training():
    """r06 and r17 finished on 4 October. The page said "still training" in
    three places, which would have been a false claim on a public URL."""
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert "still training" not in s


def test_the_griffin_lim_caveat_survives():
    """It is still reachable: a render without --vocoder produces those clips,
    and they must keep saying so."""
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert 'vocoder === "griffin-lim"' in s
    assert "Griffin-Lim placeholder" in s


def test_the_two_notes_are_visually_distinct():
    """A provenance note must not read as the same kind of thing as a
    placeholder warning, or the warning stops carrying weight."""
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert ".note {" in s and ".caveat {" in s
    note = s[s.index(".note {"):s.index(".note code {")]
    caveat = s[s.index(".caveat {"):s.index(".note {")]
    assert "--muted" in note and "--accent" in caveat


def test_the_vocoder_note_quotes_the_measured_cost():
    """The claim that the acoustic model rather than the vocoder is what you
    hear is measured, so the page carries the number rather than asserting it."""
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert "0.0005" in s and "0.0045" in s


def test_the_summary_note_names_every_vocoder_actually_used():
    """One vocoder per language, so a Hindi and a Marathi page must each name
    their own rather than a hardcoded r06."""
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert "const vocs = new Set()" in s
    assert "one vocoder per language" in s
    assert 'hifi.map(' in s


def test_the_vocoder_name_is_escaped_before_it_reaches_the_page():
    import pathlib
    s = pathlib.Path("space_static/index.html").read_text()
    assert "esc(who)" in s
    assert "esc(v.slice" in s
