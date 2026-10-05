#!/usr/bin/env python3
"""The checks that stand between a render and a public URL.

A Static Space serves files and runs nothing, so whatever is uploaded is
immediately the public artefact. Three ways this page can look finished and be
wrong, all of them reachable in this project:

  - data.json missing: the page fetches it at load, so it renders an empty
    listening test. A working demo of nothing.
  - a clip named in data.json with no file: the player shows controls that
    play silence, and a listener reads that as the model being silent.
  - a FastSpeech 2 arm with no vocoder: that arm emits a mel and is genuinely
    silent without one, which is a rendering mistake rather than a result.

Each is a refusal, not a warning.
"""

import json
import pathlib
import tempfile

import pytest

from scripts.publish_space import MEL_ONLY, audit, describe


def _write(tmp, data, clips=("audio/r02/s1.wav",), page=True):
    root = pathlib.Path(tmp)
    if page:
        (root / "index.html").write_text("<html></html>", encoding="utf-8")
    (root / "data.json").write_text(json.dumps(data), encoding="utf-8")
    for rel in clips:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"RIFF....WAVE")
    return root


def _ok_data():
    return {
        "language": "hindi", "split": "dev",
        "vocoder_bundle": "exports/r06_step18000",
        "generated_utc": "2026-10-05T12:00:00+00:00",
        "runs": [{"run_id": "r02", "architecture": "vits"},
                 {"run_id": "r01", "architecture": "fastspeech2"}],
        "sentences": [{"id": "s1", "runs": {
            "r02": {"audio": "audio/r02/s1.wav", "vocoder": "end-to-end"},
            "r01": {"audio": "audio/r01/s1.wav",
                    "vocoder": "hifigan:r06@18000"}}}],
        "failures": [],
    }


def test_a_complete_render_passes():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        bad, summary = audit(root)
        assert bad == [], bad
        assert summary["clips"] == 2
        assert summary["arms"]["r01"]["vocoder"] == ["hifigan:r06@18000"]


def test_a_missing_data_json_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "index.html").write_text("<html></html>", encoding="utf-8")
        bad, summary = audit(root)
        assert any("data.json" in b for b in bad)
        assert any("empty listening test" in b for b in bad)
        assert summary == {}


def test_a_missing_index_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"),
                      page=False)
        bad, _ = audit(root)
        assert any("index.html" in b for b in bad)


def test_a_clip_with_no_file_refuses():
    """The player would show controls that play silence."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(), clips=("audio/r02/s1.wav",))
        bad, _ = audit(root)
        assert any("have no file" in b for b in bad)
        assert any("audio/r01/s1.wav" in b for b in bad)


def test_a_mel_only_arm_with_no_audio_refuses_and_says_to_re_render():
    """FastSpeech 2 without a vocoder is silent. Publishing that as a result
    is the mistake this check exists for."""
    d = _ok_data()
    d["sentences"][0]["runs"]["r01"] = {"audio": None, "vocoder": None}
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, d, clips=("audio/r02/s1.wav",))
        bad, _ = audit(root)
        assert any("would publish as silent" in b for b in bad)
        assert any("--vocoder" in b for b in bad)


def test_a_mel_only_arm_labelled_end_to_end_refuses():
    """Its mel came from one model and its waveform from another, so the label
    is a false provenance claim on a public page."""
    d = _ok_data()
    d["sentences"][0]["runs"]["r01"]["vocoder"] = "end-to-end"
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, d,
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        bad, _ = audit(root)
        assert any("cannot be end to end" in b for b in bad)


def test_an_end_to_end_arm_is_not_required_to_have_a_vocoder():
    """VITS produces a waveform. Demanding one would refuse a correct render."""
    d = _ok_data()
    d["runs"] = [{"run_id": "r02", "architecture": "vits"}]
    d["sentences"][0]["runs"] = {
        "r02": {"audio": "audio/r02/s1.wav", "vocoder": "end-to-end"}}
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, d, clips=("audio/r02/s1.wav",))
        bad, _ = audit(root)
        assert bad == [], bad


def test_no_sentences_refuses():
    d = _ok_data()
    d["sentences"] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, d, clips=())
        bad, _ = audit(root)
        assert any("no sentences" in b for b in bad)


def test_unreadable_json_refuses_rather_than_raising():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "index.html").write_text("<html></html>", encoding="utf-8")
        (root / "data.json").write_text("{not json", encoding="utf-8")
        bad, summary = audit(root)
        assert any("not readable JSON" in b for b in bad)
        assert summary == {}


def test_matcha_counts_as_mel_only():
    """It is future work, and if it returns it must not slip past this."""
    assert "fastspeech2" in MEL_ONLY
    assert "matcha" in MEL_ONLY
    assert "vits" not in MEL_ONLY
    assert "hifigan" not in MEL_ONLY


def test_describe_names_every_arm_and_its_vocoder():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        _bad, summary = audit(root)
        text = describe(summary)
        for want in ("r01", "r02", "fastspeech2", "vits",
                     "hifigan:r06@18000", "exports/r06_step18000"):
            assert want in text, want


def test_render_failures_are_surfaced_in_the_summary():
    d = _ok_data()
    d["failures"] = [{"id": "s9", "error": "boom"}]
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, d,
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        _bad, summary = audit(root)
        assert summary["failures"] == 1
        assert "1 render failures" in describe(summary)


def test_dry_run_uploads_nothing():
    src = pathlib.Path("scripts/publish_space.py").read_text()
    body = src[src.index("def main("):]
    assert body.index("a.dry_run") < body.index("upload_folder")


def test_the_space_sdk_is_static():
    """It was first created with the Gradio SDK and the implicit create then
    refused with 402, because free cpu-basic was restricted in July 2026."""
    src = pathlib.Path("scripts/publish_space.py").read_text()
    assert 'space_sdk="static"' in src


def test_nothing_uploads_while_a_check_fails():
    src = pathlib.Path("scripts/publish_space.py").read_text()
    body = src[src.index("def main("):]
    assert body.index("NOT PUBLISHING") < body.index("upload_folder")


# -- a listing from one render and audio from another ------------------------

def _aged(root, data_offset, wav_offset):
    """A site whose data.json and newest wav have a known age difference."""
    import os
    import time
    now = time.time()
    os.utime(root / "data.json", (now + data_offset, now + data_offset))
    for w in root.rglob("*.wav"):
        os.utime(w, (now + wav_offset, now + wav_offset))


def test_a_data_json_older_than_the_newest_clip_is_refused():
    """The real failure: the render wrote fresh VITS clips, died on a vocoder
    bundle before writing data.json, and left the page labelling its
    FastSpeech 2 arms griffin-lim from the day before. Every other check
    passed, so this is the only one that would have caught it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        _aged(root, data_offset=-86400, wav_offset=0)
        bad, _ = audit(root)
        assert any("did not finish" in b for b in bad), bad
        assert any("different runs" in b for b in bad)


def test_a_finished_render_passes_the_staleness_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        _aged(root, data_offset=0, wav_offset=-5)
        assert audit(root)[0] == []


def test_a_few_seconds_of_filesystem_lag_is_not_a_stale_render():
    """data.json is written last, but not atomically with the wavs. A slow
    filesystem must not read as a dead render."""
    from scripts.publish_space import STALE_SLACK_S
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        _aged(root, data_offset=0, wav_offset=+30)
        assert STALE_SLACK_S >= 60
        assert audit(root)[0] == []


def test_a_stale_listing_is_reported_in_minutes():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        _aged(root, data_offset=-3600, wav_offset=0)
        bad, _ = audit(root)
        assert any("60 min older" in b for b in bad), bad


def test_no_wavs_is_the_audio_check_not_the_staleness_check():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write(tmp, _ok_data(),
                      clips=("audio/r02/s1.wav", "audio/r01/s1.wav"))
        for w in root.rglob("*.wav"):
            w.unlink()
        bad, _ = audit(root)
        assert any("no file" in b for b in bad)
        assert not any("did not finish" in b for b in bad)
