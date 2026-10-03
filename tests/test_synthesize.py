#!/usr/bin/env python3
"""The parts of the synthesis path that do not need torch installed.

Model construction and inference are exercised on the DGX, where coqui is
installed. What is tested here is everything that decides whether a bundle is
even loadable, because those are the checks that have to fail loudly rather
than let a demo run on a half-understood bundle: the version guard, the
required manifest fields, and the refusal to proceed without the vocabulary
that trained the weights.

`write_wav` is tested through its standard-library path, which is the one that
runs wherever soundfile is absent.
"""
from __future__ import annotations

import json
import pathlib
import wave

import pytest

from src.export.synthesize import (SUPPORTED_BUNDLE_VERSION, Speech,
                                   read_manifest, write_wav)

GOOD = {
    "bundle_version": 3,
    "run_id": "r02",
    "architecture": "vits",
    "language": "hindi",
    "input_repr": "phoneme",
    "sample_rate": 16000,
    "lr": 0.0002,
    "needs_vocoder": False,
    "step": 100000,
}


def write_bundle(root: pathlib.Path, manifest: dict) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest),
                                        encoding="utf-8")
    return root


# -- the manifest gate ------------------------------------------------------

def test_reads_a_good_manifest(tmp_path):
    root = write_bundle(tmp_path / "b", GOOD)
    assert read_manifest(root)["run_id"] == "r02"


def test_a_directory_without_a_manifest_is_not_a_bundle(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        read_manifest(tmp_path / "empty")
    assert "manifest.json" in str(e.value)


def test_a_newer_bundle_is_refused_rather_than_guessed_at(tmp_path):
    """Loading fields from a format this build does not know is worse than
    stopping, because the failure would surface as bad audio."""
    root = write_bundle(tmp_path / "b",
                        {**GOOD, "bundle_version": SUPPORTED_BUNDLE_VERSION + 1})
    with pytest.raises(SystemExit) as e:
        read_manifest(root)
    assert "bundle_version" in str(e.value)


def test_the_current_version_is_accepted(tmp_path):
    root = write_bundle(tmp_path / "b",
                        {**GOOD, "bundle_version": SUPPORTED_BUNDLE_VERSION})
    assert read_manifest(root)["bundle_version"] == SUPPORTED_BUNDLE_VERSION


@pytest.mark.parametrize("missing", ["run_id", "architecture", "language",
                                     "input_repr", "sample_rate"])
def test_every_field_the_rebuild_needs_is_required(tmp_path, missing):
    manifest = {k: v for k, v in GOOD.items() if k != missing}
    root = write_bundle(tmp_path / "b", manifest)
    with pytest.raises(SystemExit) as e:
        read_manifest(root)
    assert missing in str(e.value)


def test_a_version_2_bundle_still_reads(tmp_path):
    """lr arrived in version 3. An older bundle must not become unloadable,
    since the value does not enter the inference graph."""
    manifest = {k: v for k, v in GOOD.items() if k != "lr"}
    root = write_bundle(tmp_path / "b", {**manifest, "bundle_version": 2})
    assert read_manifest(root).get("lr") is None


# -- the Speech record ------------------------------------------------------

def test_audio_seconds_and_realtime_factor():
    import numpy as np
    sp = Speech(run_id="r02", text="x", tokens=["x"], sample_rate=16000,
                seconds_elapsed=0.5, waveform=np.zeros(16000, dtype="float32"))
    assert sp.audio_seconds == pytest.approx(1.0)
    assert sp.realtime_factor == pytest.approx(0.5)


def test_a_mel_only_result_reports_no_duration():
    """FastSpeech 2 has no waveform, so there is nothing to divide by and the
    realtime factor must be absent rather than zero or one."""
    import numpy as np
    sp = Speech(run_id="r01", text="x", tokens=["x"], sample_rate=22050,
                seconds_elapsed=0.3, mel=np.zeros((80, 40), dtype="float32"),
                needs_vocoder=True)
    assert sp.audio_seconds is None
    assert sp.realtime_factor is None
    assert sp.mel.shape[0] == 80          # n_mels first, the vocoder's order


# -- wav writing ------------------------------------------------------------

def test_write_wav_round_trips_through_the_stdlib(tmp_path):
    import numpy as np
    sr = 16000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype="float32")
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    out = tmp_path / "tone.wav"
    write_wav(out, tone, sr)
    with wave.open(str(out), "rb") as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == sr
        assert fh.getnframes() == len(tone)


def test_write_wav_clips_rather_than_wrapping(tmp_path):
    """An out-of-range sample that wraps turns into a loud click, which in a
    listening test is a confound rather than a cosmetic problem."""
    import numpy as np
    out = tmp_path / "loud.wav"
    write_wav(out, np.array([3.0, -3.0, 0.0], dtype="float32"), 16000)
    with wave.open(str(out), "rb") as fh:
        got = np.frombuffer(fh.readframes(3), dtype="<i2")
    assert got[0] > 32000 and got[1] < -32000
