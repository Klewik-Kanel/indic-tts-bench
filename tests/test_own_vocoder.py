#!/usr/bin/env python3
"""The export path must accept OUR vocoder, not only a stranger's.

resolve_vocoder was written when the vocoder was going to be a published
HiFi-GAN checkpoint, so it hands the path to read_config, which looks for
config.json, config.yaml or config.yml and raises on a directory holding none
of them. An export bundle holds manifest.json and model.pt, so
`--vocoder exports/r06_step18000` failed outright and the only vocoders the
export path could verify were ones this project did not train.

Everything here is JSON on disk: no torch, no weights.
"""

import json
import pathlib
import tempfile

import pytest

from src.export.bundle import resolve_own_bundle, resolve_vocoder
from src.export.vocoder import project_mel


def _bundle(tmp, name="r06_step18000", arch="hifigan", lang="hindi",
            sr=22050, audio=None, run_id="r06", step=18000):
    d = pathlib.Path(tmp) / name
    d.mkdir(parents=True)
    man = {"bundle_version": 3, "run_id": run_id, "architecture": arch,
           "language": lang, "input_repr": "none", "sample_rate": sr,
           "step": step, "config_hash": "abc123",
           "audio": audio if audio is not None else project_mel(sr)}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return d


def test_one_of_our_bundles_verifies():
    with tempfile.TemporaryDirectory() as tmp:
        d = _bundle(tmp)
        out = resolve_own_bundle(d, 22050, "hindi")
        assert out is not None
        assert out["mel_verified"] is True
        assert out["run_id"] == "r06"
        assert out["step"] == 18000
        assert "one of our own bundles" in out["source"]


def test_resolve_vocoder_routes_a_bundle_to_that_branch():
    """The whole bug: this used to reach read_config and raise."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _bundle(tmp)
        out = resolve_vocoder(str(d), 22050, "hindi")
        assert out["mel_verified"] is True
        assert out["name"] == "r06_step18000"


def test_a_path_that_is_not_our_bundle_returns_none_and_falls_through():
    """A stranger's checkpoint directory must still take the old path."""
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "someones_hifigan"
        d.mkdir()
        assert resolve_own_bundle(d, 22050, "hindi") is None


def test_a_non_vocoder_bundle_is_refused():
    """exports/r02_step100000 is a VITS bundle. Passing it as the vocoder is a
    mistake that would otherwise be recorded as verified."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _bundle(tmp, name="r02_step100000", arch="vits", run_id="r02")
        with pytest.raises(SystemExit):
            resolve_own_bundle(d, 22050, "hindi")


def test_a_sample_rate_mismatch_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        d = _bundle(tmp, sr=16000, audio=project_mel(16000))
        with pytest.raises(SystemExit) as e:
            resolve_own_bundle(d, 22050, "hindi")
        assert "sounds like speech" in str(e.value)


def test_a_language_mismatch_is_refused():
    """r17 on a Hindi run would put a speaker mismatch inside the control."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _bundle(tmp, name="r17_step22000", lang="marathi", run_id="r17")
        with pytest.raises(SystemExit) as e:
            resolve_own_bundle(d, 22050, "hindi")
        assert "Marathi control" in str(e.value)


def test_the_marathi_vocoder_verifies_against_a_marathi_run():
    with tempfile.TemporaryDirectory() as tmp:
        d = _bundle(tmp, name="r17_step22000", lang="marathi", run_id="r17")
        out = resolve_own_bundle(d, 22050, "marathi")
        assert out["mel_verified"] is True and out["run_id"] == "r17"


def test_a_mel_that_disagrees_is_refused_and_the_report_names_the_field():
    """The manifest's audio block was written from project_mel, so a
    disagreement means something real diverged rather than two conventions
    being compared."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = dict(project_mel(22050))
        bad["hop_length"] = 300
        d = _bundle(tmp, audio=bad)
        with pytest.raises(SystemExit) as e:
            resolve_own_bundle(d, 22050, "hindi")
        msg = str(e.value)
        assert "hop_length" in msg and "300" in msg and "256" in msg


def test_a_missing_mel_field_is_refused_rather_than_skipped():
    """Absence is a mismatch: a bundle that does not state its hop cannot be
    checked, which is the rule the external path already follows."""
    with tempfile.TemporaryDirectory() as tmp:
        short = {k: v for k, v in project_mel(22050).items() if k != "n_mels"}
        d = _bundle(tmp, audio=short)
        with pytest.raises(SystemExit) as e:
            resolve_own_bundle(d, 22050, "hindi")
        assert "n_mels" in str(e.value)


def test_a_bare_name_is_still_recorded_as_unverified():
    out = resolve_vocoder("some/published/hifigan", 22050, "hindi")
    assert out["mel_verified"] is False
    assert "NOT checked" in out["note"]


def test_no_vocoder_is_still_no_vocoder():
    out = resolve_vocoder("", 22050, "hindi")
    assert out["mel_verified"] is False
    assert out["name"] == ""


def test_the_export_passes_the_language_through():
    src = pathlib.Path("src/export/bundle.py").read_text()
    assert 'resolve_vocoder(vocoder, int(cfg["sample_rate"]), cfg["language"])' in src
