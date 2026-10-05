#!/usr/bin/env python3
"""Attaching a vocoder to a mel-only bundle, with the checks that matter.

A mel handed to a vocoder built on different analysis parameters produces
audio that is recognisably speech and quietly wrong. That is the failure the
whole export path exists to prevent, so every one of these checks raises
rather than warning.

No torch and no weights: the bundle objects here are stubs carrying only the
manifest fields the checks read.
"""

import pathlib

import pytest

from src.export.synthesize import VOCODER_ARCHITECTURES, Bundle


class Stub:
    """A Bundle with its __init__ skipped, carrying only a manifest."""

    def __init__(self, arch, lang="hindi", sr=22050, n_mels=80, run_id="rXX"):
        self.root = pathlib.Path(f"exports/{run_id}")
        self.manifest = {"architecture": arch, "language": lang,
                         "sample_rate": sr, "run_id": run_id,
                         "audio": {"n_mels": n_mels},
                         "needs_vocoder": arch in ("fastspeech2", "matcha")}
        self.needs_vocoder = self.manifest["needs_vocoder"]
        self.vocoder = None
        self.device = "cpu"

    sample_rate = property(lambda self: int(self.manifest["sample_rate"]))
    is_vocoder = Bundle.is_vocoder
    attach_vocoder = Bundle.attach_vocoder
    vocode = Bundle.vocode


def test_hifigan_is_the_vocoder_architecture():
    assert "hifigan" in VOCODER_ARCHITECTURES
    assert Stub("hifigan").is_vocoder is True
    assert Stub("fastspeech2").is_vocoder is False
    assert Stub("vits").is_vocoder is False


def test_a_matching_vocoder_attaches():
    fs2 = Stub("fastspeech2", run_id="r01")
    voc = Stub("hifigan", run_id="r06")
    assert fs2.attach_vocoder(voc) is fs2
    assert fs2.vocoder is voc


def test_a_non_vocoder_is_refused():
    fs2 = Stub("fastspeech2", run_id="r01")
    with pytest.raises(SystemExit):
        fs2.attach_vocoder(Stub("vits", run_id="r02"))


def test_an_end_to_end_model_may_not_be_given_a_vocoder():
    """VITS produces a waveform. Passing it a vocoder would mean somebody has
    misread which arm needs one."""
    vits = Stub("vits", run_id="r02")
    with pytest.raises(SystemExit):
        vits.attach_vocoder(Stub("hifigan", run_id="r06"))


def test_a_sample_rate_mismatch_is_refused():
    """A mel on the wrong rate sounds like speech and is wrong."""
    fs2 = Stub("fastspeech2", sr=22050, run_id="r01")
    with pytest.raises(SystemExit):
        fs2.attach_vocoder(Stub("hifigan", sr=16000, run_id="r06"))


def test_a_language_mismatch_is_refused():
    """Each language has its own vocoder, so crossing r06 and r17 would put a
    speaker mismatch inside the control that exists to isolate language."""
    fs2 = Stub("fastspeech2", lang="hindi", run_id="r01")
    with pytest.raises(SystemExit):
        fs2.attach_vocoder(Stub("hifigan", lang="marathi", run_id="r17"))


def test_the_marathi_pair_attaches_to_each_other():
    fs2 = Stub("fastspeech2", lang="marathi", run_id="r15")
    assert fs2.attach_vocoder(Stub("hifigan", lang="marathi", run_id="r17"))


def test_vocode_refuses_on_a_non_vocoder():
    np = pytest.importorskip("numpy")
    with pytest.raises(SystemExit):
        Stub("fastspeech2").vocode(np.zeros((80, 10), dtype="float32"))


def test_vocode_refuses_a_transposed_mel():
    """[T, n_mels] instead of [n_mels, T] is the usual mistake, and it is the
    one that produces plausible audio rather than an error."""
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        Stub("hifigan").vocode(np.zeros((10, 80), dtype="float32"))


def test_vocode_refuses_a_non_2d_mel():
    np = pytest.importorskip("numpy")
    for shape in ((80,), (1, 80, 10)):
        with pytest.raises(ValueError):
            Stub("hifigan").vocode(np.zeros(shape, dtype="float32"))


def test_vocode_refuses_a_manifest_with_no_band_count():
    """A top-level .get with a default of 80 agreed with the real value by
    accident and would have stopped checking the day it changed."""
    np = pytest.importorskip("numpy")
    voc = Stub("hifigan")
    voc.manifest["audio"] = {}
    with pytest.raises(SystemExit):
        voc.vocode(np.zeros((80, 10), dtype="float32"))


def test_the_band_count_is_read_from_audio_not_the_top_level():
    import pathlib as _p
    src = _p.Path("src/export/synthesize.py").read_text()
    body = src[src.index("    def vocode("):]
    body = body[:body.index("    def synthesize(")]
    assert 'self.manifest.get("audio")' in body
    assert 'self.manifest.get("n_mels"' not in body


def test_load_takes_a_vocoder_argument():
    import inspect

    from src.export.synthesize import load
    assert "vocoder" in inspect.signature(load).parameters
    assert inspect.signature(load).parameters["vocoder"].default is None


def test_the_mel_survives_vocoding():
    """The duration measure reads frame counts and must not start depending on
    whether a vocoder happened to be attached."""
    import pathlib as _p
    src = _p.Path("src/export/synthesize.py").read_text()
    body = src[src.index("if self.needs_vocoder:"):]
    body = body[:body.index('if "durations" in out:')]
    assert body.index("sp.mel = ") < body.index("sp.waveform = self.vocoder")
    assert "sp.mel = None" not in body


def test_the_vocoder_provenance_is_recorded_on_the_speech():
    import pathlib as _p
    src = _p.Path("src/export/synthesize.py").read_text()
    i = src.index('sp.extras["vocoder"]')
    block = src[i:i + 400]
    for field in ("run_id", "step", "config_hash", "bundle"):
        assert field in block, field


def test_every_guard_runs_before_torch_is_imported():
    """Guards behind a heavy import are untestable without the package and
    slow to fail with it. griffinlim.py had this same defect."""
    import pathlib as _p
    src = _p.Path("src/export/synthesize.py").read_text()
    body = src[src.index("    def vocode("):]
    body = body[:body.index("    def synthesize(")]
    assert body.index("raise ValueError") < body.index("import torch")
    assert body.index("n_mels") < body.index("import torch")


# --- the two callers --------------------------------------------------------

def test_render_demo_labels_a_vocoded_clip_with_its_vocoder():
    """A vocoded mel is not end to end. Labelling it so would hide which
    vocoder produced the audio, on a page whose whole job is provenance."""
    import pathlib as _p
    src = _p.Path("scripts/render_demo.py").read_text()
    i = src.index('entry["audio_seconds"] = round(sp.audio_seconds, 3)')
    block = src[i:i + 900]
    assert 'sp.extras.get("vocoder")' in block
    assert 'f"hifigan:' in block
    assert block.index("vinfo") < block.index('"end-to-end"')


def test_render_demo_records_the_vocoder_bundle_in_the_manifest():
    import pathlib as _p
    src = _p.Path("scripts/render_demo.py").read_text()
    assert '"vocoder_bundle"' in src


def test_a_vits_bundle_in_the_list_is_not_an_error_when_vocoder_is_passed():
    """attach_vocoder refuses an end-to-end model on purpose, so a mixed set
    of bundles must not reach that refusal at all.

    It used to be caught and retried without a vocoder, which also swallowed a
    real rate or language mismatch and turned it into a silent arm. The gate
    is the manifest's own needs_vocoder now, so the refusal stays fatal.
    See tests/test_render_demo_bundles.py for the behaviour."""
    import pathlib as _p
    src = _p.Path("scripts/render_demo.py").read_text()
    i = src.index("voc = a.vocoder if (a.vocoder and needs) else None")
    block = src[max(0, i - 500):i + 200]
    assert "needs_vocoder" in block
    assert "END_TO_END" in block
    assert "except SystemExit" not in src


def test_the_harness_only_skips_a_mel_only_run_without_a_vocoder():
    import pathlib as _p
    src = _p.Path("scripts/score_intelligibility.py").read_text()
    assert 'manifest.get("needs_vocoder") and not vocoder' in src


def test_the_harness_records_which_vocoder_scored_the_run():
    import pathlib as _p
    src = _p.Path("scripts/score_intelligibility.py").read_text()
    i = src.index('record["vocoder"] = {')
    block = src[i:i + 300]
    for field in ("run_id", "step", "config_hash"):
        assert field in block, field


def test_the_harness_does_not_hand_a_vocoder_to_an_end_to_end_run():
    """Passing one to VITS raises by design, so the call site must gate on
    needs_vocoder rather than passing it unconditionally."""
    import pathlib as _p
    src = _p.Path("scripts/score_intelligibility.py").read_text()
    i = src.index("b = load(bundle_dir, device=\"cpu\",")
    block = src[i:i + 200]
    assert 'needs_vocoder' in block
