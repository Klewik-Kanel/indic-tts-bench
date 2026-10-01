"""The pretrained vocoder's mel front end must match the one that trained the models.

This is the check that stands between the vocoder deviation and a set of
unusable audio metrics. A HiFi-GAN fed mels on the wrong hop or the wrong mel
band range still produces speech, and every metric downstream still produces a
number, so nothing later in the pipeline can catch it.

These tests need no torch, no audio stack and no vocoder: they exercise the
config reading and the comparison, which is where the decision is actually
made. The configs below are the real shapes of the two HiFi-GAN lineages in
circulation, not invented ones.
"""

from __future__ import annotations

import json

import pytest

from src.export import vocoder as V
from src.export.bundle import AUDIO
from src.train import features as F

SR = 22_050


def coqui(**over) -> dict:
    """coqui-ai/TTS writes its analysis parameters under an `audio` block."""
    audio = {"sample_rate": SR, "fft_size": 1024, "win_length": 1024,
             "hop_length": 256, "num_mels": 80, "mel_fmin": 0.0,
             "mel_fmax": 11025.0}
    audio.update(over)
    return {"model": "hifigan", "audio": audio}


def jik876(**over) -> dict:
    """jik876/hifi-gan writes them at the top level, with other spellings."""
    cfg = {"sampling_rate": SR, "n_fft": 1024, "win_size": 1024,
           "hop_size": 256, "num_mels": 80, "fmin": 0, "fmax": 11025}
    cfg.update(over)
    return cfg


def verdicts(cfg: dict) -> dict:
    found, where = V.resolve_defaults(*V.find_mel(cfg))
    return {r["param"]: r["verdict"] for r in V.compare(V.project_mel(SR), found, where)}


# --- both lineages are read -------------------------------------------------

def test_a_matching_coqui_config_passes():
    assert set(verdicts(coqui()).values()) == {"ok"}
    rows = V.assert_mel_matches(SR, coqui(), "test")
    assert len(rows) == len(V.CANON)


def test_a_matching_jik876_config_passes():
    assert set(verdicts(jik876()).values()) == {"ok"}


def test_int_and_float_band_edges_both_count_as_matching():
    """0 and 0.0 are the same band edge; an exact comparison would fail here."""
    assert verdicts(coqui(mel_fmin=0))["fmin"] == "ok"
    assert verdicts(jik876(fmin=0.0))["fmin"] == "ok"


# --- the realistic trap -----------------------------------------------------

def test_the_ljspeech_hifigan_mel_band_ceiling_is_caught():
    """The off-the-shelf checkpoint most likely to be reached for.

    Vocoders published for LJSpeech commonly cap the mel filter bank at 8 kHz,
    while this project builds its bank to sr/2 = 11025 Hz. Everything else
    agrees, which is what makes it dangerous: the sample rate and hop match, so
    the audio is the right length and the right pitch, and only the top of the
    spectrum is analysed differently.
    """
    assert verdicts(coqui(mel_fmax=8000.0))["fmax"] == "MISMATCH"
    assert verdicts(jik876(fmax=8000))["fmax"] == "MISMATCH"
    with pytest.raises(SystemExit, match="does not match"):
        V.assert_mel_matches(SR, coqui(mel_fmax=8000.0), "ljspeech-ish")


def test_a_16k_vocoder_is_caught_on_more_than_one_parameter():
    v = verdicts(coqui(sample_rate=16_000, mel_fmax=8000.0))
    assert v["sample_rate"] == "MISMATCH"
    assert v["fmax"] == "MISMATCH"


def test_a_different_hop_is_caught():
    assert verdicts(coqui(hop_length=300))["hop_length"] == "MISMATCH"
    assert verdicts(jik876(hop_size=512))["hop_length"] == "MISMATCH"


# --- absence is a mismatch, not a pass --------------------------------------

def test_an_unstated_hop_fails_rather_than_being_assumed():
    cfg = coqui()
    del cfg["audio"]["hop_length"]
    assert verdicts(cfg)["hop_length"] == "ABSENT"
    with pytest.raises(SystemExit):
        V.assert_mel_matches(SR, cfg, "no hop")


def test_an_empty_config_fails_on_everything():
    assert set(verdicts({}).values()) == {"ABSENT"}


# --- the one documented default ---------------------------------------------

def test_a_null_fmax_means_half_the_sample_rate():
    """coqui writes `mel_fmax: null` for sr/2, which is what this project uses."""
    assert verdicts(coqui(mel_fmax=None))["fmax"] == "ok"


def test_a_null_hop_has_no_default_and_still_fails():
    """Only fmax has a conventional default. Guessing the rest is the bug."""
    assert verdicts(coqui(hop_length=None))["hop_length"] == "ABSENT"


# --- where the project's numbers come from ----------------------------------

def test_the_project_mel_is_read_from_the_code_that_computes_the_features():
    p = V.project_mel(SR)
    assert (p["n_fft"], p["win_length"], p["hop_length"], p["n_mels"]) == \
        (F.N_FFT, F.WIN, F.HOP, F.N_MELS)
    assert p["fmin"] == 0.0 and p["fmax"] == SR / 2


def test_the_pitch_tracker_range_is_not_mistaken_for_the_mel_band_range():
    """features.FMIN_HZ/FMAX_HZ are yin's search range for one adult speaker.

    Wiring those in as mel band edges would build an 80-band filter bank over
    60-600 Hz, which is a plausible-looking mistake and would make every mel in
    the project wrong rather than just the vocoder's.
    """
    p = V.project_mel(SR)
    assert p["fmin"] != F.FMIN_HZ and p["fmax"] != F.FMAX_HZ


def test_the_two_hop_constants_must_agree(monkeypatch):
    monkeypatch.setattr(F, "HOP", 512)
    with pytest.raises(SystemExit, match="same number"):
        V.project_mel(SR)


def test_the_bundle_audio_block_still_matches_the_features(monkeypatch):
    """bundle.AUDIO is derived from project_mel; this is the old guarantee."""
    assert AUDIO["hop_length"] == F.HOP
    assert AUDIO["n_fft"] == F.N_FFT and AUDIO["n_mels"] == F.N_MELS


# --- reading a config off disk ----------------------------------------------

def test_a_config_json_is_read_from_a_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(coqui()), encoding="utf-8")
    cfg, source = V.read_config(p)
    assert cfg["audio"]["hop_length"] == 256 and str(p) in source


def test_a_directory_is_searched_for_its_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(jik876()), encoding="utf-8")
    cfg, source = V.read_config(tmp_path)
    assert cfg["hop_size"] == 256 and "config.json" in source


def test_a_directory_without_a_config_says_so(tmp_path):
    with pytest.raises(SystemExit, match="no config.json"):
        V.read_config(tmp_path)


def test_an_unreadable_extension_is_refused(tmp_path):
    p = tmp_path / "vocoder.bin"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a config or a checkpoint"):
        V.read_config(p)
