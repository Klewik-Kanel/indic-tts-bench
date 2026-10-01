"""The Matcha-TTS adapter, and the preconditions it refuses to guess at.

Matcha differs from the other two adapters in ways that are each silent when
got wrong, so each one has a test here:

  - its mel targets are normalised by corpus statistics, and using upstream's
    LJSpeech numbers, or the 0.0/1.0 defaults, trains against the wrong centre
    and scale without erroring anywhere
  - statistics computed under a different mel analysis describe a different
    quantity, so a stale stats file has to be refused rather than reused
  - its published weights were fitted to a mel bank reaching 8 kHz, so
    initialising from them under this project's sr/2 bank is a different model
    rather than a warm start
  - its encoder and cfm configs are read by attribute while its decoder config
    is splatted as a mapping, which is visible only by reading upstream

Everything except the collate test runs without torch, without the matcha
package and without audio.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.train import adapters, melstats
from src.train import features as F

SR = 22_050


def cfg(**over) -> dict:
    c = {"run_id": "r03", "architecture": "matcha", "language": "hindi",
         "input_repr": "phoneme", "data": "9h", "sample_rate": SR,
         "batch_frames": 12_000, "max_steps": 2, "lr": 2e-4,
         "warmup_steps": 1, "grad_clip": 1.0, "precision": "fp32",
         "seed": 0, "merge_nukta": False, "init_from": ""}
    c.update(over)
    return c


@pytest.fixture()
def stats_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(melstats, "PROCESSED", tmp_path)
    return tmp_path


def write_stats(root: pathlib.Path, language: str = "hindi",
                sample_rate: int = SR, **over) -> pathlib.Path:
    d = {"mel_mean": -4.1, "mel_std": 1.9,
         "mel_params": F.mel_params(sample_rate)}
    d.update(over)
    p = root / language / "mel_stats.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


# --- the statistics are a precondition, not a default -----------------------

def test_missing_statistics_stop_the_run_and_name_the_command(stats_dir):
    with pytest.raises(SystemExit, match="melstats"):
        adapters.MatchaAdapter().mel_stats(cfg())


def test_upstream_ljspeech_numbers_are_not_used_as_a_fallback(stats_dir):
    """The failure mode this guard exists for: 0.0/1.0 or LJSpeech's numbers
    would both train happily against the wrong distribution."""
    with pytest.raises(SystemExit):
        adapters.MatchaAdapter().mel_stats(cfg())
    assert adapters.MATCHA_LJSPEECH["mel_mean"] == -5.536622
    assert adapters.MATCHA_LJSPEECH["mel_std"] == 2.116101


def test_matching_statistics_are_accepted(stats_dir):
    write_stats(stats_dir)
    s = adapters.MatchaAdapter().mel_stats(cfg())
    assert s["mel_mean"] == -4.1 and s["mel_std"] == 1.9


def test_statistics_from_a_different_mel_band_are_refused(stats_dir):
    """The one that ties this adapter to the open mel-band question.

    If the project's fmax moves to 8 kHz, every stats file computed at sr/2
    describes a different quantity and has to be recomputed.
    """
    p = F.mel_params(SR) | {"fmax": 8000.0}
    write_stats(stats_dir, mel_params=p)
    with pytest.raises(SystemExit, match="different mel analysis"):
        adapters.MatchaAdapter().mel_stats(cfg())


def test_statistics_from_a_different_hop_are_refused(stats_dir):
    write_stats(stats_dir, mel_params=F.mel_params(SR) | {"hop_length": 300})
    with pytest.raises(SystemExit, match="different mel analysis"):
        adapters.MatchaAdapter().mel_stats(cfg())


def test_a_zero_standard_deviation_is_refused(stats_dir):
    write_stats(stats_dir, mel_std=0.0)
    with pytest.raises(SystemExit, match="cannot be"):
        adapters.MatchaAdapter().mel_stats(cfg())


def test_non_numeric_statistics_are_refused(stats_dir):
    write_stats(stats_dir, mel_mean="about minus four")
    with pytest.raises(SystemExit, match="numeric"):
        adapters.MatchaAdapter().mel_stats(cfg())


# --- initialising from upstream weights -------------------------------------

def test_a_bare_label_is_not_a_checkpoint():
    """r03.yaml ships init_from: matcha_ljspeech, which names nothing on disk."""
    with pytest.raises(SystemExit, match="label"):
        adapters.MatchaAdapter().assert_init_is_comparable(
            cfg(init_from="matcha_ljspeech"))


def test_no_init_from_is_allowed_and_means_from_scratch():
    assert adapters.MatchaAdapter().assert_init_is_comparable(cfg()) is None


def test_weights_from_a_different_mel_band_are_refused(tmp_path):
    ck = tmp_path / "matcha_ljspeech.ckpt"
    ck.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="different representation"):
        adapters.MatchaAdapter().assert_init_is_comparable(
            cfg(init_from=str(ck)))


def test_weights_are_allowed_when_the_mel_band_does_agree(tmp_path):
    """At 16 kHz this project's bank reaches 8 kHz, which is Matcha's.

    Not a recommendation for r03, which runs at 22.05 kHz. It tests that the
    comparison is on the number and not on the file name.
    """
    ck = tmp_path / "matcha.ckpt"
    ck.write_text("", encoding="utf-8")
    got = adapters.MatchaAdapter().assert_init_is_comparable(
        cfg(sample_rate=16_000, init_from=str(ck)))
    assert got == ck
    assert F.mel_params(16_000)["fmax"] == adapters.MATCHA_LJSPEECH["fmax"]


# --- the config shapes upstream actually requires ---------------------------

def test_encoder_and_cfm_are_read_by_attribute():
    e = adapters.attr_tree(adapters.MATCHA_ENCODER)
    assert e.encoder_type == "RoPE Encoder"
    assert e.encoder_params.n_channels == 192
    assert e.encoder_params.filter_channels == 768
    assert e.duration_predictor_params.filter_channels_dp == 256
    c = adapters.attr_tree(adapters.MATCHA_CFM)
    assert c.solver == "euler" and c.sigma_min == 1e-4


def test_the_decoder_config_stays_a_mapping():
    """CFM.__init__ does Decoder(..., **decoder_params). A namespace fails there."""
    assert isinstance(adapters.MATCHA_DECODER, dict)
    assert dict(**adapters.MATCHA_DECODER)["act_fn"] == "snakebeta"


def test_the_hyperparameters_are_upstreams_and_not_ours():
    """Pinned against matcha-tts 0.0.7.2's shipped configs.

    Decoder's own code defaults are num_heads=4 and act_fn="snake"; the shipped
    config says 2 and "snakebeta", and published Matcha is the config.
    """
    assert adapters.MATCHA_DECODER["num_heads"] == 2
    assert adapters.MATCHA_DECODER["act_fn"] == "snakebeta"
    assert adapters.MATCHA_DECODER["channels"] == [256, 256]
    assert adapters.MATCHA_DECODER["num_mid_blocks"] == 2
    ep = adapters.MATCHA_ENCODER["encoder_params"]
    assert (ep["n_layers"], ep["n_heads"], ep["kernel_size"]) == (6, 2, 3)
    assert ep["prenet"] is True and ep["p_dropout"] == 0.1


# --- the loss is upstream's unweighted sum ----------------------------------

class StubMatcha:
    """Stands in for MatchaTTS. Upstream's training_step sums the three losses."""

    def __init__(self) -> None:
        self.seen = None

    def get_losses(self, batch):
        self.seen = batch
        return {"dur_loss": 1.5, "prior_loss": 2.25, "diff_loss": 0.25}


def test_the_loss_is_the_unweighted_sum_of_the_three():
    m = StubMatcha()
    assert adapters.MatchaAdapter().loss(m, {"x": 1}) == 4.0


def test_the_batch_reaches_get_losses_unchanged():
    """prepare() is a no-op for Matcha: there is no format_batch_on_device."""
    m = StubMatcha()
    a = adapters.MatchaAdapter()
    t = {"x": 1, "y": 2}
    assert a.prepare(m, t) is t
    a.loss(m, t)
    assert m.seen is t


def test_matcha_is_not_adversarial():
    assert adapters.MatchaAdapter.n_optimizers == 1
    assert adapters.MatchaAdapter.needs == ("mel",)


def test_the_registry_no_longer_raises_for_matcha():
    a = adapters.for_config(cfg())
    assert isinstance(a, adapters.MatchaAdapter)


# --- collate, which needs the audio stack -----------------------------------

torch = pytest.importorskip("torch", reason="collate needs torch")
librosa = pytest.importorskip("librosa", reason="collate needs librosa")
sf = pytest.importorskip("soundfile", reason="collate needs soundfile")


def test_collate_normalises_before_padding_and_goes_channels_first(
        tmp_path, monkeypatch, stats_dir):
    import numpy as np
    from src.train import batching
    from src.train.text import TextEncoder

    root = tmp_path / "interim" / "hindi" / "w"
    root.mkdir(parents=True)
    rows = ["id\twav22\twav16\tseconds\ttext"]
    for i in range(6):
        secs = 1.0 + i * 0.5
        t = np.arange(int(secs * SR)) / SR
        y = (0.3 * np.sin(2 * np.pi * 150 * t)).astype("float32")
        sf.write(root / f"{i}.wav", y, SR)
        rows.append(f"u{i}\tw/{i}.wav\tw/{i}.wav\t{secs}\tकमल नगर")
    m = tmp_path / "train.tsv"
    m.write_text("\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(adapters, "HERE", tmp_path)
    monkeypatch.setattr(adapters, "INTERIM", tmp_path / "interim")
    write_stats(stats_dir)

    utts = batching.load_manifest(m, SR)
    out = adapters.MatchaAdapter().collate(
        utts, TextEncoder.for_config("hindi", "phoneme"), cfg())

    assert set(out) == {"x", "x_lengths", "y", "y_lengths", "spks", "durations"}
    assert out["spks"] is None and out["durations"] is None
    b, feats, frames = out["y"].shape
    assert b == len(utts) and feats == 80
    assert frames == int(out["y_lengths"].max())
    # Normalised, so values sit near zero rather than near a log-mel of -5.
    assert abs(float(out["y"].mean())) < 6.0
    # The pad is the corpus mean in normalised space, which is exactly zero.
    shortest = int(out["y_lengths"].argmin())
    pad = out["y"][shortest, :, int(out["y_lengths"][shortest]):]
    if pad.numel():
        assert float(pad.abs().max()) == 0.0
