"""The HiFi-GAN adapter, fine-tuned on this project's own mels.

Reinstated on 2 October: no published vocoder shares this project's mel
analysis, and fitting the vocoder to our mels removes all three differences at
once while keeping every trained acoustic parameter.

Four things here are silent when wrong, and each has a test:

  - coqui's `GAN.get_optimizer` docstring says generator first; its code
    returns discriminator first, and `train_step` indexes the code's order
  - `GAN.train_disc` defaults to False and is set by coqui's own trainer, which
    we do not use, so left alone the discriminator never trains and the GAN is
    quietly a generator with a reconstruction loss
  - a vocoder trained on padded batches learns to produce the padding, so
    collate crops fixed-length segments rather than padding to the longest item
  - the crop has to be reproducible from (seed, step) rather than drawn from a
    global RNG, because collate runs on the prefetch thread, ahead of the loop,
    and a draw there lands in a checkpoint at the wrong position

The crop and configuration tests need nothing installed. The tensor tests need
torch but not coqui: the model is a stub with the same module names.
"""

from __future__ import annotations

import math

import pytest

from src.train import adapters
from src.train import features as F

SR = 22_050


def cfg(**over) -> dict:
    c = {"run_id": "r06", "architecture": "hifigan", "language": "hindi",
         "input_repr": "none", "data": "9h", "sample_rate": SR,
         "batch_frames": 12_000, "max_steps": 2, "lr": 2e-4,
         "warmup_steps": 1, "grad_clip": 1.0, "precision": "fp32",
         "seed": 1234, "init_from": ""}
    c.update(over)
    return c


# --- the segment, and what it is derived from -------------------------------

def test_the_segment_is_coquis_seq_len_in_frames():
    """8192 samples at hop 256 is 32 frames, which is one hop per frame."""
    assert adapters.HIFIGAN_SEGMENT_SAMPLES == 8_192
    assert adapters.HiFiGanAdapter().segment_frames(cfg()) == 8_192 // F.HOP
    assert adapters.HiFiGanAdapter().segment_frames(cfg()) == 32


def test_the_mel_pad_value_is_the_floor_silence_actually_produces():
    """features.compute clamps the mel at 1e-5, so silence is log(1e-5).

    Padding with 0.0 instead would be a loud frame, and the generator would
    learn to put energy where the audio has none.
    """
    assert adapters.MEL_FLOOR == math.log(1e-5)
    assert adapters.MEL_FLOOR < -11


# --- the crop is reproducible, and varies ------------------------------------

def test_the_crop_is_the_same_on_a_resumed_run():
    a = adapters.HiFiGanAdapter()
    first = [a.crop_offset(cfg(), s, "u001", 100) for s in range(8)]
    again = [a.crop_offset(cfg(), s, "u001", 100) for s in range(8)]
    assert first == again


def test_the_crop_moves_with_the_step():
    a = adapters.HiFiGanAdapter()
    offs = {a.crop_offset(cfg(), s, "u001", 100) for s in range(20)}
    assert len(offs) > 10, "the crop barely moves across steps"


def test_the_crop_differs_between_utterances_in_one_step():
    a = adapters.HiFiGanAdapter()
    offs = {a.crop_offset(cfg(), 5, f"u{i:03d}", 100) for i in range(20)}
    assert len(offs) > 10


def test_the_crop_moves_with_the_seed():
    a = adapters.HiFiGanAdapter()
    assert (a.crop_offset(cfg(seed=1), 3, "u001", 100)
            != a.crop_offset(cfg(seed=2), 3, "u001", 100))


def test_the_crop_stays_in_range_including_the_degenerate_case():
    a = adapters.HiFiGanAdapter()
    for limit in (0, 1, 7, 100, 5_000):
        for s in range(30):
            off = a.crop_offset(cfg(), s, "u042", limit)
            assert 0 <= off <= limit
    assert a.crop_offset(cfg(), 1, "u042", 0) == 0
    assert a.crop_offset(cfg(), 1, "u042", -3) == 0


def test_the_adapter_asks_for_the_step():
    """The runner passes it only to an adapter that declares it, so the other
    three keep the collate signature their tests were written against."""
    assert adapters.HiFiGanAdapter.wants_step is True
    for other in (adapters.FastSpeech2Adapter, adapters.VitsAdapter,
                  adapters.MatchaAdapter, adapters.ToyAdapter):
        assert getattr(other, "wants_step", False) is False


def test_it_is_adversarial_and_needs_audio_as_well_as_mel():
    assert adapters.HiFiGanAdapter.n_optimizers == 2
    assert adapters.HiFiGanAdapter.needs == ("mel", "wav")
    assert adapters.ADAPTERS["hifigan"] is adapters.HiFiGanAdapter
    assert isinstance(adapters.for_config(cfg()), adapters.HiFiGanAdapter)


# --- the tensor side: a stub with coqui's module names ----------------------

torch = pytest.importorskip("torch", reason="the tensor tests need torch")


class StubGan:
    """Stands in for coqui's GAN. Same module names, same train_step contract."""

    def __init__(self) -> None:
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model_g = nn.Conv1d(80, 1, 1)
                self.model_d = nn.Conv1d(1, 1, 1)

        self.net = M()
        self.calls: list[int] = []
        self.train_disc = False

    def named_parameters(self):
        return self.net.named_parameters()

    def parameters(self):
        return self.net.parameters()

    def train_step(self, batch, criterion, optimizer_idx):
        self.calls.append(optimizer_idx)
        y = self.net.model_g(batch["input"])
        scalar = y.pow(2).mean() * (1.0 if optimizer_idx else 2.0)
        return {}, {"loss": scalar, "loss_gen": scalar}


def test_the_discriminator_comes_first_in_both_lists():
    """coqui's get_optimizer docstring says the opposite of what it returns.

    train_step's optimizer_idx == 0 is the discriminator branch and it caches
    outputs the idx-1 generator branch reuses, so index 0 must own the
    discriminator's parameters.
    """
    a = adapters.HiFiGanAdapter()
    m = StubGan()
    disc, gen = a._split(m)
    d_names = {n for n, _ in m.named_parameters() if n.startswith("model_d.")}
    assert len(disc) == len(d_names)
    groups = a.param_groups(m)
    assert len(groups) == 2
    assert [len(g) for g in groups] == [len(disc), len(gen)]
    opts = a.optimizers(m, cfg())
    assert len(opts) == 2
    first = {id(p) for g in opts[0].param_groups for p in g["params"]}
    assert first == {id(p) for p in disc}


def test_a_renamed_discriminator_stops_the_run():
    import torch.nn as nn

    class NoDisc:
        def named_parameters(self):
            return nn.Conv1d(80, 1, 1).named_parameters()

    with pytest.raises(SystemExit, match="model_d"):
        adapters.HiFiGanAdapter()._split(NoDisc())


def test_the_loss_passes_the_index_through_and_returns_the_scalar():
    a = adapters.HiFiGanAdapter()
    a._criterion = ["disc", "gen"]
    m = StubGan()
    batch = {"input": torch.zeros(2, 80, 32), "waveform": torch.zeros(2, 1, 8192)}
    d = a.loss(m, batch, 0)
    g = a.loss(m, batch, 1)
    assert m.calls == [0, 1]
    assert d.shape == torch.Size([]) and g.shape == torch.Size([])


# --- collate: fixed segments, channels-first, no text -----------------------

class FakeUtt:
    def __init__(self, uid: str, frames: int) -> None:
        self.uid = uid
        self.frames = frames
        self.wav = f"w/{uid}.wav"
        self.text = ""
        self.seconds = frames * F.HOP / SR


def fake_features(frames_list):
    import numpy as np
    out = []
    for i, n in enumerate(frames_list):
        rng = np.random.default_rng(i)
        out.append({"mel": rng.normal(-5.0, 2.0, (n, 80)).astype("float32"),
                    "wav": rng.normal(0, 0.1, n * F.HOP).astype("float32")})
    return out


def collate_with(frames_list, step=0, monkeypatch=None):
    a = adapters.HiFiGanAdapter()
    feats = fake_features(frames_list)
    a._features = lambda batch, cfg_: feats
    utts = [FakeUtt(f"u{i:03d}", n) for i, n in enumerate(frames_list)]
    return a, a.collate(utts, None, cfg(), step=step)


def test_collate_returns_one_fixed_segment_per_utterance():
    a, out = collate_with([200, 150, 90, 400])
    seg = a.segment_frames(cfg())
    assert set(out) == {"input", "waveform"}, "a vocoder batch carries no text"
    assert out["input"].shape == (4, 80, seg)
    assert out["waveform"].shape == (4, 1, seg * F.HOP)


def test_the_mel_is_channels_first_and_the_audio_matches_it_hop_for_hop():
    a, out = collate_with([300, 300])
    seg = a.segment_frames(cfg())
    assert out["input"].shape[1] == 80
    assert out["waveform"].shape[2] == out["input"].shape[2] * F.HOP == seg * F.HOP


def test_the_same_step_gives_the_same_segment_and_a_later_step_does_not():
    _, a0 = collate_with([400], step=0)
    _, b0 = collate_with([400], step=0)
    _, a9 = collate_with([400], step=9)
    assert torch.equal(a0["input"], b0["input"])
    assert not torch.equal(a0["input"], a9["input"])


def test_a_short_utterance_is_padded_with_silence_not_with_zeros():
    """Shorter than one segment: the audio pads with zeros, and the mel pads
    with the floor that silence actually produces."""
    a, out = collate_with([10])
    seg = a.segment_frames(cfg())
    assert out["input"].shape == (1, 80, seg)
    tail = out["input"][0, :, 10:]
    assert torch.allclose(tail, torch.full_like(tail, adapters.MEL_FLOOR))
    assert float(out["waveform"][0, 0, 10 * F.HOP:].abs().max()) == 0.0


def test_the_crop_never_reads_past_the_end_of_the_audio():
    """librosa's centred STFT reports a frame for the final partial hop, so
    the mel can claim a frame the waveform cannot fill."""
    import numpy as np
    a = adapters.HiFiGanAdapter()
    seg = a.segment_frames(cfg())
    n = seg + 4
    # A mel one frame longer than the audio supports, as centring produces.
    a._features = lambda batch, cfg_: [
        {"mel": np.zeros((n + 1, 80), dtype="float32"),
         "wav": np.zeros(n * F.HOP, dtype="float32")}]
    out = a.collate([FakeUtt("u000", n)], None, cfg(), step=3)
    assert out["waveform"].shape == (1, 1, seg * F.HOP)
