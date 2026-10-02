"""Moving the tensors a run holds that the model does not own.

The loop moves the model and stops, which is right for weights and wrong for
coqui's loss modules: `TorchSTFT` builds its Hann window at construction, on
the CPU, and nothing else moves it. The first spectral loss on the GPU then
raises "stft input and window must be on the same device", which is exactly
how r06's dry run failed on 2 October. VITS never hit it because its losses
take no STFT of the waveform; HiFi-GAN's L1 spectrogram term does, by default.

`module.to(device)` moves parameters and registered buffers and leaves a plain
tensor attribute alone, and coqui stores that window differently across
versions, so both shapes are covered here. The device used is "meta", which
needs no GPU and still proves the move.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="moving tensors needs torch")

import torch.nn as nn                                       # noqa: E402

from src.train.devices import move_criterion, move_module    # noqa: E402


class StftLike(nn.Module):
    def __init__(self, as_param: bool) -> None:
        super().__init__()
        if as_param:
            self.window = nn.Parameter(torch.hann_window(16), requires_grad=False)
        else:
            self.window = torch.hann_window(16)      # not a buffer; .to() ignores it
        self.lin = nn.Linear(4, 4)


class SpecLossLike(nn.Module):
    def __init__(self, as_param: bool) -> None:
        super().__init__()
        self.stft = StftLike(as_param)


class GenLossLike(nn.Module):
    def __init__(self, as_param: bool) -> None:
        super().__init__()
        self.l1_spec_loss = SpecLossLike(as_param)
        self.stray = torch.ones(3)


class FakeAdapter:
    pass


@pytest.mark.parametrize("as_param", [True, False])
def test_the_window_two_levels_down_reaches_the_device(as_param):
    a = FakeAdapter()
    a._criterion = [nn.Identity(), GenLossLike(as_param)]
    move_criterion(a, "meta")
    win = a._criterion[1].l1_spec_loss.stft.window
    assert win.device.type == "meta", "the window stayed behind"


@pytest.mark.parametrize("as_param", [True, False])
def test_ordinary_parameters_move_too(as_param):
    a = FakeAdapter()
    a._criterion = [GenLossLike(as_param)]
    move_criterion(a, "meta")
    assert a._criterion[0].l1_spec_loss.stft.lin.weight.device.type == "meta"


def test_a_stray_tensor_attribute_moves():
    """The case module.to() cannot handle, which is why this exists."""
    a = FakeAdapter()
    a._criterion = [GenLossLike(False)]
    move_criterion(a, "meta")
    assert a._criterion[0].stray.device.type == "meta"


def test_it_reports_what_it_moved():
    a = FakeAdapter()
    a._criterion = [GenLossLike(True)]
    moved = move_criterion(a, "meta")
    assert any("StftLike" in m for m in moved)
    assert any(m.endswith(".stray") for m in moved)


def test_no_criterion_is_not_an_error():
    """FastSpeech 2 and the toy adapter may hold nothing at all."""
    assert move_criterion(FakeAdapter(), "meta") == []
    a = FakeAdapter(); a._criterion = None
    assert move_criterion(a, "meta") == []


def test_a_bare_module_and_a_tuple_both_work():
    m = GenLossLike(False)
    assert move_module(m, "meta")
    assert m.l1_spec_loss.stft.window.device.type == "meta"
    t = (GenLossLike(False), GenLossLike(True))
    assert move_module(t, "meta")
    assert all(x.l1_spec_loss.stft.window.device.type == "meta" for x in t)


def test_a_cycle_does_not_hang():
    class Loop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.t = torch.ones(2)
    a, b = Loop(), Loop()
    a.other, b.other = b, a
    move_module(a, "meta")
    assert a.t.device.type == "meta" and b.t.device.type == "meta"
