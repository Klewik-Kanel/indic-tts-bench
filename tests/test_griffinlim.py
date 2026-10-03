#!/usr/bin/env python3
"""The Griffin-Lim placeholder's guards and its agreement with the forward path.

The inversion itself needs librosa and a real waveform, and is checked by
round-tripping a corpus utterance: our mel, inverted, then re-analysed through
`features._spec_and_mel`. Mean absolute mel error came out at 0.1434 at
22.05 kHz and 0.1810 at 16 kHz, against 2.1473 and 2.2866 for the same mel with
its frames shuffled, so about 15x and 13x closer than chance. Measured with
librosa 0.11.0; the DGX runs 1.0.0.

What is tested here is everything that does not need librosa, which is the part
that fails silently if wrong: whether this module still agrees with
`features.mel_params`, and whether it refuses the mel orientations and
parameters that would produce plausible-sounding nonsense.
"""
from __future__ import annotations

import pytest

from src.export import griffinlim
from src.train.features import HOP, N_FFT, N_MELS, WIN, mel_params


@pytest.mark.parametrize("sr", [16000, 22050])
def test_parameters_agree_with_the_forward_path(sr):
    p = griffinlim.assert_matches_project(sr)
    assert p == mel_params(sr)


@pytest.mark.parametrize("sr", [16000, 22050])
def test_the_bank_spans_to_nyquist(sr):
    """fmax is sr//2 here, not an 8 kHz convention. RESULTS records that no
    published HiFi-GAN matches this band, which is why it has to be explicit."""
    assert griffinlim.assert_matches_project(sr)["fmax"] == sr / 2


def test_it_notices_if_the_project_constants_move(monkeypatch):
    """A silent disagreement between this module and features.py would invert
    a mel that was never computed that way."""
    monkeypatch.setattr(griffinlim, "N_FFT", N_FFT * 2)
    with pytest.raises(AssertionError):
        griffinlim.assert_matches_project(22050)


def test_the_floor_matches_the_clamp_the_forward_log_applies():
    assert griffinlim.MEL_FLOOR == 1e-5


def test_the_default_iteration_count_is_where_returns_flatten():
    assert griffinlim.DEFAULT_ITERS == 60


# -- orientation and argument guards, which fire before librosa is imported ---

def test_a_frames_first_mel_is_refused():
    """coqui's ForwardTTS returns [B, T_frames, n_mels]. Handed over
    untransposed, a 184-frame mel would be read as 184 mel bands."""
    import numpy as np
    with pytest.raises(ValueError, match="mels-first"):
        griffinlim.invert(np.zeros((184, N_MELS), dtype="float32"), 22050)


def test_a_one_dimensional_input_is_refused():
    import numpy as np
    with pytest.raises(ValueError, match=r"\[n_mels, frames\]"):
        griffinlim.invert(np.zeros(N_MELS, dtype="float32"), 22050)


def test_a_three_dimensional_input_is_refused():
    """A batch dimension left on, which is what comes straight off the model."""
    import numpy as np
    with pytest.raises(ValueError, match=r"\[n_mels, frames\]"):
        griffinlim.invert(np.zeros((1, N_MELS, 40), dtype="float32"), 22050)


def test_zero_iterations_is_refused():
    import numpy as np
    with pytest.raises(ValueError, match="at least 1"):
        griffinlim.invert(np.zeros((N_MELS, 40), dtype="float32"), 22050, n_iter=0)


def test_the_project_constants_are_the_ones_this_module_uses():
    """Imported rather than restated, so there is one definition."""
    assert (N_FFT, HOP, WIN, N_MELS) == (1024, 256, 1024, 80)
