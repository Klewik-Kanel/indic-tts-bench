"""Known-answer tests for the objective metrics.

These are not smoke tests. Each one has an answer derivable on paper, so a
regression shows up as a wrong number rather than as an exception. Metrics that
are only ever run on real audio tend to be wrong in ways nobody notices until a
reviewer asks, which is the failure mode this file exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.f0 import compare, extract_f0
from src.eval.mcd import MCD_CONSTANT, dtw_path, mcd, mel_cepstrum

SR = 22050


def tone(freq: float, seconds: float = 0.6, sr: int = SR, amp: float = 0.5,
         harmonics: int = 6) -> np.ndarray:
    """A harmonic complex: periodic like voiced speech, so pYIN can track it."""
    t = np.arange(int(seconds * sr)) / sr
    sig = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        sig += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
    sig /= np.max(np.abs(sig))
    return (amp * sig).astype(np.float64)


def noise(seconds: float = 0.6, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, int(seconds * sr))


# --- MCD -------------------------------------------------------------------

def test_mcd_of_a_signal_against_itself_is_zero() -> None:
    x = tone(200)
    r = mcd(x, x, SR)
    assert r.mcd_db == pytest.approx(0.0, abs=1e-9)
    assert r.length_ratio == pytest.approx(1.0)


def test_mcd_ignores_amplitude_because_c0_is_dropped() -> None:
    """The defining property of a correct MCD implementation.

    Scaling a waveform multiplies its power spectrum by a constant, which adds
    a constant offset in dB across every mel bin. A constant offset lands
    entirely in the zeroth DCT coefficient, so dropping c0 makes the measure
    blind to loudness. If this test fails, c0 is leaking in and the metric is
    partly measuring gain.
    """
    x = tone(200)
    r = mcd(x, 0.25 * x, SR)
    assert r.mcd_db == pytest.approx(0.0, abs=1e-6)


def test_mcd_separates_spectra_that_differ() -> None:
    voiced = tone(200)
    hiss = noise()
    same = mcd(voiced, voiced, SR).mcd_db
    different = mcd(voiced, hiss, SR).mcd_db
    assert different > same
    assert different > 1.0, "a tone against noise should be clearly distant"


def test_dtw_absorbs_a_duration_difference() -> None:
    """A longer utterance of the same sound should not be penalised as spectral.

    This is why the metric aligns rather than comparing frame k to frame k. A
    non-autoregressive model with a duration predictor would otherwise be
    scored on its timing under the name of acoustic fidelity.
    """
    short = tone(200, seconds=0.6)
    long = tone(200, seconds=0.9)
    r = mcd(short, long, SR)
    assert r.length_ratio > 1.4
    # Steady tones: after alignment the spectral distance should stay small.
    assert r.mcd_db < mcd(short, noise(seconds=0.9), SR).mcd_db


def test_dtw_path_anchors_both_endpoints() -> None:
    a = np.arange(10, dtype=float).reshape(10, 1)
    b = np.arange(14, dtype=float).reshape(14, 1)
    path = dtw_path(a, b)
    assert path[0] == (0, 0)
    assert path[-1] == (9, 13)
    # Monotone and contiguous: no jumps larger than one frame on either axis.
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert 0 <= i1 - i0 <= 1
        assert 0 <= j1 - j0 <= 1


def test_mcd_constant_matches_the_published_definition() -> None:
    assert MCD_CONSTANT == pytest.approx(6.141851, abs=1e-5)


def test_mel_cepstrum_shape() -> None:
    x = tone(200, seconds=1.0)
    c = mel_cepstrum(x, SR, n_mfcc=25, hop_length=256)
    assert c.shape[1] == 25
    assert c.shape[0] == pytest.approx(len(x) / 256, abs=2)


# --- F0 --------------------------------------------------------------------

def test_extract_f0_recovers_a_known_pitch() -> None:
    f0, voiced = extract_f0(tone(200.0), SR)
    assert voiced.mean() > 0.7, "a steady harmonic complex should read as voiced"
    est = np.nanmedian(f0[voiced])
    assert est == pytest.approx(200.0, rel=0.03)


def test_f0_rmse_of_a_track_against_itself_is_zero() -> None:
    f0, v = extract_f0(tone(200.0), SR)
    r = compare(f0, v, f0, v)
    assert r.log_f0_rmse == pytest.approx(0.0, abs=1e-12)
    assert r.vuv_error_rate == pytest.approx(0.0)


def test_log_f0_rmse_equals_the_log_ratio_for_a_constant_offset() -> None:
    """A 10% pitch error must read as ln(1.1) regardless of the base pitch.

    That scale invariance is the whole reason the metric is computed in log
    space: the same perceptual error on a low and a high voice must produce the
    same number, which a linear RMSE would not.
    """
    n = 100
    voiced = np.ones(n, dtype=bool)
    for base in (100.0, 300.0):
        ref = np.full(n, base)
        syn = np.full(n, base * 1.1)
        r = compare(ref, voiced, syn, voiced)
        assert r.log_f0_rmse == pytest.approx(np.log(1.1), abs=1e-12)
        assert r.f0_rmse_hz == pytest.approx(base * 0.1, rel=1e-9)


def test_vuv_error_counts_voicing_disagreements() -> None:
    n = 10
    ref_f0 = np.full(n, 200.0)
    syn_f0 = np.full(n, 200.0)
    ref_v = np.ones(n, dtype=bool)
    syn_v = np.ones(n, dtype=bool)
    syn_v[:3] = False                      # three frames disagree
    r = compare(ref_f0, ref_v, syn_f0, syn_v)
    assert r.vuv_error_rate == pytest.approx(0.3)
    assert r.n_common_voiced == 7


def test_compare_refuses_to_truncate_mismatched_tracks() -> None:
    """Silently truncating would bias results toward shorter predictions."""
    with pytest.raises(ValueError, match="align them before comparing"):
        compare(
            np.full(10, 200.0), np.ones(10, dtype=bool),
            np.full(8, 200.0), np.ones(8, dtype=bool),
        )


def test_no_common_voiced_frames_yields_nan_not_zero() -> None:
    """An empty comparison must not look like a perfect score."""
    n = 10
    ref_v = np.zeros(n, dtype=bool); ref_v[:5] = True
    syn_v = np.zeros(n, dtype=bool); syn_v[5:] = True
    r = compare(np.full(n, 200.0), ref_v, np.full(n, 200.0), syn_v)
    assert np.isnan(r.log_f0_rmse)
    assert r.n_common_voiced == 0
