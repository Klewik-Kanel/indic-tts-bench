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

from src.eval import mcd as mcd_mod

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


# -- scale, which is what went wrong on 3 October ---------------------------
#
# MCD came out at 275 dB on r02. Published TTS figures are 3 to 8, and the
# cause was a double decibel conversion: librosa.feature.mfcc applies
# power_to_db, and MCD_CONSTANT converts natural-log coefficients to decibels
# again. Every test here passed throughout, because they all check relative
# behaviour and none checked magnitude. These do.

def _tone(sr, seconds, f0, harmonics=1, noise=0.0, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))
    if noise:
        y = y + noise * rng.standard_normal(len(t))
    return (0.5 * y / np.max(np.abs(y))).astype("float32")


def test_mel_cepstrum_does_not_use_librosas_db_scaled_mfcc():
    """The specific regression. librosa.feature.mfcc applies power_to_db, so
    using it means MCD_CONSTANT converts to decibels a second time."""
    import pathlib
    src = pathlib.Path(mcd_mod.__file__).read_text(encoding="utf-8")
    assert "librosa.feature.mfcc" not in src
    assert "np.log(" in src          # natural log, matching the constant


def test_mcd_orders_a_mild_degradation_below_an_unrelated_signal():
    """The ordering MCD exists to provide. A units error preserves it, which
    is why the magnitude assertions below are needed as well."""
    sr = 22050
    clean = _tone(sr, 0.6, 180.0, harmonics=6)
    mild = _tone(sr, 0.6, 180.0, harmonics=6, noise=0.02, seed=1)
    unrelated = _tone(sr, 0.6, 320.0, harmonics=3, seed=2)
    near = mcd_mod.mcd(clean, mild, sr).mcd_db
    far = mcd_mod.mcd(clean, unrelated, sr).mcd_db
    assert 0.0 < near < far


def test_an_identical_signal_scores_zero():
    sr = 22050
    y = _tone(sr, 0.5, 200.0, harmonics=4)
    assert mcd_mod.mcd(y, y, sr).mcd_db == pytest.approx(0.0, abs=1e-9)


def test_a_mild_degradation_stays_within_a_plausible_decibel_range():
    """The magnitude check. With the double conversion this came out about
    4.3x too large, which is the whole bug. A mild additive-noise degradation
    should be single or low double digits, nowhere near three figures."""
    sr = 22050
    clean = _tone(sr, 0.6, 180.0, harmonics=6)
    mild = _tone(sr, 0.6, 180.0, harmonics=6, noise=0.02, seed=1)
    got = mcd_mod.mcd(clean, mild, sr).mcd_db
    assert 0.0 < got < 60.0, got


def test_the_constant_is_the_natural_log_to_decibel_conversion():
    import numpy as np
    assert mcd_mod.MCD_CONSTANT == pytest.approx(
        10.0 * np.sqrt(2.0) / np.log(10.0))
