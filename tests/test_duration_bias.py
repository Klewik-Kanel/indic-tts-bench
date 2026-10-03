#!/usr/bin/env python3
"""Excess duration against deletion-site count: the measure that answers the question.

MCD cannot resolve the phonemic-versus-graphemic contrast at this quality
level. Measured on 3 October, r02 and r05 both sit about 55% of the way from a
faithful reconstruction to unrelated audio and differ by 0.89% of chance, which
is 0.31 standard errors. When both systems are that far from the reference, the
measure is dominated by how far they both are.

A schwa takes time. The rule deletes one in 32.0% of Hindi words, the phonemic
arm is handed that deletion and the graphemic arm must infer it, so excess
duration regressed on site count reads out in milliseconds per missed site.

What these tests hold is that the fit says "I do not know" whenever it should,
because the failure mode of a regression is a confident slope from data that
cannot support one.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.duration_bias import Fit, fit, syn_seconds_from_mel


def synth(slope_s, n=80, noise=0.02, tempo=0.0, seed=0, max_sites=6):
    rng = np.random.default_rng(seed)
    sites = rng.integers(0, max_sites, n).astype("float64")
    excess = tempo + slope_s * sites + rng.normal(0, noise, n)
    return list(sites), list(excess)


# -- it recovers the thing it is supposed to measure -------------------------

def test_a_model_that_keeps_every_schwa_shows_the_vowel_duration():
    f = fit(*synth(0.070))
    assert f.slope_ms == pytest.approx(70.0, abs=6.0)
    assert f.slope_t > 5


def test_a_model_that_deletes_correctly_shows_no_slope():
    f = fit(*synth(0.0))
    assert abs(f.slope_ms) < 6.0
    assert abs(f.slope_t) < 2


def test_a_constant_tempo_offset_does_not_look_like_kept_schwas():
    """The reason the intercept is fitted rather than assumed zero: a model
    that simply speaks 8% slow would otherwise read as one keeping schwas."""
    f = fit(*synth(0.0, tempo=0.25))
    assert abs(f.slope_ms) < 6.0
    assert f.intercept_s == pytest.approx(0.25, abs=0.02)


def test_the_two_arms_are_distinguishable_at_this_noise_level():
    """The whole point. MCD gave 0.31 sigma for this contrast."""
    keeps = fit(*synth(0.070, seed=1))
    deletes = fit(*synth(0.0, seed=2))
    gap = keeps.slope_ms - deletes.slope_ms
    se = (keeps.slope_se ** 2 + deletes.slope_se ** 2) ** 0.5 * 1000.0
    assert gap / se > 5, (gap, se)


# -- it refuses when it cannot know -----------------------------------------

def test_too_few_points_raises_rather_than_fitting_two():
    with pytest.raises(ValueError, match="at least 3"):
        fit([1.0, 2.0], [0.1, 0.2])


def test_no_variation_in_sites_raises_rather_than_dividing_by_zero():
    """If every utterance has the same site count, no slope is identifiable.
    Returning one anyway would be inventing a finding."""
    with pytest.raises(ValueError, match="no slope is identifiable"):
        fit([2.0] * 20, list(np.linspace(0, 1, 20)))


def test_non_finite_points_are_dropped_not_propagated():
    sites, excess = synth(0.070, n=40)
    sites += [3.0, 4.0]
    excess += [float("nan"), float("inf")]
    f = fit(sites, excess)
    assert f.n == 40
    assert np.isfinite(f.slope_ms)


# -- the standard error is the part that stops a null being written up ------

def test_noisier_data_widens_the_standard_error():
    tight = fit(*synth(0.070, noise=0.005, seed=3))
    loose = fit(*synth(0.070, noise=0.080, seed=3))
    assert loose.slope_se > tight.slope_se * 3


def test_more_utterances_narrow_the_standard_error():
    few = fit(*synth(0.070, n=20, seed=4))
    many = fit(*synth(0.070, n=320, seed=4))
    assert many.slope_se < few.slope_se / 2


def test_t_is_the_slope_over_its_error():
    f = fit(*synth(0.070))
    assert f.slope_t == pytest.approx(f.slope_s / f.slope_se)


# -- the no-vocoder claim ---------------------------------------------------

def test_a_mel_frame_count_is_a_duration():
    """This is why FastSpeech 2 can be measured before r06 finishes.

    Stated as the relationship rather than a rounded constant: frames times hop
    is samples, and samples over the rate is seconds."""
    frames, hop, sr = 184, 256, 22050
    assert syn_seconds_from_mel(frames, hop, sr) == \
        pytest.approx(frames * hop / sr, rel=1e-12)
    assert syn_seconds_from_mel(0, hop, sr) == 0.0
    # and it scales linearly, which is what makes the regression meaningful
    assert syn_seconds_from_mel(2 * frames, hop, sr) == \
        pytest.approx(2 * syn_seconds_from_mel(frames, hop, sr))


def test_the_fit_is_a_plain_dataclass_so_it_serialises():
    f = fit(*synth(0.070))
    assert isinstance(f, Fit)
    import dataclasses
    d = dataclasses.asdict(f)
    assert set(d) >= {"slope_s", "intercept_s", "n", "r", "slope_se"}
