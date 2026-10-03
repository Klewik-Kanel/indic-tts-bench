#!/usr/bin/env python3
"""Does the model delete the schwa? Measured in milliseconds, without an ASR.

The dissertation's question is whether explicit G2P still buys anything, and
the acoustic-distance metrics cannot answer it: measured on 3 October, r02 and
r05 both sit about 55% of the way from a faithful reconstruction to unrelated
audio, and the gap between them is 0.89% of chance. When both systems are that
far from the reference, MCD is dominated by how far they both are.

This measures the phenomenon instead of the distance, and the idea is simple
enough to state in one line: **a schwa takes time.**

Devanagari writes a word-final consonant with an inherent schwa, so the rule
deletes one in 32.0% of Hindi words. The phonemic arm is handed the deletion;
the graphemic arm has to infer it. If the graphemic arm fails to, its output is
longer than the reference by roughly one vowel per missed site. So regress
excess duration on the number of deletion sites the rule found:

    syn_seconds - ref_seconds  =  intercept + slope * n_deletion_sites

`slope` is seconds of excess per deletion site. A model that deletes correctly
has a slope near zero; one that keeps the schwas has a slope near the duration
of a short vowel, 50 to 90 ms. The intercept absorbs any constant tempo offset,
which is why it is fitted rather than assumed to be zero: a model that simply
speaks 5% slow would otherwise look like one that keeps schwas.

Three reasons this is worth more than its simplicity suggests.

**It needs no vocoder.** A mel frame count is a duration. FastSpeech 2 can be
measured now, while r06 is still training.

**It cannot saturate.** However rough the audio, a schwa is either there or it
is not, and its absence is a measurable number of milliseconds.

**It is falsifiable in the right direction.** If both arms come out with the
same slope, the front end is not buying deletion behaviour, and that is a real
answer rather than a null result dressed up.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fit:
    slope_s: float            # seconds of excess duration per deletion site
    intercept_s: float        # constant offset, i.e. overall tempo
    n: int
    r: float                  # Pearson correlation of sites against excess
    slope_se: float           # standard error of the slope
    sites_mean: float
    excess_mean_s: float

    @property
    def slope_ms(self) -> float:
        return self.slope_s * 1000.0

    @property
    def slope_t(self) -> float:
        """Slope over its standard error. About 2 is the usual threshold."""
        return self.slope_s / self.slope_se if self.slope_se else float("nan")


def fit(sites: list[float], excess_s: list[float]) -> Fit:
    """Least squares of excess duration on deletion-site count.

    Done by hand rather than with scipy so the standard error is explicit: the
    slope on its own is not a finding, and a slope reported without its error
    is how a null result gets written up as a positive one.
    """
    import numpy as np

    x = np.asarray(sites, dtype="float64")
    y = np.asarray(excess_s, dtype="float64")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        raise ValueError(f"need at least 3 points, got {n}")
    if np.allclose(x, x[0]):
        raise ValueError("every utterance has the same number of sites, so "
                         "no slope is identifiable")

    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)

    resid = y - (intercept + slope * x)
    dof = n - 2
    s2 = float((resid ** 2).sum() / dof) if dof > 0 else float("nan")
    slope_se = float((s2 / sxx) ** 0.5) if s2 == s2 else float("nan")

    sy = float(((y - ym) ** 2).sum())
    r = float(((x - xm) * (y - ym)).sum() / ((sxx * sy) ** 0.5)) if sy else 0.0

    return Fit(slope_s=slope, intercept_s=intercept, n=n, r=r,
               slope_se=slope_se, sites_mean=float(xm),
               excess_mean_s=float(ym))


def syn_seconds_from_mel(n_frames: int, hop_length: int, sample_rate: int) -> float:
    """A mel frame count is a duration. This is why no vocoder is needed."""
    return n_frames * hop_length / float(sample_rate)
