# Results log

Append only. Every number carries the commit that produced it. Nothing here is
a paper result until it is marked as such.

## 2026-09-13 — front end, provisional

Command: `python -m src.g2p.score stresstests/hindi_schwa_set.tsv --include-unvalidated`

    49/49 = 100.0%
    applies 18/18, blocked 5/5, cluster 4/4, final 6/6, loan 6/6, mono 4/4, nasal 6/6

**Not a result.** The rule implementation and the gold forms were written by
the same author in the same sitting, so this figure measures self-consistency,
not front-end quality. The scorer refuses to report a validated number because
zero of the 49 items have been checked by an independent speaker. The real
figure comes after that pass, scored without `--include-unvalidated`.

Unit tests: 26 passing, cloud container and device.

## 2026-09-13 — objective metrics, known-answer verification

MCD and F0 implemented and verified against answers derivable on paper rather
than against real audio. 13 metric tests, 39 in total, all passing.

The two that matter:

- **MCD is amplitude invariant.** Scaling a waveform by 0.25 changes MCD by
  less than 1e-6 dB. Scaling multiplies the power spectrum by a constant, which
  is a constant dB offset across mel bins, which lands entirely in the zeroth
  DCT coefficient. Dropping c0 therefore makes the measure blind to gain. If
  this ever fails, MCD is partly measuring loudness.
- **log-F0 RMSE is scale invariant.** A 10% pitch error reads as ln(1.1) at
  both 100 Hz and 300 Hz, exact to 1e-12. A linear RMSE would report 10 Hz and
  30 Hz for the same perceptual error and would let a low-pitched speaker
  dominate any average.

Two guards against silent bias, both tested: `compare` raises rather than
truncating mismatched F0 tracks, since truncation would favour systems that
predict shorter durations; and a comparison with no commonly-voiced frames
returns NaN rather than 0.0, so an empty comparison cannot look like a perfect
score.
