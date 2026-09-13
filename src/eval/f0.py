"""F0 RMSE and voiced/unvoiced error.

Prosody metrics for the evaluation matrix. Two numbers, deliberately kept
separate:

- **F0 RMSE**, computed in log-F0 over frames both systems call voiced.
  Log scale because pitch is perceived roughly logarithmically, so a 20 Hz
  error at 100 Hz and at 300 Hz are not the same error, and a linear RMSE
  would let a low-pitched speaker dominate the average.

- **V/UV error rate**, the fraction of frames where the two disagree about
  whether there is pitch at all.

They must not be combined. A system that gets voicing right and pitch wrong has
a different problem from one that does the reverse, and a single blended figure
hides which.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reasonable bounds for adult speech. Narrower than librosa's defaults, which
# otherwise produce octave errors on creaky phonation.
FMIN = 60.0
FMAX = 500.0


@dataclass(frozen=True)
class F0Result:
    log_f0_rmse: float       # in log Hz, over commonly-voiced frames
    f0_rmse_hz: float        # linear, reported alongside for readability
    vuv_error_rate: float    # fraction of frames disagreeing on voicing
    n_common_voiced: int
    n_frames: int


def extract_f0(
    wav: np.ndarray,
    sr: int,
    hop_length: int = 256,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0_hz, voiced_flag). Unvoiced frames carry NaN in f0."""
    import librosa

    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)
    f0, voiced_flag, _ = librosa.pyin(
        y=wav.astype(np.float64),
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
    )
    return f0, np.nan_to_num(voiced_flag, nan=False).astype(bool)


def compare(
    ref_f0: np.ndarray,
    ref_voiced: np.ndarray,
    syn_f0: np.ndarray,
    syn_voiced: np.ndarray,
) -> F0Result:
    """Compare two already-extracted F0 tracks of the same length.

    Callers align tracks first (the DTW path from the MCD stage is reused for
    this in the pipeline). Truncating to the shorter of the two, which is the
    tempting shortcut, silently discards the tail of the longer utterance and
    biases the result toward whichever system predicts shorter durations.
    """
    n = min(len(ref_f0), len(syn_f0))
    if n == 0:
        raise ValueError("empty F0 track")
    if len(ref_f0) != len(syn_f0):
        raise ValueError(
            f"tracks differ in length ({len(ref_f0)} vs {len(syn_f0)}); "
            "align them before comparing rather than truncating here"
        )

    both_voiced = ref_voiced & syn_voiced
    n_common = int(np.sum(both_voiced))

    if n_common == 0:
        log_rmse = float("nan")
        lin_rmse = float("nan")
    else:
        r = ref_f0[both_voiced]
        s = syn_f0[both_voiced]
        ok = np.isfinite(r) & np.isfinite(s) & (r > 0) & (s > 0)
        if not np.any(ok):
            log_rmse = float("nan")
            lin_rmse = float("nan")
        else:
            log_rmse = float(np.sqrt(np.mean((np.log(r[ok]) - np.log(s[ok])) ** 2)))
            lin_rmse = float(np.sqrt(np.mean((r[ok] - s[ok]) ** 2)))

    vuv = float(np.mean(ref_voiced != syn_voiced))

    return F0Result(
        log_f0_rmse=log_rmse,
        f0_rmse_hz=lin_rmse,
        vuv_error_rate=vuv,
        n_common_voiced=n_common,
        n_frames=n,
    )


def align_to_path(track: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Resample a per-frame track onto a DTW path's frame indices."""
    idx = np.clip(index, 0, len(track) - 1)
    return track[idx]
