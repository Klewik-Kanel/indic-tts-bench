"""Mel-cepstral distortion with DTW alignment.

MCD is the acoustic-fidelity number in the evaluation matrix. It compares a
synthesised utterance against the natural recording of the same text, frame by
frame, in mel-cepstral space.

Two decisions here matter and are easy to get silently wrong:

1. **c0 is excluded.** The zeroth coefficient tracks overall energy, so leaving
   it in makes MCD partly a loudness measure. Every published MCD figure that
   is comparable to another excludes it.

2. **Alignment is DTW, not frame-by-frame.** A synthesised utterance almost
   never has the same duration as the reference, and a non-autoregressive model
   with a duration predictor will differ systematically from an end-to-end one.
   Comparing frame k to frame k would then penalise timing differences as if
   they were spectral ones, and would favour whichever architecture happens to
   predict durations closest to the reference. DTW separates the two.

The constant 10*sqrt(2)/ln(10) is conventional and makes the result comparable
with the literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MCD_CONSTANT = 10.0 * np.sqrt(2.0) / np.log(10.0)   # ~= 6.14185


@dataclass(frozen=True)
class MCDResult:
    mcd_db: float
    n_frames: int          # frames on the alignment path
    ref_frames: int
    syn_frames: int
    length_ratio: float    # syn / ref; reported separately, never folded into MCD


def mel_cepstrum(
    wav: np.ndarray,
    sr: int,
    n_mfcc: int = 25,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """Frames x coefficients, c0 included. The caller drops c0.

    n_mfcc=25 gives 24 usable coefficients after dropping c0, which is the
    common setting for 22.05 kHz TTS evaluation.
    """
    import librosa

    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)
    mfcc = librosa.feature.mfcc(
        y=wav.astype(np.float64),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return mfcc.T


def dtw_path(ref: np.ndarray, syn: np.ndarray) -> list[tuple[int, int]]:
    """Classic DTW with the standard three-way step, Euclidean local cost.

    Written out rather than pulled from a library so the step pattern and the
    boundary conditions are visible: both endpoints are anchored, which is what
    makes the result reproducible across library versions.
    """
    n, m = len(ref), len(syn)
    if n == 0 or m == 0:
        raise ValueError("cannot align an empty sequence")

    # Local cost matrix, squared then rooted: (n, m)
    diff = ref[:, None, :] - syn[None, :, :]
    local = np.sqrt(np.sum(diff * diff, axis=2))

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = local[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )

    # Backtrace
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def mcd(
    ref_wav: np.ndarray,
    syn_wav: np.ndarray,
    sr: int,
    n_mfcc: int = 25,
    hop_length: int = 256,
    return_path: bool = False,
):
    """MCD in dB between a reference and a synthesised waveform.

    With `return_path`, also returns the DTW path. The F0 stage has to align
    its tracks the same way MCD aligned its frames, and the alternative to
    handing the path over is for the caller to recompute the cepstra and the
    DTW itself, which is a second copy of this function's body and a second
    place for the alignment to drift. One alignment, one definition.
    """
    ref = mel_cepstrum(ref_wav, sr, n_mfcc=n_mfcc, hop_length=hop_length)
    syn = mel_cepstrum(syn_wav, sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Drop c0: it is energy, not spectral shape.
    ref_c = ref[:, 1:]
    syn_c = syn[:, 1:]

    path = dtw_path(ref_c, syn_c)
    diffs = np.array([ref_c[i] - syn_c[j] for i, j in path])
    per_frame = MCD_CONSTANT * np.sqrt(np.sum(diffs * diffs, axis=1))

    result = MCDResult(
        mcd_db=float(np.mean(per_frame)),
        n_frames=len(path),
        ref_frames=len(ref_c),
        syn_frames=len(syn_c),
        length_ratio=len(syn_c) / len(ref_c),
    )
    return (result, path) if return_path else result
