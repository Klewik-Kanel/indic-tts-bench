#!/usr/bin/env python3
"""Make a mel audible without a vocoder, as a labelled placeholder.

FastSpeech 2 and Matcha emit mel spectrograms and are silent on their own. r06
and r17 are the vocoders that will give them a voice, and until those train
this is how their arms can be heard at all.

**It is not the system's vocoder and must never be presented as one.** Griffin-
Lim recovers phase by iteration from magnitudes alone, so it produces the
characteristic metallic, slightly smeared quality regardless of how good the
mel is. A listener told "this is the model" would be hearing the algorithm.
Everything that writes a Griffin-Lim clip therefore records
`vocoder: "griffin-lim"` beside it, and the player labels it. The flag that
turns it on is explicit for the same reason: it cannot happen by default.

What it is good for: checking that a mel is intelligible, that the front end and
the weights line up, and that the interface works end to end before a vocoder
exists. Those are real uses, and none of them is a quality claim.

**It inverts this project's mel, not a generic one.** `features._spec_and_mel`
computes `log(max(mel_fb @ |stft|, 1e-5))` with the bank built `fmin=0`,
`fmax=sr//2`. Three details each break the inversion silently if missed: the
forward mel is MAGNITUDE, so librosa's default `power=2.0` would square-root
the spectrum and lose roughly half the dynamic range; the bank must be built to
`sr//2`, not librosa's default `sr/2` with a different rounding or an 8 kHz
convention; and the log has to be undone with `exp` before any of it.

Verified against a real corpus utterance by round-tripping: our mel, inverted
here, then re-analysed through the identical forward path. Mean absolute mel
error 0.1434 at 22.05 kHz and 0.1810 at 16 kHz, against 2.1473 and 2.2866 for
the same mel with its frames shuffled. So about 15x and 13x closer than chance.
Iterations past 60 buy 0.0046, which is why 60 is the default.
"""

from __future__ import annotations

from ..train.features import HOP, N_FFT, N_MELS, WIN, mel_params

# The clamp `features._spec_and_mel` applies before taking the log. Frames at
# this value are silence, and exp() puts them back where they started.
MEL_FLOOR = 1e-5
DEFAULT_ITERS = 60


def assert_matches_project(sample_rate: int) -> dict:
    """The parameters this inversion uses, checked against the forward path."""
    p = mel_params(sample_rate)
    if (p["n_fft"], p["hop_length"], p["win_length"], p["n_mels"]) != \
       (N_FFT, HOP, WIN, N_MELS):
        raise AssertionError(
            f"mel_params disagrees with this module's constants: {p}")
    if p["fmin"] != 0.0:
        raise AssertionError(f"expected fmin 0, got {p['fmin']}")
    return p


def invert(mel, sample_rate: int, n_iter: int = DEFAULT_ITERS):
    """Log-magnitude mel `[n_mels, frames]` to a waveform.

    Takes the orientation `synthesize.Speech.mel` hands back, which is also
    `features.compute`'s and the vocoder's, so no caller has to transpose.
    """
    import numpy as np

    # Validated before librosa is imported, so a transposed mel or a bad
    # iteration count fails in milliseconds rather than after a multi-second
    # import, and so these guards can be tested without the audio stack.
    assert_matches_project(sample_rate)
    m = np.asarray(mel, dtype="float64")
    if m.ndim != 2:
        raise ValueError(f"expected [n_mels, frames], got shape {m.shape}")
    if m.shape[0] != N_MELS:
        raise ValueError(
            f"expected {N_MELS} mel bands first, got shape {m.shape}; "
            "coqui returns frames-first and this wants mels-first")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")

    import numba            # noqa: F401  before librosa: its lazy loader
    #                       trips a circular import inside numba.core otherwise
    import librosa

    # exp undoes features._spec_and_mel's log. power=1.0 because that mel is
    # magnitude; librosa's default of 2.0 would treat it as power.
    mel_mag = np.exp(m)
    spec = librosa.feature.inverse.mel_to_stft(
        mel_mag, sr=int(sample_rate), n_fft=N_FFT, power=1.0,
        fmin=0, fmax=int(sample_rate) // 2)
    wav = librosa.griffinlim(spec, n_iter=int(n_iter), hop_length=HOP,
                             win_length=WIN, n_fft=N_FFT)
    # Griffin-Lim returns a whole number of hops and so lands up to one frame
    # short of the original, about 2 ms at 22.05 kHz. Left as it is rather than
    # padded: a placeholder should not quietly invent samples.
    return np.clip(wav, -1.0, 1.0).astype("float32")
