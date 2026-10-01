#!/usr/bin/env python3
"""Acoustic features, computed once per utterance and cached.

The two architectures want different things from the same audio. VITS trains on
a linear spectrogram and the waveform, because its decoder is adversarial and
reconstructs samples. FastSpeech 2 trains on a mel spectrogram plus per-frame
pitch and energy, because its variance adaptor predicts them explicitly.

Computing these inside the training loop would make the step rate a function of
CPU work rather than of the GPU, which is the thing the budget is supposed to
bound. They are therefore cached to disk on first use. The cache key includes
the sample rate and the FFT settings, so a config change cannot silently reuse
features computed under different parameters, which is the kind of error that
produces audio that is recognisably speech and subtly wrong.

Pitch is extracted with librosa's YIN rather than pyin. pyin is more accurate
on breathy voices and several times slower, and this corpus is one clean studio
speaker per language. The choice is recorded rather than defended as optimal:
if the pitch predictor underperforms, this is the first thing to revisit.
"""

from __future__ import annotations

import hashlib
import pathlib

N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
FMIN_HZ, FMAX_HZ = 60.0, 600.0        # a generous range for one adult speaker


def _key(sr: int) -> str:
    return hashlib.sha256(
        f"{sr}|{N_FFT}|{HOP}|{WIN}|{N_MELS}|{FMIN_HZ}|{FMAX_HZ}".encode()
    ).hexdigest()[:8]


def cache_path(wav_path: pathlib.Path, sr: int, root: pathlib.Path) -> pathlib.Path:
    return root / _key(sr) / (wav_path.stem + ".npz")


def compute(wav_path: pathlib.Path, sr: int, want_pitch: bool = False):
    """Return a dict of float32 arrays for one utterance.

    mel   (frames, 80)   log mel, natural log, floored at 1e-5
    spec  (frames, 513)  linear magnitude, what VITS trains on
    wav   (samples,)     the waveform itself, for the adversarial decoder
    pitch (frames,)      f0 in Hz, 0 where unvoiced, only when want_pitch
    energy(frames,)      L2 norm of each linear frame
    """
    import librosa
    import numpy as np

    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    spec = np.abs(stft).astype("float32")                     # (513, frames)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS,
                                 fmin=0, fmax=sr // 2)
    mel = np.log(np.maximum(mel_fb @ spec, 1e-5)).astype("float32")
    energy = np.linalg.norm(spec, axis=0).astype("float32")

    out = {"mel": mel.T, "spec": spec.T, "wav": y.astype("float32"),
           "energy": energy}

    if want_pitch:
        f0 = librosa.yin(y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr,
                         frame_length=N_FFT, hop_length=HOP)
        # YIN reports a pitch everywhere, including silence. Gate on energy so
        # unvoiced frames read as 0 rather than as a confident wrong octave.
        thresh = float(np.percentile(energy, 15))
        f0 = np.where(energy[:len(f0)] > thresh, f0[:len(energy)], 0.0)
        out["pitch"] = f0.astype("float32")

    # Frame counts from independent code paths must agree, or the model is
    # silently trained on misaligned targets.
    n = out["mel"].shape[0]
    for k in ("spec", "energy") + (("pitch",) if want_pitch else ()):
        a = out[k]
        if abs(a.shape[0] - n) > 1:
            raise AssertionError(
                f"{wav_path.name}: {k} has {a.shape[0]} frames against mel's {n}")
        out[k] = a[:n] if a.shape[0] >= n else np.pad(a, (0, n - a.shape[0]))
    return out


def load_or_compute(wav_path: pathlib.Path, sr: int, root: pathlib.Path,
                    want_pitch: bool = False, keys: tuple[str, ...] | None = None):
    """Load one utterance's cached features, or compute and cache them.

    `keys` names the arrays the caller actually uses, and only those are read
    off disk. This is not a micro-optimisation. A cache entry holds mel, spec,
    wav, energy and pitch, and `spec` alone is (frames, 513) float32 — about
    2.6 MB for a 15-second utterance against 0.4 MB for its mel. FastSpeech 2
    never touches spec or wav, so reading the whole entry made every step
    decompress roughly four times the bytes it needed, in the training loop's
    own thread. Measured effect: the A100 sat at 66% utilisation while the loop
    waited on npz decompression, and the step rate ran about a third below the
    benchmarked figure.

    Reading a subset cannot change what a run computes: the arrays are the same
    arrays, and a key the caller does not name is one it never reads.
    """
    import numpy as np
    p = cache_path(wav_path, sr, root)
    if p.exists():
        with np.load(p) as z:
            have = set(z.files)
            if not want_pitch or "pitch" in have:
                wanted = [k for k in (keys or tuple(z.files)) if k in have]
                missing = [k for k in (keys or ()) if k not in have]
                if not missing:
                    return {k: z[k] for k in wanted}
    d = compute(wav_path, sr, want_pitch=want_pitch)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.rename(p)                       # atomic: a killed job leaves no half file
    return d
