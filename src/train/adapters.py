#!/usr/bin/env python3
"""One adapter per architecture, plus a toy adapter that depends on nothing.

The toy adapter exists to answer a question the real ones cannot answer until a
GPU is available: does the shared loop actually work, and does resume actually
resume? It is a small text-to-mel model with real parameters and a real loss,
trainable on CPU in seconds. It is never used for a result, and
`assert_not_toy` refuses to let a config name it.

The three real adapters wrap upstream implementations. Each is thin on purpose.
If an adapter starts making modelling decisions, the decision belongs in the
config and in the deviations table, not in Python nobody reads.
"""

from __future__ import annotations

import pathlib

from . import batching
from .text import TextEncoder

HERE = pathlib.Path(__file__).resolve().parents[2]
INTERIM = HERE / "data" / "interim"


# --- mel features ----------------------------------------------------------

def mel_for(utt: batching.Utterance, cfg: dict, n_mels: int = 80):
    """Log-mel for one utterance, computed with the project's own parameters.

    Deliberately not cached to disk. The cache would be another artefact to
    keep in step with the manifests, and mel extraction is not the bottleneck
    next to the model.
    """
    import librosa
    import numpy as np
    sr = int(cfg["sample_rate"])
    path = INTERIM / cfg["language"] / utt.wav
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=batching.HOP_LENGTH,
        n_mels=n_mels, fmin=0, fmax=sr // 2)
    return np.log(np.maximum(m, 1e-5)).T.astype("float32")      # (frames, mels)


def pad_stack(arrays, pad_value: float = 0.0):
    import numpy as np
    n = max(a.shape[0] for a in arrays)
    out = np.full((len(arrays), n) + arrays[0].shape[1:], pad_value, dtype="float32")
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out


def pad_ids(seqs: list[list[int]]):
    import numpy as np
    n = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), n), dtype="int64")           # pad id is 0
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


# --- toy -------------------------------------------------------------------

class ToyAdapter:
    """Text encoder plus a length regulator plus a linear mel head.

    Enough structure to exercise every part of the loop that can break:
    an embedding table sized from the vocabulary, a real optimiser state,
    padding, a masked loss, gradient clipping and checkpoint round trips.
    """

    name = "toy"

    def build(self, vocab_size: int, cfg: dict):
        import torch.nn as nn

        class Toy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
                self.enc = nn.GRU(64, 64, batch_first=True, bidirectional=True)
                self.head = nn.Linear(128, 80)

            def forward(self, ids, n_frames):
                h, _ = self.enc(self.emb(ids))
                # Crude length regulation: stretch the token sequence to the
                # frame count by nearest-neighbour index. Not a real duration
                # model, and not pretending to be one.
                import torch
                idx = (torch.linspace(0, h.shape[1] - 1, n_frames, device=h.device)
                       .round().long())
                return self.head(h[:, idx, :])

        return Toy()

    def collate(self, batch: list[batching.Utterance], enc: TextEncoder, cfg: dict) -> dict:
        import torch
        mels = [mel_for(u, cfg) for u in batch]
        ids = pad_ids([enc.encode(u.text) for u in batch])
        lens = [m.shape[0] for m in mels]
        return {"ids": torch.from_numpy(ids),
                "mel": torch.from_numpy(pad_stack(mels)),
                "lengths": torch.tensor(lens)}

    def loss(self, model, t: dict):
        import torch
        pred = model(t["ids"], t["mel"].shape[1])
        # Mask the padding, or the loss rewards predicting zeros in the pad
        # region and every batch's loss depends on its padding ratio.
        mask = (torch.arange(t["mel"].shape[1], device=t["mel"].device)[None, :]
                < t["lengths"][:, None].to(t["mel"].device))[..., None]
        err = ((pred - t["mel"]) ** 2) * mask
        return err.sum() / mask.sum().clamp(min=1) / t["mel"].shape[-1]


# --- real architectures ----------------------------------------------------

class FastSpeech2Adapter:
    """FastSpeech 2 with an internal unsupervised aligner.

    Durations come from the model's own alignment module, not from MFA; see
    ALIGNER_NOTE in src/train/config.py for why. The adapter therefore has no
    duration input and no duration file to keep in step with the manifests.
    """

    name = "fastspeech2"

    def build(self, vocab_size: int, cfg: dict):
        raise NotImplementedError(
            "FastSpeech 2 weights are built on Kaggle, where the upstream "
            "package is installed. Bind the upstream constructor here in the "
            "Kaggle job rather than vendoring the model into this repository.")

    def collate(self, batch, enc, cfg):
        raise NotImplementedError

    def loss(self, model, t):
        raise NotImplementedError


class VitsAdapter:
    """VITS, initialised from the MMS checkpoint for the language.

    Two declared deviations: a second optimiser for the discriminator, and a
    16 kHz native rate. Both are in the config's deviations list, and the 16 kHz
    rate is why `batching.load_manifest` picks the wav16 column for this run.
    """

    name = "vits"

    def build(self, vocab_size: int, cfg: dict):
        raise NotImplementedError(
            "VITS is initialised from " + str(cfg.get("init_from")) +
            ", which this environment cannot reach. Built in the Kaggle job.")

    def collate(self, batch, enc, cfg):
        raise NotImplementedError

    def loss(self, model, t):
        raise NotImplementedError


class MatchaAdapter:
    """Matcha-TTS. Inference sampling steps fixed at 10 for every evaluation."""

    name = "matcha"

    def build(self, vocab_size: int, cfg: dict):
        raise NotImplementedError(
            "Matcha-TTS is initialised from the LJSpeech checkpoint, which this "
            "environment cannot reach. Built in the Kaggle job.")

    def collate(self, batch, enc, cfg):
        raise NotImplementedError

    def loss(self, model, t):
        raise NotImplementedError


ADAPTERS = {
    "toy": ToyAdapter,
    "fastspeech2": FastSpeech2Adapter,
    "vits": VitsAdapter,
    "matcha": MatchaAdapter,
}


def assert_not_toy(cfg: dict) -> None:
    """A result must never come from the toy model."""
    if cfg.get("architecture") == "toy":
        raise SystemExit(
            f"{cfg.get('run_id')}: the toy adapter exists to test the loop and "
            "must not appear in a config that produces a number")


def for_config(cfg: dict):
    arch = cfg["architecture"]
    if arch not in ADAPTERS:
        raise SystemExit(f"{cfg.get('run_id')}: no adapter for {arch!r}")
    return ADAPTERS[arch]()
