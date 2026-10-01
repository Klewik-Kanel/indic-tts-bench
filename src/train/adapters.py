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

def coqui_characters(vocab_size: int):
    """A CharactersConfig sized from OUR inventory, not coqui's default.

    This exists because of a silent failure worth remembering.
    ForwardTTS.init_from_config and Vits.init_from_config build a tokenizer
    from the config's character set and then OVERWRITE model_args.num_chars
    with its size. Setting num_chars by hand beforehand looks like it works,
    reports no error, and is discarded. coqui's default set is 67 symbols; our
    phoneme inventory is 78. The result is an embedding table with 67 rows
    being indexed at 77, which surfaces only on the GPU, thousands of lines
    deep, as:

        vectorized_gather_kernel: index out of bounds

    We feed our own token ids and never use coqui's tokenizer, so only the
    SIZE of the table matters here, not which symbol sits in which row. The
    characters string is therefore placeholder codepoints of the right count,
    chosen from a range that cannot collide with the punctuation or the
    reserved pad, eos, bos and blank entries coqui adds on top.
    """
    from TTS.tts.configs.shared_configs import CharactersConfig

    syms = "".join(chr(0x4E00 + i) for i in range(vocab_size))
    return CharactersConfig(
        characters=syms,
        punctuations="!,.?",
        pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>",
        characters_class="TTS.tts.utils.text.characters.Graphemes",
    )


def assert_embedding_fits(model, vocab_size: int, run_id: str) -> None:
    """Fail here, on the CPU, with a readable message.

    Without this the same error arrives as a CUDA device-side assert from a
    gather kernel, which says nothing about vocabularies and cannot be caught.
    """
    rows = model.emb.num_embeddings
    if rows < vocab_size:
        raise SystemExit(
            f"{run_id}: the model's embedding has {rows} rows and the "
            f"vocabulary has {vocab_size} symbols. coqui rebuilt num_chars "
            "from its own character set. Pass coqui_characters(vocab_size).")


class CoquiAdapter:
    """Shared base for the two coqui-tts models.

    coqui's own trainer is deliberately not used. The fixed-budget claim rests
    on one loop controlling steps, batching, schedule and precision; three
    upstream trainers would each do something slightly different and the claim
    would be unverifiable. What is taken from coqui is the model and its loss,
    which is the part worth taking.

    Its batch is a dict of named tensors, documented by reading train_step
    rather than by guessing: the keys below are what the installed version
    actually indexes.
    """

    name = "coqui"
    want_pitch = False

    def _features(self, batch, cfg):
        import numpy as np
        from . import features as F
        sr = int(cfg["sample_rate"])
        root = HERE / "data" / "cache" / cfg["language"]
        return [F.load_or_compute(INTERIM / cfg["language"] / u.wav, sr, root,
                                  want_pitch=self.want_pitch) for u in batch]

    def loss(self, model, t: dict):
        raise NotImplementedError          # each subclass calls train_step itself


class FastSpeech2Adapter(CoquiAdapter):
    """FastSpeech 2, as coqui's ForwardTTS with pitch and energy enabled.

    Durations come from the model's internal aligner, not from MFA; see
    ALIGNER_NOTE in src/train/config.py. train_step takes durations=None and
    uses the aligner's own output as the target, which is what keeps the
    duration source identical across the phonemic and graphemic arms.
    """

    name = "fastspeech2"
    want_pitch = True

    def build(self, vocab_size: int, cfg: dict):
        from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
        from TTS.tts.models.forward_tts import ForwardTTS

        c = Fastspeech2Config()
        c.characters = coqui_characters(vocab_size)
        c.use_phonemes = False            # our front end already produced them
        c.model_args.use_aligner = True
        c.model_args.use_pitch = True
        c.model_args.use_energy = True
        c.audio.sample_rate = int(cfg["sample_rate"])
        c.audio.num_mels = 80
        c.audio.hop_length = 256
        c.audio.fft_size = 1024
        c.audio.win_length = 1024
        m = ForwardTTS.init_from_config(c)
        assert_embedding_fits(m, vocab_size, cfg["run_id"])
        self._criterion = m.get_criterion()
        return m

    def collate(self, batch, enc, cfg) -> dict:
        import numpy as np
        import torch
        feats = self._features(batch, cfg)
        ids = pad_ids([enc.encode(u.text) for u in batch])
        mels = pad_stack([f["mel"] for f in feats])
        return {
            "text_input": torch.from_numpy(ids),
            "text_lengths": torch.tensor([len(enc.encode(u.text)) for u in batch]),
            "mel_input": torch.from_numpy(mels),
            "mel_lengths": torch.tensor([f["mel"].shape[0] for f in feats]),
            # [B, 1, frames], NOT [B, frames, 1]. average_over_durations indexes
            # the last axis as time; with the axes swapped its cumulative-sum
            # arithmetic runs past the end and the only symptom is a CUDA
            # device-side assert inside a gather kernel.
            "pitch": torch.from_numpy(
                pad_stack([f["pitch"] for f in feats])).unsqueeze(1),
            "energy": torch.from_numpy(
                pad_stack([f["energy"] for f in feats])).unsqueeze(1),
            "durations": None,                 # the aligner supplies them
            "speaker_ids": None,               # one speaker per language
            "d_vectors": None,
        }

    def loss(self, model, t: dict):
        _outputs, loss_dict = model.train_step(t, self._criterion)
        return loss_dict["loss"]


class VitsAdapter(CoquiAdapter):
    """VITS, initialised from the MMS checkpoint for the language.

    Two declared deviations, both now real in the code rather than only in the
    table: a second optimiser for the discriminator, and a 16 kHz native rate,
    which is why batching.load_manifest selects the wav16 column for this run.
    """

    name = "vits"

    def build(self, vocab_size: int, cfg: dict):
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits

        c = VitsConfig()
        c.characters = coqui_characters(vocab_size)
        c.use_phonemes = False
        c.audio.sample_rate = int(cfg["sample_rate"])
        c.audio.hop_length = 256
        c.audio.fft_size = 1024
        c.audio.win_length = 1024
        m = Vits.init_from_config(c)
        assert_embedding_fits(m, vocab_size, cfg["run_id"])
        self._criterion = m.get_criterion()
        return m

    def collate(self, batch, enc, cfg) -> dict:
        import torch
        feats = self._features(batch, cfg)
        ids = pad_ids([enc.encode(u.text) for u in batch])
        return {
            "tokens": torch.from_numpy(ids),
            "token_lens": torch.tensor([len(enc.encode(u.text)) for u in batch]),
            "spec": torch.from_numpy(pad_stack([f["spec"] for f in feats])),
            "spec_lens": torch.tensor([f["spec"].shape[0] for f in feats]),
            # [B, 1, samples], matching the pitch convention above.
            "waveform": torch.from_numpy(
                pad_stack([f["wav"] for f in feats])).unsqueeze(1),
            "speaker_ids": None,
            "language_ids": None,
            "d_vectors": None,
        }

    def loss(self, model, t: dict):
        # optimizer_idx 0 is the generator. The discriminator step is handled
        # by the runner's two-optimiser path, which is the declared deviation.
        _, loss_dict = model.train_step(t, self._criterion, optimizer_idx=0)
        return loss_dict["loss"]


class MatchaAdapter:
    """Matcha-TTS. Deferred: not in the 3 Oct scope and not yet wired."""

    name = "matcha"

    def build(self, vocab_size: int, cfg: dict):
        raise NotImplementedError(
            "Matcha-TTS is deferred past the current deadline. It is still in "
            "the run matrix as r03 so the plan does not quietly lose it.")

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
