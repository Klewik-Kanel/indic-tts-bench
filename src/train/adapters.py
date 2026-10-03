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

import copy
import hashlib
import os
import json
import math
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


# --- optimisers ------------------------------------------------------------

def adamw(params, cfg: dict):
    """One AdamW, built from the budget rather than from an upstream default.

    Every run shares lr, betas and weight decay, because the budget is the
    thing being held constant across architectures. An architecture needing
    more than one optimiser (a GAN has two) builds several of these over
    disjoint parameter groups; it never reaches for different hyper-parameters,
    which would put a budget difference inside an architecture comparison.
    """
    import torch
    return torch.optim.AdamW(list(params), lr=float(cfg["lr"]),
                             betas=(0.9, 0.98))


class AdapterBase:
    """Defaults for the single-optimiser case, which is most architectures.

    An adversarial architecture overrides `n_optimizers`, `optimizers` and
    `param_groups`, and the loop then steps each optimiser in index order with
    that optimiser's index passed to `loss`. The ordering is the architecture's
    to define: for coqui's VITS, index 0 is the discriminator and index 1 the
    generator, and the generator step reuses outputs the discriminator step
    cached, so the order is load-bearing rather than cosmetic.
    """

    n_optimizers = 1
    # Whether this architecture takes text. A vocoder does not: it maps mel to
    # waveform. config.py enforces input_repr == "none" for hifigan and there is
    # no vocabulary for "none", so TextEncoder.for_config raises on it. The loop
    # asks this before building an encoder rather than building one and hoping.
    needs_text = True
    want_pitch = False
    # Whether this architecture's build() actually reads cfg["init_from"]. The
    # VITS adapter declared a warm start from MMS for weeks and never loaded
    # it, so eight runs trained from scratch while their configs said
    # otherwise and nothing noticed until the audio was listened to. This flag
    # is what assert_init_from_is_honest checks against.
    reads_init_from = False
    # The arrays this architecture's collate actually indexes. Only these are
    # read from the feature cache: see features.load_or_compute for why the
    # difference is measurable rather than cosmetic. A key missing from this
    # tuple is a KeyError in collate, not silent wrong data.
    needs: tuple[str, ...] = ()

    def _record(self, loss_dict) -> None:
        """Keep the scalar terms of the last loss, for the step record.

        The headline scalar hides what it is made of, and for FastSpeech 2 that
        matters: its objective is the sum of a mel L1 and four other terms, two
        of which are mean squared errors on f0 in hertz and on an unnormalised
        frame energy. Measured on the Hindi corpus, those two carry roughly 615
        of a mean predictor's 617, so the mel term is under one per cent of the
        total and the headline number is mostly a pitch error. A loss curve
        without the breakdown cannot show that, which is how the 5 h and 1 h
        rungs came to differ by 29.9x with no interpretation attached.

        Only 0-dim values are kept, and `loss` is dropped because the loop
        already logs it. Failures here are swallowed: a logging field must not
        be able to end a training run.
        """
        out = {}
        try:
            for k, v in dict(loss_dict).items():
                if k == "loss":
                    continue
                try:
                    if hasattr(v, "detach"):
                        if getattr(v, "ndim", 0) != 0:
                            continue
                        v = v.detach()
                    out[k] = round(float(v), 5)
                except Exception:
                    continue
        except Exception:
            return
        self.last_components = out

    # Set by `_record`, read by the loop. A class-level default means an
    # adapter that never records one still answers the loop's question.
    last_components: dict = {}

    def _features(self, batch, cfg):
        import numpy as np
        from . import features as F
        sr = int(cfg["sample_rate"])
        # TRAIN_CACHE_ROOT moves the feature cache off the shared filesystem.
        # Measured on the DGX on 2 October: Lustre gave 37 to 83 MB/s while a
        # VITS step needs 36.9 MB and two runs at full speed need 117 MB/s, so
        # the loop was waiting on the mount and the GPU sat at 45%.
        #
        # This cannot change what a run computes. features.cache_path keys every
        # entry on the analysis parameters, so a cache under a different root
        # holds the same arrays for the same inputs or it holds nothing and they
        # are recomputed. Like TRAIN_GPU_FRACTION and TRAIN_THREADS it is a
        # scheduling knob and not part of the budget.
        root = pathlib.Path(os.environ.get("TRAIN_CACHE_ROOT")
                            or (HERE / "data" / "cache")) / cfg["language"]
        return [F.load_or_compute(INTERIM / cfg["language"] / u.wav, sr, root,
                                  want_pitch=self.want_pitch,
                                  keys=self.needs or None) for u in batch]

    def optimizers(self, model, cfg: dict) -> list:
        return [adamw(model.parameters(), cfg)]

    def param_groups(self, model) -> list:
        """The parameters each optimiser owns, for per-optimiser grad clipping.

        Clipping `model.parameters()` once for a two-optimiser model would
        compute one global norm over both halves, so the discriminator's
        gradients would scale the generator's clip and the reverse. The budget
        says grad_clip; it has to mean the same thing per optimiser.
        """
        return [list(model.parameters())]

    def prepare(self, model, t: dict) -> dict:
        """Anything that must happen once per step, before any optimiser runs.

        Separate from `loss` because a two-optimiser step calls `loss` twice
        and must not redo the per-step batch work in between.
        """
        return t


# --- toy -------------------------------------------------------------------

class ToyAdapter(AdapterBase):
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

    def loss(self, model, t: dict, optimizer_idx: int = 0):
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
    import torch.nn as nn

    # The two architectures name it differently: ForwardTTS exposes .emb at the
    # top level, Vits keeps it at text_encoder.emb. Rather than hard-code
    # either, find every embedding table and check the largest. A speaker or
    # language table is small by construction, so the text embedding is the one
    # that has to clear the vocabulary.
    tables = {n: m.num_embeddings for n, m in model.named_modules()
              if isinstance(m, nn.Embedding)}
    if not tables:
        raise SystemExit(f"{run_id}: the model has no embedding table to check")
    biggest = max(tables.values())
    if biggest < vocab_size:
        raise SystemExit(
            f"{run_id}: the largest embedding has {biggest} rows and the "
            f"vocabulary has {vocab_size} symbols. coqui rebuilt num_chars "
            f"from its own character set. Tables found: {tables}. "
            "Pass coqui_characters(vocab_size).")


class CoquiAdapter(AdapterBase):
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

    def prepare(self, model, t: dict) -> dict:
        """Both coqui models need this, and VITS needs it more than once.

        `format_batch_on_device` is what derives `mel` from the spectrogram and
        the relative waveform lengths. coqui's own trainer calls it; we are not
        using that trainer, so we call it here, once per step for every coqui
        architecture.

        This lives on the shared base rather than on one subclass because of a
        real failure: with it defined on FastSpeech2Adapter alone, VITS inherited
        the no-op default, its idx-0 discriminator branch ran anyway (it indexes
        only keys collate already supplies), and the idx-1 generator branch died
        on `KeyError: 'mel'` four steps in. A per-architecture copy of a shared
        step is exactly the kind of thing that goes missing from one of them.
        """
        return model.format_batch_on_device(t)

    def loss(self, model, t: dict, optimizer_idx: int = 0):
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
    needs = ("mel", "pitch", "energy")        # never spec, never wav

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

    def loss(self, model, t: dict, optimizer_idx: int = 0):
        # ForwardTTS is not adversarial: one optimiser, one loss, and
        # train_step takes no optimizer_idx at all. `prepare` on CoquiAdapter
        # has already run format_batch_on_device.
        _outputs, loss_dict = model.train_step(t, self._criterion)
        self._record(loss_dict)
        return loss_dict["loss"]


class VitsAdapter(CoquiAdapter):
    """VITS, initialised from the MMS checkpoint for the language.

    Two declared deviations, both real in the code: a second optimiser for the
    discriminator, and a 16 kHz native rate, which is why
    batching.load_manifest selects the wav16 column for this run.

    The second optimiser is the architecture's, not the loop's special case:
    `n_optimizers`, `optimizers` and `param_groups` below say what the loop
    needs to know, and the loop stays the same loop for every architecture.
    """

    name = "vits"
    n_optimizers = 2
    needs = ("spec", "wav")                   # mel is derived by format_batch_on_device

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
        # The budget owns both learning rates. VitsConfig ships lr_disc and
        # lr_gen of its own, and leaving them would give the VITS runs a
        # learning rate no other run used while assert_budget_matched went on
        # passing, because it checks the config file and not coqui's defaults.
        c.lr_disc = float(cfg["lr"])
        c.lr_gen = float(cfg["lr"])
        m = Vits.init_from_config(c)
        assert_embedding_fits(m, vocab_size, cfg["run_id"])
        # get_criterion returns [VitsDiscriminatorLoss, VitsGeneratorLoss], in
        # that order, and train_step indexes it by optimizer_idx. The list is
        # passed through whole rather than unpacked.
        self._criterion = m.get_criterion()
        return m

    def _split(self, model) -> tuple[list, list]:
        """Discriminator parameters, then everything else.

        The split is by the `disc.` name prefix, which is how Vits.get_optimizer
        splits them upstream. Taking upstream's own optimisers instead would
        take upstream's learning rates with them, and the budget has to win.
        """
        disc, gen = [], []
        for name, p in model.named_parameters():
            (disc if name.startswith("disc.") else gen).append(p)
        if not disc:
            raise SystemExit(
                "vits: no parameters under 'disc.'; upstream renamed the "
                "discriminator and the two-optimiser split is now wrong")
        return disc, gen

    def optimizers(self, model, cfg: dict) -> list:
        disc, gen = self._split(model)
        # Index order matches train_step's optimizer_idx: 0 discriminator,
        # 1 generator. Verified against coqui-tts 0.27.5, whose train_step
        # docstring states it and whose idx-1 branch reads outputs the idx-0
        # branch cached.
        return [adamw(disc, cfg), adamw(gen, cfg)]

    def param_groups(self, model) -> list:
        disc, gen = self._split(model)
        return [disc, gen]

    def collate(self, batch, enc, cfg) -> dict:
        import torch
        feats = self._features(batch, cfg)
        ids = pad_ids([enc.encode(u.text) for u in batch])
        return {
            "tokens": torch.from_numpy(ids),
            "token_lens": torch.tensor([len(enc.encode(u.text)) for u in batch]),
            # [B, freq, frames]. VITS's posterior encoder is a conv1d over
            # frequency channels, so the spectrogram is channels-first here,
            # while FastSpeech 2 above takes its mel as [B, frames, mels]. The
            # two architectures really do differ; padding happens on the frame
            # axis either way, so the transpose comes after pad_stack.
            "spec": torch.from_numpy(
                pad_stack([f["spec"] for f in feats])).transpose(1, 2).contiguous(),
            "spec_lens": torch.tensor([f["spec"].shape[0] for f in feats]),
            # Relative frame lengths, as a fraction of the longest item in the
            # batch. format_batch_on_device recomputes spec_lens and mel_lens
            # from this and then asserts the two agree, so it has to be the
            # FRAME ratio: deriving it from sample counts can round to a
            # different frame and trip that assertion on a long batch.
            "waveform_rel_lens": torch.tensor(
                [f["spec"].shape[0] / max(x["spec"].shape[0] for x in feats)
                 for f in feats], dtype=torch.float32),
            # [B, 1, samples], matching the pitch convention above.
            "waveform": torch.from_numpy(
                pad_stack([f["wav"] for f in feats])).unsqueeze(1),
            "speaker_ids": None,
            "language_ids": None,
            "d_vectors": None,
        }

    def loss(self, model, t: dict, optimizer_idx: int = 0):
        """The loss for one of the two optimisers.

        **optimizer_idx 0 is the DISCRIMINATOR and 1 is the GENERATOR**, which
        is the opposite of what this adapter assumed before. Checked against the
        installed package rather than remembered: coqui-tts 0.27.5,
        `Vits.train_step`, whose docstring says "0 for the discriminator and 1
        for the generator networks", whose idx-0 branch runs the full forward
        pass and assigns `self.model_outputs_cache`, and whose idx-1 branch
        reads that cache. An idx-1 call without a preceding idx-0 call on the
        same batch therefore scores the previous batch's outputs, which is a
        silent wrong answer rather than a crash.

        So the order the loop steps these in is part of the architecture, not a
        detail: 0 then 1, on the same prepared batch. The idx-0 branch feeds the
        discriminator `model_outputs.detach()`, so backward through it does not
        free the generator's graph and the idx-1 branch can still use it.
        """
        _, loss_dict = model.train_step(t, self._criterion,
                                        optimizer_idx=optimizer_idx)
        self._record(loss_dict)
        return loss_dict["loss"]


# Matcha-TTS hyperparameters, copied from upstream's shipped configs rather
# than chosen here: configs/model/matcha.yaml and model/{encoder,decoder,cfm}/
# default.yaml at shivammehta25/Matcha-TTS, against the matcha-tts 0.0.7.2
# sdist. Hydra interpolations are resolved to their values. Choosing these
# numbers ourselves would make r03 a different model from published Matcha and
# the architecture comparison would be against something nobody else has.
MATCHA_ENCODER = {
    "encoder_type": "RoPE Encoder",
    "encoder_params": {
        "n_feats": 80, "n_channels": 192, "filter_channels": 768,
        "filter_channels_dp": 256, "n_heads": 2, "n_layers": 6,
        "kernel_size": 3, "p_dropout": 0.1, "spk_emb_dim": 64,
        "n_spks": 1, "prenet": True,
    },
    "duration_predictor_params": {
        "filter_channels_dp": 256, "kernel_size": 3, "p_dropout": 0.1,
    },
}
# A MAPPING, not a namespace: CFM.__init__ does Decoder(..., **decoder_params),
# so this one is splatted while encoder_params and cfm_params are read by
# attribute. The two shapes are not interchangeable and the difference is only
# visible by reading upstream. Note also that Decoder's own defaults are
# num_heads=4 and act_fn="snake" while the shipped config says 2 and
# "snakebeta"; the config wins, because that is what published Matcha is.
MATCHA_DECODER = {
    "channels": [256, 256], "dropout": 0.05, "attention_head_dim": 64,
    "n_blocks": 1, "num_mid_blocks": 2, "num_heads": 2, "act_fn": "snakebeta",
}
MATCHA_CFM = {"name": "CFM", "solver": "euler", "sigma_min": 1e-4}

# What upstream's LJSpeech recipe was fitted to. Kept so the mel-band condition
# on init_from can be stated in numbers: those weights saw a filter bank built
# to 8 kHz and mels normalised by LJSpeech's own statistics.
MATCHA_LJSPEECH = {"fmax": 8000.0, "mel_mean": -5.536622, "mel_std": 2.116101}


def attr_tree(obj):
    """Nested attribute access over a dict, for configs read with dots.

    MatchaTTS reads its encoder and cfm configuration by attribute, because
    upstream hands it an OmegaConf node. A dict raises AttributeError there.
    Rather than depend on omegaconf for three nested dicts, this converts.
    """
    from types import SimpleNamespace
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: attr_tree(v) for k, v in obj.items()})
    return obj


class MatchaAdapter(AdapterBase):
    """Matcha-TTS: optimal-transport flow matching over mel spectrograms.

    Not a `CoquiAdapter`. Matcha is a separate upstream package with its own
    conventions: it is a Lightning module, its losses come from `get_losses`
    rather than `train_step`, there is no `format_batch_on_device`, and it
    takes no `optimizer_idx` because it is not adversarial. One optimiser.

    Three things about it differ from the other two adapters and each is a
    place where a silent mistake was available:

    **Its mel targets are normalised.** `get_losses` expects `y` already
    centred and scaled by the corpus statistics the model was constructed
    with. Feeding raw log-mel trains against a distribution the prior loss
    does not expect. The statistics come from `src/train/melstats.py`, over
    this project's own cache, and this adapter refuses to run without them
    rather than falling back to upstream's defaults of 0.0 and 1.0.

    **Its mel is channels-first**, `[B, n_feats, frames]`, like VITS's
    spectrogram and unlike FastSpeech 2's `[B, frames, mels]`.

    **Its durations come from monotonic alignment search inside the model**,
    which is the same arrangement as FastSpeech 2's internal aligner and is why
    `durations` is passed as None. `aligner: internal_mas` in the config.
    """

    name = "matcha"
    needs = ("mel",)
    reads_init_from = True        # assert_init_is_comparable, then load

    # --- the corpus statistics, which are a precondition and not a default ---

    def mel_stats(self, cfg: dict) -> dict:
        """Read the corpus mel statistics, or stop with what to run.

        Also checks that they were computed under the analysis in force now. A
        mel mean and standard deviation are properties of the filter bank as
        much as of the audio, so a stats file written before a change to the
        mel parameters describes a different quantity. Reusing it would shift
        every target by a constant nobody would find later.
        """
        from . import features as F
        from . import melstats

        p = melstats.stats_path(cfg["language"])
        if not p.exists():
            raise SystemExit(
                f"{cfg.get('run_id')}: Matcha normalises its mel targets by the "
                f"corpus mean and standard deviation, and {p} does not exist. "
                f"Run:  python -m src.train.melstats {cfg['language']} "
                f"--sample-rate {int(cfg['sample_rate'])}\n"
                "Upstream's LJSpeech numbers are not a substitute: different "
                "corpus, different speaker, different mel band range.")
        s = json.loads(p.read_text(encoding="utf-8"))
        want = F.mel_params(int(cfg["sample_rate"]))
        got = s.get("mel_params") or {}
        bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
        if bad:
            lines = "\n".join(f"    {k}: statistics say {g!r}, this run uses {w!r}"
                               for k, (w, g) in sorted(bad.items()))
            raise SystemExit(
                f"{cfg.get('run_id')}: {p} was computed under a different mel "
                f"analysis, so its mean and standard deviation describe a "
                f"different quantity:\n{lines}\n"
                f"  Recompute:  python -m src.train.melstats {cfg['language']} "
                f"--sample-rate {int(cfg['sample_rate'])}")
        for k in ("mel_mean", "mel_std"):
            if not isinstance(s.get(k), (int, float)):
                raise SystemExit(f"{p}: no numeric {k}")
        if float(s["mel_std"]) <= 0:
            raise SystemExit(f"{p}: mel_std is {s['mel_std']}, which cannot be")
        return s

    # --- initialisation from upstream weights, under one condition -----------

    def assert_init_is_comparable(self, cfg: dict) -> pathlib.Path:
        """`init_from` has to be a checkpoint, and it has to share the mel band.

        Matcha's published LJSpeech weights were fitted to mels from a filter
        bank built to 8 kHz. This project builds to sr/2. Initialising from
        those weights and then training on these mels starts the model from a
        representation of a different frequency range, which is not an error
        and not visible in a loss curve. So it stops here instead, and the
        choice between changing the mel band and training from scratch is made
        in the config rather than by accident.
        """
        src = str(cfg.get("init_from") or "")
        if not src:
            return None
        p = pathlib.Path(src).expanduser()
        if not p.exists():
            raise SystemExit(
                f"{cfg.get('run_id')}: init_from is {src!r}, which is a label "
                "and not a checkpoint on this machine. Point it at the "
                "downloaded Matcha checkpoint, or set it empty to train from "
                "scratch, which is a budget change and belongs in the "
                "deviations table.")
        from . import features as F
        ours = F.mel_params(int(cfg["sample_rate"]))["fmax"]
        theirs = MATCHA_LJSPEECH["fmax"]
        if abs(float(ours) - float(theirs)) > 0.5:
            raise SystemExit(
                f"{cfg.get('run_id')}: init_from points at Matcha weights "
                f"fitted to a mel bank reaching {theirs:.0f} Hz, and this run "
                f"computes mels to {ours:.0f} Hz. Those weights describe a "
                "different representation, so initialising from them is not a "
                "warm start, it is a different model. Either settle the mel "
                "band question first, or train r03 from scratch and record "
                "that as the deviation.")
        return p

    # --- the three methods the loop calls ------------------------------------

    def build(self, vocab_size: int, cfg: dict):
        from matcha.models.matcha_tts import MatchaTTS

        stats = self.mel_stats(cfg)
        ckpt = self.assert_init_is_comparable(cfg)

        enc = copy.deepcopy(MATCHA_ENCODER)
        enc["encoder_params"]["n_feats"] = int(stats["mel_params"]["n_mels"])
        m = MatchaTTS(
            n_vocab=vocab_size,
            n_spks=1,                       # one speaker per language
            spk_emb_dim=int(MATCHA_ENCODER["encoder_params"]["spk_emb_dim"]),
            n_feats=int(stats["mel_params"]["n_mels"]),
            encoder=attr_tree(enc),
            decoder=dict(MATCHA_DECODER),   # splatted upstream: keep it a dict
            cfm=attr_tree(MATCHA_CFM),
            data_statistics={"mel_mean": float(stats["mel_mean"]),
                             "mel_std": float(stats["mel_std"])},
            # None is upstream's shipped default: train on the whole mel rather
            # than a cut segment. With a 12,000-frame budget that is the larger
            # memory footprint of the two, so it is the thing to watch first if
            # the dry run runs out of memory.
            out_size=None,
            prior_loss=True,
            use_precomputed_durations=False,
        )
        if ckpt is not None:
            import torch
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = blob.get("state_dict", blob)
            own = m.state_dict()
            taken = {k: v for k, v in sd.items()
                     if k in own and own[k].shape == v.shape}
            skipped = sorted(set(own) - set(taken))
            m.load_state_dict(taken, strict=False)
            print(f"{cfg.get('run_id')}: initialised {len(taken)} tensors from "
                  f"{ckpt}, {len(skipped)} left at their initial values "
                  f"(the text embedding differs in size: this vocabulary is "
                  f"{vocab_size})", flush=True)
        return m

    def collate(self, batch, enc, cfg) -> dict:
        import torch
        feats = self._features(batch, cfg)
        stats = self.mel_stats(cfg)
        mean, std = float(stats["mel_mean"]), float(stats["mel_std"])
        ids = pad_ids([enc.encode(u.text) for u in batch])
        # Normalise before padding, so the pad value 0.0 is the corpus mean in
        # normalised space rather than a log-mel of zero, which would be a loud
        # frame. Upstream masks the padding by y_lengths anyway; this way the
        # pad is harmless even where a mask is missed.
        mels = pad_stack([(f["mel"] - mean) / std for f in feats])
        return {
            "x": torch.from_numpy(ids),
            "x_lengths": torch.tensor([len(enc.encode(u.text)) for u in batch]),
            # [B, n_feats, frames]: channels-first, as VITS's spec is, and
            # unlike FastSpeech 2's [B, frames, mels]. The transpose comes
            # after padding, which happens on the frame axis.
            "y": torch.from_numpy(mels).transpose(1, 2).contiguous(),
            "y_lengths": torch.tensor([f["mel"].shape[0] for f in feats]),
            "spks": None,                  # one speaker, so no embedding
            "durations": None,             # monotonic alignment search, in-model
        }

    def loss(self, model, t: dict, optimizer_idx: int = 0):
        """The sum of Matcha's three losses, which is what upstream optimises.

        `get_losses` returns duration, prior and flow-matching losses
        separately and upstream's training_step optimises `sum(...)` of them
        unweighted. Summing here rather than weighting keeps r03 the published
        model; a weighting would be a modelling decision and would belong in
        the config and the deviations table.
        """
        losses = model.get_losses(t)
        self._record(losses)
        return sum(losses.values())


# --- hifigan ---------------------------------------------------------------

# coqui's HifiganConfig documents seq_len 8192 samples of audio per training
# item, which at hop 256 is exactly 32 mel frames. A vocoder trains on fixed
# segments rather than whole utterances: padding a batch to the longest item
# and training on the padding teaches the generator to produce the padding.
HIFIGAN_SEGMENT_SAMPLES = 8_192

# log(1e-5), the floor features.compute clamps the mel to, which is what the
# mel of digital silence comes out as. A short utterance is padded with this
# rather than with 0.0, which would be a loud frame.
MEL_FLOOR = math.log(1e-5)


class HiFiGanAdapter(AdapterBase):
    """HiFi-GAN, fine-tuned on this project's own ground-truth mels.

    Reinstated on 2 October after the mel survey: no published vocoder shares
    this project's mel analysis, and the three axes it differs on (band,
    amplitude convention, STFT framing) all disappear if the vocoder is fitted
    to our mels instead of our mels being refitted to a vocoder. That also
    keeps every trained acoustic parameter, which is the constraint that
    decided it, and removes the "vocoder not matched to the corpus" deviation
    rather than adding one.

    **The mel handed to the generator is ours, unnormalised.** Not coqui's
    AudioProcessor scale. That is the entire point: the generator learns the
    representation our acoustic models actually emit. A warm start from weights
    fitted to a different scale is what the fine-tuning is for.

    Two details read out of coqui's source rather than assumed, each of which
    would be silent if taken on trust:

    **`GAN.get_optimizer`'s docstring is wrong.** It says "First one is for the
    generator and the second one is for the discriminator", and the code
    returns `[optimizer2, optimizer1]`, which is discriminator first. The code
    is what `train_step` indexes: `optimizer_idx == 0` takes the discriminator
    branch and caches `y_hat_g` for the generator branch at index 1. So the
    order here is discriminator then generator, as it is for VITS, and it comes
    from reading `train_step`, not the docstring.

    **`GAN.train_disc` defaults to False** and is set by coqui's own trainer
    from `trainer.total_steps_done >= config.steps_to_start_discriminator`. We
    do not use that trainer, so left alone it stays False for the whole run,
    the discriminator never trains, and a GAN quietly becomes a generator with
    a reconstruction loss. It is set explicitly in `build`.
    """

    name = "hifigan"
    n_optimizers = 2
    needs = ("mel", "wav")
    needs_text = False
    reads_init_from = True        # _warm_start, verified on 2 Oct
    # The crop offset is derived from (seed, step, utterance id): see the note
    # in runner._collated for why this is not drawn from a global RNG.
    wants_step = True

    def segment_frames(self, cfg: dict) -> int:
        from . import features as F
        return int(HIFIGAN_SEGMENT_SAMPLES // int(F.HOP))

    def build(self, vocab_size: int, cfg: dict):
        from TTS.vocoder.configs.hifigan_config import HifiganConfig
        from TTS.vocoder.models.gan import GAN
        from . import features as F

        mel = F.mel_params(int(cfg["sample_rate"]))
        c = HifiganConfig()
        c.audio.sample_rate = mel["sample_rate"]
        c.audio.hop_length = mel["hop_length"]
        c.audio.fft_size = mel["n_fft"]
        c.audio.win_length = mel["win_length"]
        c.audio.num_mels = mel["n_mels"]
        c.audio.mel_fmin = mel["fmin"]
        # None is coqui's way of saying sr/2, which is this project's band. It
        # is written as None rather than as 11025.0 so the two agree even if
        # the sample rate ever changes.
        c.audio.mel_fmax = None
        # The L1 spectrogram term in GeneratorLoss computes its own mel, and at
        # its defaults that mel is already this project's: fmin 0.0 and
        # mel_fmax None, which librosa reads as sr/2. Set explicitly anyway,
        # because a silent disagreement here would train the generator against
        # a different analysis than the one it is being fitted to.
        c.l1_spec_loss_params = dict(c.l1_spec_loss_params)
        c.l1_spec_loss_params.update({
            "use_mel": True, "sample_rate": mel["sample_rate"],
            "n_fft": mel["n_fft"], "hop_length": mel["hop_length"],
            "win_length": mel["win_length"], "n_mels": mel["n_mels"],
            "mel_fmin": mel["fmin"], "mel_fmax": None,
        })
        # The budget owns both learning rates, as it does for VITS: coqui's own
        # lr_gen and lr_disc would give this run a rate no other run used.
        c.lr_gen = float(cfg["lr"])
        c.lr_disc = float(cfg["lr"])
        c.steps_to_start_discriminator = 0

        m = GAN.init_from_config(c)

        # The generator's upsampling has to reconstruct exactly one hop per mel
        # frame. If the factors stop multiplying to the hop, the waveform comes
        # out a different length than the mel implies and train_step's
        # y_hat[:, :, :y.size(2)] truncation hides it.
        ups = list(c.generator_model_params["upsample_factors"])
        prod = 1
        for u in ups:
            prod *= int(u)
        if prod != int(mel["hop_length"]):
            raise SystemExit(
                f"{cfg.get('run_id')}: the generator's upsample factors {ups} "
                f"multiply to {prod} and the hop length is {mel['hop_length']}. "
                "One sample of waveform per hop is what makes the mel frames "
                "and the waveform line up.")

        # See the class docstring: left at its default this is False for the
        # whole run and the discriminator never trains.
        m.train_disc = True
        self._criterion = m.get_criterion()   # [DiscriminatorLoss, GeneratorLoss]

        if cfg.get("init_from"):
            self._warm_start(m, cfg)
        return m

    def _warm_start(self, model, cfg: dict) -> None:
        """Load a published generator, renaming its tensors onto ours.

        A straight `load_state_dict` matches nothing here, and the reason is not
        the lineage but torch: `weight_norm` used to store `X.weight_g` and
        `X.weight_v`, and since the move to `parametrizations` it stores
        `X.parametrizations.weight.original0` and `.original1`. Every published
        checkpoint predates that. Add speechbrain's extra `.conv` level and
        coqui's `model_g.` prefix and three renames stand between the file and
        the model. `vocoder_init.remap_generator_state` does them, deciding the
        direction from this model's own keys rather than from a torch version.

        Measured against speechbrain/tts-hifigan-ljspeech and coqui's generator
        built from HifiganConfig: 234 of 234 tensors, no shape disagreements,
        and the loaded generator turns 32 mel frames into 8,192 samples.

        It refuses rather than proceeding when no generator tensor matched,
        because that is a cold start wearing a checkpoint's name, and the step
        count it would then need is not the one the schedule assumes.
        """
        import torch

        from .vocoder_init import describe, remap_generator_state

        p = pathlib.Path(str(cfg["init_from"])).expanduser()
        if not p.exists():
            raise SystemExit(
                f"{cfg.get('run_id')}: init_from is {cfg['init_from']!r}, which "
                "is not a checkpoint on this machine. Point it at the "
                "downloaded vocoder, or set it empty to train from scratch, "
                "which costs the warm start and belongs in the deviations "
                "table.")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        for key in ("model", "state_dict", "generator"):
            if isinstance(blob, dict) and isinstance(blob.get(key), dict):
                blob = blob[key]
                break
        if not isinstance(blob, dict):
            raise SystemExit(f"{p}: not a state dict this can read")

        own = model.state_dict()
        renamed, report = remap_generator_state(blob, own.keys())

        # Shapes, after the names agree. A tensor of the right name and the
        # wrong size means a different architecture, not a different spelling.
        wrong = [k for k, v in renamed.items()
                 if hasattr(v, "shape") and tuple(v.shape) != tuple(own[k].shape)]
        for k in wrong:
            renamed.pop(k)
        model.load_state_dict(renamed, strict=False)

        print(f"{cfg.get('run_id')}: warm start from {p}")
        print(describe(report), flush=True)
        if wrong:
            print(f"  {len(wrong)} tensor(s) matched by name but not by shape and "
                  f"were skipped, e.g. {wrong[0]}", flush=True)
        gen = sum(1 for k in renamed if k.startswith("model_g."))
        if gen == 0:
            raise SystemExit(
                f"{cfg.get('run_id')}: no generator tensor in {p} could be "
                "mapped onto this model, so this is not a warm start. Either "
                "the checkpoint is a different architecture, or it names its "
                "tensors in a way vocoder_init does not yet cover. Training "
                "from scratch is the honest alternative and should be chosen "
                "deliberately, not reached by accident.")
        print(f"  {gen} generator tensor(s) loaded; the discriminator starts "
              "fresh unless the checkpoint carried one", flush=True)

    def _split(self, model) -> tuple[list, list]:
        """Discriminator parameters, then the generator's.

        By the `model_d.` prefix, which is how GAN.__init__ names the two
        submodules: model_g and model_d.
        """
        disc, gen = [], []
        for name, p in model.named_parameters():
            (disc if name.startswith("model_d.") else gen).append(p)
        if not disc:
            raise SystemExit(
                "hifigan: no parameters under 'model_d.'; upstream renamed the "
                "discriminator and the two-optimiser split is now wrong")
        return disc, gen

    def optimizers(self, model, cfg: dict) -> list:
        disc, gen = self._split(model)
        # Discriminator first. train_step's optimizer_idx == 0 is the
        # discriminator branch and it caches outputs the idx-1 generator branch
        # reuses, so the order is load-bearing. coqui's get_optimizer docstring
        # says the opposite of what its code returns; the code is right.
        return [adamw(disc, cfg), adamw(gen, cfg)]

    def param_groups(self, model) -> list:
        disc, gen = self._split(model)
        return [disc, gen]

    def crop_offset(self, cfg: dict, step: int, uid: str, limit: int) -> int:
        """Where this utterance is cropped at this step, reproducibly.

        Derived from (seed, step, utterance id) by hashing, not drawn from an
        RNG: collate runs on the prefetch thread, ahead of the training loop,
        so a global draw would land in a checkpoint at the wrong position and
        the bit-exact resume guarantee would stop holding without any symptom.
        Hashing gives a different crop every step, the same crop on a resumed
        run, and no RNG state to keep in step with anything.
        """
        if limit <= 0:
            return 0
        h = hashlib.blake2b(f"{int(cfg['seed'])}:{int(step)}:{uid}".encode("utf-8"),
                            digest_size=8).digest()
        return int.from_bytes(h, "big") % (limit + 1)

    def collate(self, batch, enc, cfg, step: int = 0) -> dict:
        """One fixed-length segment per utterance: mel and the matching audio.

        No text. `config.py` enforces input_repr == "none" for this
        architecture, so the text encoder is not involved and the vocoder sees
        the same audio whichever arm it will later be used to synthesise. That
        is what lets one vocoder serve both arms of a comparison.
        """
        import numpy as np
        import torch
        from . import features as F

        hop = int(F.HOP)
        seg = self.segment_frames(cfg)
        feats = self._features(batch, cfg)
        mels, waves = [], []
        for u, f in zip(batch, feats):
            mel = np.asarray(f["mel"], dtype="float32")        # (frames, mels)
            wav = np.asarray(f["wav"], dtype="float32").reshape(-1)
            # Frames whose full hop of audio actually exists. librosa's
            # centred STFT reports a frame for the final partial hop, and
            # cropping up to it would ask for samples past the end.
            usable = min(mel.shape[0], wav.size // hop)
            if usable >= seg:
                off = self.crop_offset(cfg, step, u.uid, usable - seg)
                mel_c = mel[off:off + seg]
                wav_c = wav[off * hop:(off + seg) * hop]
            else:
                # Shorter than one segment: pad the audio with silence and the
                # mel with the floor silence actually produces.
                mel_c = np.full((seg, mel.shape[1]), MEL_FLOOR, dtype="float32")
                mel_c[:mel.shape[0]] = mel
                wav_c = np.zeros(seg * hop, dtype="float32")
                wav_c[:wav.size] = wav[:seg * hop]
            mels.append(mel_c)
            waves.append(wav_c)

        return {
            # [B, n_mels, frames]: the generator is a conv1d stack over mel
            # channels, so channels-first, as VITS's spectrogram is.
            "input": torch.from_numpy(np.stack(mels)).transpose(1, 2).contiguous(),
            # [B, 1, samples], which is what train_step's y.size(2) indexes.
            "waveform": torch.from_numpy(np.stack(waves)).unsqueeze(1),
        }

    def validate(self, model, cfg: dict, n_items: int = 8):
        """Held-out mel reconstruction error, which is what stops this run.

        Generate audio from dev-set ground-truth mels, recompute the mel of
        what came out, and take the mean absolute difference against the mel
        that went in. Analysis-consistent by construction: the recomputation
        goes through `features.mel_from_array`, which is the same code path
        that produced every training mel, so the number cannot drift because of
        a second mel implementation.

        The dev manifest, which no run trains on. `step=0` is passed to collate
        deliberately, so every check scores the same segments and the number
        moves because the model moved rather than because the crop did.

        Returns None when it cannot run, which the criterion treats as no
        evidence rather than as a failure to improve.
        """
        import numpy as np
        import torch
        from . import batching
        from . import features as F

        mpath = HERE / "data" / "processed" / cfg["language"] / "dev.tsv"
        if not mpath.exists():
            print(f"{cfg.get('run_id')}: no {mpath}, so there is no "
                  "early-stopping signal and this run will use its step count",
                  flush=True)
            return None
        utts = batching.load_manifest(mpath, int(cfg["sample_rate"]))[:int(n_items)]
        if not utts:
            return None

        batch = self.collate(utts, None, cfg, step=0)
        gen = getattr(model, "model_g", model)
        device = next(gen.parameters()).device
        x = batch["input"].to(device)
        was_training = bool(getattr(gen, "training", False))
        gen.eval()
        try:
            with torch.no_grad():
                y_hat = gen(x)
        finally:
            if was_training:
                gen.train()

        audio = y_hat.squeeze(1).float().cpu().numpy()
        want = x.transpose(1, 2).float().cpu().numpy()        # (B, frames, mels)
        sr = int(cfg["sample_rate"])
        errs = []
        for i in range(audio.shape[0]):
            got = F.mel_from_array(np.ascontiguousarray(audio[i]), sr)
            n = min(got.shape[0], want[i].shape[0])
            if n:
                errs.append(float(np.abs(got[:n] - want[i][:n]).mean()))
        return float(np.mean(errs)) if errs else None

    def loss(self, model, t: dict, optimizer_idx: int = 0):
        """Discriminator loss at index 0, generator loss at index 1."""
        _outputs, loss_dict = model.train_step(t, self._criterion, optimizer_idx)
        self._record(loss_dict)
        return loss_dict["loss"]


ADAPTERS = {
    "toy": ToyAdapter,
    "fastspeech2": FastSpeech2Adapter,
    "vits": VitsAdapter,
    "matcha": MatchaAdapter,
    "hifigan": HiFiGanAdapter,
}


def assert_not_toy(cfg: dict) -> None:
    """A result must never come from the toy model."""
    if cfg.get("architecture") == "toy":
        raise SystemExit(
            f"{cfg.get('run_id')}: the toy adapter exists to test the loop and "
            "must not appear in a config that produces a number")


def assert_init_from_is_honest(cfg: dict, adapter) -> None:
    """A config may not claim a warm start its adapter never performs.

    This is the check that would have caught the MMS gap on day one. r02
    declared `init_from: facebook/mms-tts-hin`, `VitsAdapter.build` never read
    it, and the config, the deviations table and the class docstring all said
    the 16 kHz rate was inherited from that checkpoint. Nothing failed, nothing
    warned, and eight runs' provenance was wrong until someone listened to the
    audio and worked backwards.

    The acknowledged case is allowed through. On 3 October the decision was to
    keep those weights rather than re-run them, and that decision is recorded
    in the non-hashed `corrections` field of every affected config. So a
    mismatch passes only when a correction mentions `init_from`, which means
    somebody wrote down what is wrong and why. An unacknowledged mismatch
    refuses to start.
    """
    declared = str(cfg.get("init_from") or "").strip()
    if not declared or getattr(adapter, "reads_init_from", False):
        return
    corrections = cfg.get("corrections") or ()
    if any("init_from" in str(c) for c in corrections):
        return
    raise SystemExit(
        f"{cfg.get('run_id')}: config declares init_from={declared!r} but "
        f"{type(adapter).__name__} never reads it, so this run would train "
        "from scratch while its provenance says otherwise. Fix one of three "
        "things: implement the warm start in the adapter, set init_from to '' "
        "in config.py and regenerate, or add a correction mentioning "
        "init_from that records what is wrong and why.")


def for_config(cfg: dict):
    arch = cfg["architecture"]
    if arch not in ADAPTERS:
        raise SystemExit(f"{cfg.get('run_id')}: no adapter for {arch!r}")
    adapter = ADAPTERS[arch]()
    assert_init_from_is_honest(cfg, adapter)
    return adapter
