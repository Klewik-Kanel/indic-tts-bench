#!/usr/bin/env python3
"""Text in, speech out, from a bundle and nothing else.

`bundle.py` exists so a finished run can be demonstrated on a machine with no
GPU, no corpus and no copy of this repository. This is the other half of that:
the thing that reads a bundle and speaks. It is also what the demo app calls,
so there is one synthesis path rather than one for the app and another for the
listening test.

Three things it refuses to do, each because the alternative fails quietly:

**It does not rebuild the vocabulary.** `Vocab.load` reads the table that
trained these weights. A rebuilt one is correct only if the inventory has not
changed since, and when it has, every embedding row is off by one and the model
speaks fluent nonsense with no error raised anywhere.

**It does not reimplement the front end.** The same `TextEncoder` that fed
training encodes the demo text, through the same normaliser, the same G2P and
the same schwa configuration. A demo whose front end differs from training is
demonstrating something other than the run.

**It does not pretend a mel is audio.** FastSpeech 2 and Matcha return mel
spectrograms and are silent on their own. `synthesize` hands back the mel and
says so rather than inventing a waveform, because an unlabelled Griffin-Lim
rendering presented as the system's output is the kind of claim this project
cannot make.

The mel orientation is handled here, once. coqui's ForwardTTS returns
`[B, T_frames, n_mels]`; the HiFi-GAN generator takes `[B, n_mels, T_frames]`.
Mels leave this module as `[n_mels, T_frames]`, matching the vocoder and
`features.compute`, so no caller has to decide which way round it is.

    python -m src.export.synthesize exports/r02_step100000 \
        --text "नमक कमल" --out demo.wav
    python -m src.export.synthesize exports/r02_step100000 --benchmark
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass, field

from ..train.features import mel_params
from ..train.text import TextEncoder, Vocab

# Architectures that return a waveform, against those that return a mel and
# need a vocoder downstream. Taken from the manifest's own `needs_vocoder`
# where present; this is the fallback for a bundle that predates the field.
END_TO_END = ("vits",)

SUPPORTED_BUNDLE_VERSION = 3

# Architectures that map mel to waveform and carry no text side.
VOCODER_ARCHITECTURES = frozenset({"hifigan"})

# How a synthesis draw is seeded. Keyed on the run, the text and the draw
# index, so an utterance's audio does not depend on how many utterances came
# before it and scoring a subset reproduces the full set's audio exactly.
SEED_SALT = "indic-tts-bench/synthesis/v1"


def draw_seed(run_id: str, text: str, draw: int = 0) -> int:
    """A reproducible 63-bit seed for one (run, text, draw).

    sha256 rather than hash(), because Python salts str hashing per process
    and a per-process seed would make every re-run a different measurement,
    which is the defect this exists to close.
    """
    key = f"{SEED_SALT}|{run_id}|{draw}|{text}".encode("utf-8")
    import hashlib
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


@dataclass
class Speech:
    """One synthesis. Either `waveform` or `mel` is set, never neither."""

    run_id: str
    text: str
    tokens: list[str]
    sample_rate: int
    seconds_elapsed: float
    waveform: object | None = None        # np.ndarray [N], float32 in [-1, 1]
    mel: object | None = None             # np.ndarray [n_mels, T_frames]
    needs_vocoder: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def audio_seconds(self) -> float | None:
        if self.waveform is None:
            return None
        return len(self.waveform) / self.sample_rate

    @property
    def realtime_factor(self) -> float | None:
        """Below 1.0 means faster than real time, which is what a demo needs."""
        dur = self.audio_seconds
        if not dur:
            return None
        return self.seconds_elapsed / dur


def read_manifest(root: pathlib.Path | str) -> dict:
    """The manifest, with the version checked. No torch, so it fails fast.

    Kept separate from `Bundle` so that a listing UI can describe every bundle
    in a directory without loading a single set of weights, and so that a
    bundle written by a newer build is rejected in milliseconds rather than
    after a multi-second torch import.
    """
    root = pathlib.Path(root)
    man_path = root / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"{root}: no manifest.json; not a bundle")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    version = int(manifest.get("bundle_version", 0))
    if version > SUPPORTED_BUNDLE_VERSION:
        raise SystemExit(
            f"{root}: bundle_version {version}, but this build understands "
            f"{SUPPORTED_BUNDLE_VERSION}. Update the code rather than loading "
            "it and hoping the fields line up.")
    for key in ("run_id", "architecture", "language", "input_repr",
                "sample_rate"):
        if key not in manifest:
            raise SystemExit(f"{root}: manifest has no {key!r}")
    return manifest


class Bundle:
    """A loaded bundle: manifest, model on a device, and its text encoder."""

    def __init__(self, root: pathlib.Path, device: str = "cpu") -> None:
        self.root = pathlib.Path(root)
        self.manifest = read_manifest(self.root)
        self.device = device
        self.vocoder: "Bundle | None" = None
        arch = self.manifest["architecture"]
        self.needs_vocoder = bool(
            self.manifest.get("needs_vocoder", arch not in END_TO_END))

        # The vocabulary that trained these weights, never a rebuilt one.
        vocab_path = self.root / "vocab.json"
        if self.manifest["input_repr"] == "none":
            self.encoder = None            # a vocoder bundle: no text side
        else:
            if not vocab_path.exists():
                raise SystemExit(
                    f"{self.root}: input_repr is {self.manifest['input_repr']!r} "
                    "but there is no vocab.json. Rebuilding one here could be "
                    "off by a row against these weights, so this is fatal.")
            self.encoder = TextEncoder(
                self.manifest["language"], self.manifest["input_repr"],
                Vocab.load(vocab_path),
                merge_nukta=bool(self.manifest.get("merge_nukta", False)))

        import torch

        self.model = self._build(arch)
        state = torch.load(self.root / "model.pt", map_location="cpu",
                           weights_only=True)
        # strict: a silently partial load is how a demo ends up running a
        # half-initialised model and nobody can say why it sounds wrong.
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.loaded_tensors = len(state)

    # -- construction -------------------------------------------------------

    def _build(self, arch: str):
        """Rebuild through the adapter that trained it, not a parallel path."""
        from ..train import adapters

        cfg = {
            "run_id": self.manifest["run_id"],
            "architecture": arch,
            "language": self.manifest["language"],
            "input_repr": self.manifest["input_repr"],
            "sample_rate": int(self.manifest["sample_rate"]),
            "merge_nukta": bool(self.manifest.get("merge_nukta", False)),
            # VitsAdapter.build reads cfg["lr"] to set coqui's lr_disc and
            # lr_gen. Neither enters the inference graph, but the manifest
            # carries the real value from bundle_version 3 on so that nothing
            # here has to invent a number that looks like a hyperparameter.
            "lr": float(self.manifest.get("lr", 0.0)),
        }
        adapter = adapters.for_config(cfg)
        vocab_size = len(self.encoder.vocab) if self.encoder is not None else 0
        return adapter.build(vocab_size, cfg)

    # -- synthesis ----------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return int(self.manifest["sample_rate"])

    @property
    def describes(self) -> str:
        return self.manifest.get("describes", self.manifest["run_id"])

    def tokens(self, text: str) -> list[str]:
        if self.encoder is None:
            raise SystemExit(f"{self.root}: this bundle has no text front end")
        return self.encoder.tokens(text)

    @property
    def is_vocoder(self) -> bool:
        """A bundle that maps mel to waveform and has no text side."""
        return self.manifest["architecture"] in VOCODER_ARCHITECTURES

    def attach_vocoder(self, voc: "Bundle") -> "Bundle":
        """Give a mel-only bundle the vocoder that turns its mels into audio.

        Checked rather than assumed, because a mel handed to a vocoder built on
        different analysis parameters produces audio that is recognisably
        speech and quietly wrong, which is the one failure mode this whole
        export path exists to prevent.

        The sample rate has to match: r06 is the Hindi vocoder at 22.05 kHz and
        r17 the Marathi one, and crossing them would put a speaker mismatch
        inside the control that exists to isolate language. The language is
        checked too, for the same reason, and a mismatch raises rather than
        warning.
        """
        if not voc.is_vocoder:
            raise SystemExit(
                f"{voc.root}: architecture {voc.manifest['architecture']!r} is "
                f"not a vocoder; expected one of {sorted(VOCODER_ARCHITECTURES)}")
        if not self.needs_vocoder:
            raise SystemExit(
                f"{self.root}: {self.manifest['architecture']} is end to end "
                "and must not be given a vocoder")
        if int(voc.sample_rate) != int(self.sample_rate):
            raise SystemExit(
                f"sample rate mismatch: {self.manifest['run_id']} is "
                f"{self.sample_rate} Hz and {voc.manifest['run_id']} is "
                f"{voc.sample_rate} Hz. A mel on the wrong rate sounds like "
                "speech and is wrong.")
        if voc.manifest["language"] != self.manifest["language"]:
            raise SystemExit(
                f"language mismatch: {self.manifest['run_id']} is "
                f"{self.manifest['language']} and {voc.manifest['run_id']} is "
                f"{voc.manifest['language']}. Each language has its own "
                "vocoder so a speaker mismatch cannot enter the control.")
        self.vocoder = voc
        return self

    def vocode(self, mel):
        """Waveform from a log-mel, shaped [n_mels, T].

        coqui's generator takes [B, n_mels, T]. The mel this project produces
        and the mel `synthesize` returns are both [n_mels, T], so the batch
        axis is added here and nowhere else.
        """
        # Every check runs BEFORE torch is imported. Guards behind a heavy
        # import are untestable without the package and slow to fail with it,
        # which is the same mistake src/export/griffinlim.py already had.
        import numpy as np

        if not self.is_vocoder:
            raise SystemExit(f"{self.root}: not a vocoder bundle")
        arr = np.asarray(mel, dtype="float32")
        if arr.ndim != 2:
            raise ValueError(f"mel must be 2-D [n_mels, T], got {arr.shape}")
        # The band count lives under "audio", not at the top level. A
        # top-level .get with a default of 80 would have agreed with the real
        # value by accident and stopped checking the day it changed.
        audio = self.manifest.get("audio") or {}
        if "n_mels" not in audio:
            raise SystemExit(
                f"{self.root}: manifest has no audio.n_mels, so the mel this "
                "vocoder expects cannot be checked")
        want = int(audio["n_mels"])
        if arr.shape[0] != want:
            raise ValueError(
                f"mel has {arr.shape[0]} bands, this vocoder expects {want}. "
                "A transposed mel is the usual cause.")

        import torch
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.inference(x)
        return out.detach().cpu().reshape(-1).numpy().astype("float32")

    def synthesize(self, text: str, draw: int = 0,
                   seed: int | None = None) -> Speech:
        """Synthesise `text`, reproducibly.

        **Why this takes a seed.** VITS samples: its stochastic duration
        predictor and its flow both draw noise at inference, so two calls on
        the same text give two different utterances. Measured on 3 October by
        running scripts/score_intelligibility.py twice with identical
        arguments, code, weights and ten utterances: the character error rate
        of r05 under IndicConformer moved from 0.1330 to 0.1478, its
        medial-site errors fell from 16 of 96 characters to 5, and the
        medial-site excess that the two recognisers had agreed on at +0.1147
        and +0.1150 came back as -0.0569 and +0.0661, reversing sign on one of
        them. The ground-truth floor was byte-identical across both passes,
        because it reads fixed files. An unseeded measurement of a sampling
        model is not a measurement of the model; it is one draw from it.

        The seed is derived from the text and the draw index rather than taken
        from a global RNG, so one utterance's result does not depend on how
        many utterances were synthesised before it, and a re-run of a subset
        reproduces the same audio as the full set. `draw` selects which sample,
        so a caller that wants the distribution asks for several and reports
        the spread; the default of 0 is one fixed, reproducible draw rather
        than a claim that one draw is enough.

        A deterministic architecture ignores all of this, which is why the
        seeding is unconditional: it costs nothing there and removes a
        difference between architectures that is otherwise silent.
        """
        import hashlib

        import numpy as np
        import torch

        if self.encoder is None:
            raise SystemExit(f"{self.root}: this bundle has no text front end")
        if not text.strip():
            raise ValueError("nothing to synthesise")

        toks = self.encoder.tokens(text)
        ids = self.encoder.encode(text)
        x = torch.LongTensor([ids]).to(self.device)

        if seed is None:
            seed = draw_seed(self.manifest["run_id"], text, draw)
        torch.manual_seed(seed)
        # `device` is whatever the caller passed, and every caller passes a
        # string: `.to("cpu")` works, so nothing had ever needed it to be a
        # torch.device. Asking it for `.type` raised AttributeError on all 250
        # synthesis calls per run on 4 October, and because the harness logs
        # only its first two tracebacks the rest failed in silence and the run
        # reported "nothing transcribed". Tested on the string.
        if str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.inference(x)
        elapsed = time.perf_counter() - t0

        got = out["model_outputs"].detach().cpu()
        sp = Speech(run_id=self.manifest["run_id"], text=text, tokens=toks,
                    sample_rate=self.sample_rate, seconds_elapsed=elapsed,
                    needs_vocoder=self.needs_vocoder)

        if self.needs_vocoder:
            # [B, T_frames, n_mels] from coqui -> [n_mels, T_frames], the
            # orientation the vocoder and features.compute both use.
            sp.mel = got[0].transpose(0, 1).numpy().astype(np.float32)
            sp.extras["mel_params"] = mel_params(self.sample_rate)
            if self.vocoder is not None:
                # The mel stays on the Speech alongside the waveform. The
                # duration measure reads frame counts and must not start
                # depending on whether a vocoder happened to be attached.
                sp.waveform = self.vocoder.vocode(sp.mel)
                sp.extras["vocoder"] = {
                    "run_id": self.vocoder.manifest["run_id"],
                    "step": self.vocoder.manifest.get("step"),
                    "config_hash": self.vocoder.manifest.get("config_hash"),
                    "bundle": self.vocoder.root.name,
                }
        else:
            # [B, 1, N] -> [N]
            sp.waveform = got.reshape(-1).numpy().astype(np.float32)
        if "durations" in out:
            sp.extras["frames"] = int(out["durations"].sum().item())
        return sp


def load(root: pathlib.Path | str, device: str = "cpu",
         vocoder: pathlib.Path | str | None = None) -> Bundle:
    """A bundle, with its vocoder attached when one is given.

    `vocoder` is another bundle directory, not a bare checkpoint: a vocoder
    exported as a bundle carries its own manifest, so the rate, language and
    provenance checks in `attach_vocoder` have something to check against.
    """
    b = Bundle(pathlib.Path(root), device=device)
    if vocoder:
        b.attach_vocoder(Bundle(pathlib.Path(vocoder), device=device))
    return b


def write_wav(path: pathlib.Path, waveform, sample_rate: int) -> None:
    """Write a float32 waveform. soundfile if present, else the stdlib."""
    import numpy as np

    w = np.clip(np.asarray(waveform, dtype="float32"), -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(str(path), w, sample_rate)
        return
    except ImportError:
        pass
    import wave
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes((w * 32767.0).astype("<i2").tobytes())


# -- command line -----------------------------------------------------------

DEFAULT_BENCH = [
    "नमक कमल",
    "आज सुबह बहुत ठंड थी।",
    "यह वाक्य थोड़ा लंबा है और इसमें कई शब्द हैं, ताकि गति मापी जा सके।",
]


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=pathlib.Path)
    ap.add_argument("--text")
    ap.add_argument("--out", type=pathlib.Path,
                    help="wav for an end-to-end model, .npy for a mel")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--benchmark", action="store_true",
                    help="time three sentences, after one warm-up pass")
    a = ap.parse_args(argv)

    b = load(a.bundle, device=a.device)
    print(f"{b.manifest['run_id']} step {b.manifest.get('step')}  "
          f"{b.loaded_tensors} tensors  {b.sample_rate} Hz  "
          f"{'mel, needs a vocoder' if b.needs_vocoder else 'end to end'}")
    print(f"  {b.describes}")

    if a.benchmark:
        b.synthesize(DEFAULT_BENCH[0])            # warm up, not counted
        print(f"  {'chars':>6} {'tokens':>7} {'audio s':>9} "
              f"{'wall s':>8} {'xRT':>6}")
        for text in DEFAULT_BENCH:
            sp = b.synthesize(text)
            dur = sp.audio_seconds
            rtf = sp.realtime_factor
            print(f"  {len(text):>6} {len(sp.tokens):>7} "
                  f"{(f'{dur:.2f}' if dur else '-'):>9} "
                  f"{sp.seconds_elapsed:>8.2f} "
                  f"{(f'{rtf:.2f}' if rtf else '-'):>6}")
        return 0

    if not a.text:
        ap.error("--text is required unless --benchmark is given")
    sp = b.synthesize(a.text)
    print(f"  {len(sp.tokens)} tokens: {' '.join(sp.tokens[:24])}"
          f"{' ...' if len(sp.tokens) > 24 else ''}")
    if sp.waveform is not None:
        print(f"  {sp.audio_seconds:.2f} s of audio in "
              f"{sp.seconds_elapsed:.2f} s ({sp.realtime_factor:.2f}x realtime)")
        out = a.out or pathlib.Path(f"{sp.run_id}.wav")
        write_wav(out, sp.waveform, sp.sample_rate)
        print(f"  wrote {out}")
    else:
        print(f"  mel {sp.mel.shape} (n_mels, frames) in "
              f"{sp.seconds_elapsed:.2f} s; silent without a vocoder")
        out = a.out or pathlib.Path(f"{sp.run_id}_mel.npy")
        np.save(out, sp.mel)
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
