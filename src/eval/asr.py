#!/usr/bin/env python3
"""Automatic speech recognition, used as an intelligibility proxy.

The measure is: synthesise the test sentences, transcribe the synthesis, and
compare the transcript with the text that was fed in. A system whose speech is
hard to understand produces a worse transcript. That is the whole idea, and it
is only sound if the recogniser does not quietly fix the errors being counted.

**Why connectionist temporal classification and not Whisper.** A
sequence-to-sequence recogniser carries an implicit language model in its
decoder. Hand it a slurred Hindi word and it will emit the correct spelling
anyway, because the spelling is what its training data makes likely. The schwa
contrast this project exists to measure is exactly the kind of thing such a
decoder repairs, so a stronger sequence-to-sequence model is a WEAKER
instrument here. A CTC model decoded greedily, with no external language
model, is frame-local: it reports what it heard. Both backends below are CTC,
decoded greedily, and that is a methodological choice rather than a
convenience.

**Two backends, two training sets, no averaging.** IndicConformer is primary:
MIT licensed, built for 22 Indian languages, Devanagari output. MMS is the
cross-check: a different architecture trained on different data. If the two
disagree about which arm is more intelligible, the disagreement is reported.
Averaging two recognisers would hide precisely the instrument-dependence that
makes an ASR proxy worth distrusting.

**Both are pinned by revision.** `trust_remote_code=True` on IndicConformer
executes code from that repository at load time, so an unpinned load is a
result that cannot be reproduced once upstream moves. The revisions here were
read from the Hugging Face API on 3 October 2026 and are recorded beside the
run's config_hash in the results table.

**Licences differ and both go in the methods section.** IndicConformer is MIT.
MMS is CC-BY-NC-4.0, which covers academic use and forbids commercial use.

**Sample rate.** Both want 16 kHz mono. The corpus already has
data/interim/<lang>/wav16, so REFERENCES are never resampled by this module.
Synthesis comes out at the run's own rate, 22.05 kHz for the VITS arms, so only
HYPOTHESES are resampled, and `RESAMPLER` records how. Resampling both would
put a filter between the reference and itself; resampling neither would feed
the recogniser audio at the wrong rate. The asymmetry is deliberate and is
stated in the table.
"""

from __future__ import annotations

import unicodedata

TARGET_SR = 16000

# How a hypothesis gets to TARGET_SR. soxr_hq is librosa's default high quality
# resampler; naming it here means the results table can state it rather than
# leaving it as whatever the installed default happened to be.
RESAMPLER = "soxr_hq"

# Read from https://huggingface.co/api/models/<id> on 2026-10-03. A load
# without one of these is refused by `backend`, because an unpinned revision
# makes the number it produces unreproducible.
MODELS = {
    "indicconformer": {
        "repo": "ai4bharat/indic-conformer-600m-multilingual",
        "revision": "e9b71b369c048e2c6b634d4c131061c34e441179",
        "licence": "MIT",
        "gated": True,          # gating is "auto": accepting the terms suffices
        "role": "primary",
        "runtime": "onnxruntime via trust_remote_code",
        "lang_arg": {"hindi": "hi", "marathi": "mr"},
    },
    "mms": {
        "repo": "facebook/mms-1b-all",
        "revision": "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
        "licence": "CC-BY-NC-4.0",
        "gated": False,
        "role": "cross-check",
        "runtime": "transformers Wav2Vec2ForCTC",
        # MMS uses three-letter codes. "hin", not "hi": the two-letter code
        # loads a DIFFERENT adapter or none at all, and the failure is a
        # transcript in the wrong language rather than an exception.
        "lang_arg": {"hindi": "hin", "marathi": "mar"},
    },
}


def model_card(name: str) -> dict:
    """The pinned record for one backend, for the results table.

    Separate from loading it so the table can be written, and the licences
    cited, on a machine that has neither model downloaded.
    """
    if name not in MODELS:
        raise KeyError(f"unknown ASR backend {name!r}; have {sorted(MODELS)}")
    return dict(MODELS[name])


def lang_code(name: str, language: str) -> str:
    card = model_card(name)
    codes = card["lang_arg"]
    if language not in codes:
        raise KeyError(
            f"{name} has no recorded code for {language!r}; have {sorted(codes)}")
    return codes[language]


# --- the comparison --------------------------------------------------------

def _chars(text: str, drop_spaces: bool) -> list[str]:
    from src.g2p import normalize as _norm
    text = _norm.normalize(text)
    text = unicodedata.normalize("NFC", text)
    if drop_spaces:
        text = "".join(text.split())
    return list(text)


def edit_distance(a, b) -> int:
    """Levenshtein distance, two rows rather than a full matrix.

    Written here rather than taken from a dependency because the error rate is
    a headline number and a silent change in someone else's tie-breaking would
    move it.
    """
    a, b = list(a), list(b)
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str, drop_spaces: bool = True) -> float:
    """Character error rate, on text both sides of which went through
    `normalize()`.

    Both sides are normalised with the project's own chain, so the number is
    not partly measuring digit expansion or nukta composition: a reference
    written with a decomposed nukta and a transcript written with a composed
    one are the same string here, as they are to a listener.

    `drop_spaces` defaults to True, which is the choice that matters. Devanagari
    word boundaries interact with the sandhi and schwa phenomena under test, so
    a rate that counts spaces is partly measuring this project's own
    tokenisation rather than the audio. Both are reported; the space-free one
    leads.

    An empty reference has no rate. It raises rather than returning 0.0, which
    would read as a perfect score.
    """
    ref = _chars(reference, drop_spaces)
    hyp = _chars(hypothesis, drop_spaces)
    if not ref:
        raise ValueError("empty reference after normalisation: no rate exists")
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, on whitespace tokens after `normalize()`.

    Reported beside CER rather than instead of it, for comparability with
    published numbers. It is the weaker instrument here for the reason in
    `cer`: one slurred schwa inside a word costs a whole word.
    """
    from src.g2p import normalize as _norm
    ref = _norm.normalize(reference).split()
    hyp = _norm.normalize(hypothesis).split()
    if not ref:
        raise ValueError("empty reference after normalisation: no rate exists")
    return edit_distance(ref, hyp) / len(ref)


def corpus_cer(pairs, drop_spaces: bool = True) -> float:
    """CER over a whole set, pooled rather than averaged per utterance.

    A mean of per-utterance rates weights a three-character utterance like a
    forty-character one. The pooled rate is total edits over total reference
    characters, which is what every published CER means.
    """
    edits = chars = 0
    for reference, hypothesis in pairs:
        ref = _chars(reference, drop_spaces)
        hyp = _chars(hypothesis, drop_spaces)
        if not ref:
            raise ValueError("empty reference after normalisation")
        edits += edit_distance(ref, hyp)
        chars += len(ref)
    if not chars:
        raise ValueError("no reference characters: no rate exists")
    return edits / chars


# --- resampling ------------------------------------------------------------

def to_target_sr(wav, sample_rate: int):
    """A hypothesis waveform at TARGET_SR, mono, float32.

    Returns the array unchanged when it is already at the target rate, so a
    16 kHz run is not pushed through a filter the 22.05 kHz runs needed.
    """
    import numpy as np
    wav = np.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[-1] else -1)
    if int(sample_rate) == TARGET_SR:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=int(sample_rate), target_sr=TARGET_SR,
                            res_type=RESAMPLER).astype("float32")


# --- backends --------------------------------------------------------------

class _Backend:
    name = ""

    def __init__(self, language: str, device: str = "cpu"):
        self.language = language
        self.device = device
        self.card = model_card(self.name)
        self.code = lang_code(self.name, language)
        self.revision = self.card["revision"]
        self._loaded = None

    def describe(self) -> dict:
        return {"backend": self.name, "repo": self.card["repo"],
                "revision": self.revision, "licence": self.card["licence"],
                "role": self.card["role"], "lang_arg": self.code,
                "decoding": "ctc-greedy", "target_sr": TARGET_SR,
                "resampler": RESAMPLER}

    def transcribe(self, wav, sample_rate: int) -> str:
        raise NotImplementedError


class IndicConformerCTC(_Backend):
    """AI4Bharat's IndicConformer, CTC head, greedy.

    The checkpoint is ONNX behind `trust_remote_code`, so the revision pin is
    load-bearing: it fixes the executed code, not only the weights. The
    repository ships ONE ctc_decoder.onnx and a per-language joint_post_net for
    the RNNT head, so under CTC the language argument may select nothing at
    all. It is passed as documented and recorded in `describe`; what it changes
    is a question for the first run, not an assumption made here.
    """

    name = "indicconformer"

    def _model(self):
        if self._loaded is None:
            import torch
            from transformers import AutoModel
            m = AutoModel.from_pretrained(self.card["repo"],
                                          revision=self.revision,
                                          trust_remote_code=True)
            if hasattr(m, "eval"):
                m.eval()
            self._loaded = (m, torch)
        return self._loaded

    def transcribe(self, wav, sample_rate: int) -> str:
        m, torch = self._model()
        y = to_target_sr(wav, sample_rate)
        t = torch.from_numpy(y).unsqueeze(0)          # [1, samples], mono
        with torch.no_grad():
            out = m(t, self.code, "ctc")
        return out if isinstance(out, str) else str(out[0])


class MmsCtc(_Backend):
    """Meta's MMS-1B-all, the Hindi adapter, argmax over CTC logits.

    `ignore_mismatched_sizes=True` is required: the language head is resized to
    the adapter's vocabulary and transformers warns about the shapes. The
    warning is expected and is not evidence of a broken load.
    """

    name = "mms"

    def _model(self):
        if self._loaded is None:
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            proc = AutoProcessor.from_pretrained(
                self.card["repo"], revision=self.revision, target_lang=self.code)
            model = Wav2Vec2ForCTC.from_pretrained(
                self.card["repo"], revision=self.revision, target_lang=self.code,
                ignore_mismatched_sizes=True)
            model.eval().to(self.device)
            self._loaded = (proc, model, torch)
        return self._loaded

    def transcribe(self, wav, sample_rate: int) -> str:
        proc, model, torch = self._model()
        y = to_target_sr(wav, sample_rate)
        inputs = proc(y, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(self.device)
                              for k, v in inputs.items()}).logits
        ids = torch.argmax(logits, dim=-1)[0]
        return proc.decode(ids)


_BACKENDS = {IndicConformerCTC.name: IndicConformerCTC, MmsCtc.name: MmsCtc}


def backend(name: str, language: str, device: str = "cpu") -> _Backend:
    """One backend by name, with its revision already pinned.

    The registry is the only way in, so a backend cannot be constructed
    without a recorded revision and licence.
    """
    if name not in _BACKENDS:
        raise KeyError(f"unknown ASR backend {name!r}; have {sorted(_BACKENDS)}")
    if not MODELS[name]["revision"]:
        raise SystemExit(
            f"{name} has no pinned revision. Read it from "
            f"https://huggingface.co/api/models/{MODELS[name]['repo']} and put "
            f"the sha in MODELS before scoring anything with it.")
    return _BACKENDS[name](language, device=device)
