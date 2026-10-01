#!/usr/bin/env python3
"""Text to token ids, and the vocabulary that pins them down.

The ablation compares phonemic against graphemic input. If the two arms built
their vocabularies differently, or if a vocabulary drifted between training and
inference, the comparison would measure bookkeeping rather than linguistics.
Three rules keep it honest.

**One vocabulary per (language, input representation), built from the
inventory, not from the corpus.** A corpus-derived vocabulary changes when the
ladder rung changes, so the 10-minute run and the 9-hour run would have
different embedding tables and different vocabulary sizes. Building from the
declared inventory makes every rung share one table. A symbol that never occurs
in 10 minutes of speech simply gets an untrained embedding, which is the honest
outcome.

**The vocabulary is written into the checkpoint directory.** Resuming a run, or
synthesising from a checkpoint months later, reads the ids from the file that
trained them rather than rebuilding and hoping the order matched.

**Ids are assigned by sorted symbol order, not by insertion order.** Insertion
order depends on a dict traversal somewhere upstream, which is the kind of
thing that silently changes between Python versions.

Reserved ids 0 to 2 are pad, bos and eos. Pad must be 0 so an all-zero padded
tensor is unambiguous.
"""

from __future__ import annotations

import json
import pathlib

from ..g2p import G2P
from ..g2p.phoneset import PUNCTUATION, SPECIALS, WORD_BOUNDARY, build_inventory

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
RESERVED = [PAD, BOS, EOS]

# Devanagari block plus the punctuation and the word boundary. The grapheme arm
# is character level over the SAME normalised text the phoneme arm sees, so the
# normaliser cannot be what separates the two arms.
DEVANAGARI_RANGE = range(0x0900, 0x0980)


def grapheme_inventory() -> list[str]:
    chars = [chr(c) for c in DEVANAGARI_RANGE]
    return sorted(set(chars) | set(PUNCTUATION) | {WORD_BOUNDARY})


def phoneme_inventory(merge_nukta: bool = False) -> list[str]:
    return sorted(set(build_inventory(merge_nukta=merge_nukta)) | set(SPECIALS))


class Vocab:
    """Symbol table. Immutable once built."""

    def __init__(self, symbols: list[str]) -> None:
        extra = [s for s in symbols if s not in RESERVED]
        self.symbols: list[str] = RESERVED + sorted(set(extra))
        self.ids: dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        if self.ids[PAD] != 0:
            raise AssertionError("pad must be id 0")

    def __len__(self) -> int:
        return len(self.symbols)

    @classmethod
    def build(cls, input_repr: str, merge_nukta: bool = False) -> "Vocab":
        if input_repr == "phoneme":
            return cls(phoneme_inventory(merge_nukta=merge_nukta))
        if input_repr == "grapheme":
            return cls(grapheme_inventory())
        raise ValueError(f"no vocabulary for input_repr={input_repr!r}")

    def encode(self, tokens: list[str], bos_eos: bool = True) -> list[int]:
        """Map symbols to ids.

        An unknown symbol raises rather than falling back to <unk>. The
        coverage run already proved zero unknown symbols over 870,547 phone
        tokens, so an unknown one here means the front end or the normaliser
        changed, and silently substituting <unk> would hide that.
        """
        out: list[int] = []
        if bos_eos:
            out.append(self.ids[BOS])
        for t in tokens:
            try:
                out.append(self.ids[t])
            except KeyError:
                raise KeyError(
                    f"symbol {t!r} is not in the {len(self)}-symbol vocabulary; "
                    "the front end and the vocabulary have diverged"
                ) from None
        if bos_eos:
            out.append(self.ids[EOS])
        return out

    def decode(self, ids: list[int]) -> list[str]:
        return [self.symbols[i] for i in ids
                if self.symbols[i] not in (PAD, BOS, EOS)]

    # -- persistence --------------------------------------------------------

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"symbols": self.symbols}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "Vocab":
        symbols = json.loads(path.read_text(encoding="utf-8"))["symbols"]
        v = cls.__new__(cls)
        v.symbols = symbols
        v.ids = {s: i for i, s in enumerate(symbols)}
        if v.ids.get(PAD) != 0:
            raise AssertionError(f"{path}: pad is not id 0; the table is corrupt")
        return v


class TextEncoder:
    """The one place a config's input_repr turns into integers.

    Holds the front end and the vocabulary together so a run cannot pair a
    phoneme vocabulary with a grapheme tokeniser.
    """

    def __init__(self, language: str, input_repr: str, vocab: Vocab,
                 merge_nukta: bool = False) -> None:
        if input_repr not in ("phoneme", "grapheme"):
            raise ValueError(f"unknown input_repr {input_repr!r}")
        self.language = language
        self.input_repr = input_repr
        self.vocab = vocab
        self.g2p = G2P.for_language(language, merge_nukta=merge_nukta)

    @classmethod
    def for_config(cls, language: str, input_repr: str,
                   merge_nukta: bool = False) -> "TextEncoder":
        return cls(language, input_repr,
                   Vocab.build(input_repr, merge_nukta=merge_nukta),
                   merge_nukta=merge_nukta)

    def tokens(self, text: str) -> list[str]:
        if self.input_repr == "phoneme":
            return self.g2p.phonemize(text)
        return self.g2p.graphemes(text)

    def encode(self, text: str) -> list[int]:
        return self.vocab.encode(self.tokens(text))
