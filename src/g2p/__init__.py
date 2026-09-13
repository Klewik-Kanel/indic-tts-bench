"""Devanagari grapheme-to-phoneme front end.

Public surface:

    g2p = G2P.for_language("hindi")
    g2p.phonemize("नमकीन कमल")        -> ['n', 'ə', 'm', 'k', 'iː', 'n', '|', ...]
    g2p.phonemize_word("नमकीन")
    g2p.graphemes("नमकीन कमल")         -> character-level tokens, the ablation arm

The grapheme path exists so that the phoneme-vs-grapheme comparison runs
through the same normaliser and the same token conventions. If the two arms
differed in normalisation, the ablation would measure the normaliser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import normalize as _norm
from .devanagari import Segment, word_to_segments
from .phoneset import WORD_BOUNDARY, build_inventory
from .schwa import HINDI, MARATHI, NO_DELETION, SchwaConfig, delete_schwas

__all__ = [
    "G2P",
    "Segment",
    "SchwaConfig",
    "HINDI",
    "MARATHI",
    "NO_DELETION",
    "build_inventory",
]

LANGUAGE_CONFIGS: dict[str, SchwaConfig] = {
    "hindi": HINDI,
    "hi": HINDI,
    "marathi": MARATHI,
    "mr": MARATHI,
    "none": NO_DELETION,
}


@dataclass
class G2P:
    """Configured front end for one language."""

    language: str
    schwa: SchwaConfig
    merge_nukta: bool = False
    lexicon: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def for_language(
        cls,
        language: str,
        merge_nukta: bool = False,
        lexicon: dict[str, list[str]] | None = None,
    ) -> "G2P":
        key = language.strip().lower()
        if key not in LANGUAGE_CONFIGS:
            raise ValueError(
                f"unknown language {language!r}; known: {sorted(LANGUAGE_CONFIGS)}"
            )
        return cls(
            language=key,
            schwa=LANGUAGE_CONFIGS[key],
            merge_nukta=merge_nukta,
            lexicon=dict(lexicon or {}),
        )

    # -- phoneme path -------------------------------------------------------

    def phonemize_word(self, word: str) -> list[str]:
        """One orthographic word to phones. Lexicon wins over rules."""
        word = _norm.normalize(word)
        if not word:
            return []
        if word in self.lexicon:
            return list(self.lexicon[word])
        segs = word_to_segments(word, merge_nukta=self.merge_nukta)
        survivors, _ = delete_schwas(segs, self.schwa)
        return [s.phone for s in survivors if s.phone]

    def phonemize(self, text: str) -> list[str]:
        """Sentence to phones, with explicit word boundaries."""
        text = _norm.normalize(text)
        out: list[str] = []
        for i, word in enumerate(text.split()):
            phones = self.phonemize_word(word)
            if not phones:
                continue
            if i > 0 and out:
                out.append(WORD_BOUNDARY)
            out.extend(phones)
        return out

    def segments(self, word: str) -> list[Segment]:
        """Pre-deletion segments, for inspection and for stress-test scoring."""
        return word_to_segments(_norm.normalize(word), merge_nukta=self.merge_nukta)

    # -- grapheme path, the ablation arm ------------------------------------

    def graphemes(self, text: str) -> list[str]:
        """Character tokens after the identical normalisation chain.

        Combining marks stay attached to nothing: each Unicode character is one
        token. That is the usual grapheme baseline, and it is what a model
        trained on raw text would see.
        """
        text = _norm.normalize(text)
        out: list[str] = []
        for i, word in enumerate(text.split()):
            if i > 0 and out:
                out.append(WORD_BOUNDARY)
            out.extend(list(word))
        return out

    # -- symbol table -------------------------------------------------------

    def inventory(self) -> list[str]:
        return build_inventory(merge_nukta=self.merge_nukta)
