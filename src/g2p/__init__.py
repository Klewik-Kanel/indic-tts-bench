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
from .numbers import expand as _expand_numbers
from .devanagari import Segment, word_to_segments
from .phoneset import PUNCTUATION, WORD_BOUNDARY, build_inventory
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
    # Digit strings the number expander could not handle, collected rather
    # than dropped so a coverage run can count them.
    unexpanded_numbers: list[str] = field(default_factory=list)

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
        """One orthographic word to phones. Lexicon wins over rules.

        Punctuation is peeled off the edges before the schwa stage and put
        back afterwards. Leaving it attached breaks the rule in two ways at
        once: a trailing comma means the word-final schwa is no longer final,
        so it survives, and its survival changes the context of the schwa to
        its left, which can then delete. नमक is /nəmək/ but नमक, came out as
        /nəmkə/ — a different word. 16.5% of Hindi word tokens and 14.9% of
        Marathi carry punctuation, and the damage would have landed on the
        phoneme arm alone, which is one half of the comparison this study
        exists to make.
        """
        phones, _ = self._phonemize_word_with_sites(word)
        return phones

    def _phonemize_word_with_sites(self, word: str) -> tuple[list[str], list[int]]:
        """The body of `phonemize_word`, keeping the deletion sites.

        `delete_schwas` already returns which segment indices it deleted, and
        the only reason that was unavailable outside this function was that it
        was discarded here. Analysis needs the sites — selecting demo sentences
        by how much schwa deletion they actually exercise, and scoring
        predicted against gold deletion sites — and the one thing worse than
        not having them is a second copy of this punctuation-and-lexicon
        handling that can drift away from the path that trained the models.

        Sites index the pre-deletion segments of the core word, so they are
        comparable to `segments()` and unaffected by the punctuation peeled
        off either end. A lexicon hit reports no sites, because no schwa rule
        ran: that is the real answer, not a missing one.
        """
        word = _norm.normalize(word)
        if not word:
            return [], []

        lead, core, trail = [], word, []
        while core and core[0] in PUNCTUATION:
            lead.append(core[0])
            core = core[1:]
        while core and core[-1] in PUNCTUATION:
            trail.insert(0, core[-1])
            core = core[:-1]
        if not core:
            return lead + trail, []

        if core in self.lexicon:
            return lead + list(self.lexicon[core]) + trail, []
        segs = word_to_segments(core, merge_nukta=self.merge_nukta)
        survivors, deleted = delete_schwas(segs, self.schwa)
        phones = lead + [s.phone for s in survivors if s.phone] + trail
        return phones, deleted

    def deletion_sites(self, word: str) -> list[int]:
        """Which pre-deletion segment indices the schwa rules removed."""
        return self._phonemize_word_with_sites(word)[1]

    def phonemize(self, text: str) -> list[str]:
        """Sentence to phones, with explicit word boundaries.

        Digits are expanded to number words first. A digit reaching the phone
        sequence would become an embedding row trained on a handful of
        examples, and the model would have no way to pronounce it.
        """
        text, unexpanded = _expand_numbers(_norm.normalize(text), self.language)
        if unexpanded:
            self.unexpanded_numbers.extend(unexpanded)
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

        Numbers are expanded first, exactly as `phonemize` does. This is not a
        convenience: `normalize` runs `devanagari_digits_to_ascii`, and the
        grapheme inventory is the U+0900 block plus punctuation, so an ASCII
        digit is outside it by construction. Leaving the expansion out raised
        on the first Marathi transcript containing a numeral, and, worse, meant
        the phoneme arm heard a spoken number while this arm saw a digit glyph.
        The two arms must be handed the same string or the ablation measures
        the front end.
        """
        text, unexpanded = _expand_numbers(_norm.normalize(text), self.language)
        if unexpanded:
            self.unexpanded_numbers.extend(unexpanded)
        out: list[str] = []
        for i, word in enumerate(text.split()):
            if i > 0 and out:
                out.append(WORD_BOUNDARY)
            out.extend(list(word))
        return out

    # -- symbol table -------------------------------------------------------

    def inventory(self) -> list[str]:
        return build_inventory(merge_nukta=self.merge_nukta)
