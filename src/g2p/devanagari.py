"""Devanagari grapheme sequence to a phone sequence with inherent schwas intact.

This stage is deliberately dumb about schwa. It emits every inherent schwa the
orthography implies, including the ones Hindi deletes. Deletion happens in
schwa.py, as a separate, switchable, testable stage. That separation is what
makes the Marathi control possible: same converter, deletion stage disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .phoneset import (
    ANUSVARA,
    AVAGRAHA,
    CHANDRABINDU,
    CONSONANTS,
    HOMORGANIC_NASAL,
    INDEPENDENT_VOWELS,
    INHERENT_VOWEL,
    NASALISATION,
    NUKTA_CONSONANTS,
    NUKTA_MERGE,
    VIRAMA,
    VISARGA,
    VISARGA_PHONE,
    VOWEL_SIGNS,
)


@dataclass
class Segment:
    """One phone plus the provenance needed to debug and to apply schwa rules.

    ``inherent`` marks a schwa that the orthography never wrote down. Only
    those are candidates for deletion. A schwa written as a matra or as the
    independent vowel is never deleted.
    """

    phone: str
    kind: str                      # "consonant" | "vowel" | "other"
    inherent: bool = False
    source_index: int = -1         # index into the input word, for tracing
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        tag = "*" if self.inherent else ""
        return f"{self.phone}{tag}"


def _consonant_phone(ch: str, merge_nukta: bool) -> str | None:
    if ch in NUKTA_CONSONANTS:
        phone = NUKTA_CONSONANTS[ch]
        return NUKTA_MERGE[phone] if merge_nukta else phone
    return CONSONANTS.get(ch)


def word_to_segments(word: str, merge_nukta: bool = False) -> list[Segment]:
    """Convert one orthographic word to segments, schwas included.

    Handles: consonant plus matra, consonant plus virama (cluster), bare
    consonant (inherent schwa), independent vowels, anusvara with homorganic
    assimilation, chandrabindu, visarga, avagraha.
    """
    segs: list[Segment] = []
    i = 0
    n = len(word)

    while i < n:
        ch = word[i]

        cons = _consonant_phone(ch, merge_nukta)
        if cons is not None:
            segs.append(Segment(cons, "consonant", source_index=i))
            nxt = word[i + 1] if i + 1 < n else ""

            if nxt == VIRAMA:
                # Explicit cluster: no vowel follows this consonant.
                i += 2
                continue
            if nxt in VOWEL_SIGNS:
                segs.append(Segment(VOWEL_SIGNS[nxt], "vowel", source_index=i + 1))
                i += 2
                continue
            # Nothing written: the consonant carries the inherent schwa.
            segs.append(
                Segment(INHERENT_VOWEL, "vowel", inherent=True, source_index=i)
            )
            i += 1
            continue

        if ch in INDEPENDENT_VOWELS:
            segs.append(Segment(INDEPENDENT_VOWELS[ch], "vowel", source_index=i))
            i += 1
            continue

        if ch == ANUSVARA:
            # Homorganic nasal before a stop, otherwise vowel nasalisation.
            following = _next_consonant_phone(word, i + 1, merge_nukta)
            if following in HOMORGANIC_NASAL:
                segs.append(
                    Segment(HOMORGANIC_NASAL[following], "consonant", source_index=i,
                            notes=["anusvara->homorganic"])
                )
            else:
                _nasalise_previous_vowel(segs, "anusvara->nasalisation")
            i += 1
            continue

        if ch == CHANDRABINDU:
            _nasalise_previous_vowel(segs, "chandrabindu")
            i += 1
            continue

        if ch == VISARGA:
            segs.append(Segment(VISARGA_PHONE, "consonant", source_index=i,
                                notes=["visarga"]))
            i += 1
            continue

        if ch == AVAGRAHA:
            i += 1
            continue

        if ch == VIRAMA:
            # A stray virama with no consonant before it. Skip, but record it.
            if segs:
                segs[-1].notes.append("stray-virama")
            i += 1
            continue

        # Anything else (Latin, punctuation, digits) is left for the caller.
        segs.append(Segment(ch, "other", source_index=i))
        i += 1

    return segs


def _next_consonant_phone(word: str, start: int, merge_nukta: bool) -> str | None:
    for j in range(start, len(word)):
        phone = _consonant_phone(word[j], merge_nukta)
        if phone is not None:
            return phone
        if word[j] in INDEPENDENT_VOWELS or word[j] in VOWEL_SIGNS:
            return None
    return None


def _nasalise_previous_vowel(segs: list[Segment], note: str) -> None:
    for seg in reversed(segs):
        if seg.kind == "vowel":
            if not seg.phone.endswith(NASALISATION):
                seg.phone = seg.phone + NASALISATION
            seg.notes.append(note)
            return
    # No preceding vowel: nasalisation has nothing to attach to. Record it
    # rather than silently dropping, so corpus oddities surface in QA.
    if segs:
        segs[-1].notes.append(f"{note}-orphan")


def segments_to_phones(segs: list[Segment]) -> list[str]:
    return [s.phone for s in segs if s.phone]
