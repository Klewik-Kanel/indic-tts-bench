"""Devanagari text normalisation, shared by every language in the study.

Runs before G2P and before any model sees text. Deterministic and reversible
enough to debug: every transformation is a named step so a surprising
pronunciation can be traced back to the step that caused it.
"""

from __future__ import annotations

import re
import unicodedata

from .phoneset import DIGITS, NUKTA

# Canonical nukta compositions. Devanagari allows both a precomposed character
# and a base plus combining nukta. Two spellings of the same word would become
# two different token sequences, so they are collapsed here.
# Targets are explicit codepoints: U+0958..U+095F are composition exclusions,
# so a literal pasted into source is often already decomposed and the mapping
# would be a silent no-op.
NUKTA_COMPOSE: dict[str, str] = {
    "क" + NUKTA: "क़",   # क़
    "ख" + NUKTA: "ख़",   # ख़
    "ग" + NUKTA: "ग़",   # ग़
    "ज" + NUKTA: "ज़",   # ज़
    "ड" + NUKTA: "ड़",   # ड़
    "ढ" + NUKTA: "ढ़",   # ढ़
    "फ" + NUKTA: "फ़",   # फ़
    "य" + NUKTA: "य",   # य़ has no distinct Hindi pronunciation
}

# Characters that carry no pronunciation but appear in corpora.
ZERO_WIDTH = "​‌‍﻿­"

# Danda and double danda behave as sentence punctuation.
DANDA = "।"
DOUBLE_DANDA = "॥"

# Every mark folds into the four the model actually gets an embedding for.
# Quotation marks carry no prosody of their own and are dropped; a hyphen
# inside a compound (बार-बार, पुढे-मागे) is a word boundary, not a pause.
PUNCT_MAP: dict[str, str] = {
    DANDA: ".", DOUBLE_DANDA: ".",
    "‘": "", "’": "", "“": "", "”": "", "'": "", '"': "",
    "–": ",", "—": ",", "…": ".",
    ";": ",", ":": ",", "-": " ",
}

KEEP_PUNCT = set(".,?!")

# Corpus spellings repaired before anything else looks at them. आ + candra-e
# and अ + candra-o are keyboard slips for ऑ, seen in आॅन and अॉस्कर. RRA is a
# Marathi letter that only ever appears in the rya cluster (करणाऱ्या), where
# plain र is the standard spelling.
TYPO_MAP: dict[str, str] = {
    "आॅ": "ऑ",     # आॅ -> ऑ
    "अॉ": "ऑ",     # अॉ -> ऑ
    "ऱ": "र",           # ऱ -> र
    "\u0950": "\u0913\u092e\u094d",   # ॐ -> ओम्, a ligature for the syllable
    # Southern Devanagari short vowels, which neither Hindi nor Marathi
    # distinguishes. Seven tokens across both corpora; they fold onto the
    # long counterparts rather than becoming unknown symbols.
    "\u0912": "\u0913",           # ऒ short O -> ओ
    "\u090e": "\u090f",           # ऎ short E -> ए
    "\u0944": "\u0943",           # ॄ vocalic RR matra -> ृ
}

# Unicode general categories to keep. Mn and Mc matter enormously here: every
# Devanagari matra, the virama, anusvara and chandrabindu are combining marks,
# and str.isalnum() is False for all of them. Filtering on isalnum() alone
# deletes the vowels from the text and leaves a bare consonant skeleton.
KEEP_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Nd", "Nl", "No"})

DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")
_WS = re.compile(r"\s+")


def strip_zero_width(text: str) -> str:
    return text.translate({ord(c): None for c in ZERO_WIDTH})


def compose_nukta(text: str) -> str:
    for decomposed, composed in NUKTA_COMPOSE.items():
        text = text.replace(decomposed, composed)
    return text


def repair_typos(text: str) -> str:
    for src, dst in TYPO_MAP.items():
        text = text.replace(src, dst)
    return text


def strip_orphan_nukta(text: str) -> str:
    """Drop a nukta sitting on a consonant that has no nukta form.

    The corpora contain nine Hindi tokens where the nukta landed on the wrong
    letter (व़क्त for वक़्त, गल़ती for ग़लती). Left in place the mark becomes an
    unknown symbol and, worse, breaks the consonant-plus-matra parse of the
    following character. Dropping it yields the plain consonant, which is the
    pronunciation most speakers use for these words anyway.
    """
    out: list[str] = []
    for ch in text:
        if ch == NUKTA:
            continue          # composable nukta was already joined in NFC step
        out.append(ch)
    return "".join(out)


def colon_to_visarga(text: str) -> str:
    """A colon between Devanagari letters is a typed visarga.

    Marathi स्वत:च is स्वतःच. A colon elsewhere is punctuation and is dropped
    by the punctuation fold.
    """
    return re.sub(r"(?<=[\u0900-\u097F]):", "\u0903", text)


def map_punctuation(text: str) -> str:
    for src, dst in PUNCT_MAP.items():
        text = text.replace(src, dst)
    return "".join(
        c
        for c in text
        if unicodedata.category(c) in KEEP_CATEGORIES
        or c.isspace()
        or c in KEEP_PUNCT
    )


def devanagari_digits_to_ascii(text: str) -> str:
    return text.translate({ord(k): v for k, v in DIGITS.items()})


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Full normalisation chain. Order matters.

    NFC first so that combining marks sit in canonical order, then nukta
    composition, then digits, then punctuation, then whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_zero_width(text)
    text = compose_nukta(text)
    text = strip_orphan_nukta(text)
    text = repair_typos(text)
    text = colon_to_visarga(text)
    text = devanagari_digits_to_ascii(text)
    text = map_punctuation(text)
    return collapse_whitespace(text)


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RANGE.search(text))


def devanagari_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Devanagari.

    Used to filter corpus lines that are mostly Latin transliteration or
    code-switched English, which would otherwise pollute the phone statistics.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    hits = sum(1 for c in alpha if DEVANAGARI_RANGE.match(c))
    return hits / len(alpha)
