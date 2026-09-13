"""Expand ASCII digit strings into Devanagari number words.

Scope is deliberately narrow. The corpora contain digits in 15 Marathi
utterances out of 11,044 and none in Hindi, so a full number grammar would be
effort spent far from the research question. What matters is that no bare digit
reaches the phone sequence, and that anything this module cannot expand is
reported rather than silently mangled.

Covered: 0 to 20 by lookup, exact tens, and the round hundred and thousand
compositions the corpora actually use. Everything else raises Unexpandable so
the caller can count it.

The 21 to 99 range is irregular in both languages (Marathi 21 is एकवीस, not
"twenty one"), so it is not composed from parts. Adding it means writing out
roughly 160 forms, which is worth doing only if a corpus needs it.
"""

from __future__ import annotations

import re


class Unexpandable(ValueError):
    """The number is outside the covered range."""


UNITS: dict[str, list[str]] = {
    "hindi": [
        "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
        "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह",
        "अठारह", "उन्नीस", "बीस",
    ],
    "marathi": [
        "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ",
        "दहा", "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा",
        "अठरा", "एकोणीस", "वीस",
    ],
}

TENS: dict[str, dict[int, str]] = {
    "hindi": {30: "तीस", 40: "चालीस", 50: "पचास", 60: "साठ",
              70: "सत्तर", 80: "अस्सी", 90: "नब्बे"},
    "marathi": {30: "तीस", 40: "चाळीस", 50: "पन्नास", 60: "साठ",
                70: "सत्तर", 80: "ऐंशी", 90: "नव्वद"},
}

# The 21-99 range is irregular in both languages and is NOT composed from
# parts. This table is deliberately incomplete: it holds only forms confident
# enough to put into a corpus, and anything absent raises rather than being
# guessed at. A wrong number word would be a pronunciation error baked into
# training data and invisible afterwards, which is far worse than a flagged
# refusal. Extend it from a dictionary, not from memory.
IRREGULAR: dict[str, dict[int, str]] = {
    "hindi": {21: "इक्कीस", 22: "बाईस", 23: "तेईस", 24: "चौबीस", 25: "पच्चीस"},
    "marathi": {21: "एकवीस", 22: "बावीस", 23: "तेवीस", 24: "चोवीस",
                25: "पंचवीस"},
}

HUNDRED: dict[str, str] = {"hindi": "सौ", "marathi": "शंभर"}
# Hindi writes multiples of a hundred as two words (दो सौ); Marathi fuses them
# into one (दोनशे). The leading space carries that difference.
HUNDRED_SUFFIX: dict[str, str] = {"hindi": " सौ", "marathi": "शे"}
THOUSAND: dict[str, str] = {"hindi": "हज़ार", "marathi": "हजार"}

_DIGITS = re.compile(r"\d+")


def _lang(language: str) -> str:
    key = {"hi": "hindi", "mr": "marathi"}.get(language, language)
    if key not in UNITS:
        raise Unexpandable(f"no number words for {language!r}")
    return key


def number_to_words(n: int, language: str) -> str:
    lang = _lang(language)
    if n < 0:
        raise Unexpandable("negative")
    if n <= 20:
        return UNITS[lang][n]
    if n < 100:
        if n in TENS[lang]:
            return TENS[lang][n]
        if n in IRREGULAR[lang]:
            return IRREGULAR[lang][n]
        raise Unexpandable(f"{n}: 21-99 is irregular and not tabulated")
    if n == 100:
        return HUNDRED[lang]
    if n < 1000:
        head, rest = divmod(n, 100)
        if head > 9:
            raise Unexpandable(str(n))
        word = UNITS[lang][head] + HUNDRED_SUFFIX[lang]
        if rest == 0:
            return word
        return word + " " + number_to_words(rest, lang)
    if n < 100000:
        head, rest = divmod(n, 1000)
        word = number_to_words(head, lang) + " " + THOUSAND[lang]
        if rest == 0:
            return word
        return word + " " + number_to_words(rest, lang)
    raise Unexpandable(f"{n}: above the covered range")


def expand(text: str, language: str) -> tuple[str, list[str]]:
    """Replace digit runs with number words.

    Returns the rewritten text and a list of the digit strings that could not
    be expanded, which are left in place so the caller can decide what to do
    rather than having them disappear.
    """
    failed: list[str] = []

    def sub(m: re.Match[str]) -> str:
        raw = m.group(0)
        try:
            return number_to_words(int(raw), language)
        except Unexpandable:
            failed.append(raw)
            return raw

    return _DIGITS.sub(sub, text), failed
