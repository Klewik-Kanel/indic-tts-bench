"""Shared IPA phone inventory for Devanagari languages.

One inventory across Hindi and Marathi so that results from both sit in the
same symbol space. That matters for the control comparison: if the phone sets
differed, a difference in the phoneme-vs-grapheme effect could be an artefact
of inventory size rather than of phonology.

Reference for the Hindi inventory: Ohala (1999), Illustrations of the IPA:
Hindi. Marathi additions follow Dhongde & Wali (2009).
"""

from __future__ import annotations

# --- consonants -----------------------------------------------------------
# Dental stops carry the dental diacritic. Retroflex stops do not.
CONSONANTS: dict[str, str] = {
    # velar
    "क": "k",    "ख": "kʰ",   "ग": "ɡ",    "घ": "ɡʱ",   "ङ": "ŋ",
    # palatal / post-alveolar affricates
    "च": "tʃ",   "छ": "tʃʰ",  "ज": "dʒ",   "झ": "dʒʱ",  "ञ": "ɲ",
    # retroflex
    "ट": "ʈ",    "ठ": "ʈʰ",   "ड": "ɖ",    "ढ": "ɖʱ",   "ण": "ɳ",
    # dental
    "त": "t̪",    "थ": "t̪ʰ",   "द": "d̪",    "ध": "d̪ʱ",   "न": "n",
    # labial
    "प": "p",    "फ": "pʰ",   "ब": "b",    "भ": "bʱ",   "म": "m",
    # approximants, fricatives
    "य": "j",    "र": "r",    "ल": "l",    "व": "ʋ",
    "श": "ʃ",    "ष": "ʂ",    "स": "s",    "ह": "ɦ",
    # retroflex lateral: marginal in Hindi, phonemic in Marathi
    "ळ": "ɭ",
}

# Perso-Arabic borrowings written with nukta. Many Hindi speakers merge these
# with the plain counterpart. The merge is a config switch, not a hard rule,
# because it changes the phone inventory size and therefore the model.
# Written as explicit codepoints on purpose. U+0958..U+095F are on the Unicode
# composition-exclusion list, so a literal typed into source may already be the
# decomposed base-plus-nukta pair and NFC will not put it back together. Pinning
# the codepoints removes that ambiguity from the lookup table.
NUKTA_CONSONANTS: dict[str, str] = {
    "क़": "q",    # क़
    "ख़": "x",    # ख़
    "ग़": "ɣ",    # ग़
    "ज़": "z",    # ज़
    "ड़": "ɽ",    # ड़
    "ढ़": "ɽʱ",   # ढ़
    "फ़": "f",    # फ़
}

# What each nukta phone collapses to when merge_nukta is on.
NUKTA_MERGE: dict[str, str] = {
    "q": "k", "x": "kʰ", "ɣ": "ɡ", "z": "dʒ", "f": "pʰ",
    # the flaps do not merge: /ɽ/ is contrastive in Hindi
    "ɽ": "ɽ", "ɽʱ": "ɽʱ",
}

# --- vowels ---------------------------------------------------------------
# Independent (word-initial or post-vowel) forms.
INDEPENDENT_VOWELS: dict[str, str] = {
    "अ": "ə",  "आ": "aː", "इ": "ɪ",  "ई": "iː", "उ": "ʊ",  "ऊ": "uː",
    "ऋ": "ri", "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː",
    "ऑ": "ɒ",  "ऍ": "æ",  "ॲ": "æ",
}

# Dependent forms (matras) that attach to a consonant.
VOWEL_SIGNS: dict[str, str] = {
    "ा": "aː", "ि": "ɪ",  "ी": "iː", "ु": "ʊ",  "ू": "uː",
    "ृ": "ri", "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː",
    "ॉ": "ɒ",  "ॅ": "æ",
}

# The vowel a bare consonant carries when no matra and no virama follow.
# Deleting this in the right places is the entire subject of the study.
INHERENT_VOWEL = "ə"

VIRAMA = "्"
ANUSVARA = "ं"
CHANDRABINDU = "ँ"
VISARGA = "ः"
NUKTA = "़"
AVAGRAHA = "ऽ"

NASALISATION = "̃"          # combining tilde, applied to the preceding vowel
VISARGA_PHONE = "h"

# Nasal consonants keyed by the place of the following stop, for anusvara
# assimilation. Anusvara before a stop surfaces as the homorganic nasal;
# elsewhere it nasalises the preceding vowel.
HOMORGANIC_NASAL: dict[str, str] = {
    "k": "ŋ", "kʰ": "ŋ", "ɡ": "ŋ", "ɡʱ": "ŋ",
    "tʃ": "ɲ", "tʃʰ": "ɲ", "dʒ": "ɲ", "dʒʱ": "ɲ",
    "ʈ": "ɳ", "ʈʰ": "ɳ", "ɖ": "ɳ", "ɖʱ": "ɳ",
    "t̪": "n", "t̪ʰ": "n", "d̪": "n", "d̪ʱ": "n",
    "p": "m", "pʰ": "m", "b": "m", "bʱ": "m",
}

DIGITS: dict[str, str] = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

ALL_CONSONANT_PHONES = set(CONSONANTS.values()) | set(NUKTA_CONSONANTS.values())
ALL_VOWEL_PHONES = set(INDEPENDENT_VOWELS.values()) | set(VOWEL_SIGNS.values()) | {INHERENT_VOWEL}

# Special symbols emitted by the front end.
WORD_BOUNDARY = "|"
SILENCE = "sil"
UNKNOWN = "<unk>"

# Punctuation is kept rather than stripped, because it is the only prosodic
# signal the text carries: a comma and a full stop are where a reader pauses,
# and a question mark changes the contour of the whole phrase. Coverage over
# the real corpora found punctuation to be 4.1% of Hindi phone tokens and 2.7%
# of Marathi, so discarding it would throw away the phrasing cue on roughly one
# token in thirty.
#
# The set is small on purpose. Every symbol here becomes a row in the model's
# embedding table, and a rare symbol is a row that receives almost no gradient.
# Rarer marks are folded into these four during normalisation.
PUNCTUATION = [",", ".", "?", "!"]

SPECIALS = [WORD_BOUNDARY, SILENCE, UNKNOWN] + PUNCTUATION


def is_vowel(phone: str) -> bool:
    """True for a vowel, with or without a nasalisation diacritic."""
    return phone.rstrip(NASALISATION) in ALL_VOWEL_PHONES


def is_consonant(phone: str) -> bool:
    return phone in ALL_CONSONANT_PHONES or phone in {"ŋ", "ɲ", "ɳ", "n", "m", "h"}


def build_inventory(merge_nukta: bool = False) -> list[str]:
    """The full symbol table a model is trained against.

    Sorted and deterministic, because the symbol-to-index mapping has to be
    stable across runs or checkpoints stop being loadable.
    """
    phones: set[str] = set()
    phones |= set(CONSONANTS.values())
    if merge_nukta:
        phones |= {NUKTA_MERGE[p] for p in NUKTA_CONSONANTS.values()}
    else:
        phones |= set(NUKTA_CONSONANTS.values())
    phones |= set(INDEPENDENT_VOWELS.values())
    phones |= set(VOWEL_SIGNS.values())
    phones.add(INHERENT_VOWEL)
    phones |= {v + NASALISATION for v in list(phones) if v in ALL_VOWEL_PHONES}
    phones |= set(HOMORGANIC_NASAL.values())
    phones.add(VISARGA_PHONE)
    return SPECIALS + sorted(phones)
