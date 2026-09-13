"""Hindi schwa deletion, isolated as a switchable stage.

This module is the subject of the study, so it is kept small, explicit and
independently testable. Everything about it is measurable on its own, before
any TTS model is trained.

The core rule follows Narasimhan, Sproat and Kiraz (2004): an inherent schwa
deletes in the context V C _ C V, applied iteratively from right to left.
Word-final inherent schwa deletes near-obligatorily in polysyllables. A lexicon
overrides both, because the rule has a known residual error rate and the point
of the study is to know what that rate is rather than to pretend it is zero.

Marathi uses the same code path with ``delete_medial=False``. That single
switch is the control in the experiment: same script, same converter, same
inventory, only the phonological rule differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .devanagari import Segment
from .phoneset import NASALISATION


@dataclass(frozen=True)
class SchwaConfig:
    """Per-language switches.

    delete_medial: the V C _ C V rule. True for Hindi, False for Marathi.
    delete_final:  word-final inherent schwa. True for both.
    min_vowels:    never reduce a word below this many vowels. Guards against
                   deleting the only vowel in a monosyllable.
    block_before:  consonants after which a preceding schwa is protected.
                   Populated from error analysis, empty by default so the
                   baseline rule is reported unmodified.
    """

    delete_medial: bool = True
    delete_final: bool = True
    min_vowels: int = 1
    block_before: frozenset[str] = frozenset()


HINDI = SchwaConfig(delete_medial=True, delete_final=True)
MARATHI = SchwaConfig(delete_medial=False, delete_final=True)
NO_DELETION = SchwaConfig(delete_medial=False, delete_final=False)


def _is_vowel_seg(seg: Segment) -> bool:
    return seg.kind == "vowel"


def _is_consonant_seg(seg: Segment) -> bool:
    return seg.kind == "consonant"


def _count_vowels(segs: list[Segment]) -> int:
    return sum(1 for s in segs if _is_vowel_seg(s))


def _is_deletable_schwa(seg: Segment) -> bool:
    """Only an unwritten schwa is a candidate.

    A schwa spelled with a matra, or the independent vowel, is never deleted.
    The nasalisation check matters: a nasalised inherent schwa carries a
    written anusvara or chandrabindu, so the vowel is not really unwritten.
    """
    return (
        seg.inherent
        and seg.kind == "vowel"
        and seg.phone == "ə"
        and NASALISATION not in seg.phone
    )


def delete_schwas(
    segs: list[Segment],
    config: SchwaConfig = HINDI,
) -> tuple[list[Segment], list[int]]:
    """Return the surviving segments and the indices that were deleted.

    The deleted-index list is returned rather than discarded because the
    stress-test scoring compares predicted deletion sites against gold
    deletion sites, not just final phone strings.
    """
    alive = [True] * len(segs)
    deleted: list[int] = []

    def live_indices() -> list[int]:
        return [i for i, ok in enumerate(alive) if ok]

    # --- word-final schwa --------------------------------------------------
    if config.delete_final:
        idxs = live_indices()
        if idxs:
            last = idxs[-1]
            if _is_deletable_schwa(segs[last]):
                remaining = _count_vowels([segs[i] for i in idxs]) - 1
                if remaining >= config.min_vowels:
                    alive[last] = False
                    deleted.append(last)

    # --- medial schwa, right to left ---------------------------------------
    if config.delete_medial:
        changed = True
        while changed:
            changed = False
            idxs = live_indices()
            # Right to left. Iterative, because each deletion changes the
            # context for the schwa to its left.
            for pos in range(len(idxs) - 1, -1, -1):
                i = idxs[pos]
                if not _is_deletable_schwa(segs[i]):
                    continue
                if pos - 2 < 0 or pos + 2 >= len(idxs):
                    continue

                prev_c = segs[idxs[pos - 1]]
                prev_v = segs[idxs[pos - 2]]
                next_c = segs[idxs[pos + 1]]
                next_v = segs[idxs[pos + 2]]

                context_ok = (
                    _is_vowel_seg(prev_v)
                    and _is_consonant_seg(prev_c)
                    and _is_consonant_seg(next_c)
                    and _is_vowel_seg(next_v)
                )
                if not context_ok:
                    continue
                if next_c.phone in config.block_before:
                    continue

                remaining = _count_vowels([segs[j] for j in idxs]) - 1
                if remaining < config.min_vowels:
                    continue

                alive[i] = False
                deleted.append(i)
                changed = True
                break  # restart the scan, contexts have shifted

    survivors = [segs[i] for i in range(len(segs)) if alive[i]]
    return survivors, sorted(deleted)


def deletion_sites(segs: list[Segment], config: SchwaConfig = HINDI) -> set[int]:
    """Source-character indices at which a schwa was deleted.

    Expressed in input-string coordinates so that gold annotations can be
    written against the orthography rather than against our segment indices.
    """
    _, deleted = delete_schwas(segs, config)
    return {segs[i].source_index for i in deleted}
