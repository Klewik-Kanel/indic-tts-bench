#!/usr/bin/env python3
"""Frame-budget batching, deterministic and identical across architectures.

The study claims every architecture trains under one budget. Expressing that
budget in utterances would break the claim: a batch of 16 long utterances is
several times the gradient signal of a batch of 16 short ones, and the mix
differs per architecture only because the data loader shuffled differently.
Batching to a fixed number of mel frames makes the quantity of speech per step
the controlled variable, which is what `batch_frames` in the config means.

Three properties this file guarantees, each with a test.

**Determinism.** Given a seed and a manifest, the sequence of batches is a pure
function of the two. A resumed run at step 40,000 must see the same batch the
uninterrupted run would have seen, or resume quietly changes the data order and
the run is no longer the run described by its config.

**A frame budget that is respected, not approximated.** Every batch is at or
under `batch_frames`, with the single documented exception of an utterance
longer than the whole budget, which is emitted alone and counted.

**Length bucketing without length bias.** Padding to the longest item in a
batch wastes compute, so items are grouped by length. Pure sorting by length,
though, makes every epoch present short utterances first, which is a curriculum
nobody asked for. Items are therefore shuffled, chunked into a pool, sorted
within the pool, cut into batches, and the batch order is shuffled again.
"""

from __future__ import annotations

import csv
import math
import pathlib
import random
from dataclasses import dataclass

HOP_LENGTH = 256          # matches src/eval/mcd.py, so frame counts agree
POOL_BATCHES = 32         # pool size, in batches, for length bucketing


@dataclass(frozen=True)
class Utterance:
    uid: str
    wav: str
    seconds: float
    text: str
    frames: int


def frames_for(seconds: float, sample_rate: int, hop_length: int = HOP_LENGTH) -> int:
    """Mel frames an utterance occupies.

    Ceiling, not rounding: a partial final frame is still computed, and the
    budget must not be exceeded because of a rounding convention.
    """
    return int(math.ceil(seconds * sample_rate / hop_length))


def load_manifest(path: pathlib.Path, sample_rate: int) -> list[Utterance]:
    """Read a split manifest into utterances, at the run's sample rate.

    The manifest carries both a 22.05 kHz and a 16 kHz path for every item,
    because the MMS-initialised VITS runs natively at 16 kHz. Picking the column
    from the run's sample rate here means no training script ever has to know
    about that, and a run cannot accidentally mix rates.
    """
    col = "wav22" if sample_rate == 22050 else "wav16"
    out: list[Utterance] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if col not in row:
                raise SystemExit(f"{path}: no {col} column; manifest predates dual-rate prep")
            secs = float(row["seconds"])
            out.append(Utterance(
                uid=row["id"], wav=row[col], seconds=secs, text=row["text"],
                frames=frames_for(secs, sample_rate),
            ))
    if not out:
        raise SystemExit(f"{path}: empty manifest")
    return out


def make_batches(utts: list[Utterance], batch_frames: int, seed: int,
                 epoch: int) -> list[list[Utterance]]:
    """One epoch of batches. A pure function of (utts, batch_frames, seed, epoch).

    Padding is accounted for honestly: a batch costs len(batch) * max_frames,
    not the sum of its frames, because that is what the GPU actually computes.
    Budgeting on the sum would let a batch with one long item and twenty short
    ones blow past the memory the budget is supposed to bound.
    """
    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")
    # Derived, not stateful: the same (seed, epoch) must give the same order
    # on any machine and in any process, so no global RNG is touched.
    rng = random.Random(seed * 1_000_003 + epoch)
    order = list(utts)
    rng.shuffle(order)

    pool_size = max(1, POOL_BATCHES * max(1, batch_frames // max(1, _median_frames(utts))))
    batches: list[list[Utterance]] = []
    for start in range(0, len(order), pool_size):
        pool = sorted(order[start:start + pool_size], key=lambda u: u.frames)
        cur: list[Utterance] = []
        cur_max = 0
        for u in pool:
            if u.frames > batch_frames:
                # Longer than the whole budget. Emit alone rather than drop it:
                # dropping would silently change the training set between runs
                # with different budgets.
                if cur:
                    batches.append(cur)
                    cur, cur_max = [], 0
                batches.append([u])
                continue
            new_max = max(cur_max, u.frames)
            if cur and new_max * (len(cur) + 1) > batch_frames:
                batches.append(cur)
                cur, cur_max = [u], u.frames
            else:
                cur.append(u)
                cur_max = new_max
        if cur:
            batches.append(cur)

    rng.shuffle(batches)
    return batches


def _median_frames(utts: list[Utterance]) -> int:
    f = sorted(u.frames for u in utts)
    return f[len(f) // 2] if f else 1


def oversized(utts: list[Utterance], batch_frames: int) -> list[Utterance]:
    """Utterances that cannot share a batch with anything. Reported, not hidden."""
    return [u for u in utts if u.frames > batch_frames]


def batch_stream(utts: list[Utterance], batch_frames: int, seed: int,
                 max_steps: int, start_step: int = 0):
    """Yield (step, batch) from start_step to max_steps, crossing epochs.

    Resume correctness lives here. The epoch and the position inside it are
    derived from the step number, so resuming at step N reproduces the batch
    the uninterrupted run saw at step N. Nothing about the data order is stored
    in the checkpoint, which means a checkpoint cannot disagree with the data.
    """
    step = 0
    epoch = 0
    while step < max_steps:
        batches = make_batches(utts, batch_frames, seed, epoch)
        if not batches:
            raise SystemExit("no batches; batch_frames smaller than every utterance")
        for b in batches:
            if step >= max_steps:
                return
            if step >= start_step:
                yield step, b
            step += 1
        epoch += 1
