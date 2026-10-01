#!/usr/bin/env python3
"""Learning-rate schedule, shared by every architecture.

`warmup_inverse_sqrt` is the schedule named in every config. It is written here
once rather than taken from each architecture's own repository, because the
fixed-budget claim covers the schedule shape and three upstream codebases will
not implement it identically.

    lr(step) = peak * min(step / warmup, sqrt(warmup / step))

Both branches equal `peak` at step == warmup, so the curve is continuous at the
join. That continuity is the property worth testing: a discontinuity there is a
silent order-of-magnitude jump in the middle of training.

Step numbering is 1-based inside the formula. Step 0 would be a division by
zero in the decay branch and a zero learning rate in the warmup branch, so the
first optimiser step is step 1.
"""

from __future__ import annotations

import math


def lr_at(step: int, peak_lr: float, warmup_steps: int) -> float:
    """Learning rate for a 1-based step number."""
    if step < 1:
        raise ValueError("steps are 1-based; step 0 has no learning rate")
    if warmup_steps <= 0:
        return peak_lr
    if step <= warmup_steps:
        return peak_lr * (step / warmup_steps)
    return peak_lr * math.sqrt(warmup_steps / step)


def schedule_table(peak_lr: float, warmup_steps: int, max_steps: int,
                   points: int = 10) -> list[tuple[int, float]]:
    """A few (step, lr) pairs, for the run log and for the paper's appendix."""
    steps = sorted({1, warmup_steps, warmup_steps + 1, max_steps} |
                   {max(1, round(max_steps * i / points)) for i in range(points + 1)})
    return [(s, lr_at(s, peak_lr, warmup_steps)) for s in steps if 1 <= s <= max_steps]
