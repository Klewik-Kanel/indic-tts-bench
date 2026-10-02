#!/usr/bin/env python3
"""Stop a run when it stops improving, for the one run that has no step budget.

Every acoustic run trains for exactly `max_steps` because the fixed-budget
claim depends on it, and `assert_budget_matched` enforces that. The vocoder is
different and already excluded from that assertion: it is not an object of
comparison in the study, it exists so that both arms of every comparison are
synthesised through the same analysis. Giving it 100,000 steps because the
others have 100,000 steps would spend GPU time on a number nobody reports.

So it gets a stopping criterion instead, and the step count it reached is
reported rather than chosen. That exemption belongs in the methods table
stated outright, which is why this module says it here too.

Deliberately knows nothing about tensors or models: a sequence of numbers goes
in, a decision comes out, and it is tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plateau:
    """Stop after `patience` consecutive checks that fail to beat the best.

    `min_delta` is what counts as beating it. Without one, noise in the fourth
    decimal reads as improvement forever and the criterion never fires, which
    is the failure mode that makes people give up on early stopping and go back
    to a step count.
    """

    patience: int = 3
    min_delta: float = 0.0
    best: float | None = None
    best_step: int = 0
    strikes: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    def update(self, step: int, value: float) -> bool:
        """Record a check. True means stop.

        A value that is not a number is ignored rather than treated as a
        failure to improve: a validation pass that could not run is not
        evidence that the run has converged.
        """
        if value is None or value != value:            # None, or NaN
            return False
        value = float(value)
        self.history.append((int(step), value))
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.best_step = int(step)
            self.strikes = 0
            return False
        self.strikes += 1
        return self.strikes >= self.patience

    def why(self) -> str:
        return (f"no improvement over {self.best:.6f} at step {self.best_step} "
                f"for {self.strikes} consecutive checks "
                f"(patience {self.patience}, min_delta {self.min_delta})")
