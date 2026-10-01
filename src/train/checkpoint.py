#!/usr/bin/env python3
"""Checkpoint bookkeeping: what is saved, what is kept, and what resume means.

A Kaggle session is killed without warning. Whether that costs N steps or a
whole run is decided entirely by this file, which is why it is separate from
the training loop and tested on its own.

**What a checkpoint must contain.** Model weights, optimiser state and the step
number. Weights alone are not a resumable checkpoint: Adam's moment estimates
are state, and restarting them from zero after a kill puts a visible transient
in the loss curve that is indistinguishable, later, from a real effect.

**What is deliberately not in it.** The data order. `batching.batch_stream`
derives the batch at step N from (seed, N), so the data position is recomputed
rather than stored, and a checkpoint cannot disagree with the manifest it was
trained on.

**Retention.** Keeping every checkpoint fills the disk; keeping only the last
one means a corrupt write destroys the run. Keep the most recent `keep_last`
and every `keep_every`-th step, so there is always an older good checkpoint.

**Atomicity.** Write to a temporary name and rename. A rename within one
filesystem is atomic, so a kill during a save leaves the previous checkpoint
intact instead of a half-written file that loads as garbage.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass

STEP_RE = re.compile(r"^step_(\d+)$")


@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int = 2
    keep_every: int = 25_000

    def survivors(self, steps: list[int]) -> list[int]:
        steps = sorted(set(steps))
        keep = set(steps[-self.keep_last:]) if self.keep_last > 0 else set()
        keep |= {s for s in steps if self.keep_every > 0 and s % self.keep_every == 0}
        return sorted(keep)

    def to_delete(self, steps: list[int]) -> list[int]:
        keep = set(self.survivors(steps))
        return [s for s in sorted(set(steps)) if s not in keep]


class CheckpointDir:
    """A run's checkpoint directory. Knows nothing about tensors.

    The tensor blob is written by a saver callable the trainer supplies, so
    this module imports no deep-learning framework and its tests run anywhere.
    """

    def __init__(self, root: pathlib.Path, policy: RetentionPolicy | None = None) -> None:
        self.root = pathlib.Path(root)
        self.policy = policy or RetentionPolicy()

    def path_for(self, step: int) -> pathlib.Path:
        return self.root / f"step_{step}"

    def steps(self) -> list[int]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.iterdir():
            m = STEP_RE.match(p.name)
            # A directory without its marker is a half-written save. Ignoring it
            # is the point of the marker: resume must never pick one up.
            if m and p.is_dir() and (p / "COMPLETE").exists():
                out.append(int(m.group(1)))
        return sorted(out)

    def latest(self) -> int | None:
        s = self.steps()
        return s[-1] if s else None

    def save(self, step: int, saver, meta: dict) -> pathlib.Path:
        """Write a checkpoint atomically and apply the retention policy.

        `saver` receives the destination directory and writes the tensors.
        The COMPLETE marker is written last, after the rename, so a directory
        that exists without it is visibly unfinished.
        """
        final = self.path_for(step)
        tmp = self.root / f".tmp_step_{step}"
        if tmp.exists():
            raise SystemExit(f"{tmp} exists; a previous save was interrupted mid-write")
        tmp.mkdir(parents=True)
        saver(tmp)
        (tmp / "meta.json").write_text(
            json.dumps({**meta, "step": step}, indent=1, sort_keys=True),
            encoding="utf-8")
        tmp.rename(final)
        (final / "COMPLETE").write_text("", encoding="utf-8")
        return final

    def prune(self) -> list[int]:
        """Delete checkpoints the policy does not keep. Returns what it removed.

        Deletion is by rename into a `_trash` directory rather than unlink: the
        device shell cannot delete inside a connected folder, and an operation
        that silently fails to free space is worse than one that moves it.
        """
        gone = self.policy.to_delete(self.steps())
        trash = self.root / "_trash"
        for s in gone:
            trash.mkdir(exist_ok=True)
            self.path_for(s).rename(trash / f"step_{s}")
        return gone

    def resume_step(self) -> int:
        """The step to resume from. 0 means start fresh."""
        latest = self.latest()
        return latest if latest is not None else 0
