#!/usr/bin/env python3
"""Live training progress, in a form that works with and without a terminal.

The DGX has no tmux and no screen, so a long run goes under nohup and its
output is read from a file afterwards. That makes the usual progress bar
actively harmful: a bar redraws by rewriting the same line with carriage
returns, and a file full of carriage returns is unreadable, megabytes long and
hides the one line that mattered.

So this reports twice, differently, from one call site.

**To a terminal** it rewrites a single line in place: step, percentage, loss,
step rate, elapsed and the estimated finish, plus GPU memory. Watching a run
live, that is what you want.

**To a file or a pipe** it writes a plain line every `file_every` steps, each
one complete and on its own line, so `tail -f` during the run and `grep` after
it both behave.

Either way the JSONL record in train_log.jsonl is the machine-readable truth;
this is the human-readable view and is never parsed by anything.

The rate is a moving average over the recent window rather than the whole run,
because the first steps include CUDA warmup and graph compilation and would
otherwise drag the estimate for hours.
"""

from __future__ import annotations

import collections
import shutil
import sys
import time


def _hms(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Progress:
    """One call per step. Decides for itself how loudly to report."""

    def __init__(self, run_id: str, total_steps: int, start_step: int = 0,
                 stream=None, file_every: int = 200, window: int = 50) -> None:
        self.run_id = run_id
        self.total = total_steps
        self.start_step = start_step
        self.stream = stream or sys.stdout
        self.file_every = max(1, file_every)
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.t0 = time.time()
        self.recent: collections.deque = collections.deque(maxlen=max(2, window))
        self.last_draw = 0.0
        self._wrote_line = False

    # -- the one method the training loop calls ----------------------------

    def update(self, step: int, loss: float, lr: float,
               gpu_mb: float | None = None) -> None:
        now = time.time()
        self.recent.append((now, step))
        rate = self._rate()
        left = (self.total - step) / rate if rate > 0 else float("nan")

        if self.tty:
            # Redraw at most five times a second. More than that is invisible
            # to a human and, over 100,000 steps, is real time spent on IO.
            if now - self.last_draw < 0.2 and step != self.total:
                return
            self.last_draw = now
            self._draw_line(step, loss, lr, rate, left, gpu_mb)
        elif step % self.file_every == 0 or step == 1 or step == self.total:
            self._write_record(step, loss, lr, rate, left, gpu_mb)

    def close(self, final_loss: float) -> None:
        if self.tty and self._wrote_line:
            self.stream.write("\n")
        elapsed = time.time() - self.t0
        done = self.total - self.start_step
        rate = done / elapsed if elapsed > 0 else 0.0
        self.stream.write(
            f"{self.run_id}: finished {self.total:,} steps in {_hms(elapsed)} "
            f"({rate:.2f} steps/s), final loss {final_loss:.4f}\n")
        self.stream.flush()

    # -- internals ---------------------------------------------------------

    def _rate(self) -> float:
        if len(self.recent) < 2:
            return 0.0
        (t_first, s_first), (t_last, s_last) = self.recent[0], self.recent[-1]
        dt = t_last - t_first
        return (s_last - s_first) / dt if dt > 0 else 0.0

    def _draw_line(self, step, loss, lr, rate, left, gpu_mb) -> None:
        pct = 100.0 * step / self.total if self.total else 0.0
        width = shutil.get_terminal_size((100, 24)).columns
        bar_room = max(10, width - 78)
        filled = int(bar_room * step / self.total) if self.total else 0
        bar = "#" * filled + "." * (bar_room - filled)
        mem = f" {gpu_mb/1024:.1f}G" if gpu_mb else ""
        line = (f"\r{self.run_id} [{bar}] {step:>7,}/{self.total:,} {pct:5.1f}% "
                f"loss {loss:8.4f} {rate:5.2f}it/s eta {_hms(left)}{mem}")
        self.stream.write(line[:width - 1])
        self.stream.flush()
        self._wrote_line = True

    def _write_record(self, step, loss, lr, rate, left, gpu_mb) -> None:
        mem = f"  gpu {gpu_mb/1024:.1f}G" if gpu_mb else ""
        self.stream.write(
            f"{self.run_id}  step {step:>7,}/{self.total:,}  "
            f"{100.0*step/self.total:5.1f}%  loss {loss:8.4f}  lr {lr:.3e}  "
            f"{rate:5.2f} it/s  elapsed {_hms(time.time()-self.t0)}  "
            f"eta {_hms(left)}{mem}\n")
        self.stream.flush()
