#!/usr/bin/env python3
"""A stopping criterion has to keep the weights it stopped for.

Measured on 4 October. r06 and r17 both stopped correctly on the plateau
criterion and neither kept the weights it stopped for. Evaluation ran every
2,000 steps and checkpoints every 5,000, so the two intervals never coincided:
r06's best held-out mel L1 was 0.246545 at step 12,000, its saved checkpoints
were 15,000 and 18,000, and the generator that would have been exported was
step 18,000 at 0.262312. That is 6.40% worse than the best and 2.25 standard
deviations of the tail noise.

Separately, min_delta was 1e-4 against a mean check-to-check movement of
0.012466 for r06 and 0.010220 for r17, so the threshold was 100 to 125 times
smaller than the noise. That is recorded here as a measurement rather than
fixed, because changing it changes what the two finished vocoders would have
done and they are not being re-run.
"""

import pathlib

import pytest

from src.train.earlystop import Plateau

RUNNER = pathlib.Path("src/train/runner.py").read_text()

# The histories as the two runs recorded them.
R06 = [(2000, 0.2915978357195854), (4000, 0.2535130064934492),
       (6000, 0.26335070095956326), (8000, 0.25260191783308983),
       (10000, 0.24822324514389038), (12000, 0.2465445213019848),
       (14000, 0.26330992951989174), (16000, 0.25369551219046116),
       (18000, 0.2623116299510002)]


def test_the_best_is_saved_before_the_stop_is_decided():
    """Order matters: a save after `break` never runs."""
    body = RUNNER[RUNNER.index("if stopper is not None and n % eval_every"):]
    body = body[:body.index("prog.close(")]
    assert body.index("was_best") < body.index("stopper.update(")


def test_a_new_best_writes_a_checkpoint():
    body = RUNNER[RUNNER.index("was_best = ("):]
    body = body[:body.index("if stopper.update(")]
    assert "save_checkpoint(n)" in body
    assert "best.json" in body


def test_the_best_test_does_not_use_min_delta():
    """min_delta exists so noise does not read as progress, which is a
    question about when to STOP. Any improvement at all is worth keeping on
    disk, so the checkpoint test is a plain comparison."""
    body = RUNNER[RUNNER.index("was_best = ("):]
    body = body[:body.index("if stopper.update(")]
    assert "min_delta" not in body


def test_the_best_is_read_before_the_stopper_is_updated():
    """Comparing against stopper.best AFTER update would compare the value
    with itself and never save anything."""
    body = RUNNER[RUNNER.index("if stopper is not None and n % eval_every"):]
    body = body[:body.index("prog.close(")]
    assert body.index("stopper.best is None") < body.index("stopper.update(")


def test_r06_would_now_keep_its_best_step():
    """Replay r06's own history and check step 12,000 is a save point."""
    st = Plateau(patience=3, min_delta=1e-4)
    saved = []
    for step, val in R06:
        if st.best is None or val < st.best:
            saved.append(step)
        if st.update(step, val):
            break
    assert 12000 in saved, f"the best step is not saved: {saved}"
    assert saved[-1] == 12000, "the last save must be the best, not a later one"


def test_the_replay_still_stops_where_it_stopped():
    """The fix must not change the stopping decision, only what is kept."""
    st = Plateau(patience=3, min_delta=1e-4)
    stop_at = None
    for step, val in R06:
        if st.update(step, val):
            stop_at = step
            break
    assert stop_at == 18000
    assert st.best_step == 12000
    assert st.best == pytest.approx(0.2465445213019848)


def test_min_delta_was_far_below_the_measured_noise():
    """Recorded so the figure is in the repository rather than in a chat.

    If a vocoder is ever re-run, min_delta has to come from this number and
    not from a round guess, and eval_every has to divide ckpt_every.
    """
    vals = [v for _, v in R06]
    moves = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    mean_move = sum(moves) / len(moves)
    assert mean_move == pytest.approx(0.012466, abs=1e-5)
    assert mean_move / 1e-4 > 100


def test_a_failed_validation_does_not_count_as_a_best():
    """None means the check could not run. It must not be treated as an
    improvement any more than as a failure to improve."""
    st = Plateau(patience=3, min_delta=1e-4)
    st.update(2000, 0.30)
    before = st.best
    assert st.update(4000, None) is False
    assert st.best == before
