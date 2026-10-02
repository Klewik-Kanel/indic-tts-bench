"""The vocoder's stopping criterion. Numbers in, a decision out, no GPU."""

from __future__ import annotations

from src.train.earlystop import Plateau


def run(values, **kw):
    p = Plateau(**kw)
    for i, v in enumerate(values, 1):
        if p.update(i * 100, v):
            return p, i
    return p, None


def test_it_stops_after_patience_checks_without_improvement():
    p, stopped_at = run([1.0, 0.9, 0.8, 0.81, 0.82, 0.83], patience=3)
    assert stopped_at == 6
    assert p.best == 0.8 and p.best_step == 300


def test_it_keeps_going_while_the_value_falls():
    p, stopped_at = run([1.0, 0.9, 0.8, 0.7, 0.6], patience=2)
    assert stopped_at is None and p.strikes == 0


def test_the_strike_count_resets_on_a_new_best():
    p, stopped_at = run([1.0, 1.1, 1.2, 0.5, 1.3], patience=3)
    assert stopped_at is None
    assert p.best == 0.5 and p.strikes == 1


def test_noise_below_min_delta_does_not_count_as_improvement():
    """Without min_delta, fourth-decimal wobble reads as progress forever."""
    p, stopped_at = run([1.0, 0.99999, 0.99998, 0.99997], patience=2,
                        min_delta=1e-3)
    assert stopped_at == 3


def test_a_validation_pass_that_could_not_run_is_not_a_strike():
    """A missing number is not evidence of convergence."""
    p, stopped_at = run([1.0, None, None, None, None], patience=2)
    assert stopped_at is None and p.strikes == 0
    assert len(p.history) == 1


def test_a_nan_is_ignored_the_same_way():
    p, stopped_at = run([1.0, float("nan"), float("nan"), float("nan")],
                        patience=2)
    assert stopped_at is None and p.strikes == 0


def test_the_history_is_kept_for_the_record():
    """The step count the vocoder reached is reported, not chosen, so the
    sequence it was decided from has to survive the run."""
    p, _ = run([1.0, 0.9, 0.95], patience=5)
    assert p.history == [(100, 1.0), (200, 0.9), (300, 0.95)]


def test_why_names_the_best_and_the_strikes():
    p, _ = run([1.0, 1.1, 1.2, 1.3], patience=3)
    msg = p.why()
    assert "1.000000" in msg and "step 100" in msg and "patience 3" in msg
