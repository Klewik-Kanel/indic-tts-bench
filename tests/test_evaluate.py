#!/usr/bin/env python3
"""The evaluation harness's bookkeeping, which is where it can go wrong quietly.

The metrics themselves are tested in test_eval_metrics.py. What is tested here
is the layer around them, because its failure modes are silent rather than loud:
a run dropped from the table instead of marked unscorable, a reference pulled at
the wrong sample rate, or a spread computed over values that include a NaN and
so reports a floor of nan while looking like a number.
"""
from __future__ import annotations

import math
import pathlib

import pytest

from scripts import evaluate as ev


# -- picking the reference ---------------------------------------------------

def test_a_16k_run_is_compared_against_16k_audio():
    """Comparing a 16 kHz synthesis against a 22.05 kHz reference would make
    MCD a resampling measurement. The VITS runs are the 16 kHz ones."""
    assert ev.reference_column(16000) == "wav16"


def test_a_22k_run_is_compared_against_22k_audio():
    assert ev.reference_column(22050) == "wav22"


# -- the summary, which is what the results table reads ----------------------

def test_mean_and_spread_over_a_known_set():
    rows = [{"mcd_db": 4.0}, {"mcd_db": 5.0}, {"mcd_db": 6.0}]
    s = ev.summarise(rows)
    assert s["n_utterances"] == 3
    assert s["mcd_db"]["mean"] == pytest.approx(5.0)
    assert s["mcd_db"]["median"] == pytest.approx(5.0)
    assert s["mcd_db"]["sd"] == pytest.approx(1.0)      # ddof=1
    assert s["mcd_db"]["n"] == 3


def test_the_spread_uses_the_sample_estimator():
    """ddof=1, because these are samples of a run's behaviour, not a
    population. With ddof=0 a two-utterance spread reads 30% too small."""
    s = ev.summarise([{"mcd_db": 4.0}, {"mcd_db": 6.0}])
    assert s["mcd_db"]["sd"] == pytest.approx(math.sqrt(2.0))


def test_a_single_utterance_reports_zero_spread_rather_than_nan():
    s = ev.summarise([{"mcd_db": 4.2}])
    assert s["mcd_db"]["sd"] == 0.0


def test_non_finite_values_are_excluded_rather_than_poisoning_the_mean():
    """One NaN through numpy's mean makes every number in the row nan, which
    looks like a value in a table and is not one."""
    rows = [{"mcd_db": 4.0}, {"mcd_db": float("nan")}, {"mcd_db": 6.0},
            {"mcd_db": float("inf")}]
    s = ev.summarise(rows)
    assert s["mcd_db"]["n"] == 2
    assert s["mcd_db"]["mean"] == pytest.approx(5.0)
    assert s["n_utterances"] == 4          # the count is of attempts, not of wins


def test_a_metric_absent_from_every_row_is_absent_from_the_summary():
    s = ev.summarise([{"mcd_db": 4.0}])
    assert "log_f0_rmse" not in s


def test_an_empty_set_summarises_without_raising():
    s = ev.summarise([])
    assert s["n_utterances"] == 0
    assert "mcd_db" not in s


def test_every_reported_metric_is_summarised():
    """If a metric is computed per utterance but never summarised, it silently
    never reaches the results table."""
    rows = [{"mcd_db": 4.0, "log_f0_rmse": 0.2, "f0_rmse_hz": 20.0,
             "vuv_error_rate": 0.1, "length_ratio": 0.98}]
    s = ev.summarise(rows)
    for k in ("mcd_db", "log_f0_rmse", "f0_rmse_hz", "vuv_error_rate",
              "length_ratio"):
        assert k in s, k


# -- the default split -------------------------------------------------------

def test_the_default_split_is_test_not_dev():
    """dev.tsv chose the demo sentences on 3 Oct and is no longer neutral;
    test.tsv was frozen on 14 Sep before any training. Scoring on dev by
    default would mean reporting numbers from a split that has already been
    selected on, which is the quiet version of testing on your training set."""
    args = ev.main.__wrapped__ if hasattr(ev.main, "__wrapped__") else None
    text = pathlib.Path(ev.__file__).read_text(encoding="utf-8")
    assert '"--split", default="test"' in text
    assert args is None     # main takes argv, it is not decorated
