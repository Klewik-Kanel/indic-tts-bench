#!/usr/bin/env python3
"""The status formatter, which does arithmetic and runs unattended for days."""

import json
import os
import subprocess
import sys


def run(record, rid="r22", mx="100000"):
    env = dict(os.environ, RID=rid, MAX_STEPS=mx)
    out = subprocess.run([sys.executable, "scripts/_status_line.py"],
                         input=json.dumps(record) if isinstance(record, dict)
                         else record,
                         capture_output=True, text=True, env=env, cwd=".")
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_the_rate_is_steps_over_elapsed_seconds():
    line = run({"step": 1900, "loss": 36.111, "elapsed_s": 1253.9})
    assert "1.52 it/s" in line          # 1900 / 1253.9 = 1.5153


def test_the_eta_is_the_remaining_steps_at_that_rate():
    line = run({"step": 1900, "loss": 36.111, "elapsed_s": 1253.9})
    assert "17.98 h" in line            # (100000-1900) / 1.5153 / 3600


def test_the_percentage_is_of_the_step_budget():
    assert "  1.9%" in run({"step": 1900, "loss": 1.0, "elapsed_s": 100.0})
    assert "100.0%" in run({"step": 100000, "loss": 1.0, "elapsed_s": 100.0})


def test_a_zero_rate_prints_no_eta_rather_than_infinity():
    """Step 0 is a real state at launch, and 'inf h' in a watch pane reads as
    a crash."""
    line = run({"step": 0, "loss": 0, "elapsed_s": 0})
    assert "n/a" in line
    assert "inf" not in line


def test_an_unreadable_line_names_the_run_and_exits_zero():
    """A half-written log line is normal: the trainer appends while this
    reads. It must not take the status pane down."""
    line = run("not json at all")
    assert "r22" in line
    assert "unreadable" in line


def test_the_largest_loss_components_are_shown_when_present():
    """The headline loss is a sum, and for FastSpeech 2 the mel term is under
    one per cent of it, so the breakdown is the part worth watching."""
    line = run({"step": 200, "loss": 26.48, "elapsed_s": 83.2,
                "components": {"loss_spec": 1.97, "loss_pitch": 653.99,
                               "loss_dur": 0.4, "loss_energy": 45.6}})
    assert "pitch=654" in line
    assert "energy=45.6" in line
    assert "spec=1.97" in line
    assert "dur=" not in line, "only the three largest, to keep one line"


def test_the_loss_prefix_is_stripped_so_the_line_fits():
    line = run({"step": 1, "loss": 1.0, "elapsed_s": 1.0,
                "components": {"loss_spec": 2.0}})
    assert "spec=2" in line
    assert "loss_spec" not in line


def test_a_run_without_components_prints_no_breakdown():
    """Runs launched before component logging landed have none, and an empty
    breakdown must not become a trailing separator."""
    line = run({"step": 10, "loss": 1.0, "elapsed_s": 5.0})
    assert line.endswith("h")


def test_the_budget_comes_from_the_environment():
    line = run({"step": 500, "loss": 1.0, "elapsed_s": 100.0}, mx="1000")
    assert "500/1000" in line
    assert " 50.0%" in line
