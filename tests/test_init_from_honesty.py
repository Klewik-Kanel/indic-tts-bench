#!/usr/bin/env python3
"""A config may not claim a warm start its adapter never performs.

This is the check that would have caught the MMS gap on day one. r02 declared
`init_from: facebook/mms-tts-hin`, `VitsAdapter.build` never read it, and the
config, the deviations table and the class docstring all stated that the 16 kHz
rate was inherited from that checkpoint. Nothing failed and nothing warned;
eight runs carried wrong provenance until the audio was listened to and the
cause traced backwards.

The decided case has to pass, though, or the remaining VITS runs cannot start:
on 3 October the choice was to keep those weights, recorded in each config's
non-hashed `corrections`. So the rule is that a mismatch is allowed exactly
when somebody has written down what is wrong. These tests pin both halves,
because a check that refuses everything gets deleted and a check that allows
everything is decoration.
"""
from __future__ import annotations

import pytest

from src.train.adapters import assert_init_from_is_honest


class Reads:
    reads_init_from = True


class DoesNot:
    reads_init_from = False


class Silent:
    """An adapter predating the flag. Absent means does not read."""


# -- the cases that must pass ------------------------------------------------

def test_no_init_from_declared_is_fine():
    assert_init_from_is_honest({"run_id": "x", "init_from": ""}, DoesNot())


def test_a_missing_key_is_fine():
    assert_init_from_is_honest({"run_id": "x"}, DoesNot())


def test_whitespace_only_counts_as_no_claim():
    assert_init_from_is_honest({"run_id": "x", "init_from": "   "}, DoesNot())


def test_an_adapter_that_reads_it_is_fine():
    assert_init_from_is_honest({"run_id": "r06", "init_from": "/w/x.ckpt"},
                               Reads())


def test_an_acknowledged_mismatch_passes():
    """The 3 Oct decision. Without this the queued VITS runs cannot start."""
    assert_init_from_is_honest(
        {"run_id": "r16", "init_from": "facebook/mms-tts-mar",
         "corrections": ("3 Oct 2026: init_from names a checkpoint but no warm "
                         "start was ever implemented.",)},
        DoesNot())


# -- the cases that must refuse ---------------------------------------------

def test_an_unacknowledged_mismatch_refuses():
    with pytest.raises(SystemExit) as e:
        assert_init_from_is_honest(
            {"run_id": "rXX", "init_from": "facebook/mms-tts-hin"}, DoesNot())
    assert "never reads it" in str(e.value)


def test_the_message_says_what_to_do_about_it():
    """A refusal that does not name the remedy gets worked around."""
    with pytest.raises(SystemExit) as e:
        assert_init_from_is_honest(
            {"run_id": "rXX", "init_from": "something"}, DoesNot())
    msg = str(e.value)
    assert "implement the warm start" in msg
    assert "set init_from to ''" in msg
    assert "add a correction" in msg


def test_corrections_about_something_else_do_not_excuse_it():
    """Any correction at all would make this check decoration."""
    with pytest.raises(SystemExit):
        assert_init_from_is_honest(
            {"run_id": "rXX", "init_from": "something",
             "corrections": ("the sample rate is unmotivated",)}, DoesNot())


def test_an_adapter_without_the_flag_is_treated_as_not_reading_it():
    """Fail safe: a new adapter that forgets the flag refuses rather than
    silently claiming to honour a warm start."""
    with pytest.raises(SystemExit):
        assert_init_from_is_honest(
            {"run_id": "rXX", "init_from": "something"}, Silent())


# -- against the real matrix -------------------------------------------------

def test_every_run_in_the_matrix_passes_the_check():
    """The point of the whole exercise: nothing queued is blocked by it."""
    import dataclasses

    from src.train.adapters import ADAPTERS
    from src.train.config import plan_runs

    for r in plan_runs():
        cfg = dataclasses.asdict(r)
        adapter = ADAPTERS[r.architecture]()
        assert_init_from_is_honest(cfg, adapter)


def test_the_vocoders_really_do_read_it():
    from src.train.adapters import ADAPTERS
    assert ADAPTERS["hifigan"].reads_init_from is True


def test_vits_really_does_not():
    from src.train.adapters import ADAPTERS
    assert ADAPTERS["vits"].reads_init_from is False
