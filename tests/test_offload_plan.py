#!/usr/bin/env python3
"""What the offload pass has to carry for the training box to be disposable.

The point of this file: offload.py ran on a timer for days and nobody checked
WHICH files it had been sending. It covered RESULTS.md, the configs, each run's
logs, one checkpoint per run and the export bundles, and it silently covered
neither the scored tables nor the demo audio. A reader restoring from it would
have had the weights and no record of what was measured from them, and none of
the audio.
"""

import pathlib

import pytest

from scripts import offload

SRC = pathlib.Path("scripts/offload.py").read_text()


def test_the_scored_tables_are_carried():
    """They are the numbers the paper quotes."""
    assert 'results/tables/' in SRC
    assert '"results" / "tables"' in SRC


def test_the_demo_is_carried():
    """A Static Space serves files and runs nothing, so these wavs are the
    deliverable rather than a build artefact."""
    assert 'space_static/' in SRC
    assert '"space_static"' in SRC


def test_the_small_per_run_files_are_still_carried():
    for name in ("config.json", "vocab.json", "train_log.jsonl",
                 "stopped.json"):
        assert name in offload.SMALL, name


def test_stopped_json_is_among_them():
    """The vocoder runs report their step and reason only there, and that file
    is the whole evidence for how r06 and r17 ended."""
    assert "stopped.json" in offload.SMALL


def test_one_checkpoint_per_run_is_the_default():
    """This runs on a timer between pairs, so the routine pass stays cheap."""
    import inspect
    sig = inspect.signature(offload.plan)
    assert sig.parameters["all_checkpoints"].default is False


def test_all_checkpoints_is_available_and_documented_as_expensive():
    assert "--all-checkpoints" in SRC
    i = SRC.index('"--all-checkpoints"')
    assert "gigabytes" in SRC[i:i + 400]


def test_plan_is_called_once_with_the_flag():
    """It walks the filesystem, and a second shadowed call walked it twice."""
    assert SRC.count("jobs = plan(") == 1
    assert "jobs = plan(all_checkpoints=a.all_checkpoints)" in SRC


def test_hidden_files_are_not_uploaded():
    """.offloaded.json is the local ledger. Uploading it would overwrite the
    record of what has already been sent."""
    body = SRC[SRC.index("def plan("):SRC.index("def load_state(")]
    assert body.count('startswith(".")') >= 2


def test_the_plan_runs_on_a_machine_with_no_runs_directory():
    """It is also run from the laptop clone, where runs/ does not exist. A
    missing directory is the normal state there, not an error."""
    jobs = offload.plan()
    assert isinstance(jobs, list)
    for local, remote in jobs:
        assert isinstance(remote, str) and not remote.startswith("/")


def test_every_remote_path_is_relative_and_has_no_parent_escape():
    for _local, remote in offload.plan():
        assert ".." not in remote.split("/"), remote
