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


# -- the splits ---------------------------------------------------------------

def test_the_frozen_splits_are_carried():
    """Weights trained on an unrecorded split are not reproducible by anyone.
    The audio is a public corpus; which utterance landed in train, dev or test
    is not, and it exists only on the machine that is going away."""
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        repo = pathlib.Path(tmp)
        for rel in ("data/processed/hindi/train.tsv",
                    "data/processed/hindi/dev.tsv",
                    "data/processed/hindi/test.tsv",
                    "data/processed/hindi/SPLITS.lock",
                    "data/processed/hindi/ladder/1h.tsv",
                    "data/interim/hindi/manifest.tsv",
                    "data/raw/dataset_profile.json"):
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
        from scripts import offload
        old_repo, old_runs = offload.REPO, offload.RUNS
        offload.REPO = repo
        offload.RUNS = repo / "runs"
        try:
            remote = [r for _, r in offload.plan()]
        finally:
            offload.REPO, offload.RUNS = old_repo, old_runs
    assert "data/processed/hindi/SPLITS.lock" in remote, remote
    assert "data/processed/hindi/train.tsv" in remote
    assert "data/processed/hindi/ladder/1h.tsv" in remote, "a ladder rung"
    assert "data/interim/hindi/manifest.tsv" in remote
    assert "data/raw/dataset_profile.json" in remote


def test_no_file_is_queued_twice():
    """SPLITS.lock matches one pattern and *.tsv another. A duplicate uploads
    the same bytes twice and makes the dry run's total a lie."""
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        repo = pathlib.Path(tmp)
        for rel in ("data/processed/hindi/train.tsv",
                    "data/processed/hindi/SPLITS.lock",
                    "space_static/index.html", "RESULTS.md"):
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
        from scripts import offload
        old_repo, old_runs = offload.REPO, offload.RUNS
        offload.REPO = repo
        offload.RUNS = repo / "runs"
        try:
            remote = [r for _, r in offload.plan()]
        finally:
            offload.REPO, offload.RUNS = old_repo, old_runs
    assert len(remote) == len(set(remote)), \
        [r for r in remote if remote.count(r) > 1]


def test_no_audio_is_dragged_in_by_the_split_patterns():
    """data/interim holds the wavs. Only its manifest goes."""
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        repo = pathlib.Path(tmp)
        for rel in ("data/interim/hindi/manifest.tsv",
                    "data/interim/hindi/hi_0001.wav",
                    "data/processed/hindi/train.tsv"):
            f = repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")
        from scripts import offload
        old_repo, old_runs = offload.REPO, offload.RUNS
        offload.REPO = repo
        offload.RUNS = repo / "runs"
        try:
            remote = [r for _, r in offload.plan()]
        finally:
            offload.REPO, offload.RUNS = old_repo, old_runs
    assert not any(r.endswith(".wav") and r.startswith("data/")
                   for r in remote), remote
