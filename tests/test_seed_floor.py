#!/usr/bin/env python3
"""The seed-variance runs must differ from r01 in the seed and nothing else.

Plan v3's standing rule is that no difference between two runs is reportable
until the variance floor exists. The floor is the spread across r01, r20 and
r21, which are the same cell at seeds 0, 1 and 2, so the measurement is only
valid if nothing else differs. A stray change to one of them would turn the
floor into a comparison of two things at once, and it would do so silently,
because the number that came out would still look like a floor.

seed is inside config_hash, so these carry their own hashes while every
hash-bearing field stays identical to r01's. That is the property tested here.
"""
from __future__ import annotations

import dataclasses

import pytest

from src.train.config import COSMETIC, plan_runs

# Two floors, because a floor is architecture-specific: r01's cell for
# FastSpeech 2 and r02's for VITS. The VITS one matters more in practice, since
# VITS is end to end and is the half that can be scored without a vocoder.
FS2_SEEDS = {"r01": 0, "r20": 1, "r21": 2}
VITS_SEEDS = {"r02": 0, "r22": 1, "r23": 2}
SEEDS = FS2_SEEDS


def by_id() -> dict:
    return {r.run_id: r for r in plan_runs()}


def test_all_three_exist_with_the_seeds_they_claim():
    runs = by_id()
    for rid, seed in SEEDS.items():
        assert rid in runs, rid
        assert runs[rid].seed == seed, rid


@pytest.mark.parametrize("rid", ["r20", "r21"])
def test_nothing_but_the_seed_differs_from_r01(rid):
    runs = by_id()
    a = dataclasses.asdict(runs["r01"])
    b = dataclasses.asdict(runs[rid])
    differing = {k for k in a if a[k] != b[k]}
    # run_id and notes are cosmetic and outside the hash; seed is the variable.
    assert differing <= ({"seed"} | COSMETIC), differing
    assert "seed" in differing


@pytest.mark.parametrize("rid", ["r20", "r21"])
def test_the_hash_differs_so_the_runs_are_distinct(rid):
    runs = by_id()
    assert runs[rid].config_hash() != runs["r01"].config_hash()


def test_the_three_hashes_are_all_different():
    runs = by_id()
    hashes = {runs[rid].config_hash() for rid in SEEDS}
    assert len(hashes) == 3


def test_they_share_r01s_budget():
    """A floor measured under a different budget is not this study's floor."""
    runs = by_id()
    r01 = runs["r01"]
    for rid in ("r20", "r21"):
        r = runs[rid]
        assert (r.max_steps, r.batch_frames, r.lr, r.lr_schedule,
                r.warmup_steps, r.precision, r.grad_clip) == \
               (r01.max_steps, r01.batch_frames, r01.lr, r01.lr_schedule,
                r01.warmup_steps, r01.precision, r01.grad_clip), rid


def test_they_are_the_only_seeds_other_than_zero():
    """If a seed drifts into another run, the floor stops being a floor and the
    ladder starts mixing seed variance into its data trend."""
    off = {r.run_id: r.seed for r in plan_runs() if r.seed != 0}
    assert off == {"r20": 1, "r21": 2, "r22": 1, "r23": 2}, off


@pytest.mark.parametrize("rid", ["r22", "r23"])
def test_the_vits_floor_differs_from_r02_only_in_the_seed(rid):
    runs = by_id()
    a = dataclasses.asdict(runs["r02"])
    b = dataclasses.asdict(runs[rid])
    differing = {k for k in a if a[k] != b[k]}
    assert differing <= ({"seed"} | COSMETIC), differing
    assert "seed" in differing


def test_the_vits_floor_shares_r02s_rate_and_corrections():
    """A floor measured at a different sample rate, or without the same
    declared provenance, is not r02's floor."""
    runs = by_id()
    for rid in ("r22", "r23"):
        assert runs[rid].sample_rate == runs["r02"].sample_rate
        assert runs[rid].corrections == runs["r02"].corrections
        assert runs[rid].init_from == runs["r02"].init_from


def test_there_is_a_floor_for_each_architecture_being_compared():
    """One floor does not cover two architectures. Reporting the VITS gap
    against the FastSpeech 2 spread would be using the wrong ruler."""
    runs = by_id()
    assert {runs[r].architecture for r in FS2_SEEDS} == {"fastspeech2"}
    assert {runs[r].architecture for r in VITS_SEEDS} == {"vits"}


def test_the_pair_is_queued():
    import pathlib
    q = (pathlib.Path(__file__).resolve().parents[1]
         / "scripts" / "queue_all.sh").read_text(encoding="utf-8")
    assert '"r20 r21"' in q
