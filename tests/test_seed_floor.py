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

SEEDS = {"r01": 0, "r20": 1, "r21": 2}


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
    assert off == {"r20": 1, "r21": 2}, off


def test_the_pair_is_queued():
    import pathlib
    q = (pathlib.Path(__file__).resolve().parents[1]
         / "scripts" / "queue_all.sh").read_text(encoding="utf-8")
    assert '"r20 r21"' in q
