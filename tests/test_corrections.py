#!/usr/bin/env python3
"""A declared deviation found false is corrected, not rewritten.

`deviations` and `init_from` are inside `config_hash`, so editing them would
change the hash of runs that have already trained and orphan their checkpoints
from their own provenance. On 3 October r02 and r05 were found to have trained
from scratch while their config declared `init_from: facebook/mms-tts-hin`, and
the 16 kHz rate they justify by that checkpoint turned out to rest on nothing.

So the declaration stays exactly as it was and `corrections` carries what was
found. These tests hold that line: a correction must never move a hash, and the
VITS runs must actually carry one, because the failure mode is a correction that
quietly stops being emitted and a paper that prints only the false claim.
"""
from __future__ import annotations

import dataclasses
import pathlib

import pytest

from src.train.config import COSMETIC, RunConfig, plan_runs

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"


def a_run(**kw) -> RunConfig:
    base = dict(run_id="x", architecture="vits", language="hindi",
                input_repr="phoneme", data="9h")
    base.update(kw)
    return RunConfig(**base)


def test_corrections_are_outside_the_hash():
    assert "corrections" in COSMETIC


def test_adding_a_correction_does_not_move_the_hash():
    """The whole point. If this fails, r02's checkpoint no longer matches its
    own config and the provenance chain is broken."""
    plain = a_run()
    fixed = a_run(corrections=("found something untrue",))
    assert plain.config_hash() == fixed.config_hash()


def test_editing_a_deviation_DOES_move_the_hash():
    """The reason corrections exist rather than editing in place."""
    before = a_run(deviations=("a",))
    after = a_run(deviations=("b",))
    assert before.config_hash() != after.config_hash()


def test_editing_init_from_DOES_move_the_hash():
    assert a_run(init_from="").config_hash() != \
        a_run(init_from="facebook/mms-tts-hin").config_hash()


def test_every_vits_run_carries_the_corrections():
    """Including the ones that have not started, and the two seed-floor runs
    added later: they share the finding because they share the adapter.

    The count is asserted so that a VITS run added without corrections shows up
    here rather than reaching the paper with only the false claim attached."""
    vits = [r for r in plan_runs() if r.architecture == "vits"]
    assert len(vits) == 10, [r.run_id for r in vits]
    for r in vits:
        assert r.corrections, r.run_id
        assert any("no warm start was ever implemented" in c for c in r.corrections), r.run_id
        assert any("unmotivated" in c for c in r.corrections), r.run_id


def test_no_other_architecture_carries_them():
    """A correction is about specific runs, not a banner on the matrix."""
    for r in plan_runs():
        if r.architecture != "vits":
            assert not r.corrections, r.run_id


def test_the_declaration_is_left_intact():
    """The false claim stays in deviations. Removing it would hide that it was
    ever made, which is the opposite of what the table is for."""
    r02 = next(r for r in plan_runs() if r.run_id == "r02")
    assert any("inherited from the MMS checkpoint" in d for d in r02.deviations)
    assert r02.init_from == "facebook/mms-tts-hin"


def test_the_correction_reaches_the_yaml():
    text = (CONFIGS / "r02.yaml").read_text(encoding="utf-8")
    assert "corrections:" in text
    assert "no warm start was ever implemented" in text


@pytest.mark.parametrize("rid", ["r02", "r05", "r11", "r16", "r19"])
def test_the_written_hash_still_matches_the_matrix(rid):
    text = (CONFIGS / f"{rid}.yaml").read_text(encoding="utf-8")
    on_disk = next(l.split(":", 1)[1].strip() for l in text.splitlines()
                   if l.startswith("config_hash:"))
    assert on_disk == next(r for r in plan_runs() if r.run_id == rid).config_hash()


def test_the_loader_reads_corrections_as_a_list_of_strings():
    """runner.load_config hardens strings against being parsed as numbers; a
    new list field must not trip it, or every VITS run fails at startup."""
    from src.train.runner import load_config
    cfg = load_config(CONFIGS / "r02.yaml")
    assert isinstance(cfg["corrections"], list)
    assert all(isinstance(c, str) for c in cfg["corrections"])
    assert cfg["config_hash"] == "d3873342a10d"
    assert cfg["sample_rate"] == 16000
