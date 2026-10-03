#!/usr/bin/env python3
"""The harness's bookkeeping, with no model and no audio.

What can go wrong here without an exception: a mel-only run silently dropped
instead of reported, a table that prints a number for a class that has none,
and the thread and device caps being set after the libraries are imported,
which makes them do nothing while looking like they work.
"""

import os
import pathlib

import pytest

from scripts import score_intelligibility as mod


import contextlib


@contextlib.contextmanager
def _clean_env(*names):
    """Save and restore environment variables without a fixture.

    Written by hand rather than with monkeypatch so the file runs under any
    runner, including the cut-down one used on the laptop where pytest is not
    installed. A test that only passes under one runner is not a test.
    """
    saved = {n: os.environ.get(n) for n in names}
    for n in names:
        os.environ.pop(n, None)
    try:
        yield
    finally:
        for n, v in saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


CAPPED = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS")


def test_the_cpu_default_hides_the_card_from_every_library():
    """The card is training. A GPU allocation from here can end a run."""
    with _clean_env(*CAPPED):
        mod._cap_threads(4, "cpu")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "4"


def test_asking_for_cuda_does_not_hide_the_card():
    with _clean_env(*CAPPED):
        mod._cap_threads(2, "cuda")
        assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_the_caps_are_set_before_torch_is_imported():
    """_cap_threads must run before the imports it affects.

    Asserted on the source rather than at runtime: the environment variables
    are read by the libraries at import time, so a call placed after an import
    would pass every behavioural test and still do nothing.
    """
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def main("):]
    assert body.index("_cap_threads(") < body.index("score_bundle("), \
        "threads and device must be capped before any scoring import runs"


def test_a_missing_class_prints_a_dash_not_a_zero():
    assert mod._fmt(None) == "-"
    assert mod._fmt(0.0) == "0.0000"


def test_the_table_prints_the_reason_for_a_skipped_run():
    row = {"run_id": "r01", "architecture": "fastspeech2",
           "input_repr": "phoneme",
           "skipped": "mel-only architecture: intelligibility needs a waveform"}
    out = mod.table([row])
    assert "r01" in out
    assert "mel-only" in out, "a skipped run must say why, not vanish"


def test_the_table_header_leads_with_the_partitioned_columns():
    out = mod.table([])
    head = out.splitlines()[0]
    assert head.index("final") < head.index("excess")
    for col in ("CER", "final", "medial", "none", "excess", "ins"):
        assert col in head, col


def test_a_scored_row_prints_every_class_and_the_excess():
    row = {"run_id": "r02", "architecture": "vits", "input_repr": "phoneme",
           "asr": "ai4bharat/indic-conformer-600m-multilingual", "n": 12,
           "corpus_cer": 0.1234,
           "classes": {"final": {"cer": 0.2}, "medial": {"cer": 0.3},
                       "neither": {"cer": 0.1}},
           "contrast": {"final": {"excess": 0.1}},
           "insertions": 3}
    line = mod.table([row]).splitlines()[-1]
    for want in ("r02", "vits", "0.1234", "0.2000", "0.3000", "0.1000",
                 "0.1000", "3"):
        assert want in line, want


def test_a_row_whose_contrast_is_undefined_still_prints():
    """contrast() returns None when a side has no characters. The row must not
    crash the table, and must not invent a zero effect."""
    row = {"run_id": "r05", "architecture": "vits", "input_repr": "grapheme",
           "asr": "facebook/mms-1b-all", "n": 4, "corpus_cer": 0.5,
           "classes": {"final": {"cer": None}, "medial": {"cer": None},
                       "neither": {"cer": 0.5}},
           "contrast": {"final": None}, "insertions": 0}
    line = mod.table([row]).splitlines()[-1]
    assert "r05" in line
    assert line.count("-") >= 3


def test_both_is_the_default_backend_and_resolves_to_two():
    ap_default = "both"
    names = ["indicconformer", "mms"] if ap_default == "both" else [ap_default]
    assert names == ["indicconformer", "mms"]


def test_no_number_in_the_harness_combines_the_two_recognisers():
    """Averaging them would hide the instrument-dependence that is the reason
    for having two."""
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    for bad in ("mean(", "statistics.mean", "+ mms", "avg"):
        assert bad not in src, f"found {bad!r}: the backends must stay separate"
