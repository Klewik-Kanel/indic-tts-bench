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
    for having two.

    Checked structurally rather than by looking for the word "mean", which
    gave a false positive once the across-DRAWS average was added: that one
    averages inside a single recogniser's record, which is the point of it.
    What must hold is that every number is scoped to one backend.
    """
    # Every record carries exactly one backend, and the pairing refuses to
    # cross them (see test_pairing_does_not_cross_recognisers).
    utts = [{"id": "u1"}]
    a = _rec("r02", "phoneme", asr="ai4bharat/x", utts=utts)
    b = _rec("r05", "grapheme", asr="facebook/y", utts=utts)
    assert mod.pair_records([a, b]) == []

    # And the table prints one row per record rather than merging them.
    rows = mod.table([a, b]).splitlines()[2:]
    assert len(rows) == 2
    assert "ai4bharat" not in rows[1] and "facebook" not in rows[0]


def test_the_across_draw_average_stays_inside_one_record():
    """The only averaging in the harness is over draws of one run under one
    recogniser. If it ever spanned records, two systems or two instruments
    would be silently pooled."""
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def score_bundle("):src.index("def reference_floor(")]
    assert "_mean(" in body, "the draw average lives in score_bundle"
    after = src[src.index("def reference_floor("):]
    assert "_mean(" not in after, "no averaging outside one run's record"


# --- bundle selection -------------------------------------------------------

def _fake_bundle(tmp, dirname, run_id):
    import json
    d = pathlib.Path(tmp) / dirname
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "bundle_version": 3, "run_id": run_id, "architecture": "vits",
        "language": "hindi", "input_repr": "phoneme", "sample_rate": 22050,
    }), encoding="utf-8")
    return d


def test_runs_accepts_a_run_id_not_only_a_directory_name():
    """`--runs r02` must find exports/r02_step100000.

    This failed once and read as a missing export rather than a filter that
    matched nothing, which cost a round trip to the box.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a = _fake_bundle(tmp, "r02_step100000", "r02")
        b = _fake_bundle(tmp, "r05_step100000", "r05")
        assert mod._select([a, b], ["r02"]) == [a]
        assert mod._select([a, b], ["r02", "r05"]) == [a, b]


def test_runs_still_accepts_the_directory_name():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a = _fake_bundle(tmp, "r02_step100000", "r02")
        assert mod._select([a], ["r02_step100000"]) == [a]


def test_an_unmatched_name_is_reported_with_what_is_on_disk(capsys=None):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a = _fake_bundle(tmp, "r02_step100000", "r02")
        assert mod._select([a], ["r99"]) == []


def test_a_name_given_twice_selects_the_bundle_once():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a = _fake_bundle(tmp, "r02_step100000", "r02")
        assert mod._select([a], ["r02", "r02_step100000"]) == [a]


def test_an_exact_directory_name_wins_over_a_run_id_collision():
    """A directory literally called `r02` must not be shadowed by another
    bundle whose manifest says run_id r02."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        exact = _fake_bundle(tmp, "r02", "r02")
        other = _fake_bundle(tmp, "r02_step100000", "r02")
        assert mod._select([exact, other], ["r02"]) == [exact]
        assert mod._select([other, exact], ["r02"]) == [exact]


# --- the reference floor ----------------------------------------------------

def test_the_floor_table_is_labelled_as_ground_truth():
    """A reader must not mistake the recogniser's own rate for a run's."""
    out = mod.floor_table([])
    assert "GROUND TRUTH" in out
    assert "refCER" in out.splitlines()[1]


def test_the_floor_table_prints_every_class_and_the_excess():
    f = {"backend": "mms", "n": 10, "corpus_cer": 0.1812,
         "classes": {"final": 0.2, "medial": 0.25, "neither": 0.15},
         "contrast": {"final": {"excess": 0.05}}}
    line = mod.floor_table([f]).splitlines()[-1]
    for want in ("mms", "10", "0.1812", "0.2000", "0.2500", "0.1500", "0.0500"):
        assert want in line, want


def test_a_floor_that_failed_prints_its_reason():
    f = {"backend": "mms", "n": 0, "error": "nothing transcribed"}
    assert "nothing transcribed" in mod.floor_table([f])


def test_a_floor_class_with_no_rate_prints_a_dash():
    f = {"backend": "mms", "n": 3, "corpus_cer": 0.1,
         "classes": {"final": None, "medial": None, "neither": 0.1},
         "contrast": {"final": None}}
    line = mod.floor_table([f]).splitlines()[-1]
    assert line.count("-") >= 3


def test_the_floor_reads_wav16_and_never_resamples_a_reference():
    """Both recognisers want 16 kHz and the corpus already has it. A resampled
    reference would put a filter between the reference and itself."""
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    floor = src[src.index("def reference_floor("):src.index("def _select(")]
    assert '"wav16"' in floor
    assert "wav22" not in floor
    assert "to_target_sr" not in floor, "a reference must not be resampled"


def test_the_floor_is_computed_once_per_backend_not_once_per_bundle():
    """Both arms of a pair read the same references, so a per-bundle pass
    would repeat identical work and invite two different floors."""
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def main("):]
    assert body.index("reference_floor(") < body.index("score_bundle(")


def test_floor_only_stops_before_any_bundle_is_loaded():
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def main("):]
    assert body.index("a.floor_only") < body.index("score_bundle(")


def test_no_floor_is_documented_as_debugging_only():
    """Without the floor a synthesis CER has no denominator, so the flag must
    not read as an ordinary option."""
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    i = src.index('"--no-floor"')
    assert "debugging" in src[i:i + 400]


# --- counts and the paired bootstrap ----------------------------------------

def _rec(rid, repr_, asr="ai4bharat/x", n=10, utts=None):
    return {"run_id": rid, "architecture": "vits", "language": "hindi",
            "input_repr": repr_, "asr": asr, "step": 100000, "split": "test",
            "n": n, "corpus_cer": 0.1,
            "classes": {"final": {"edits": 92, "chars": 327, "cer": 0.2813},
                        "medial": {"edits": 4, "chars": 96, "cer": 0.0417},
                        "neither": {"edits": 47, "chars": 389, "cer": 0.1208}},
            "contrast": {"final": {"excess": 0.16}}, "insertions": 3,
            "utterances": utts or []}


def test_the_counts_table_prints_edits_over_characters():
    """A rate whose denominator is 96 is a different claim from one whose
    denominator is 4000, and the rates alone do not say which."""
    out = mod.counts_table([_rec("r02", "phoneme")])
    assert "COUNTS" in out
    assert "4/96" in out
    assert "92/327" in out
    assert "143/812" in out, "the all-classes total must be the sum"


def test_the_counts_table_skips_a_run_that_was_skipped():
    out = mod.counts_table([{"run_id": "r01", "skipped": "mel-only"}])
    assert "r01" not in out


def test_pairing_matches_the_two_arms_of_one_cell():
    utts = [{"id": "u1"}]
    ph, gr = _rec("r02", "phoneme", utts=utts), _rec("r05", "grapheme", utts=utts)
    pairs = mod.pair_records([gr, ph])
    assert len(pairs) == 1
    assert pairs[0][0]["input_repr"] == "phoneme"
    assert pairs[0][1]["input_repr"] == "grapheme"


def test_pairing_does_not_cross_recognisers():
    """Two arms scored by different recognisers are not one contrast."""
    utts = [{"id": "u1"}]
    ph = _rec("r02", "phoneme", asr="ai4bharat/x", utts=utts)
    gr = _rec("r05", "grapheme", asr="facebook/y", utts=utts)
    assert mod.pair_records([ph, gr]) == []


def test_pairing_does_not_cross_architectures_or_languages():
    utts = [{"id": "u1"}]
    ph = _rec("r02", "phoneme", utts=utts)
    gr = _rec("r18", "grapheme", utts=utts)
    gr["language"] = "marathi"
    assert mod.pair_records([ph, gr]) == []


def test_an_unpaired_arm_is_not_bootstrapped():
    out = mod.bootstrap_table([_rec("r02", "phoneme", utts=[{"id": "u1"}])],
                              100, 0)
    assert "no phonemic/graphemic pair" in out


def test_the_bootstrap_table_reports_an_interval_and_whether_it_clears_zero():
    pytest.importorskip("numpy")
    utts_a, utts_b = [], []
    for i in range(12):
        utts_a.append({"id": f"u{i}", "final_edits": 1, "final_chars": 20,
                       "medial_edits": 0, "medial_chars": 8,
                       "neither_edits": 1, "neither_chars": 20})
        utts_b.append({"id": f"u{i}", "final_edits": 9, "final_chars": 20,
                       "medial_edits": 0, "medial_chars": 8,
                       "neither_edits": 1, "neither_chars": 20})
    ph = _rec("r02", "phoneme", utts=utts_a)
    gr = _rec("r05", "grapheme", utts=utts_b)
    out = mod.bootstrap_table([ph, gr], 300, 0)
    assert "PAIRED BOOTSTRAP" in out
    assert "final" in out
    assert "yes" in out, "a large uniform effect must read as clear of zero"


def test_the_bootstrap_names_its_resample_count_and_seed():
    """A resampled interval is not reproducible unless both are recorded."""
    out = mod.bootstrap_table([], 1234, 99)
    assert "1234 resamples" in out
    assert "seed 99" in out


def test_boot_zero_skips_the_interval_entirely():
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def main("):]
    assert "if a.boot:" in body, "--boot 0 must skip the resampling"


def test_the_per_utterance_rows_are_stripped_from_the_combined_file():
    """They are already in the per-run JSON; keeping them writes every
    utterance of every run twice."""
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def main("):]
    assert 'r.pop("utterances", None)' in body
    assert body.index("bootstrap_table(") < body.index('r.pop("utterances"')


# --- merging draws for the bootstrap ----------------------------------------

def _pass(draw, rows):
    return {"draw": draw, "per_utt": rows, "parts": [], "pairs": []}


def _u(uid, fe, fc, me=0, mc=0, ne=0, nc=0, ins=0):
    return {"id": uid, "final_edits": fe, "final_chars": fc,
            "medial_edits": me, "medial_chars": mc,
            "neither_edits": ne, "neither_chars": nc, "insertions": ins}


def test_draws_are_summed_into_one_row_per_utterance():
    """The bootstrap read draw 0 alone while the table pooled all five, so on
    4 October the table said +0.0263 and the bootstrap beside it said +0.0057
    for the same quantity."""
    passes = [_pass(0, [_u("a", 1, 10), _u("b", 2, 20)]),
              _pass(1, [_u("a", 3, 10), _u("b", 4, 20)])]
    out = mod._merge_draws(passes)
    assert [r["id"] for r in out] == ["a", "b"]
    assert out[0]["final_edits"] == 4 and out[0]["final_chars"] == 20
    assert out[1]["final_edits"] == 6 and out[1]["final_chars"] == 40
    assert out[0]["draws"] == 2


def test_the_merged_order_follows_the_first_draw():
    """Two arms scored on the same split must come out in the same order, or
    the paired bootstrap's order check passes only by luck."""
    passes = [_pass(0, [_u("x", 1, 10), _u("y", 1, 10), _u("z", 1, 10)])]
    assert [r["id"] for r in mod._merge_draws(passes)] == ["x", "y", "z"]


def test_an_utterance_missing_from_a_later_draw_keeps_the_draws_it_has():
    """Its denominator is smaller, which is correct. Dropping the row would
    silently change which utterances the interval is over."""
    passes = [_pass(0, [_u("a", 1, 10), _u("b", 1, 10)]),
              _pass(1, [_u("a", 1, 10)])]
    out = mod._merge_draws(passes)
    assert [r["id"] for r in out] == ["a", "b"]
    assert out[0]["draws"] == 2 and out[0]["final_chars"] == 20
    assert out[1]["draws"] == 1 and out[1]["final_chars"] == 10


def test_insertions_are_summed_too():
    passes = [_pass(0, [_u("a", 0, 10, ins=2)]),
              _pass(1, [_u("a", 0, 10, ins=3)])]
    assert mod._merge_draws(passes)[0]["insertions"] == 5


def test_every_class_is_merged_not_just_final():
    passes = [_pass(0, [_u("a", 1, 10, me=2, mc=8, ne=3, nc=20)]),
              _pass(1, [_u("a", 1, 10, me=2, mc=8, ne=3, nc=20)])]
    r = mod._merge_draws(passes)[0]
    assert (r["medial_edits"], r["medial_chars"]) == (4, 16)
    assert (r["neither_edits"], r["neither_chars"]) == (6, 40)


def test_a_single_draw_merges_to_itself():
    passes = [_pass(0, [_u("a", 1, 10, me=1, mc=5, ne=1, nc=10)])]
    r = mod._merge_draws(passes)[0]
    assert r["final_edits"] == 1 and r["final_chars"] == 10
    assert r["draws"] == 1


def test_the_record_uses_the_merge_rather_than_the_first_draw():
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("def score_bundle("):src.index("def reference_floor(")]
    assert 'record["utterances"] = _merge_draws(passes)' in body
    assert 'passes[0]["per_utt"]' not in body.split('record["utterances"]')[1]


def test_the_merge_and_the_table_pool_the_same_characters():
    """The whole point: the interval and the rate beside it must be over the
    same sample. Five draws of 50 utterances is 5x the characters."""
    rows = [_u(f"u{i}", 1, 10) for i in range(50)]
    passes = [_pass(d, rows) for d in range(5)]
    out = mod._merge_draws(passes)
    assert len(out) == 50
    assert sum(r["final_chars"] for r in out) == 50 * 10 * 5
