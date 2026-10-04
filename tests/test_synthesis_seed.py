#!/usr/bin/env python3
"""Synthesis must be reproducible, or a measurement is one draw from a model.

The defect this file exists for, measured on 3 October: two runs of
scripts/score_intelligibility.py with identical arguments, code, weights and
ten utterances gave different answers. r05's character error rate under
IndicConformer moved from 0.1330 to 0.1478, its medial-site errors fell from
16 of 96 characters to 5, and the medial-site excess the two recognisers had
agreed on at +0.1147 and +0.1150 came back as -0.0569 and +0.0661, reversing
sign on one. The ground-truth floor was byte-identical across both passes,
because it reads fixed files.

The cause is that VITS samples: its stochastic duration predictor and its flow
both draw noise at inference, and `Bundle.synthesize` called `inference` with
no seed.
"""

import pytest

from src.export.synthesize import SEED_SALT, draw_seed


def test_the_same_run_text_and_draw_give_the_same_seed():
    a = draw_seed("r02", "कमल खिला", 0)
    b = draw_seed("r02", "कमल खिला", 0)
    assert a == b


def test_the_draw_index_changes_the_seed():
    """Otherwise --draws 5 would synthesise one utterance five times."""
    assert draw_seed("r02", "कमल", 0) != draw_seed("r02", "कमल", 1)


def test_the_run_changes_the_seed():
    """Two arms must not share a noise draw, or the comparison is partly a
    shared accident rather than two independent samples of each system."""
    assert draw_seed("r02", "कमल", 0) != draw_seed("r05", "कमल", 0)


def test_the_text_changes_the_seed():
    assert draw_seed("r02", "कमल", 0) != draw_seed("r02", "नमक", 0)


def test_the_seed_does_not_depend_on_what_was_synthesised_before():
    """Keyed on (run, text, draw) and nothing else, so scoring a subset
    reproduces the same audio as the full set. A global RNG advanced per
    utterance would make utterance 7 depend on utterances 1 to 6, and
    --limit 10 would then measure different audio from the full split."""
    first = draw_seed("r02", "तीसरा वाक्य", 0)
    for other in ("पहला", "दूसरा", "चौथा"):
        draw_seed("r02", other, 0)
    assert draw_seed("r02", "तीसरा वाक्य", 0) == first


def test_the_seed_is_stable_across_processes():
    """sha256, not hash(): Python salts str hashing per process, so hash()
    would make every re-run a different measurement, which is the defect."""
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, '.');"
            "from src.export.synthesize import draw_seed;"
            "print(draw_seed('r02', 'कमल खिला', 0))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == draw_seed("r02", "कमल खिला", 0)


def test_the_seed_fits_what_torch_accepts():
    """torch.manual_seed takes a 64-bit value; a negative or oversized seed
    raises, and it would raise only on the machine that has torch."""
    for args in (("r02", "क", 0), ("r05", "कमल खिला है", 4),
                 ("r23", "x" * 500, 99)):
        v = draw_seed(*args)
        assert isinstance(v, int)
        assert 0 <= v < 2 ** 63


def test_the_salt_is_versioned_so_a_change_is_visible():
    """If the seeding scheme changes, the audio changes. The salt carries a
    version so that is a recorded change rather than a silent one."""
    assert "v1" in SEED_SALT


def test_synthesize_seeds_before_it_calls_inference():
    """Asserted on the source, because the ordering is the whole point and a
    seed set after the forward pass would pass every behavioural test on a
    deterministic model."""
    import pathlib
    src = pathlib.Path("src/export/synthesize.py").read_text()
    body = src[src.index("def synthesize("):]
    body = body[:body.index("elapsed = time.perf_counter()")]
    assert body.index("torch.manual_seed(") < body.index("self.model.inference(")


def test_synthesize_takes_a_draw_and_an_explicit_seed():
    import inspect

    from src.export.synthesize import Bundle
    sig = inspect.signature(Bundle.synthesize)
    assert "draw" in sig.parameters
    assert "seed" in sig.parameters
    assert sig.parameters["draw"].default == 0
    assert sig.parameters["seed"].default is None


def test_the_harness_warns_when_a_single_draw_is_used_on_vits():
    """A single draw is reproducible and is still not the system. The warning
    is what stops the next reader repeating my mistake."""
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    assert "a.draws < 2" in src
    assert "reversed the sign" in src


# --- the device check, and failing loudly -----------------------------------

def test_the_cuda_check_works_on_a_string_and_on_a_device_object():
    """Every caller passes a string. `.to("cpu")` works, so nothing had ever
    needed Bundle.device to be a torch.device, and asking it for `.type`
    raised AttributeError on all 250 synthesis calls per run on 4 October.
    """
    import pathlib
    src = pathlib.Path("src/export/synthesize.py").read_text()
    body = src[src.index("def synthesize("):]
    body = body[:body.index("t0 = time.perf_counter()")]
    assert "self.device.type" not in body, "device may be a plain string"
    assert 'str(self.device).startswith("cuda")' in body

    class Devicey:
        def __init__(self, t):
            self._t = t

        def __str__(self):
            return self._t

    for spelling, want in (("cpu", False), ("cuda", True), ("cuda:0", True),
                           ("cuda:3", True)):
        assert str(spelling).startswith("cuda") is want, spelling
        assert str(Devicey(spelling)).startswith("cuda") is want, spelling


def test_load_passes_device_as_a_string_so_the_check_must_accept_one():
    import inspect

    from src.export.synthesize import Bundle, load
    assert inspect.signature(Bundle.__init__).parameters["device"].default == "cpu"
    assert inspect.signature(load).parameters["device"].default == "cpu"


def test_a_draw_in_which_everything_fails_abandons_the_rest():
    """40 passes ran in silence on 4 October because the loop kept going and
    the progress counter kept printing 50/50."""
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    body = src[src.index("for d in range(max(1, int(draws))):"):]
    body = body[:body.index("record[\"failures\"]")]
    assert "ALL" in body and "abandoning" in body
    assert "break" in body


def test_failures_are_summarised_by_distinct_cause():
    """One AttributeError repeated 250 times should read as one line naming
    the bug, not as two tracebacks and 248 silences."""
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    assert "failure_kinds" in src
    assert "distinct" in src


def test_the_skip_reason_names_the_cause_not_just_the_count():
    import pathlib
    src = pathlib.Path("scripts/score_intelligibility.py").read_text()
    i = src.index('"nothing transcribed')
    assert "failure_kinds" in src[i:i + 500]
