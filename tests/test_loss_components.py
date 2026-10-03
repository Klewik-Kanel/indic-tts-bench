#!/usr/bin/env python3
"""The step record has to say what the headline loss is made of.

Why this test exists: r07 at the 5 h rung ends on 299.85 and r08 at the 1 h
rung on 10.04, a factor of 29.9, and for weeks there was no way to say what
had changed. The loss both runs report is coqui's ForwardTTS sum, and measured
on 12 utterances of the Hindi corpus a mean predictor's share of it is
0.1*var(f0 in Hz) = 653.99, 0.1*var(frame energy) = 45.65, and a mel L1 of
1.97. The mel term is 0.3 per cent of that. So the headline number is almost
entirely a pitch error, and the cliff is the 1 h rung memorising f0 over 3870
passes of 541 utterances rather than anything about spectral quality.

The fix is not to change the objective mid-matrix. It is to log the terms, so
the claim is checkable from the run and not from a reconstruction.
"""

import math

from src.train.adapters import AdapterBase


class _Fake:
    """A 0-dim tensor's interface, without torch."""

    def __init__(self, v, ndim=0):
        self._v, self.ndim = v, ndim

    def detach(self):
        return self

    def __float__(self):
        return float(self._v)


def test_scalar_terms_are_kept_and_rounded():
    a = AdapterBase()
    a._record({"loss": _Fake(300.0), "loss_spec": _Fake(1.9718239),
               "loss_pitch": _Fake(653.9954)})
    assert a.last_components == {"loss_spec": 1.97182, "loss_pitch": 653.9954}


def test_the_headline_loss_is_not_duplicated():
    a = AdapterBase()
    a._record({"loss": _Fake(1.0), "loss_dur": _Fake(2.0)})
    assert "loss" not in a.last_components
    assert a.last_components["loss_dur"] == 2.0


def test_non_scalars_are_dropped_rather_than_stringified():
    a = AdapterBase()
    a._record({"alignments": _Fake(0.0, ndim=3), "loss_spec": _Fake(1.0)})
    assert a.last_components == {"loss_spec": 1.0}


def test_plain_floats_work_so_a_non_torch_adapter_can_record():
    a = AdapterBase()
    a._record({"duration_error": 0.125})
    assert a.last_components == {"duration_error": 0.125}


def test_a_term_that_raises_does_not_lose_the_others():
    class Angry:
        ndim = 0

        def detach(self):
            raise RuntimeError("no")

    a = AdapterBase()
    a._record({"bad": Angry(), "loss_spec": 1.5})
    assert a.last_components == {"loss_spec": 1.5}


def test_a_broken_loss_dict_cannot_end_a_run():
    a = AdapterBase()
    a.last_components = {"loss_spec": 1.0}

    class NotADict:
        def items(self):
            raise RuntimeError("no")

        def keys(self):
            raise RuntimeError("no")

    a._record(NotADict())              # must not raise
    assert a.last_components == {"loss_spec": 1.0}


def test_the_default_is_falsy_so_the_loop_logs_nothing_extra():
    assert not AdapterBase.last_components


def test_recording_is_per_instance_not_shared_across_adapters():
    one, two = AdapterBase(), AdapterBase()
    one._record({"loss_spec": 1.0})
    assert two.last_components == {}


def test_nan_survives_as_nan_rather_than_being_swallowed():
    a = AdapterBase()
    a._record({"loss_spec": float("nan")})
    assert math.isnan(a.last_components["loss_spec"])


def test_every_coqui_loss_site_records():
    """Each train_step call that gets a dict must record it.

    Four sites return `loss_dict["loss"]` or sum a dict: FastSpeech 2, VITS,
    the vocoder and Matcha. One of them missing is a run whose components are
    silently absent, which is the state this test exists to prevent.
    """
    import pathlib
    src = pathlib.Path("src/train/adapters.py").read_text()
    assert src.count("self._record(") == 4          # FastSpeech 2, VITS, vocoder, Matcha
