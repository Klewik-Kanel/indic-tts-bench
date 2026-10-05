#!/usr/bin/env python3
"""What the listening page actually renders, run rather than read as text.

The page is the deliverable: a Static Space serves files and runs nothing, so
whatever this HTML says is a claim on a public URL. Grepping the source for a
string says the string is present. It does not say the page reaches it, puts
the right arms under it, or declines to make it when the data does not support
it. These tests execute the page's own script under node and assert on the
markup it produces.

Guarded here:

  - the headline group is the contrast with the fewest things in it that are
    not the ablation, which means an end-to-end architecture over one that
    needs a vocoder in the audio path;
  - a seed replica is labelled as one only when it matches the headline's
    architecture AND rung, because a different run carrying a different seed
    is not a measure of training noise;
  - a measured error rate is attached to the run it was measured on, and the
    arms that were not measured are named as not measured;
  - the page does not say an arm "plays" when it has no audio;
  - "less data" is claimed only of a rung genuinely below the headline's.

Skipped where node is not installed.
"""

import json
import pathlib
import re
import shutil
import subprocess
import tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
PAGE = HERE / "space_static" / "index.html"

HARNESS = r"""
import fs from "fs";
const js = fs.readFileSync(process.argv[2], "utf8");
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

const sink = {}, nodes = {};
const node = id => ({
  set innerHTML(v) { sink[id] = v; },
  get innerHTML() { return sink[id] || ""; },
  set textContent(v) { sink[id + ":text"] = v; },
  addEventListener() {}, after() {},
});
globalThis.document = {
  querySelector: s => (nodes[s.slice(1)] ||= node(s.slice(1))),
  querySelectorAll: () => [],
  createElement: () => ({ style: {}, classList: { add() {} },
                          set innerHTML(v) {} }),
};

const mod = new Function(js + "\nreturn {boot};")();
mod.boot(data);

const list = sink["list"] || "";
const groups = [...list.matchAll(
  /<div class="grouphead([^"]*)">([\s\S]*?)<\/div>/g)].map(m => ({
    headline: m[1].includes("is-headline"),
    text: m[2].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim(),
}));
const arms = [...list.matchAll(
  /<div class="name">([^<]*)<small>([^<]*)<\/small>(?:<span class="rid">([^<]*)<\/span>)?/g
)].map(m => ({ label: m[1], sub: m[2], run_id: m[3] || "" }));
console.log(JSON.stringify({
  groups, arms,
  warnings: (sink["warnings"] || "").replace(/<[^>]+>/g, " ")
                                    .replace(/\s+/g, " ").trim(),
  provenance: (sink["provenance"] || "").replace(/<[^>]+>/g, " ")
                                        .replace(/\s+/g, " ").trim(),
  footer: sink["footer:text"] || "",
  runs_dl: (sink["runs"] || "").replace(/<[^>]+>/g, " ")
                               .replace(/\s+/g, " ").trim(),
}));
"""


def _script() -> str:
    """The page's own script, with the fetch tail removed."""
    html = PAGE.read_text(encoding="utf-8")
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "the page has no script block"
    js = m.group(1)
    return js[:js.index('fetch("data.json")')]


def _run(data: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        raise pytest.importorskip("node")      # not installed: skip
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        (t / "page.js").write_text(_script(), encoding="utf-8")
        (t / "harness.mjs").write_text(HARNESS, encoding="utf-8")
        (t / "data.json").write_text(json.dumps(data), encoding="utf-8")
        proc = subprocess.run(
            [node, str(t / "harness.mjs"), str(t / "page.js"),
             str(t / "data.json")],
            capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


# -- the data these tests drive the page with -------------------------------

def _run_rec(rid, arch, repr_, data, seed):
    return {"run_id": rid, "architecture": arch, "input_repr": repr_,
            "data": data, "seed": seed, "step": 100000, "config_hash": "h",
            "git_commit": "0123456789ab", "sample_rate": 22050,
            "describes": f"{arch} on hindi with {repr_} input"}


def _clip(arch, audio=True):
    return {"audio": "audio/x/s1.wav" if audio else None,
            "tokens": ["n", "ə", "m", "k"],
            "vocoder": "end-to-end" if arch == "vits"
                       else ("hifigan:r06@18000" if audio else None)}


def _page(runs, audio=True):
    per = {r["run_id"]: _clip(r["architecture"], audio) for r in runs}
    return {"language": "hindi", "split": "dev", "criterion": "most sites",
            "generated_utc": "2026-10-05T12:40:30+00:00",
            "vocoder_bundle": "exports/r06_step18000", "runs": runs,
            "sentences": [{"id": "hi_1", "text": "नमक",
                           "seconds": 1.2, "words": 1, "medial": 1, "final": 0,
                           "medial_words": ["नमक"],
                           "reference": {"22050": "audio/reference_22050/hi_1.wav"},
                           "runs": per}],
            "failures": []}


FULL = [
    _run_rec("r01", "fastspeech2", "phoneme", "9h", 0),
    _run_rec("r04", "fastspeech2", "grapheme", "9h", 0),
    _run_rec("r07", "fastspeech2", "phoneme", "5h", 0),
    _run_rec("r08", "fastspeech2", "phoneme", "1h", 0),
    _run_rec("r02", "vits", "phoneme", "9h", 0),
    _run_rec("r05", "vits", "grapheme", "9h", 0),
    _run_rec("r22", "vits", "phoneme", "9h", 1),
    _run_rec("r23", "vits", "phoneme", "9h", 2),
]


# -- the headline ------------------------------------------------------------

def test_the_headline_is_the_end_to_end_contrast_not_the_vocoded_one():
    """Both VITS and FastSpeech 2 hold a phonemic and a graphemic arm at 9 h.
    The one a listener should start from has no vocoder between the model and
    what they hear."""
    got = _run(_page(FULL))
    head = [g for g in got["groups"] if g["headline"]]
    assert len(head) == 1, got["groups"]
    assert "VITS" in head[0]["text"]
    assert "FastSpeech 2" not in head[0]["text"]
    ids = [a["run_id"] for a in got["arms"]]
    assert ids[:3] == ["", "r02", "r05"], ids


def test_no_group_is_the_headline_when_no_group_holds_both_arms():
    got = _run(_page([_run_rec("r02", "vits", "phoneme", "9h", 0),
                      _run_rec("r22", "vits", "phoneme", "9h", 1)]))
    assert not any(g["headline"] for g in got["groups"])
    assert not any("The comparison" in g["text"] for g in got["groups"])


def test_the_headline_falls_to_a_mel_only_group_when_that_is_all_there_is():
    got = _run(_page([_run_rec("r01", "fastspeech2", "phoneme", "9h", 0),
                      _run_rec("r04", "fastspeech2", "grapheme", "9h", 0)]))
    head = [g for g in got["groups"] if g["headline"]]
    assert len(head) == 1
    assert "FastSpeech 2" in head[0]["text"]


# -- seed replicas -----------------------------------------------------------

def test_a_seed_replica_is_named_as_training_noise():
    got = _run(_page(FULL))
    texts = [g["text"] for g in got["groups"]]
    assert any("seed 1" in t and "training noise" in t for t in texts), texts
    assert any("seed 2" in t for t in texts)


def test_another_architecture_at_another_rung_is_not_called_a_seed_replica():
    """r07 and r08 carry seed 0, the same as the headline, and are a different
    rung. Nothing about them measures training noise."""
    got = _run(_page(FULL))
    for g in got["groups"]:
        if "5 hour" in g["text"] or "1 hour" in g["text"]:
            assert "training noise" not in g["text"], g["text"]


def test_a_replica_at_a_different_rung_is_not_a_replica():
    runs = [_run_rec("r02", "vits", "phoneme", "9h", 0),
            _run_rec("r05", "vits", "grapheme", "9h", 0),
            _run_rec("rxx", "vits", "phoneme", "1h", 7)]
    got = _run(_page(runs))
    for g in got["groups"]:
        if "1 hour" in g["text"]:
            assert "training noise" not in g["text"], g["text"]


# -- measured against unmeasured --------------------------------------------

def test_a_measured_error_rate_names_the_run_it_was_measured_on():
    got = _run(_page(FULL))
    w = got["warnings"]
    assert "r01 at 0.993" in w, w
    assert "r04 at 0.992" in w


def test_the_unmeasured_arms_are_named_as_unmeasured():
    """r07 and r08 share the objective, and no error rate was computed for
    them. Lending them their neighbour's number would be a fabricated
    measurement on a public page."""
    got = _run(_page(FULL))
    w = got["warnings"]
    assert "were not measured" in w, w
    assert "r07" in w and "r08" in w
    assert "r07 at" not in w
    assert "r08 at" not in w


def test_an_unmeasured_architecture_claims_no_rate_at_all():
    got = _run(_page([_run_rec("r99", "matcha", "phoneme", "9h", 0)]))
    w = got["warnings"]
    assert "No intelligibility measurement has been run" in w, w
    assert "0.99" not in w


def test_an_end_to_end_only_page_carries_no_unintelligibility_banner():
    got = _run(_page([_run_rec("r02", "vits", "phoneme", "9h", 0),
                      _run_rec("r05", "vits", "grapheme", "9h", 0)]))
    assert "not intelligible" not in got["warnings"], got["warnings"]


# -- the page does not describe audio that is not there ---------------------

def test_a_silent_mel_arm_is_not_said_to_play():
    got = _run(_page([_run_rec("r01", "fastspeech2", "phoneme", "9h", 0)],
                     audio=False))
    w = got["warnings"]
    assert "have no audio here" in w, w
    assert "Those arms play" not in w


def test_an_audible_mel_arm_is_said_to_play():
    got = _run(_page([_run_rec("r01", "fastspeech2", "phoneme", "9h", 0)]))
    assert "Those arms play" in got["warnings"]


# -- rungs -------------------------------------------------------------------

def test_one_hour_is_one_hour_not_one_hours():
    got = _run(_page(FULL))
    blob = " ".join(g["text"] for g in got["groups"]) + got["runs_dl"]
    assert "1 hours" not in blob, blob
    assert "1 hour" in blob


def test_less_data_is_claimed_only_below_the_headline_rung():
    got = _run(_page(FULL))
    for g in got["groups"]:
        if "Less data" in g["text"]:
            assert "5 hour" in g["text"] or "1 hour" in g["text"], g["text"]


def test_a_lone_arm_at_the_top_rung_is_not_called_less_data():
    got = _run(_page([_run_rec("r02", "vits", "phoneme", "9h", 0),
                      _run_rec("r05", "vits", "grapheme", "9h", 0),
                      _run_rec("r09", "matcha", "phoneme", "9h", 0)]))
    for g in got["groups"]:
        if "Matcha" in g["text"]:
            assert "Less data" not in g["text"], g["text"]
            assert "no contrast to read" in g["text"], g["text"]


def test_an_unrecorded_rung_says_so_and_does_not_group_by_it():
    runs = [_run_rec("r02", "vits", "phoneme", "", None),
            _run_rec("r05", "vits", "grapheme", "", None)]
    got = _run(_page(runs))
    assert "No ladder rung is recorded" in got["warnings"]
    head = [g for g in got["groups"] if g["headline"]]
    assert len(head) == 1
    assert "rung not recorded" not in head[0]["text"], head[0]["text"]


# -- the arm labels ----------------------------------------------------------

def test_every_arm_says_which_input_it_was_given():
    got = _run(_page(FULL))
    model_arms = [a for a in got["arms"] if a["run_id"]]
    assert len(model_arms) == len(FULL)
    for a in model_arms:
        assert a["label"] in ("Phonemic front end", "Graphemic, raw characters")


def test_every_arm_says_its_architecture_rung_and_seed():
    got = _run(_page(FULL))
    for a in got["arms"]:
        if not a["run_id"]:
            continue
        assert "seed" in a["sub"], a
        assert ("hour" in a["sub"] or "minute" in a["sub"]), a
        assert ("VITS" in a["sub"] or "FastSpeech 2" in a["sub"]), a


def test_the_natural_recording_comes_first_and_is_labelled():
    got = _run(_page(FULL))
    assert got["arms"][0]["label"] == "Natural speech"
    assert got["groups"][0]["text"].startswith("The speaker")


# -- which vocoder made the waveforms, page-level ---------------------------

def test_the_page_names_the_vocoder_it_actually_used():
    """One vocoder per language. A Marathi page must name its own rather than
    inherit a run id written into the HTML."""
    got = _run(_page(FULL))
    assert "r06@18000" in got["provenance"], got["provenance"]
    assert "HiFi-GAN" in got["provenance"]


def test_a_marathi_page_names_the_marathi_vocoder():
    runs = [_run_rec("r11", "fastspeech2", "phoneme", "9h", 0)]
    data = _page(runs)
    data["language"] = "marathi"
    for sent in data["sentences"]:
        sent["runs"]["r11"]["vocoder"] = "hifigan:r17@18000"
    got = _run(data)
    assert "r17@18000" in got["provenance"], got["provenance"]
    assert "r06" not in got["provenance"]


def test_the_page_says_end_to_end_arms_have_no_vocoder_in_the_path():
    got = _run(_page([_run_rec("r02", "vits", "phoneme", "9h", 0)]))
    assert "no vocoder" in got["provenance"], got["provenance"]
    assert "HiFi-GAN" not in got["provenance"]


def test_a_griffin_lim_page_says_placeholder_at_the_page_level_too():
    runs = [_run_rec("r01", "fastspeech2", "phoneme", "9h", 0)]
    data = _page(runs)
    for sent in data["sentences"]:
        sent["runs"]["r01"]["vocoder"] = "griffin-lim"
    got = _run(data)
    assert "Griffin-Lim" in got["provenance"]
    assert "placeholder" in got["provenance"]


def test_a_page_with_no_recorded_vocoder_says_that_rather_than_guessing():
    runs = [_run_rec("r01", "fastspeech2", "phoneme", "9h", 0)]
    data = _page(runs, audio=False)
    got = _run(data)
    assert "No vocoder is recorded" in got["provenance"] \
        or "unprocessed" in got["provenance"], got["provenance"]
    assert "HiFi-GAN" not in got["provenance"]
