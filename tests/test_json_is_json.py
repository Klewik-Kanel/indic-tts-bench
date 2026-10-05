#!/usr/bin/env python3
"""data.json has to be readable by a browser, not only by Python.

Python's json writes Infinity, -Infinity and NaN, and reads all three back
without complaint. None of them is JSON. A browser's JSON.parse refuses the
whole document on any one of them, so a single bad field in a single run blanks
the entire listening test: the page showed "Could not load data.json" and no
arms at all.

The route in: r08 trained with `cfg["config_hash"] = inf` on 2 October. That
float went into its config, its checkpoints, its bundle manifest, and finally
into data.json as the bare token `Infinity`.

Two holes, both closed here. render_demo must not write such a token. And
publish_space must not accept one: it audited the file with Python's permissive
reader, so it reported "checks pass" on a page that could not load.
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import publish_space                                            # noqa: E402
import render_demo                                              # noqa: E402

INF, NINF, NAN = float("inf"), float("-inf"), float("nan")


# -- the sanitiser -----------------------------------------------------------

def test_infinity_becomes_a_string_that_says_so():
    out = render_demo.json_safe({"config_hash": INF})
    assert out == {"config_hash": "not-recoverable"}
    assert isinstance(out["config_hash"], str)


def test_negative_infinity_and_nan_too():
    out = render_demo.json_safe({"a": NINF, "b": NAN})
    assert out == {"a": "not-recoverable", "b": "not-recoverable"}


def test_it_reaches_inside_lists_and_nested_dicts():
    out = render_demo.json_safe(
        {"runs": [{"run_id": "r08", "config_hash": INF},
                  {"run_id": "r01", "config_hash": "e3bbebc98825"}]})
    assert out["runs"][0]["config_hash"] == "not-recoverable"
    assert out["runs"][1]["config_hash"] == "e3bbebc98825"


def test_ordinary_values_are_untouched():
    src = {"s": "x", "i": 3, "f": 1.5, "t": True, "n": None, "l": [1, "two"]}
    assert render_demo.json_safe(src) == src


def test_a_finite_float_survives_including_zero_and_negatives():
    src = {"a": 0.0, "b": -2.5, "c": 1e308}
    assert render_demo.json_safe(src) == src


def test_the_replacement_is_a_string_not_a_null():
    """null would say the field was empty. It was not: it held a number that
    is not a hash, and RESULTS.md records why."""
    out = render_demo.json_safe({"config_hash": INF})
    assert out["config_hash"] is not None


# -- what a browser would do with the result ---------------------------------

def _js_parses(text: str) -> bool:
    import shutil
    node = shutil.which("node")
    if node is None:
        raise pytest.importorskip("node")
    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "d.json"
        f.write_text(text, encoding="utf-8")
        p = subprocess.run(
            [node, "-e",
             "const fs=require('fs');"
             "try{JSON.parse(fs.readFileSync(process.argv[1],'utf8'));"
             "console.log('OK');}catch(e){console.log('FAIL '+e.message);}",
             str(f)],
            capture_output=True, text=True, timeout=30)
        return p.stdout.strip().startswith("OK")


def test_the_raw_python_output_is_what_the_browser_refused():
    raw = json.dumps({"config_hash": INF})
    assert "Infinity" in raw
    assert _js_parses(raw) is False


def test_the_sanitised_output_is_what_the_browser_accepts():
    safe = json.dumps(render_demo.json_safe({"config_hash": INF}),
                      allow_nan=False)
    assert _js_parses(safe) is True


def test_allow_nan_false_is_the_backstop():
    """If anything ever gets past json_safe, writing must fail loudly rather
    than publish a page that cannot load."""
    src = pathlib.Path(HERE / "scripts" / "render_demo.py").read_text()
    assert "allow_nan=False" in src
    with pytest.raises(ValueError):
        json.dumps({"x": INF}, allow_nan=False)


# -- the audit must be as strict as the browser ------------------------------

def _site(tmp, data_text):
    root = pathlib.Path(tmp)
    (root / "index.html").write_text("<html></html>", encoding="utf-8")
    (root / "data.json").write_text(data_text, encoding="utf-8")
    f = root / "audio" / "r02" / "s1.wav"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"RIFF....WAVE")
    return root


def _ok():
    return {"language": "hindi", "split": "dev", "generated_utc": "now",
            "runs": [{"run_id": "r02", "architecture": "vits"}],
            "sentences": [{"id": "s1", "runs": {"r02": {
                "audio": "audio/r02/s1.wav", "vocoder": "end-to-end"}}}],
            "failures": []}


def test_the_audit_refuses_an_infinity_the_browser_would_refuse():
    """This is the hole: json.loads accepted it, so the dry run said checks
    pass on a page that rendered nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _ok()
        d["runs"][0]["config_hash"] = INF
        root = _site(tmp, json.dumps(d))
        bad, _ = publish_space.audit(root)
        assert any("not readable JSON" in b for b in bad), bad
        assert any("Infinity" in b for b in bad), bad


def test_the_audit_refuses_nan_too():
    with tempfile.TemporaryDirectory() as tmp:
        d = _ok()
        d["runs"][0]["config_hash"] = NAN
        root = _site(tmp, json.dumps(d))
        bad, _ = publish_space.audit(root)
        assert any("not readable JSON" in b for b in bad), bad


def test_the_audit_still_accepts_an_ordinary_file():
    with tempfile.TemporaryDirectory() as tmp:
        d = _ok()
        d["runs"][0]["config_hash"] = "e3bbebc98825"
        root = _site(tmp, json.dumps(d))
        import os
        import time
        now = time.time()
        os.utime(pathlib.Path(root) / "data.json", (now, now))
        for w in pathlib.Path(root).rglob("*.wav"):
            os.utime(w, (now - 5, now - 5))
        assert publish_space.audit(root)[0] == []


def test_python_would_have_accepted_the_bad_file_which_is_the_point():
    assert json.loads('{"h": Infinity}')["h"] == INF


# -- the manifest ------------------------------------------------------------

def test_a_hash_leaves_the_bundle_as_text():
    from src.export.bundle import hash_text
    assert hash_text(INF) == "not-recoverable"
    assert hash_text(NINF) == "not-recoverable"
    assert hash_text(NAN) == "not-recoverable"
    assert hash_text("e3bbebc98825") == "e3bbebc98825"
    assert hash_text("") == ""
    assert hash_text(None) == ""


def test_a_finite_number_is_kept_rather_than_discarded():
    """An all-digit hash parsed as a number is still the hash."""
    from src.export.bundle import hash_text
    assert hash_text(1234.0) == "1234.0"
    assert hash_text(0.0) == "0.0"
