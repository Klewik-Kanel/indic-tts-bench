#!/usr/bin/env python3
"""The Pages workflow's own gate, executed rather than read.

Two states look alike from a distance and must not be treated alike.

Nothing rendered yet is normal. The site is produced on the GPU machine and
committed from there, so between a page edit and the next render the
repository holds index.html and nothing to put in it. Failing there emailed a
failure on every push and said nothing true: nothing was published, so nothing
was broken.

A half-committed site is a fault. Wavs without data.json renders an empty
listening test, data.json without wavs renders players that fall silent, and a
data.json carrying Infinity renders nothing at all, because JSON.parse refuses
the whole document on a token Python's json accepts.

The gate used `set -u` and not `set -e`, so the strict JSON check printed its
error, exited 1, and the next line wrote ready=true anyway. Reading the script
would not have shown that. Running it does, which is what these tests do.

Skipped where PyYAML or bash is unavailable.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
WF = HERE / ".github" / "workflows" / "pages.yml"


def _gate() -> str:
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    steps = doc["jobs"]["publish"]["steps"]
    gate = [s for s in steps if s.get("id") == "check"]
    assert len(gate) == 1, "the workflow has no single check step"
    return gate[0]["run"]


def _site(root: pathlib.Path, data=None, clips=(), page=True):
    if page:
        (root / "space_static").mkdir(parents=True, exist_ok=True)
        (root / "space_static" / "index.html").write_text("<html>", encoding="utf-8")
    for rel in clips:
        f = root / "space_static" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"RIFF....WAVE")
    if data is not None:
        (root / "space_static" / "data.json").write_text(
            json.dumps(data), encoding="utf-8")
    return root


def _run(root: pathlib.Path):
    """(exit code, the outputs the step set)."""
    bash = shutil.which("bash")
    if bash is None:
        raise pytest.importorskip("bash")
    script = root / "gate.sh"
    script.write_text(_gate(), encoding="utf-8")
    out = root / "gh_output"
    out.write_text("", encoding="utf-8")
    proc = subprocess.run([bash, str(script)], cwd=root, capture_output=True,
                          text=True, timeout=120,
                          env={"PATH": "/usr/bin:/bin:/usr/local/bin",
                               "GITHUB_OUTPUT": str(out)})
    got = dict(line.split("=", 1) for line in
               out.read_text(encoding="utf-8").splitlines() if "=" in line)
    return proc.returncode, got, proc.stdout + proc.stderr


def _ok_data(audio="audio/r02/s1.wav"):
    return {"language": "hindi",
            "runs": [{"run_id": "r02", "config_hash": "a1"}],
            "sentences": [{"id": "s1", "runs": {"r02": {"audio": audio}}}]}


# -- nothing rendered yet is not a failure ----------------------------------

def test_a_page_with_no_render_skips_rather_than_failing():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp))
        code, out, log = _run(root)
        assert code == 0, log
        assert out.get("ready") == "false", out
        assert "Skipping the deploy" in log


def test_the_skip_says_what_to_do_about_it():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp))
        _, _, log = _run(root)
        assert "render_demo.py" in log
        assert "data.json" in log


# -- a half-committed site is ----------------------------------------------

def test_wavs_without_data_json_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp), clips=["audio/r02/s1.wav"])
        code, out, log = _run(root)
        assert code == 1, log
        assert out.get("ready") is None, out


def test_data_json_without_wavs_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp), data=_ok_data())
        code, out, log = _run(root)
        assert code == 1, log
        assert "silent" in log
        assert out.get("ready") is None


def test_a_missing_index_html_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "space_static").mkdir()
        code, _, log = _run(root)
        assert code == 1, log
        assert "index.html" in log


# -- the JSON a browser would refuse ----------------------------------------

def test_infinity_in_data_json_fails_and_sets_nothing():
    """The bug this catches: the check printed its error, exited 1, and the
    next line wrote ready=true anyway, so it would have deployed."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _ok_data()
        d["runs"][0]["config_hash"] = float("inf")
        root = _site(pathlib.Path(tmp), data=d, clips=["audio/r02/s1.wav"])
        code, out, log = _run(root)
        assert code == 1, log
        assert out.get("ready") is None, f"would have deployed: {out}"
        assert "Infinity" in log


def test_nan_in_data_json_fails():
    with tempfile.TemporaryDirectory() as tmp:
        d = _ok_data()
        d["runs"][0]["config_hash"] = float("nan")
        root = _site(pathlib.Path(tmp), data=d, clips=["audio/r02/s1.wav"])
        code, out, _ = _run(root)
        assert code == 1
        assert out.get("ready") is None


def test_unparseable_json_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp), clips=["audio/r02/s1.wav"])
        (root / "space_static" / "data.json").write_text("{nope", encoding="utf-8")
        code, out, log = _run(root)
        assert code == 1, log
        assert out.get("ready") is None


def test_a_clip_named_but_not_committed_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp), data=_ok_data("audio/r02/GONE.wav"),
                     clips=["audio/r02/s1.wav"])
        code, out, log = _run(root)
        assert code == 1, log
        assert "GONE.wav" in log
        assert out.get("ready") is None


# -- a complete site publishes ----------------------------------------------

def test_a_complete_site_is_ready():
    with tempfile.TemporaryDirectory() as tmp:
        root = _site(pathlib.Path(tmp), data=_ok_data(),
                     clips=["audio/r02/s1.wav"])
        code, out, log = _run(root)
        assert code == 0, log
        assert out.get("ready") == "true", out


def test_a_clip_recorded_as_silent_is_not_a_missing_file():
    """A mel-only arm rendered without a vocoder records audio: null. That is
    a rendering gap the page labels, not a file this gate should hunt for."""
    with tempfile.TemporaryDirectory() as tmp:
        d = _ok_data()
        d["sentences"][0]["runs"]["r01"] = {"audio": None, "vocoder": None}
        root = _site(pathlib.Path(tmp), data=d, clips=["audio/r02/s1.wav"])
        code, out, log = _run(root)
        assert code == 0, log
        assert out.get("ready") == "true"


# -- the deploy is gated on all of that -------------------------------------

def test_every_deploy_step_is_gated_on_the_check():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    steps = doc["jobs"]["publish"]["steps"]
    deploying = [s for s in steps
                 if "pages" in str(s.get("uses", "")) and "checkout" not in str(s.get("uses", ""))]
    assert len(deploying) == 3, deploying
    for s in deploying:
        assert s.get("if") == "steps.check.outputs.ready == 'true'", s


def test_the_gate_stops_on_the_first_error():
    assert "set -eu" in _gate()


def test_an_upload_is_never_cancelled_mid_flight():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(WF.read_text(encoding="utf-8"))
    assert doc["concurrency"]["cancel-in-progress"] is False
