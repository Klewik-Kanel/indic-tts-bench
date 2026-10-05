#!/usr/bin/env python3
"""Which bundles become arms of the listening test, and which get a vocoder.

exports/ holds the vocoders beside the voices. A vocoder has no text side, so
asking it to speak raises SystemExit, and SystemExit is not an Exception: it
walks straight past the per-sentence handler and kills the whole render. The
cost is not the one bad arm. It is that data.json is written at the end, so a
render that had already produced every clip of every real arm leaves the page
serving whatever it served yesterday, which reads as a result and is not one.

So the vocoder is dropped from the bundle list by its manifest, before any
weights load, and the vocoder is attached only to the arms whose manifest says
they need one.

No torch and no audio: `load` and `write_wav` are replaced with stubs that
record what they were asked for.
"""

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import render_demo                                                # noqa: E402


class _Mel:
    shape = (80, 7)


class _Spoken:
    def __init__(self, voiced, vocoded):
        self.tokens = ["a", "b"]
        self.seconds_elapsed = 0.25
        self.mel = _Mel()
        self.sample_rate = 22050
        self.waveform = [0.0] * 16 if voiced else None
        self.audio_seconds = 0.5 if voiced else 0.0
        # Only a vocoded mel reports a vocoder. VITS makes its own waveform
        # and reports none, which is what "end-to-end" on the page means.
        self.extras = ({"vocoder": {"run_id": "r06", "step": 18000}}
                       if vocoded else {})


class _Loaded:
    """What render_demo gets back from `load`, carrying its own manifest."""

    def __init__(self, man, root, vocoder):
        self.manifest = man
        self.root = root
        self.vocoder = vocoder
        self.needs_vocoder = bool(man["needs_vocoder"])
        self.describes = man["run_id"] + " described"
        self.sample_rate = int(man["sample_rate"])
        # VITS makes its own waveform; a mel-only arm only does with a vocoder.
        self._voiced = (not self.needs_vocoder) or vocoder is not None

    def synthesize(self, text, **kw):
        return _Spoken(self._voiced, self.vocoder is not None)


def _bundle(root: pathlib.Path, run_id, arch, needs, lang="hindi"):
    root.mkdir(parents=True, exist_ok=True)
    man = {"bundle_version": 3, "run_id": run_id, "architecture": arch,
           "language": lang, "input_repr": "none" if needs is None else "phoneme",
           "sample_rate": 22050, "needs_vocoder": bool(needs),
           "audio": {"n_mels": 80}}
    (root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return root


def _demo_set(path: pathlib.Path):
    path.write_text(json.dumps({"split": "dev", "criterion": "x", "chosen": [
        {"id": "hi_1", "text": "नमक", "seconds": 1.0, "words": 1,
         "medial": 1, "final": 0, "medial_words": ["नमक"]},
    ]}), encoding="utf-8")
    return path


def _run(tmp_path, monkeypatch, bundles, vocoder=None):
    asked = []

    def fake_load(root, device="cpu", vocoder=None):
        asked.append((pathlib.Path(root).name,
                      pathlib.Path(vocoder).name if vocoder else None))
        man = json.loads((pathlib.Path(root) / "manifest.json")
                         .read_text(encoding="utf-8"))
        return _Loaded(man, pathlib.Path(root), vocoder)

    written = []
    monkeypatch.setattr(render_demo, "load", fake_load)
    monkeypatch.setattr(render_demo, "write_wav",
                        lambda p, w, sr: written.append(pathlib.Path(p).name))

    out = tmp_path / "site"
    argv = ["--lang", "hindi", "--set", str(_demo_set(tmp_path / "set.json")),
            "--out", str(out), "--bundles", *[str(b) for b in bundles]]
    if vocoder:
        argv += ["--vocoder", str(vocoder)]
    code = render_demo.main(argv)
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    return code, asked, written, data


# -- the vocoder is not an arm ---------------------------------------------

def test_a_vocoder_bundle_never_reaches_load(tmp_path, monkeypatch):
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)
    code, asked, _, data = _run(tmp_path, monkeypatch, [fs2, voc], vocoder=voc)
    names = [n for n, _ in asked]
    assert "r06" not in names, "the vocoder was loaded as if it could speak"
    assert names == ["r01"]
    assert [r["run_id"] for r in data["runs"]] == ["r01"]


def test_data_json_is_written_even_with_a_vocoder_in_the_list(tmp_path,
                                                              monkeypatch):
    """The defect this guards: the render died on the vocoder and the page
    kept serving the previous day's data.json."""
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    vits = _bundle(tmp_path / "r02", "r02", "vits", False)
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)
    code, _, _, data = _run(tmp_path, monkeypatch, [fs2, vits, voc],
                            vocoder=voc)
    assert code == 0
    assert len(data["sentences"]) == 1
    assert set(data["sentences"][0]["runs"]) == {"r01", "r02"}


def test_the_vocoder_is_skipped_without_a_vocoder_flag_too(tmp_path,
                                                           monkeypatch):
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    code, asked, _, data = _run(tmp_path, monkeypatch, [voc, fs2])
    assert [n for n, _ in asked] == ["r01"]


# -- who gets the vocoder --------------------------------------------------

def test_only_a_mel_only_arm_is_given_the_vocoder(tmp_path, monkeypatch):
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    vits = _bundle(tmp_path / "r02", "r02", "vits", False)
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)
    _, asked, _, _ = _run(tmp_path, monkeypatch, [fs2, vits], vocoder=voc)
    assert dict(asked) == {"r01": "r06", "r02": None}


def test_a_mel_only_arm_is_labelled_with_the_vocoder_not_end_to_end(
        tmp_path, monkeypatch):
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)
    _, _, written, data = _run(tmp_path, monkeypatch, [fs2], vocoder=voc)
    entry = data["sentences"][0]["runs"]["r01"]
    assert entry["vocoder"] == "hifigan:r06@18000"
    assert entry["audio"] == "audio/r01/hi_1.wav"
    assert written == ["hi_1.wav"]
    assert data["vocoder_bundle"] == str(voc)


def test_an_end_to_end_arm_stays_end_to_end(tmp_path, monkeypatch):
    vits = _bundle(tmp_path / "r02", "r02", "vits", False)
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)
    _, _, _, data = _run(tmp_path, monkeypatch, [vits], vocoder=voc)
    assert data["sentences"][0]["runs"]["r02"]["vocoder"] == "end-to-end"


def test_without_a_vocoder_a_mel_only_arm_is_silent_and_says_so(tmp_path,
                                                                monkeypatch):
    """Not a regression: the honest state the publish audit then refuses."""
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    _, _, written, data = _run(tmp_path, monkeypatch, [fs2])
    entry = data["sentences"][0]["runs"]["r01"]
    assert entry["audio"] is None
    assert entry["vocoder"] is None
    assert written == []


# -- the language filter still works, and reads the manifest ----------------

def test_another_language_is_skipped_before_its_weights_load(tmp_path,
                                                             monkeypatch):
    mar = _bundle(tmp_path / "r11", "r11", "fastspeech2", True, lang="marathi")
    hin = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    _, asked, _, data = _run(tmp_path, monkeypatch, [mar, hin])
    assert [n for n, _ in asked] == ["r01"], "marathi weights were loaded"
    assert [r["run_id"] for r in data["runs"]] == ["r01"]


def test_a_rate_mismatch_is_still_fatal(tmp_path, monkeypatch):
    """The old code caught SystemExit from attach_vocoder and retried without
    a vocoder, so a genuinely wrong pairing became a silent arm. The decision
    is made from the manifest now, so the refusal reaches the caller."""
    fs2 = _bundle(tmp_path / "r01", "r01", "fastspeech2", True)
    voc = _bundle(tmp_path / "r06", "r06", "hifigan", None)

    def angry_load(root, device="cpu", vocoder=None):
        if vocoder is not None:
            raise SystemExit("sample rate mismatch: r01 is 22050 Hz")
        man = json.loads((pathlib.Path(root) / "manifest.json")
                         .read_text(encoding="utf-8"))
        return _Loaded(man, pathlib.Path(root), None)

    monkeypatch.setattr(render_demo, "load", angry_load)
    monkeypatch.setattr(render_demo, "write_wav", lambda *a: None)
    with pytest.raises(SystemExit):
        render_demo.main(["--lang", "hindi",
                          "--set", str(_demo_set(tmp_path / "s.json")),
                          "--out", str(tmp_path / "site"),
                          "--bundles", str(fs2), "--vocoder", str(voc)])


def test_every_vocoder_in_the_list_is_skipped(tmp_path, monkeypatch):
    from src.export.synthesize import VOCODER_ARCHITECTURES
    bundles = [_bundle(tmp_path / f"v{i}", f"v{i}", arch, None)
               for i, arch in enumerate(sorted(VOCODER_ARCHITECTURES))]
    bundles.append(_bundle(tmp_path / "r02", "r02", "vits", False))
    _, asked, _, _ = _run(tmp_path, monkeypatch, bundles)
    assert [n for n, _ in asked] == ["r02"]
