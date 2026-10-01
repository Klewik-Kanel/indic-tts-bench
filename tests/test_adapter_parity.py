"""Two arms of a comparison must see the same data, and where they cannot, the
difference must be the one that was declared.

`assert_budget_matched` already refuses runs whose budget fields differ.
Nothing checked that two runs being compared actually saw the same utterances
in the same order, which is the one place this study cannot afford a silent
divergence: a data difference inside what the paper reports as an architecture
or input-representation difference.

The checks are in two layers, because only one of them is guaranteeable.

**Within an architecture, across input representations** -- r01 against r04,
r02 against r05, and every ladder pair -- the batches must be identical. These
runs share a sample rate, and the input representation is applied after
batching, so anything other than identical batches is a bug.

**Across architectures** the batches are NOT identical and cannot be, because
VITS runs natively at 16 kHz while FastSpeech 2 and Matcha run at 22.05 kHz.
That is a declared deviation. What the tests here pin down is that the sample
rate is the ONLY reason they differ, and the exact size of the difference, so
that a future edit which changes it fails a test rather than quietly changing
what the paper compares.

The collate-level tests need torch and an audio stack; the batching-level ones
need neither and run anywhere.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from src.train import batching
from src.train.batching import HOP_LENGTH, frames_for

BUDGET_FRAMES = 12_000          # configs/*.yaml, identical across every run
SEED = 1234


def manifest(tmp: pathlib.Path, n: int = 24) -> pathlib.Path:
    """Utterances of varied length, so batching has real decisions to make."""
    p = tmp / "train.tsv"
    rows = ["id\twav22\twav16\tseconds\ttext"]
    for i in range(n):
        secs = 1.0 + (i % 7) * 1.5
        rows.append(f"u{i:03d}\tw/{i}.wav\tw/{i}.wav\t{secs}\tकमल नगर")
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def uid_stream(mpath: pathlib.Path, sample_rate: int, steps: int = 30):
    utts = batching.load_manifest(mpath, sample_rate)
    return [tuple(u.uid for u in b)
            for _, b in batching.batch_stream(utts, BUDGET_FRAMES, SEED, steps)]


# --- the comparison the dissertation actually makes --------------------------

def test_the_two_input_arms_of_one_architecture_see_identical_batches(tmp_path):
    """r01 vs r04, r02 vs r05: the ablation arms differ only in input.

    The input representation is applied by the text encoder inside collate,
    after the batch exists. If it ever reached batching, the ablation would
    compare input representations AND data order at the same time.
    """
    m = manifest(tmp_path)
    for sr in (22_050, 16_000):
        assert uid_stream(m, sr) == uid_stream(m, sr), \
            "the same run configuration produced two different batch streams"


def test_batching_reads_nothing_that_the_input_representation_changes(tmp_path):
    """Stated as a property of the signature, not of one run.

    make_batches takes the utterances, the frame budget, the seed and the
    epoch. None of those carries the input representation, which is what makes
    the arms comparable.
    """
    m = manifest(tmp_path)
    utts = batching.load_manifest(m, 22_050)
    a = batching.make_batches(utts, BUDGET_FRAMES, SEED, 0)
    b = batching.make_batches(utts, BUDGET_FRAMES, SEED, 0)
    assert [[u.uid for u in x] for x in a] == [[u.uid for u in x] for x in b]


def test_a_different_seed_really_does_change_the_order(tmp_path):
    """The negative control for the test above: if nothing changed the batches,
    the parity tests would pass for the wrong reason."""
    m = manifest(tmp_path)
    utts = batching.load_manifest(m, 22_050)
    a = batching.make_batches(utts, BUDGET_FRAMES, SEED, 0)
    b = batching.make_batches(utts, BUDGET_FRAMES, SEED + 1, 0)
    assert [[u.uid for u in x] for x in a] != [[u.uid for u in x] for x in b]


# --- across architectures: the declared 16 kHz deviation, measured ----------

def test_both_rates_draw_from_the_same_utterance_set(tmp_path):
    """The grouping differs; the corpus must not."""
    m = manifest(tmp_path)
    at22 = {u for b in uid_stream(m, 22_050) for u in b}
    at16 = {u for b in uid_stream(m, 16_000) for u in b}
    assert at22 == at16


def test_the_frame_count_scales_with_the_sample_rate_and_nothing_else(tmp_path):
    for secs in (1.0, 2.5, 7.3, 15.0):
        f22 = frames_for(secs, 22_050)
        f16 = frames_for(secs, 16_000)
        assert f22 == math.ceil(secs * 22_050 / HOP_LENGTH)
        assert f16 == math.ceil(secs * 16_000 / HOP_LENGTH)
        # Within one frame of the rate ratio, the rest being the ceiling.
        assert abs(f16 - f22 * 16_000 / 22_050) <= 1


def test_the_frame_budget_is_not_an_audio_budget():
    """The confound recorded in RESULTS.md on 2 October, pinned as a test.

    batch_frames is 12,000 for every run and `assert_budget_matched` enforces
    that the NUMBER is identical. A frame is 256 samples at the run's own rate,
    so the same number of frames is a different amount of audio: 139.32 s at
    22.05 kHz against 192.00 s at 16 kHz, which is 37.8% more audio per step
    for every VITS run.

    This test does not assert that the design is right. It asserts the size of
    the asymmetry, so that changing it is a deliberate act that updates this
    number rather than a silent change to what the paper compares.
    """
    secs22 = BUDGET_FRAMES / (22_050 / HOP_LENGTH)
    secs16 = BUDGET_FRAMES / (16_000 / HOP_LENGTH)
    assert round(secs22, 2) == 139.32
    assert round(secs16, 2) == 192.00
    assert round(secs16 / secs22, 4) == round(22_050 / 16_000, 4) == 1.3781
    # What an audio-equalised 16 kHz budget would have been, if that route is
    # ever taken: it is not the current design.
    assert round(BUDGET_FRAMES * 16_000 / 22_050) == 8_707


# --- collate: the same batch through two adapters ---------------------------

torch = pytest.importorskip("torch", reason="collate parity needs torch")
librosa = pytest.importorskip("librosa", reason="collate parity needs librosa")
sf = pytest.importorskip("soundfile", reason="collate parity needs soundfile")


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """A tiny corpus on disk, with the adapters pointed at it.

    Both wav columns name the same file and both configs below use 22.05 kHz,
    so the rate deviation is held constant and the test measures the collate
    contract rather than re-measuring the sample rate.
    """
    import numpy as np
    from src.train import adapters

    root = tmp_path / "interim" / "hindi" / "w"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(24):
        secs = 1.0 + (i % 7) * 1.5
        t = np.arange(int(secs * 22_050)) / 22_050
        # A couple of harmonics plus a little noise: enough structure for the
        # pitch tracker to return something finite.
        y = (0.3 * np.sin(2 * np.pi * 140 * t)
             + 0.1 * np.sin(2 * np.pi * 280 * t)
             + 0.01 * rng.standard_normal(t.size)).astype("float32")
        sf.write(root / f"{i}.wav", y, 22_050)

    monkeypatch.setattr(adapters, "HERE", tmp_path)
    monkeypatch.setattr(adapters, "INTERIM", tmp_path / "interim")
    return manifest(tmp_path)


def _cfg(arch: str, repr_: str) -> dict:
    return {"run_id": f"parity_{arch}_{repr_}", "architecture": arch,
            "language": "hindi", "input_repr": repr_, "data": "9h",
            "sample_rate": 22_050, "batch_frames": BUDGET_FRAMES,
            "max_steps": 2, "lr": 1e-4, "warmup_steps": 1, "grad_clip": 1.0,
            "precision": "fp32", "seed": SEED, "merge_nukta": False}


def _first_batch(mpath, cfg):
    utts = batching.load_manifest(mpath, int(cfg["sample_rate"]))
    _, batch = next(iter(batching.batch_stream(
        utts, int(cfg["batch_frames"]), int(cfg["seed"]), 1)))
    return batch


def test_one_batch_through_both_collates_carries_the_same_utterances(corpus):
    """The check PLAN-phase3.md asked for, with the rate held constant.

    FastSpeech 2 returns mel as [B, frames, mels] and VITS returns spec as
    [B, freq, frames]. Different layouts of the same frames: the frame counts
    must agree item by item, or an architecture comparison contains a data
    difference.
    """
    from src.train import adapters
    from src.train.text import TextEncoder

    fs_cfg, vi_cfg = _cfg("fastspeech2", "phoneme"), _cfg("vits", "phoneme")
    batch = _first_batch(corpus, fs_cfg)
    assert batch == _first_batch(corpus, vi_cfg), \
        "the two adapters were not even given the same batch"

    enc = TextEncoder.for_config("hindi", "phoneme")
    fs = adapters.FastSpeech2Adapter().collate(batch, enc, fs_cfg)
    vi = adapters.VitsAdapter().collate(batch, enc, vi_cfg)

    assert fs["mel_lengths"].tolist() == vi["spec_lens"].tolist()
    assert torch.equal(fs["text_input"], vi["tokens"])
    assert fs["text_lengths"].tolist() == vi["token_lens"].tolist()
    assert fs["mel_input"].shape[1] == vi["spec"].shape[2]


def test_the_two_input_arms_tokenise_differently_but_frame_identically(corpus):
    """The ablation, at the level of one batch: same audio, different tokens.

    If the frame counts moved with the input representation, the ablation would
    not be an ablation.
    """
    from src.train import adapters
    from src.train.text import TextEncoder

    ph_cfg, gr_cfg = _cfg("fastspeech2", "phoneme"), _cfg("fastspeech2", "grapheme")
    batch = _first_batch(corpus, ph_cfg)
    assert batch == _first_batch(corpus, gr_cfg)

    ph = adapters.FastSpeech2Adapter().collate(
        batch, TextEncoder.for_config("hindi", "phoneme"), ph_cfg)
    gr = adapters.FastSpeech2Adapter().collate(
        batch, TextEncoder.for_config("hindi", "grapheme"), gr_cfg)

    assert ph["mel_lengths"].tolist() == gr["mel_lengths"].tolist()
    assert ph["mel_input"].shape == gr["mel_input"].shape
    assert not torch.equal(ph["text_input"], gr["text_input"]), \
        "the phonemic and graphemic arms produced identical token ids"
