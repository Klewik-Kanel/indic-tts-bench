#!/usr/bin/env python3
"""r24 and r25 z-score FastSpeech 2's pitch and energy; nothing earlier moves.

On 10 October the objective defect recorded on 3 October was acted on for the
Hindi FastSpeech 2 pair only. Two things have to hold at once, and both are
tested by executing the code rather than reading it:

1. Every one of the 23 configs that trained before the change keeps the hash
   it trained under, and its YAML is unchanged. The new field is out of the
   hash, and out of the file, at its default.
2. A run that sets variance_norm: zscore hands the loss pitch and energy
   z-scored exactly as coqui's own dataset classes would, with statistics over
   its own training rung, while a run that does not hands the loss exactly
   the arrays it always did.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from src.train import adapters, features
from src.train.config import RunConfig, plan_runs
from src.train.runner import load_config

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"

# Copied from handoff/02-STATE-OF-PLAY.md, which copied them from the run
# directories. Literal on purpose: a test that recomputed them from the same
# code it is checking would pass whatever that code did to the hash.
TRAINED_HASHES = {
    "r01": "e3bbebc98825", "r02": "d3873342a10d", "r03": "1d85771a0e91",
    "r04": "6aeae2e66a19", "r05": "05eaea2642e8", "r06": "3120211db4c7",
    "r07": "c1b1edce3d2d", "r08": "52e245223208", "r09": "ed76b539d2bc",
    "r10": "1c378a7604cf", "r11": "93bae73db260", "r12": "414a33d2879d",
    "r13": "40e2939875a1", "r14": "348e61158e4b", "r15": "d72a3563cbfc",
    "r16": "983453d32f43", "r17": "99a3999ca13e", "r18": "3cccd1725aeb",
    "r19": "c2ba36578a66", "r20": "0476f129360d", "r21": "ce7dc788bc86",
    "r22": "500e5e3eae82", "r23": "72f4c9aa7494",
}


def from_yaml(path: pathlib.Path) -> RunConfig:
    d = load_config(path)
    d.pop("config_hash")
    for k in ("deviations", "corrections"):
        d[k] = tuple(d.get(k) or ())
    return RunConfig(**d)


# -- 1. nothing that trained moves -------------------------------------------

@pytest.mark.parametrize("rid", sorted(TRAINED_HASHES))
def test_every_trained_config_still_hashes_to_what_it_trained_under(rid):
    path = CONFIGS / f"{rid}.yaml"
    assert load_config(path)["config_hash"] == TRAINED_HASHES[rid]
    assert from_yaml(path).config_hash() == TRAINED_HASHES[rid]


def test_the_plan_reproduces_every_trained_hash():
    plan = {r.run_id: r.config_hash() for r in plan_runs()}
    for rid, h in TRAINED_HASHES.items():
        if rid == "r03":
            continue                      # future work, not in the plan
        assert plan[rid] == h, rid


def test_the_plan_rewrites_every_trained_yaml_byte_for_byte():
    for r in plan_runs():
        if r.run_id in TRAINED_HASHES:
            on_disk = (CONFIGS / f"{r.run_id}.yaml").read_text(encoding="utf-8")
            assert r.to_yaml() == on_disk, r.run_id
            assert "variance_norm" not in on_disk, r.run_id


# -- the two new runs ----------------------------------------------------------

def test_r24_and_r25_differ_from_their_parents_only_in_the_objective():
    runs = {r.run_id: r for r in plan_runs()}
    allowed = {"run_id", "notes", "variance_norm", "deviations"}
    for new, parent in (("r24", "r01"), ("r25", "r04")):
        a = dataclasses.asdict(runs[new])
        b = dataclasses.asdict(runs[parent])
        assert {k for k in a if a[k] != b[k]} == allowed, new
        assert a["variance_norm"] == "zscore" and b["variance_norm"] == ""
        assert runs[new].config_hash() != runs[parent].config_hash()


def test_the_new_yaml_says_what_it_is_and_reads_back_as_text():
    for rid in ("r24", "r25"):
        cfg = load_config(CONFIGS / f"{rid}.yaml")
        assert cfg["variance_norm"] == "zscore"
        assert cfg["config_hash"] == next(
            r for r in plan_runs() if r.run_id == rid).config_hash()
        assert any("z-scored" in d for d in cfg["deviations"])


def test_variance_norm_is_refused_off_fastspeech2_and_for_unknown_values():
    with pytest.raises(ValueError):
        RunConfig(run_id="x", architecture="vits", language="hindi",
                  input_repr="phoneme", data="9h", variance_norm="zscore")
    with pytest.raises(ValueError):
        RunConfig(run_id="x", architecture="fastspeech2", language="hindi",
                  input_repr="phoneme", data="9h", variance_norm="minmax")


# -- 2. the arithmetic, against coqui's rule ---------------------------------

def coqui_normalize(x, mean, std):
    """coqui-tts 0.27.5, F0Dataset.normalize, transcribed."""
    zero_idxs = np.where(x == 0.0)[0]
    x = x - mean
    x = x / std
    x[zero_idxs] = 0.0
    return x


def test_nonzero_stats_known_answer():
    mean, std, n = features.nonzero_stats([np.array([0.0, 2.0, 4.0]),
                                           np.array([0.0, 6.0])])
    assert n == 3
    assert mean == pytest.approx(4.0)
    assert std == pytest.approx((8.0 / 3.0) ** 0.5)


def test_nonzero_stats_refuses_nothing_to_measure():
    with pytest.raises(ValueError):
        features.nonzero_stats([np.zeros(5)])
    with pytest.raises(ValueError):
        features.nonzero_stats([np.full(5, 3.0)])


def test_zscore_matches_coqui_and_leaves_its_input_alone():
    rng = np.random.default_rng(0)
    x = rng.uniform(60, 300, 400).astype("float32")
    x[rng.random(400) < 0.2] = 0.0
    before = x.copy()
    got = features.zscore_nonzero(x, 110.0, 40.0)
    want = coqui_normalize(x.astype("float64"), 110.0, 40.0)
    assert got.dtype == np.float32
    assert np.array_equal(x, before)                   # cache arrays untouched
    assert np.array_equal(got == 0.0, x == 0.0)        # unvoiced stays unvoiced
    assert np.allclose(got, want, atol=1e-5)


# -- the adapter path ----------------------------------------------------------

@pytest.fixture
def fake_corpus(monkeypatch):
    """Three utterances' pitch and energy, in hertz and raw norm."""
    rng = np.random.default_rng(1)
    store = {}
    for i in range(3):
        f0 = rng.normal(100, 80, 50).clip(60, 400).astype("float32")
        f0[rng.random(50) < 0.15] = 0.0
        store[f"u{i}.wav"] = {"pitch": f0,
                              "energy": rng.gamma(2, 12, 50).astype("float32")}
    calls = {"n": 0}

    class U:
        def __init__(self, wav):
            self.wav = wav

    def load_manifest(path, sr):
        assert path.name == "9h.tsv" and sr == 22050
        return [U(w) for w in store]

    def load_or_compute(wav_path, sr, root, want_pitch=False, keys=None):
        calls["n"] += 1
        return store[pathlib.Path(wav_path).name]

    monkeypatch.setattr(adapters.batching, "load_manifest", load_manifest)
    monkeypatch.setattr(features, "load_or_compute", load_or_compute)
    return store, calls


CFG = {"run_id": "r24", "language": "hindi", "data": "9h",
       "sample_rate": 22050, "architecture": "fastspeech2"}


def test_without_the_field_the_loss_sees_exactly_what_it_always_did(fake_corpus):
    store, calls = fake_corpus
    feats = list(store.values())
    pitch, energy = adapters.FastSpeech2Adapter()._variance_targets(
        feats, dict(CFG, run_id="r01"))
    assert all(p is f["pitch"] for p, f in zip(pitch, feats))
    assert all(e is f["energy"] for e, f in zip(energy, feats))
    assert calls["n"] == 0                 # no statistics pass at all


def test_zscore_uses_the_rungs_own_statistics_once(fake_corpus, capsys):
    store, calls = fake_corpus
    feats = list(store.values())
    a = adapters.FastSpeech2Adapter()
    cfg = dict(CFG, variance_norm="zscore")
    pitch, energy = a._variance_targets(feats, cfg)
    n_first = calls["n"]
    a._variance_targets(feats, cfg)
    assert n_first == 3 and calls["n"] == 3          # computed once, then held

    pm, ps, _ = features.nonzero_stats([f["pitch"] for f in feats])
    st = a.variance_stats
    assert st["pitch_mean"] == pm and st["pitch_std"] == ps
    assert st["utterances"] == 3 and st["manifest"] == "hindi/ladder/9h.tsv"

    voiced = np.concatenate(pitch)[np.concatenate([f["pitch"] for f in feats]) != 0]
    assert abs(voiced.mean()) < 1e-5 and abs(voiced.std() - 1.0) < 1e-5
    e = np.concatenate(energy)
    assert abs(e.mean()) < 1e-5 and abs(e.std() - 1.0) < 1e-5
    assert "variance_norm zscore, statistics" in capsys.readouterr().out


def test_an_unknown_mode_stops_the_run(fake_corpus):
    with pytest.raises(SystemExit):
        adapters.FastSpeech2Adapter()._variance_targets(
            list(fake_corpus[0].values()), dict(CFG, variance_norm="minmax"))


def test_collate_hands_the_normalised_arrays_to_the_model(fake_corpus, monkeypatch):
    torch = pytest.importorskip("torch")
    store, _ = fake_corpus
    feats = [dict(v, mel=np.zeros((50, 80), "float32")) for v in store.values()]
    a = adapters.FastSpeech2Adapter()
    monkeypatch.setattr(a, "_features", lambda batch, cfg: feats)

    class Enc:
        def encode(self, text):
            return [1, 2, 3]

    class B:
        text = "x"

    t = a.collate([B(), B(), B()], Enc(), dict(CFG, variance_norm="zscore"))
    assert tuple(t["pitch"].shape) == (3, 1, 50)
    got = t["pitch"][:, 0, :].numpy()
    want = np.stack([features.zscore_nonzero(f["pitch"], a.variance_stats["pitch_mean"],
                                             a.variance_stats["pitch_std"]) for f in feats])
    assert np.array_equal(got, want)
    assert float(t["pitch"].abs().max()) < 10      # not hertz any more
