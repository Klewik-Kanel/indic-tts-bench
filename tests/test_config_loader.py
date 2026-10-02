"""The config reader must not turn a hash into a number.

`load_config` used to try int() then float() on every unquoted value, so a
12-hex `config_hash` that is also a valid float literal became one. r08's real
hash is 52e245223208, which float() reads as scientific notation and returns inf
for. That run recorded `config_hash: Infinity` in its config.json and in every
checkpoint's meta.json: the single field that ties weights to the config they
came from, replaced by a value that is not even valid strict JSON.

Needs nothing installed.
"""

from __future__ import annotations

import pathlib

from src.train.runner import load_config

HERE = pathlib.Path(__file__).resolve().parents[1]


def write(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    p = tmp_path / "r99.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_the_hash_that_broke_r08_stays_a_string(tmp_path):
    cfg = load_config(write(tmp_path, "config_hash: 52e245223208\n"))
    assert cfg["config_hash"] == "52e245223208"


def test_an_all_digit_hash_stays_a_string_too(tmp_path):
    """The same defect with int() rather than float()."""
    cfg = load_config(write(tmp_path, "config_hash: 523456789012\n"))
    assert cfg["config_hash"] == "523456789012"


def test_a_hash_that_reads_as_infinity_or_nan_stays_a_string(tmp_path):
    for v in ("1e999", "inf", "nan", "Infinity"):
        cfg = load_config(write(tmp_path, f"config_hash: {v}\n"))
        assert cfg["config_hash"] == v, v


def test_the_numbers_are_still_numbers(tmp_path):
    cfg = load_config(write(tmp_path, "\n".join([
        "max_steps: 100000", "batch_frames: 12000", "lr: 0.0002",
        "grad_clip: 1.0", "warmup_steps: 4000", "sample_rate: 22050",
        "merge_nukta: false", "seed: 0"]) + "\n"))
    assert cfg["max_steps"] == 100_000 and isinstance(cfg["max_steps"], int)
    assert cfg["lr"] == 0.0002 and isinstance(cfg["lr"], float)
    assert cfg["grad_clip"] == 1.0 and isinstance(cfg["grad_clip"], float)
    assert cfg["merge_nukta"] is False
    assert cfg["seed"] == 0 and isinstance(cfg["seed"], int)


def test_negative_and_exponent_floats_still_parse(tmp_path):
    cfg = load_config(write(tmp_path, "a: -1.5\nb: 1.0e-4\nc: 0.5E3\nd: .25\n"))
    assert cfg["a"] == -1.5 and cfg["b"] == 1e-4
    assert cfg["c"] == 500.0 and cfg["d"] == 0.25


def test_text_fields_are_never_coerced(tmp_path):
    cfg = load_config(write(tmp_path, "\n".join([
        "run_id: r99", "architecture: vits", "language: hindi",
        "data: 30min", "precision: bf16", "init_from: 1e5",
        "vocoder: 12345", "notes: ''"]) + "\n"))
    for k in ("run_id", "architecture", "language", "data", "precision"):
        assert isinstance(cfg[k], str), k
    assert cfg["init_from"] == "1e5"
    assert cfg["vocoder"] == "12345"


def test_every_real_config_loads_with_a_string_hash():
    """The regression, over the configs actually in the repository."""
    for p in sorted((HERE / "configs").glob("*.yaml")):
        cfg = load_config(p)
        assert isinstance(cfg["config_hash"], str), p.name
        assert len(cfg["config_hash"]) == 12, (p.name, cfg["config_hash"])
        assert isinstance(cfg["lr"], float), p.name
        assert isinstance(cfg["max_steps"], int), p.name
