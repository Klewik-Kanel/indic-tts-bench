#!/usr/bin/env python3
"""Bundle discovery and the path arithmetic underneath it.

`hf_hub_download(..., local_dir=X)` writes to `X/<path in repo>`, reproducing
the repo prefix beneath the cache directory. A caller that assumes the file
landed directly in `X` reports "not a bundle" for a bundle it has just
downloaded, which is a confusing failure a long way from its cause. This got it
wrong once in the Space app, so the arithmetic now lives in `src.export.hub`
with these tests on it.

Nothing here touches the network: `bundle_names` is handed a listing rather
than fetching one, which is why it can be tested at all.
"""
from __future__ import annotations

import pathlib

import pytest

from src.export import hub

LISTING = [
    ".gitattributes",
    "README.md",
    "RESULTS.md",
    "configs/r02.yaml",
    "runs/r02/config.json",
    "runs/r02/checkpoints/step_100000/state.pt",
    "exports/r02_step100000/manifest.json",
    "exports/r02_step100000/model.pt",
    "exports/r02_step100000/vocab.json",
    "exports/r05_step100000/manifest.json",
    "exports/r05_step100000/model.pt",
]


def test_finds_every_bundle_once():
    assert hub.bundle_names(LISTING) == ["r02_step100000", "r05_step100000"]


def test_ignores_everything_outside_the_exports_prefix():
    """runs/ holds training checkpoints with optimiser state. They are not
    bundles and must never be offered to a demo."""
    names = hub.bundle_names(LISTING)
    assert not any("r02/" in n or n.startswith("runs") for n in names)


def test_a_bare_file_at_the_prefix_root_is_not_a_bundle():
    assert hub.bundle_names(["exports/stray.txt"]) == []


def test_an_empty_listing_yields_nothing_rather_than_raising():
    assert hub.bundle_names([]) == []


def test_bundle_root_reproduces_the_repo_prefix_under_the_cache():
    """The bug this module exists to prevent."""
    root = hub.bundle_root("/tmp/bundles", "r02_step100000")
    assert root == pathlib.Path(
        "/tmp/bundles/r02_step100000/exports/r02_step100000")


def test_bundle_root_accepts_a_path_or_a_string():
    a = hub.bundle_root(pathlib.Path("/c"), "r02")
    b = hub.bundle_root("/c", "r02")
    assert a == b


def test_repo_path_matches_what_offload_uploads():
    """offload.py writes each file to exports/<relative path>, so this is the
    contract between the training box and every consumer."""
    assert hub.repo_path("r02_step100000", "model.pt") == \
        "exports/r02_step100000/model.pt"


def test_the_manifest_is_fetched_first():
    """It is kilobytes and decides whether the 330 MB file is worth pulling."""
    assert hub.FILES[0] == "manifest.json"


def test_only_the_vocabulary_is_optional():
    """A vocoder bundle has no vocabulary. Weights and manifest are not
    optional, and a missing one must raise rather than yield a half bundle."""
    assert hub.OPTIONAL == {"vocab.json"}
    assert "model.pt" not in hub.OPTIONAL
    assert "manifest.json" not in hub.OPTIONAL


def test_fetch_bundle_tolerates_a_missing_vocabulary(monkeypatch, tmp_path):
    asked = []

    def fake(repo_id, name, filename, cache):
        asked.append(filename)
        if filename == "vocab.json":
            raise RuntimeError("404")
        return pathlib.Path(cache) / filename

    monkeypatch.setattr(hub, "fetch", fake)
    root = hub.fetch_bundle("repo", "r06_step1000", tmp_path)
    assert asked == list(hub.FILES)
    assert root == hub.bundle_root(tmp_path, "r06_step1000")


def test_fetch_bundle_propagates_a_missing_model(monkeypatch, tmp_path):
    def fake(repo_id, name, filename, cache):
        if filename == "model.pt":
            raise RuntimeError("404")
        return pathlib.Path(cache) / filename

    monkeypatch.setattr(hub, "fetch", fake)
    with pytest.raises(RuntimeError):
        hub.fetch_bundle("repo", "r02_step100000", tmp_path)
