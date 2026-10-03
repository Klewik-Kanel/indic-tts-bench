#!/usr/bin/env python3
"""Find and fetch export bundles from a Hugging Face model repo.

`offload.py` uploads anything under `REPO/exports` to `exports/<bundle>/...`,
so that prefix is the contract between the machine that trains and anything
that consumes the weights afterwards: the demo Space, the listening test, and a
laptop in a viva.

The path arithmetic lives here rather than in the Space because it is where the
mistake is easy. `hf_hub_download(..., local_dir=X)` writes to
`X/<path in repo>`, reproducing the repo prefix under the cache directory, so a
caller that assumes the file landed directly in `X` gets "not a bundle" for a
bundle it has just downloaded successfully. One definition, and a test.

Nothing here imports torch or gradio, so it can be exercised without either.
"""

from __future__ import annotations

import pathlib

PREFIX = "exports/"
# manifest first: it is kilobytes and decides whether the rest is worth pulling.
FILES = ("manifest.json", "vocab.json", "model.pt")
OPTIONAL = frozenset({"vocab.json"})        # a vocoder bundle carries none


def bundle_names(repo_files: list[str]) -> list[str]:
    """Bundle directory names, from a flat listing of a repo's files.

    Takes the listing rather than fetching it, so the parsing is testable
    without a network call.
    """
    return sorted({f.split("/")[1] for f in repo_files
                   if f.startswith(PREFIX) and f.count("/") >= 2 and
                   not f.endswith("/")})


def repo_path(name: str, filename: str) -> str:
    return f"{PREFIX}{name}/{filename}"


def bundle_root(cache: pathlib.Path | str, name: str) -> pathlib.Path:
    """Where `hf_hub_download(local_dir=cache / name)` leaves the bundle."""
    return pathlib.Path(cache) / name / PREFIX.rstrip("/") / name


def fetch(repo_id: str, name: str, filename: str,
          cache: pathlib.Path | str) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    return pathlib.Path(hf_hub_download(
        repo_id, repo_path(name, filename),
        local_dir=pathlib.Path(cache) / name, repo_type="model"))


def fetch_bundle(repo_id: str, name: str,
                 cache: pathlib.Path | str) -> pathlib.Path:
    """Download every file a bundle needs and return its root."""
    for filename in FILES:
        try:
            fetch(repo_id, name, filename, cache)
        except Exception:                                     # noqa: BLE001
            if filename in OPTIONAL:
                continue
            raise
    return bundle_root(cache, name)


def discover(repo_id: str, cache: pathlib.Path | str) -> dict[str, dict]:
    """Manifests of every bundle in the repo, by bundle name. No weights.

    A bundle whose manifest this build cannot read is skipped with a note
    rather than raising: one bad bundle should not take a demo offline.
    """
    from huggingface_hub import HfApi

    from .synthesize import read_manifest

    try:
        files = HfApi().list_repo_files(repo_id)
    except Exception as exc:                                  # noqa: BLE001
        print(f"could not list {repo_id}: {type(exc).__name__}: {exc}")
        return {}

    found: dict[str, dict] = {}
    for name in bundle_names(files):
        try:
            fetch(repo_id, name, "manifest.json", cache)
            manifest = read_manifest(bundle_root(cache, name))
        except SystemExit as exc:
            print(f"skipping {name}: {exc}")
            continue
        except Exception as exc:                              # noqa: BLE001
            print(f"skipping {name}: {type(exc).__name__}: {exc}")
            continue
        manifest["_bundle"] = name
        found[name] = manifest
    return found
