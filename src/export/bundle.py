#!/usr/bin/env python3
"""Package a finished run as a self-contained bundle that runs anywhere.

The demo this exists for is a conference laptop: Hindi text goes in, a model is
chosen, speech comes out. That laptop has no GPU, no copy of this repository,
no corpus and no network. So a bundle has to carry everything inference needs
and depend on nothing that produced it.

What goes in, and why each piece is not optional:

**Weights, in fp32.** Training runs in bf16 on an A100. A MacBook has no bf16
in its CPU path, and a bf16 tensor loaded there is either refused or silently
upcast with a surprise in the middle. Casting once, here, makes the bundle
portable and makes the cast visible in the manifest rather than implicit.

**The vocabulary that trained these weights**, not a rebuilt one. Rebuilding
assumes the inventory has not changed since; if it has, every embedding row is
off by one and the model speaks nonsense with no error anywhere.

**The front-end settings**, because the same text produces different phonemes
depending on language and whether schwa deletion ran. A bundle that forgets
whether it was the phonemic or the graphemic arm cannot be demonstrated: that
distinction IS the dissertation.

**The audio parameters.** Sample rate, hop, FFT size and mel count must match
training exactly. A vocoder fed mels on a different hop produces audio that is
recognisably speech and subtly wrong, which is the worst failure mode because
nobody notices it in a demo room.

**The vocoder**, when the architecture needs one. FastSpeech 2 and Matcha
produce mel spectrograms and are mute without HiFi-GAN; VITS is end to end and
carries none. The manifest states which case it is so the player does not have
to guess.

**Provenance**: run id, config hash, step count, git commit, and the date. A
demo that cannot say which version it is, is the thing this project set out to
avoid.

    python -m src.export.bundle runs/r01 --out exports/
    python -m src.export.bundle runs/r01 --out exports/ --step 50000
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess

from ..train.batching import HOP_LENGTH
from ..train.checkpoint import CheckpointDir

AUDIO = {"n_fft": 1024, "hop_length": HOP_LENGTH, "n_mels": 80, "win_length": 1024}
BUNDLE_VERSION = 1


def git_commit(repo: pathlib.Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                        # noqa: BLE001
        return "unknown"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(run_dir: pathlib.Path, out_root: pathlib.Path,
           step: int | None = None, vocoder_bundle: str = "") -> pathlib.Path:
    import torch

    run_dir = pathlib.Path(run_dir)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{run_dir}: no config.json; the run did not record what it was")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    ck = CheckpointDir(run_dir / "checkpoints")
    step = step or ck.latest()
    if not step:
        raise SystemExit(f"{run_dir}: no completed checkpoint to export")

    blob = torch.load(ck.path_for(step) / "state.pt", map_location="cpu",
                      weights_only=False)
    # fp32 for portability. The optimiser state is deliberately dropped: it is
    # large, it is useless for inference, and shipping it to a conference
    # laptop would quadruple the download for nothing.
    weights = {k: v.float() if hasattr(v, "float") else v
               for k, v in blob["model"].items()}

    out = pathlib.Path(out_root) / f"{cfg['run_id']}_step{step}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out / "model.pt")
    shutil.copy2(run_dir / "vocab.json", out / "vocab.json")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "run_id": cfg["run_id"],
        "config_hash": cfg.get("config_hash", ""),
        "step": step,
        "architecture": cfg["architecture"],
        "language": cfg["language"],
        "input_repr": cfg["input_repr"],
        "merge_nukta": bool(cfg.get("merge_nukta", False)),
        # The one line a demo audience actually needs, in words.
        "describes": (
            f"{cfg['architecture']} trained on {cfg['language']} with "
            f"{cfg['input_repr']} input"
            + (", schwa deletion applied" if cfg["language"] == "hindi"
               and cfg["input_repr"] == "phoneme"
               else ", no medial schwa deletion" if cfg["input_repr"] == "phoneme"
               else ", raw characters, no front end")),
        "sample_rate": cfg["sample_rate"],
        "audio": AUDIO,
        "needs_vocoder": cfg["architecture"] in ("fastspeech2", "matcha"),
        "vocoder_bundle": vocoder_bundle,
        "trained_precision": cfg.get("precision", ""),
        "stored_precision": "fp32",
        "train_steps_planned": cfg.get("max_steps"),
        "git_commit": git_commit(pathlib.Path(__file__).resolve().parents[2]),
        "exported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }
    for name in ("model.pt", "vocab.json"):
        f = out / name
        manifest["files"][name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    size_mb = sum(v["bytes"] for v in manifest["files"].values()) / 1e6
    print(f"{out}  {size_mb:.1f} MB  {manifest['describes']}")
    if manifest["needs_vocoder"] and not vocoder_bundle:
        print("  WARNING: this architecture produces mel spectrograms and is "
              "silent without a vocoder bundle. Pass --vocoder.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("exports"))
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--vocoder", default="",
                    help="bundle name of the HiFi-GAN this model needs")
    a = ap.parse_args(argv)
    export(a.run_dir, a.out, a.step, a.vocoder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
