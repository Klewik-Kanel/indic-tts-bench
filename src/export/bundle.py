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

from ..train.checkpoint import CheckpointDir
from .vocoder import assert_mel_matches, project_mel, read_config

# Kept as a flat constant for the analysis parameters that do not depend on the
# sample rate, derived from project_mel so there is one definition of them in
# the repository rather than two that can drift apart.
AUDIO = {k: v for k, v in project_mel(22_050).items()
         if k in ("n_fft", "hop_length", "n_mels", "win_length")}
# 2: the manifest's "vocoder_bundle" string became a "vocoder" block that
# records whether the mel front end was actually checked.
# 3: "lr" is recorded. VitsAdapter.build reads cfg["lr"] to set coqui's lr_disc
# and lr_gen, so rebuilding the model for inference needs the value even though
# neither learning rate enters the inference graph. Carrying the real number is
# better than having src/export/synthesize.py invent one that then looks like a
# hyperparameter to anyone reading it.
BUNDLE_VERSION = 3


def resolve_own_bundle(path: pathlib.Path, sample_rate: int,
                       language: str) -> dict | None:
    """One of OUR vocoder bundles, verified from its own manifest.

    This branch did not exist. `resolve_vocoder` was written when the vocoder
    was going to be a published HiFi-GAN checkpoint, so it hands the path to
    `read_config`, which looks for config.json, config.yaml or config.yml and
    raises on a directory holding none of them. An export bundle holds
    manifest.json and model.pt, so `--vocoder exports/r06_step18000` failed
    outright: the export path could accept a stranger's vocoder and not ours.

    The manifest's own `audio` block is the authority here, because it was
    written at export time from `project_mel`, the same function this compares
    it against. A disagreement therefore means something real diverged, such
    as a bundle built under different constants or at a different rate, rather
    than two conventions being compared.

    Returns None when the path is not one of our bundles, so the caller falls
    through to the external-checkpoint path unchanged.
    """
    if not path.is_dir() or not (path / "manifest.json").exists():
        return None
    from .synthesize import VOCODER_ARCHITECTURES, read_manifest

    man = read_manifest(path)
    arch = man.get("architecture")
    if arch not in VOCODER_ARCHITECTURES:
        raise SystemExit(
            f"{path}: this is a bundle for {arch!r}, which is not a vocoder. "
            f"Expected one of {sorted(VOCODER_ARCHITECTURES)}.")
    if int(man["sample_rate"]) != int(sample_rate):
        raise SystemExit(
            f"{path}: the vocoder is {man['sample_rate']} Hz and this run is "
            f"{sample_rate} Hz. A mel on the wrong rate sounds like speech "
            "and is wrong.")
    if man.get("language") != language:
        raise SystemExit(
            f"{path}: the vocoder was trained on {man.get('language')!r} and "
            f"this run is {language!r}. Each language has its own vocoder so "
            "a speaker mismatch cannot enter the Marathi control.")
    want = project_mel(int(sample_rate))
    got = man.get("audio") or {}
    bad = {k: (v, got.get(k)) for k, v in want.items()
           if k not in got or float(got[k]) != float(v)}
    if bad:
        raise SystemExit(
            f"{path}: its mel does not match this project's: "
            + "; ".join(f"{k} wants {w} and the bundle says {g}"
                        for k, (w, g) in sorted(bad.items())))
    return {"name": path.name, "path": str(path),
            "source": f"{path / 'manifest.json'} (one of our own bundles)",
            "mel_verified": True,
            "run_id": man.get("run_id"),
            "step": man.get("step"),
            "config_hash": man.get("config_hash"),
            "mel": dict(want)}


def resolve_vocoder(spec: str, sample_rate: int, language: str = "") -> dict:
    """What the bundle records about its vocoder, and whether it was verified.

    `--vocoder` takes either a path to a real checkpoint, config, or directory,
    which is read and checked parameter by parameter, or a bare name, which
    cannot be. A bare name is still allowed, because a vocoder may ship
    separately from the bundle, but it is recorded as unverified: a later
    reader has to be able to tell "checked and matching" from "nobody looked".

    A mismatch raises rather than warning. The whole point of the pretrained
    vocoder decision is that the vocoder cancels out of the phonemic-versus-
    graphemic contrast, and it only cancels if it is the same analysis on both
    arms as the models were trained with.
    """
    if not spec:
        return {"name": "", "mel_verified": False, "note": "no vocoder attached"}
    p = pathlib.Path(spec).expanduser()
    if not p.exists():
        return {"name": spec, "mel_verified": False,
                "note": ("a name, not a path on this machine, so its mel front "
                         "end was NOT checked against this project's")}
    own = resolve_own_bundle(p, sample_rate, language)
    if own is not None:
        return own
    cfg, source = read_config(p)
    rows = assert_mel_matches(sample_rate, cfg, source)
    return {"name": p.name, "path": str(p), "source": source,
            "mel_verified": True,
            "mel": {r["param"]: r["got"] for r in rows}}


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
           step: int | None = None, vocoder: str = "") -> pathlib.Path:
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

    # Before anything is written: if a vocoder was given as a path, its mel
    # front end is read and checked here, so an export with a mismatched
    # vocoder fails instead of producing a bundle that synthesises wrongly.
    audio = project_mel(int(cfg["sample_rate"]))
    voc = resolve_vocoder(vocoder, int(cfg["sample_rate"]), cfg["language"])

    out = pathlib.Path(out_root) / f"{cfg['run_id']}_step{step}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out / "model.pt")
    # A vocoder has no vocabulary, so its run writes no vocab.json and its
    # bundle cannot carry one. Copying unconditionally raised FileNotFoundError
    # and took the export down with it.
    has_vocab = (run_dir / "vocab.json").exists()
    if has_vocab:
        shutil.copy2(run_dir / "vocab.json", out / "vocab.json")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "run_id": cfg["run_id"],
        "config_hash": cfg.get("config_hash", ""),
        "step": step,
        "architecture": cfg["architecture"],
        "language": cfg["language"],
        "input_repr": cfg["input_repr"],
        # The two fields that say which cell of the matrix this is. A page
        # listing eight arms cannot group them without the rung and the seed,
        # and recovering either from configs/ later needs the config file to
        # still match the hash. Carried here so the bundle answers it alone.
        "data": cfg.get("data", ""),
        "seed": cfg.get("seed"),
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
        "lr": cfg.get("lr"),
        "audio": audio,
        "needs_vocoder": cfg["architecture"] in ("fastspeech2", "matcha"),
        "vocoder": voc,
        "trained_precision": cfg.get("precision", ""),
        "stored_precision": "fp32",
        "train_steps_planned": cfg.get("max_steps"),
        "git_commit": git_commit(pathlib.Path(__file__).resolve().parents[2]),
        "exported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }
    for name in ("model.pt", "vocab.json") if has_vocab else ("model.pt",):
        f = out / name
        manifest["files"][name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    size_mb = sum(v["bytes"] for v in manifest["files"].values()) / 1e6
    print(f"{out}  {size_mb:.1f} MB  {manifest['describes']}")
    if manifest["needs_vocoder"] and not vocoder:
        print("  WARNING: this architecture produces mel spectrograms and is "
              "silent without a vocoder bundle. Pass --vocoder.")
    elif manifest["needs_vocoder"] and not voc["mel_verified"]:
        print(f"  WARNING: vocoder recorded as {voc['name']!r} but its mel "
              "front end was not checked, because that is a name and not a "
              "path on this machine. Pass the checkpoint or its config.json to "
              "--vocoder so the analysis parameters are verified before any "
              "audio from it is scored.")
    elif manifest["needs_vocoder"]:
        print(f"  vocoder {voc['name']}: mel front end verified against "
              "this project's")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("exports"))
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--vocoder", default="",
                    help="path to the HiFi-GAN checkpoint, its config.json, or "
                         "the directory holding them; its mel front end is "
                         "checked against this project's and the export fails "
                         "on a mismatch. A bare name is accepted and recorded "
                         "as unverified.")
    a = ap.parse_args(argv)
    export(a.run_dir, a.out, a.step, a.vocoder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
