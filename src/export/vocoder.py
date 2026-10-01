#!/usr/bin/env python3
"""The pretrained vocoder's mel front end, checked before it is trusted.

RESULTS.md, 2 October: this study synthesises with a pretrained HiFi-GAN rather
than training one, and that is a declared deviation. The deviation is only
survivable if the vocoder's mel front end is the same front end the acoustic
models were trained against.

**Why this file exists rather than a line in the methods table.** A HiFi-GAN fed
mels computed on a different hop, window or mel band range does not raise. It
produces audio that is recognisably speech, somewhat muffled or somewhat bright,
and every objective metric downstream reports a number. MCD will be worse for
both arms, by an amount nobody can attribute, and the comparison the
dissertation rests on would be made through a lens of unknown curvature. There
is no later step that catches this: the audio is plausible and the metrics are
numbers. So it is checked once, here, at the only moment where the two
configurations are both in hand.

**Where the project's parameters come from.** `src/train/features.py`, which is
what actually computes the spectrograms the models trained on, not
`adapters.mel_for` and not a constant retyped here. One trap worth naming:
`features.FMIN_HZ, FMAX_HZ = 60.0, 600.0` are the *pitch tracker's* search
range for one adult speaker. They are not mel band edges. The mel filter bank
in `features.py` is built with `fmin=0, fmax=sr // 2`, and those are the numbers
a vocoder has to agree with.

**Absence is a mismatch.** A config that does not state its hop length is not a
config that happens to match; it is a config that cannot be checked. It fails
the same way a wrong value does, because the alternative is to assume the
convenient answer about the one thing this file exists to not assume.

Key spellings differ between the two HiFi-GAN lineages in circulation, so both
are read: coqui-ai/TTS writes `fft_size`, `num_mels`, `mel_fmin`, `mel_fmax`
under an `audio` block, while jik876/hifi-gan writes `n_fft`, `num_mels`,
`hop_size`, `win_size`, `fmin`, `fmax` at the top level.

    python -m src.export.vocoder /path/to/hifigan/config.json --sample-rate 22050
"""

from __future__ import annotations

import json
import pathlib

from ..train import features as F
from ..train.batching import HOP_LENGTH

# Canonical names, in the order a human wants to read them.
CANON = ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels",
         "fmin", "fmax")

# Every spelling seen in the wild, mapped to one canonical name. Longest and
# most specific first within each tuple, because `fmin` also matches as a
# substring of nothing here but the order documents intent.
ALIASES: dict[str, tuple[str, ...]] = {
    "sample_rate": ("sample_rate", "sampling_rate", "sr"),
    "n_fft": ("n_fft", "fft_size", "filter_length"),
    "win_length": ("win_length", "win_size", "window_length"),
    "hop_length": ("hop_length", "hop_size", "hop"),
    "n_mels": ("n_mels", "num_mels", "n_mel_channels", "num_mel_bins"),
    "fmin": ("fmin", "mel_fmin", "f_min"),
    "fmax": ("fmax", "mel_fmax", "f_max"),
}

# Blocks a checkpoint's config may nest its audio parameters under. Searched in
# this order after the top level, so a top-level key wins over a nested one.
BLOCKS = ("audio", "data", "model_params", "dataset", "preprocessing")

# Hz. Band edges are floats in one lineage and ints in the other, so an exact
# comparison would fail on 0 versus 0.0; a tolerance this tight still catches
# every real mismatch, the smallest of which is thousands of Hz.
HZ_TOL = 0.5


def project_mel(sample_rate: int) -> dict:
    """The mel configuration the acoustic models were actually trained with.

    `features.HOP` and `batching.HOP_LENGTH` are two constants that must hold
    the same number: one computes the frames, the other decides what counts as
    an oversized utterance and is what `src/eval/mcd.py` aligns against. They
    are checked against each other here because this is the one function that
    reads both, and a run where they disagree has frame counts that do not mean
    what the evaluation thinks they mean.
    """
    if int(F.HOP) != int(HOP_LENGTH):
        raise SystemExit(
            f"features.HOP is {F.HOP} and batching.HOP_LENGTH is {HOP_LENGTH}. "
            "These must be the same number: one computes the frames and the "
            "other is what the evaluation aligns against. Fix the duplication "
            "before exporting anything.")
    sr = int(sample_rate)
    return {"sample_rate": sr, "n_fft": int(F.N_FFT), "win_length": int(F.WIN),
            "hop_length": int(F.HOP), "n_mels": int(F.N_MELS),
            "fmin": 0.0, "fmax": sr / 2}


def find_mel(cfg: dict) -> tuple[dict, dict]:
    """Pull the mel parameters out of a vocoder config of either lineage.

    Returns (values, where) where `values` maps each canonical name to what was
    found or None, and `where` records the key path it came from so a mismatch
    report can name the line the number came from rather than asserting it.
    """
    found: dict[str, object] = {}
    where: dict[str, str] = {}
    scopes: list[tuple[str, dict]] = [("", cfg)]
    for b in BLOCKS:
        v = cfg.get(b)
        if isinstance(v, dict):
            scopes.append((f"{b}.", v))
    for key in CANON:
        for prefix, scope in scopes:
            hit = next((a for a in ALIASES[key] if a in scope), None)
            if hit is not None and scope[hit] is not None:
                found[key] = scope[hit]
                where[key] = f"{prefix}{hit}"
                break
            if hit is not None:
                # Present and explicitly null. coqui writes `mel_fmax: null` to
                # mean sr/2, so it is recorded as a default rather than as a
                # missing value, and the report says which it was.
                found.setdefault(key, None)
                where.setdefault(key, f"{prefix}{hit} (null)")
        else:
            found.setdefault(key, None)
            where.setdefault(key, "absent")
    return found, where


def resolve_defaults(found: dict, where: dict) -> tuple[dict, dict]:
    """Apply the one documented default: a null fmax means sr/2.

    Only fmax. A null hop or a null mel count has no conventional default and
    is left as a failure, because guessing one is the behaviour this module
    exists to prevent.
    """
    out, w = dict(found), dict(where)
    sr = out.get("sample_rate")
    if out.get("fmax") is None and "(null)" in str(w.get("fmax", "")) and sr:
        out["fmax"] = float(sr) / 2
        w["fmax"] = f"{w['fmax']} -> sr/2"
    return out, w


def compare(project: dict, found: dict, where: dict | None = None) -> list[dict]:
    """One row per parameter: what we need, what the vocoder says, the verdict."""
    where = where or {}
    rows = []
    for key in CANON:
        want = project[key]
        got = found.get(key)
        if got is None:
            verdict = "ABSENT"
        elif key in ("fmin", "fmax"):
            verdict = "ok" if abs(float(got) - float(want)) <= HZ_TOL else "MISMATCH"
        else:
            try:
                verdict = "ok" if int(got) == int(want) else "MISMATCH"
            except (TypeError, ValueError):
                verdict = "MISMATCH"
        rows.append({"param": key, "want": want, "got": got,
                     "from": where.get(key, ""), "verdict": verdict})
    return rows


def describe(rows: list[dict]) -> str:
    out = [f"  {'parameter':<12} {'training':>10} {'vocoder':>12}  {'verdict':<9} source",
           f"  {'-' * 12} {'-' * 10:>10} {'-' * 12:>12}  {'-' * 9:<9} {'-' * 20}"]
    for r in rows:
        got = "absent" if r["got"] is None else str(r["got"])
        out.append(f"  {r['param']:<12} {str(r['want']):>10} {got:>12}  "
                   f"{r['verdict']:<9} {r['from']}")
    return "\n".join(out)


def assert_mel_matches(sample_rate: int, cfg: dict, source: str) -> list[dict]:
    """Raise unless every mel parameter agrees. Returns the rows when it does."""
    project = project_mel(sample_rate)
    found, where = resolve_defaults(*find_mel(cfg))
    rows = compare(project, found, where)
    bad = [r for r in rows if r["verdict"] != "ok"]
    if bad:
        raise SystemExit(
            f"the vocoder at {source} does not match this project's mel front "
            f"end, so audio from it would be scored through a different "
            f"analysis than the models were trained on:\n{describe(rows)}\n"
            f"  {len(bad)} parameter(s) wrong or unstated. Either find a "
            f"checkpoint whose front end matches, or record the resampling "
            f"step that converts between them as a deviation and implement it "
            f"before any synthesis.")
    return rows


def read_config(path: pathlib.Path) -> tuple[dict, str]:
    """Read a vocoder config from a json file, a directory, or a checkpoint.

    A coqui checkpoint carries its own config inside the blob under "config",
    which is the only copy that is certainly the one the weights were trained
    with: a config.json sitting beside it may have been edited since. So when
    given a checkpoint, the embedded config wins and the source string says so.

    torch is imported only in the checkpoint branch, so reading a plain
    config.json needs nothing installed.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        for name in ("config.json", "config.yaml", "config.yml"):
            if (path / name).exists():
                return read_config(path / name)
        raise SystemExit(f"{path}: no config.json beside the checkpoint")
    if path.suffix in (".json",):
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    if path.suffix in (".pth", ".pt", ".ckpt", ".tar"):
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and isinstance(blob.get("config"), dict):
            return blob["config"], f"{path} (config embedded in the checkpoint)"
        beside = path.parent / "config.json"
        if beside.exists():
            return (json.loads(beside.read_text(encoding="utf-8")),
                    f"{beside} (the checkpoint carries no config of its own)")
        raise SystemExit(
            f"{path}: the checkpoint has no embedded config and there is no "
            "config.json beside it, so its mel front end cannot be read. "
            "Download the config that belongs to these weights.")
    raise SystemExit(f"{path}: not a config or a checkpoint this can read")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Check a pretrained vocoder's mel front end")
    ap.add_argument("vocoder", type=pathlib.Path,
                    help="config.json, a checkpoint, or the directory holding them")
    ap.add_argument("--sample-rate", type=int, default=22050)
    a = ap.parse_args(argv)
    cfg, source = read_config(a.vocoder)
    rows = assert_mel_matches(a.sample_rate, cfg, source)
    print(f"the vocoder at {source} matches this project's mel front end:")
    print(describe(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
