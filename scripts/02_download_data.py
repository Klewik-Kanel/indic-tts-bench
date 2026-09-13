#!/usr/bin/env python3
"""Download the IndicTTS corpora, export WAVs, and profile them.

Run in YOUR OWN macOS Terminal. Claude's environments cannot reach
huggingface.co: the session egress policy returns 403 at CONNECT.

    source .venv/bin/activate
    python scripts/02_download_data.py --limit 50      # trial run first
    python scripts/02_download_data.py                 # full export

Why this does not use the `datasets` audio decoder: since `datasets` 4.x,
decoding an Audio column goes through torchcodec, which dlopens FFmpeg shared
libraries. Those are absent on a stock macOS install, so touching `ds[0]`
raises `Could not load libtorchcodec` before any work happens. Passing
`Audio(decode=False)` hands back the raw encoded bytes instead, which soundfile
reads directly. That removes FFmpeg from the dependency chain altogether.

Output, per language:
    data/raw/<lang>/wavs/<id>.wav     native sampling rate, untouched
    data/raw/<lang>/metadata.tsv      id, text, duration, sr, plus any
                                      speaker or gender column the set carries
    data/raw/dataset_profile.json     schema, hours, speakers, sampling rates
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw"

# The collection is inconsistent about hyphens and underscores, so each
# language lists candidates and the first that resolves wins.
DATASETS: dict[str, list[str]] = {
    "hindi": ["SPRINGLab/IndicTTS-Hindi", "SPRINGLab/IndicTTS_Hindi"],
    "marathi": ["SPRINGLab/IndicTTS_Marathi", "SPRINGLab/IndicTTS-Marathi"],
}

TEXT_KEYS = ("text", "transcript", "sentence", "transcription", "normalized_text")
SPEAKER_KEYS = ("speaker", "speaker_id", "spk", "gender")


def resolve(candidates: list[str]):
    from datasets import load_dataset

    last: Exception | None = None
    for repo in candidates:
        try:
            ds = load_dataset(repo, split="train")
            return repo, ds
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  {repo}: not usable ({type(exc).__name__})")
    raise RuntimeError(f"none of {candidates} resolved") from last


def pick(cols: list[str], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in cols:
            return k
    return None


def export(name: str, repo: str, ds, limit: int | None) -> dict:
    import numpy as np
    import soundfile as sf
    from datasets import Audio

    out = RAW / name
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    cols = list(ds.features.keys())
    text_key = pick(cols, TEXT_KEYS)
    spk_key = pick(cols, SPEAKER_KEYS)
    if text_key is None:
        raise RuntimeError(f"no text column found among {cols}")

    # The line that avoids torchcodec: hand back encoded bytes, not arrays.
    ds = ds.cast_column("audio", Audio(decode=False))

    n_total = len(ds)
    n = n_total if limit is None else min(limit, n_total)

    rows: list[dict] = []
    srs: dict[int, int] = {}
    speakers: dict[str, int] = {}
    total_seconds = 0.0
    failures = 0

    meta_path = out / "metadata.tsv"
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "wav", "seconds", "sample_rate", "speaker", "text"])

        for i in range(n):
            ex = ds[i]
            a = ex["audio"]
            raw = a.get("bytes")
            try:
                if raw is None:
                    data, sr = sf.read(a["path"], dtype="float32", always_2d=False)
                else:
                    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception:  # noqa: BLE001
                failures += 1
                if failures <= 3:
                    print(f"  decode failed on row {i}:")
                    traceback.print_exc(limit=1)
                continue

            if data.ndim > 1:
                data = data.mean(axis=1)

            uid = f"{name[:2]}_{i:06d}"
            path = wavs / f"{uid}.wav"
            sf.write(path, data, sr, subtype="PCM_16")

            secs = len(data) / sr
            total_seconds += secs
            srs[sr] = srs.get(sr, 0) + 1
            spk = str(ex.get(spk_key, "")) if spk_key else ""
            if spk:
                speakers[spk] = speakers.get(spk, 0) + 1

            w.writerow([uid, f"wavs/{uid}.wav", f"{secs:.3f}", sr, spk,
                        str(ex[text_key]).replace("\t", " ").strip()])
            rows.append({"seconds": secs})

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n}  ({total_seconds / 3600:.2f} h so far)", flush=True)

    durs = np.array([r["seconds"] for r in rows]) if rows else np.array([0.0])
    info = {
        "repo": repo,
        "columns": cols,
        "text_column": text_key,
        "speaker_column": spk_key,
        "n_rows_in_split": n_total,
        "n_exported": len(rows),
        "n_decode_failures": failures,
        "sampling_rates": srs,
        "speakers": speakers,
        "total_hours": round(total_seconds / 3600, 3),
        "duration_seconds": {
            "min": round(float(durs.min()), 3),
            "p50": round(float(np.percentile(durs, 50)), 3),
            "p95": round(float(np.percentile(durs, 95)), 3),
            "max": round(float(durs.max()), 3),
            "mean": round(float(durs.mean()), 3),
        },
        "metadata_tsv": str(meta_path.relative_to(HERE)),
        "partial": limit is not None,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="export only the first N utterances; use for a trial run")
    ap.add_argument("--only", choices=sorted(DATASETS), default=None)
    args = ap.parse_args(argv)

    RAW.mkdir(parents=True, exist_ok=True)
    profile: dict[str, dict] = {}
    profile_path = RAW / "dataset_profile.json"
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            profile = {}

    wanted = [args.only] if args.only else list(DATASETS)
    rc = 0
    for name in wanted:
        print(f"\n=== {name} ===", flush=True)
        try:
            repo, ds = resolve(DATASETS[name])
            print(f"  resolved: {repo}  ({len(ds)} rows)")
            profile[name] = export(name, repo, ds, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            profile[name] = {"error": str(exc)}
            rc = 1
        # Write after each language so a later failure does not lose the first.
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\nwrote {profile_path}")
    print("Tell Claude the export finished; it reads this file from the shared folder.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
