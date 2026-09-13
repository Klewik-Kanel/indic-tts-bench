#!/usr/bin/env python3
"""Download the IndicTTS corpora and profile them.

Run in YOUR OWN macOS Terminal, not inside Claude's workspace. Claude's
environments cannot reach huggingface.co: the session egress policy returns 403
at CONNECT for that host. This script is the hand-off point.

    source .venv/bin/activate
    python scripts/02_download_data.py

Writes into data/raw/ and prints a profile Claude then picks up from the
shared folder.
"""

from __future__ import annotations

import json
import pathlib
import sys

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    "hindi": "SPRINGLab/IndicTTS-Hindi",
    "marathi": "SPRINGLab/IndicTTS_Marathi",
}


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install 'datasets[audio]' soundfile librosa", file=sys.stderr)
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    profile: dict[str, dict] = {}

    for name, repo in DATASETS.items():
        print(f"\n=== {name}: {repo} ===", flush=True)
        try:
            ds = load_dataset(repo, split="train")
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            profile[name] = {"repo": repo, "error": str(exc)}
            continue

        # Profile without decoding every file: read the first example for the
        # sampling rate, then use the duration column if one exists.
        first = ds[0]
        audio = first.get("audio", {})
        sr = audio.get("sampling_rate")

        cols = list(ds.features.keys())
        n = len(ds)

        total_seconds = None
        if "audio" in cols:
            # Sum over the array lengths. Streaming the whole set is slow, so
            # this samples unless the set is small.
            import numpy as np
            idx = range(n) if n <= 2000 else np.linspace(0, n - 1, 2000).astype(int)
            secs = []
            for i in idx:
                a = ds[int(i)]["audio"]
                secs.append(len(a["array"]) / a["sampling_rate"])
            mean = sum(secs) / len(secs)
            total_seconds = mean * n
            sampled = len(secs)
        else:
            mean = None
            sampled = 0

        speakers = None
        for key in ("speaker", "speaker_id", "gender"):
            if key in cols:
                try:
                    speakers = sorted({str(v) for v in ds[key]})
                except Exception:                      # noqa: BLE001
                    speakers = None
                break

        info = {
            "repo": repo,
            "n_utterances": n,
            "columns": cols,
            "sampling_rate": sr,
            "mean_utt_seconds": round(mean, 3) if mean else None,
            "estimated_total_hours": round(total_seconds / 3600, 2) if total_seconds else None,
            "duration_sampled_from": sampled,
            "speaker_values": speakers,
            "cache_dir": str(ds.cache_files[0]["filename"]) if ds.cache_files else None,
        }
        profile[name] = info
        print(json.dumps(info, ensure_ascii=False, indent=2))

    out = RAW / "dataset_profile.json"
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print("Tell Claude the download finished; it reads this file from the shared folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
