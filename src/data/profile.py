#!/usr/bin/env python3
"""Profile the exported corpora: per-speaker hours, length filters, QA checks.

Everything the speaker-selection and length-filter decisions rest on, in one
reproducible place. Writes results/tables/corpus_profile.json.

    python -m src.data.profile
    python -m src.data.profile --no-pitch     # skip the F0 pass, which is slow
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import statistics as st

HERE = pathlib.Path(__file__).resolve().parents[2]
RAW = HERE / "data" / "raw"
OUT = HERE / "results" / "tables" / "corpus_profile.json"

LANGS = ("hindi", "marathi")
FILTERS = ((1, 12), (1, 15), (1, 20), (2, 15))

# Above this median F0 a speaker is reported as likely female. The corpora
# label speakers 0 and 1 with no key, and the study needs Hindi and Marathi
# matched on speaker characteristics, so the labels have to be recovered.
# 165 Hz sits in the gap between typical adult male and female ranges; the
# measured medians are far enough from it that the exact threshold does not
# matter, which the recorded p10/p90 spread lets a reader confirm.
FEMALE_F0_THRESHOLD = 165.0

# A transcript whose characters-per-second is wildly off the corpus median is
# the signature of text that does not match its audio. Bounds are relative to
# the median so they adapt to each language's orthographic density.
CPS_LOW, CPS_HIGH = 0.4, 2.0


def read_metadata(lang: str) -> list[dict[str, str]]:
    path = RAW / lang / "metadata.tsv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def speaker_stats(rows: list[dict[str, str]]) -> dict:
    by: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(float(r["seconds"]))
    out = {}
    for spk, durs in sorted(by.items()):
        d = sorted(durs)
        out[spk] = {
            "n_utterances": len(d),
            "hours_raw": round(sum(d) / 3600, 3),
            "seconds": {
                "mean": round(st.mean(d), 2),
                "median": round(st.median(d), 2),
                "min": round(d[0], 2),
                "p95": round(d[int(0.95 * len(d))], 2),
                "max": round(d[-1], 2),
            },
            "hours_after_filter": {
                f"{lo}-{hi}s": round(sum(x for x in d if lo <= x <= hi) / 3600, 3)
                for lo, hi in FILTERS
            },
        }
    return out


def transcript_qa(rows: list[dict[str, str]]) -> dict:
    """Flag transcripts that cannot plausibly match their audio."""
    cps = [(len(r["text"]) / float(r["seconds"]), r["id"]) for r in rows]
    vals = sorted(c for c, _ in cps)
    med = st.median(vals)
    low = [i for c, i in cps if c < med * CPS_LOW]
    high = [i for c, i in cps if c > med * CPS_HIGH]
    lengths = [len(r["text"]) for r in rows]
    return {
        "chars_per_second": {
            "median": round(med, 2),
            "p1": round(vals[len(vals) // 100], 2),
            "p99": round(vals[-len(vals) // 100], 2),
        },
        "suspicious_low": low[:20],
        "suspicious_high": high[:20],
        "n_suspicious": len(low) + len(high),
        "text_chars": {
            "mean": round(st.mean(lengths)),
            "median": round(st.median(lengths)),
            "max": max(lengths),
        },
    }


def pitch_by_speaker(lang: str, rows: list[dict[str, str]], n: int = 25) -> dict:
    import numpy as np
    import librosa
    import soundfile as sf

    random.seed(0)
    by: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in rows:
        if 2.0 <= float(r["seconds"]) <= 8.0:
            by[r["speaker"]].append(r)

    out = {}
    for spk, pool in sorted(by.items()):
        meds: list[float] = []
        for r in random.sample(pool, min(n, len(pool))):
            y, sr = sf.read(RAW / lang / r["wav"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            f0, _, _ = librosa.pyin(y.astype(np.float64), fmin=60, fmax=400,
                                    sr=16000, frame_length=1024, hop_length=256)
            vals = f0[np.isfinite(f0)]
            if len(vals) > 20:
                meds.append(float(np.median(vals)))
        if not meds:
            out[spk] = {"error": "no pitch recovered"}
            continue
        m = st.median(meds)
        out[spk] = {
            "median_f0_hz": round(m, 1),
            "p10": round(float(np.percentile(meds, 10)), 1),
            "p90": round(float(np.percentile(meds, 90)), 1),
            "n_files": len(meds),
            "inferred_gender": "female" if m >= FEMALE_F0_THRESHOLD else "male",
            "note": "inferred from median F0, not a corpus label",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-pitch", action="store_true")
    args = ap.parse_args(argv)

    profile: dict[str, dict] = {}
    for lang in LANGS:
        rows = read_metadata(lang)
        entry = {
            "n_utterances": len(rows),
            "hours_total": round(
                sum(float(r["seconds"]) for r in rows) / 3600, 3),
            "sampling_rates": dict(collections.Counter(
                r["sample_rate"] for r in rows)),
            "speakers": speaker_stats(rows),
            "transcript_qa": transcript_qa(rows),
        }
        if not args.no_pitch:
            entry["pitch"] = pitch_by_speaker(lang, rows)
        profile[lang] = entry
        print(f"{lang}: {entry['n_utterances']} utts, "
              f"{entry['hours_total']} h, "
              f"{entry['transcript_qa']['n_suspicious']} suspicious transcripts")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
