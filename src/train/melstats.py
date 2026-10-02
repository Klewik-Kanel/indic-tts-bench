#!/usr/bin/env python3
"""Mel mean and standard deviation over one corpus, which Matcha-TTS needs.

Matcha normalises its mel targets by a single scalar mean and standard
deviation of the whole corpus, held in the model as `data_statistics` and
registered as buffers. Upstream ships LJSpeech's numbers
(mel_mean -5.536622, mel_std 2.116101) in `configs/data/ljspeech.yaml`, and
those are LJSpeech's: a different corpus, a different speaker, and a mel
filter bank built to 8 kHz rather than to sr/2. Training on this project's
mels with LJSpeech's statistics would normalise to the wrong centre and scale,
which is not an error anywhere, just a worse model.

So the statistics are computed here, from this project's own cache, by the
same `features.compute` that produced the mels the model will see. They are
written beside the manifests with the mel parameters they were computed under,
so a stats file that predates a change to the analysis is detectable rather
than silently reused. `MatchaAdapter.mel_stats` refuses to run against one.

Scalars, not per-band vectors, because that is what upstream's normalise and
denormalise do.

    python -m src.train.melstats hindi
    python -m src.train.melstats marathi --limit 500      # a quick estimate
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib

from . import batching, features as F

HERE = pathlib.Path(__file__).resolve().parents[2]
INTERIM = HERE / "data" / "interim"
PROCESSED = HERE / "data" / "processed"


def stats_path(language: str) -> pathlib.Path:
    return PROCESSED / language / "mel_stats.json"


def compute(language: str, sample_rate: int = 22_050,
            manifest: pathlib.Path | None = None,
            limit: int | None = None) -> dict:
    """Accumulate over every mel frame of every utterance in the manifest.

    One pass, keeping the count, the sum and the sum of squares in float64.
    The variance then comes from E[x^2] - E[x]^2, which is numerically fine
    here: log-mel values sit around -5 with a spread near 2, so there is no
    cancellation of the kind that makes that formula a bad idea.
    """
    import numpy as np

    mpath = manifest or (PROCESSED / language / "train.tsv")
    utts = batching.load_manifest(mpath, int(sample_rate))
    if limit:
        utts = utts[:int(limit)]
    if not utts:
        raise SystemExit(f"{mpath}: no utterances")

    # Same knob the training loop honours; see AdapterBase._features.
    root = pathlib.Path(os.environ.get("TRAIN_CACHE_ROOT")
                        or (HERE / "data" / "cache")) / language
    n = 0
    total = 0.0
    total_sq = 0.0
    lo, hi = math.inf, -math.inf
    for i, u in enumerate(utts, 1):
        mel = F.load_or_compute(INTERIM / language / u.wav, int(sample_rate),
                                root, keys=("mel",))["mel"]
        a = np.asarray(mel, dtype="float64")
        n += a.size
        total += float(a.sum())
        total_sq += float(np.square(a).sum())
        lo = min(lo, float(a.min()))
        hi = max(hi, float(a.max()))
        if i % 500 == 0:
            print(f"  {i}/{len(utts)} utterances, {n:,} frames-by-bands",
                  flush=True)

    mean = total / n
    var = max(total_sq / n - mean * mean, 0.0)
    return {
        "mel_mean": round(mean, 6),
        "mel_std": round(math.sqrt(var), 6),
        "mel_min": round(lo, 6),
        "mel_max": round(hi, 6),
        "n_utterances": len(utts),
        "n_values": n,
        "manifest": str(mpath.relative_to(HERE)),
        # The analysis these numbers describe. A stats file whose parameters do
        # not match the current ones is refused rather than reused, because the
        # mean and standard deviation of a log-mel depend on every one of them.
        "mel_params": F.mel_params(int(sample_rate)),
        "computed_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Corpus mel statistics for Matcha-TTS")
    ap.add_argument("language")
    ap.add_argument("--sample-rate", type=int, default=22_050)
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N utterances only, for a quick estimate")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)

    stats = compute(a.language, a.sample_rate, a.manifest, a.limit)
    out = a.out or stats_path(a.language)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: mel_mean {stats['mel_mean']}, mel_std {stats['mel_std']} "
          f"over {stats['n_utterances']} utterances "
          f"({stats['n_values']:,} values)")
    if a.limit:
        print("  NOTE: computed from a --limit subset, so this is an estimate. "
              "Recompute over the full manifest before training a real run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
