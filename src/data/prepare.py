#!/usr/bin/env python3
"""Standardise the raw corpora into the training masters.

Produces, per language, from one chosen speaker:

    data/interim/<lang>/wav22/<id>.wav    22.05 kHz, the master
    data/interim/<lang>/wav16/<id>.wav    16 kHz, for the MMS-VITS arm only
    data/interim/<lang>/manifest.tsv      id, durations, text, gain applied

Both rates come from the same 48 kHz source with the same trim boundaries and
the same gain, rather than the 16 kHz copy being derived from the 22.05 kHz
one. Cascading two resamples would put a different signal in front of the VITS
arm than the other architectures see, and that difference would show up in the
results as if it were a modelling effect.

Order of operations, and why:

1. mono, then trim. Silence is trimmed on the 48 kHz signal so both outputs
   share identical boundaries to the sample.
2. measure loudness, compute one gain, apply it before resampling. Measuring
   after resampling would give the two copies slightly different gains.
3. resample to each target with a high-quality filter.
4. peak guard. Loudness normalisation to a fixed LUFS target can push peaks
   past full scale on utterances with a high crest factor. Clipping there
   would be inaudible in the file listing and very audible in training.
   Scaled files are counted and reported rather than silently fixed.

Long job. Run it detached and watch the log:

    nohup python -m src.data.prepare --lang hindi > logs/prepare_hindi.log 2>&1 &
    tail -f logs/prepare_hindi.log
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[2]
RAW = HERE / "data" / "raw"
INTERIM = HERE / "data" / "interim"

MASTER_SR = 22050
VITS_SR = 16000
TARGET_LUFS = -23.0          # EBU R128 broadcast reference
TRIM_TOP_DB = 40.0           # below this relative to peak counts as silence
EDGE_SILENCE_S = 0.05        # leave 50 ms at each end rather than a hard cut
MIN_S, MAX_S = 1.0, 15.0     # 15 s is 1292 mel frames at hop 256
PEAK_CEILING = 0.99

# Speaker 1 is male in both corpora, recovered from median F0 in
# src/data/profile.py. The two male voices sit 17 Hz apart across the two
# languages against 38 Hz for the female pair, so matching on male leaves less
# acoustic difference between the Hindi and Marathi arms that has nothing to do
# with schwa deletion.
DEFAULT_SPEAKER = "1"


def load_rows(lang: str, speaker: str) -> list[dict[str, str]]:
    path = RAW / lang / "metadata.tsv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [
        r for r in rows
        if r["speaker"] == speaker and MIN_S <= float(r["seconds"]) <= MAX_S
    ]


def process(lang: str, speaker: str, resume: bool, limit: int | None) -> dict:
    import numpy as np
    import librosa
    import soundfile as sf
    import pyloudnorm as pyln

    rows = load_rows(lang, speaker)
    if limit:
        rows = rows[:limit]

    out = INTERIM / lang
    d22, d16 = out / "wav22", out / "wav16"
    for d in (d22, d16):
        d.mkdir(parents=True, exist_ok=True)

    meter = pyln.Meter(48000)
    manifest = out / "manifest.tsv"
    written = 0
    skipped = 0
    peak_scaled = 0
    too_quiet = 0
    total_s = 0.0
    t0 = time.time()

    mode = "a" if (resume and manifest.exists()) else "w"
    done: set[str] = set()
    if mode == "a":
        with manifest.open(encoding="utf-8") as fh:
            done = {r["id"] for r in csv.DictReader(fh, delimiter="\t")}
        print(f"resuming; {len(done)} already done", flush=True)

    with manifest.open(mode, newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        if mode == "w":
            w.writerow(["id", "wav22", "wav16", "seconds", "gain_db",
                        "lufs_in", "peak_scaled", "text"])

        for n, r in enumerate(rows):
            uid = r["id"]
            if uid in done:
                skipped += 1
                continue

            y, sr = sf.read(RAW / lang / r["wav"], dtype="float64")
            if y.ndim > 1:
                y = y.mean(axis=1)

            # 1. trim on the source rate so both outputs share boundaries
            yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
            pad = int(EDGE_SILENCE_S * sr)
            yt = np.concatenate([np.zeros(pad), yt, np.zeros(pad)])

            secs = len(yt) / sr
            if not (MIN_S <= secs <= MAX_S):
                # Trimming can push an utterance out of the window.
                skipped += 1
                continue

            # 2. one loudness measurement, one gain, applied before resampling
            try:
                lufs = meter.integrated_loudness(yt)
            except Exception:                                   # noqa: BLE001
                lufs = float("-inf")
            if not np.isfinite(lufs) or lufs < -70:
                too_quiet += 1
                skipped += 1
                continue
            gain_db = TARGET_LUFS - lufs
            yg = yt * (10.0 ** (gain_db / 20.0))

            # 3. resample each target from the same gained 48 kHz signal
            scaled = False
            for target, folder in ((MASTER_SR, d22), (VITS_SR, d16)):
                z = librosa.resample(yg, orig_sr=sr, target_sr=target,
                                     res_type="soxr_hq")
                # 4. peak guard
                peak = float(np.max(np.abs(z))) if len(z) else 0.0
                if peak > PEAK_CEILING:
                    z = z * (PEAK_CEILING / peak)
                    scaled = True
                sf.write(folder / f"{uid}.wav", z.astype(np.float32), target,
                         subtype="PCM_16")
            if scaled:
                peak_scaled += 1

            w.writerow([uid, f"wav22/{uid}.wav", f"wav16/{uid}.wav",
                        f"{secs:.3f}", f"{gain_db:.2f}", f"{lufs:.2f}",
                        int(scaled), r["text"]])
            written += 1
            total_s += secs

            if written % 250 == 0:
                fh.flush()
                rate = written / max(time.time() - t0, 1e-9)
                left = (len(rows) - n - 1) / max(rate, 1e-9)
                print(f"  {written} written, {total_s / 3600:.2f} h, "
                      f"{rate:.1f}/s, ~{left / 60:.0f} min left", flush=True)

    info = {
        "language": lang,
        "speaker": speaker,
        "candidates_after_filter": len(rows),
        "written": written,
        "skipped": skipped,
        "too_quiet": too_quiet,
        "peak_scaled": peak_scaled,
        "hours": round(total_s / 3600, 3),
        "master_sr": MASTER_SR,
        "vits_sr": VITS_SR,
        "target_lufs": TARGET_LUFS,
        "trim_top_db": TRIM_TOP_DB,
        "duration_window_s": [MIN_S, MAX_S],
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(info, indent=2), flush=True)
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["hindi", "marathi"])
    ap.add_argument("--speaker", default=DEFAULT_SPEAKER)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    info = process(args.lang, args.speaker, args.resume, args.limit)

    report = HERE / "results" / "tables" / "prepare_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    all_info = {}
    if report.exists():
        try:
            all_info = json.loads(report.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            pass
    all_info[args.lang] = info
    report.write_text(json.dumps(all_info, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
