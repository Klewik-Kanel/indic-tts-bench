#!/usr/bin/env python3
"""Score finished runs against the frozen test split, and write the table.

This is the thing that turns trained weights into numbers. It walks the export
bundles, synthesises the held-out test split through `src.export.synthesize`,
compares each utterance to the natural recording, and writes one JSON per run
plus a combined table. No GPU: inference is a CPU workload and the card is for
training.

Four things it is careful about.

**One synthesis path.** It calls the same `Bundle.synthesize` the demo and the
listening test call, so a number in the results table came from the same code
that produced the audio a listener judged.

**The test split, never the dev split.** `dev.tsv` chose the demo sentences and
is therefore no longer neutral; `test.tsv` was frozen on 14 September before any
training and nothing has selected on it since.

**One alignment per utterance.** MCD aligns reference to synthesis with DTW, and
the F0 comparison reuses that same path rather than computing its own. Two
alignments would let the spectral and prosodic numbers disagree about which
frame matches which.

**Every MCD figure carries its chance level.** An MCD number on its own is
uninterpretable: measured on 3 October, an identical signal scores 0, a severe
but faithful reconstruction about 19, and two unrelated real recordings about
107. A run at 70 is either most of the way to unrelated noise or not, depending
entirely on a baseline nobody had computed. So each run is also scored against
MISMATCHED references, rotating the reference by one utterance, which gives the
chance level on the same material and the same alignment. The table prints it,
and the normalised position between the two.

**A mel-only run is skipped, loudly.** FastSpeech 2 and Matcha need a vocoder,
and until r06 and r17 exist their waveform metrics cannot be computed at all.
The run is recorded as unscorable with the reason rather than quietly omitted,
because a results table with a silently missing row is worse than one with a
stated gap.

    python scripts/evaluate.py --lang hindi
    python scripts/evaluate.py --lang hindi --limit 25 --threads 8
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

SPLITS = HERE / "data" / "processed"
INTERIM = HERE / "data" / "interim"
TABLES = HERE / "results" / "tables"


def load_split(lang: str, split: str) -> list[dict]:
    path = SPLITS / lang / f"{split}.tsv"
    if not path.exists():
        raise SystemExit(f"{path}: no such split")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def reference_column(sample_rate: int) -> str:
    return "wav22" if sample_rate == 22050 else "wav16"


def score_one(ref_wav, syn_wav, sr: int) -> dict:
    """MCD and F0 for one utterance, sharing a single DTW alignment."""
    import numpy as np

    from src.eval import f0 as f0mod
    from src.eval.mcd import mcd

    m, path = mcd(ref_wav, syn_wav, sr, return_path=True)
    ref_idx = np.array([i for i, _ in path])
    syn_idx = np.array([j for _, j in path])

    ref_f0, ref_v = f0mod.extract_f0(ref_wav, sr)
    syn_f0, syn_v = f0mod.extract_f0(syn_wav, sr)
    out = {
        "mcd_db": m.mcd_db,
        "length_ratio": m.length_ratio,
        "ref_frames": m.ref_frames,
        "syn_frames": m.syn_frames,
        "path_frames": m.n_frames,
    }
    try:
        f = f0mod.compare(
            f0mod.align_to_path(ref_f0, ref_idx),
            f0mod.align_to_path(ref_v, ref_idx),
            f0mod.align_to_path(syn_f0, syn_idx),
            f0mod.align_to_path(syn_v, syn_idx),
        )
        out.update({"log_f0_rmse": f.log_f0_rmse, "f0_rmse_hz": f.f0_rmse_hz,
                    "vuv_error_rate": f.vuv_error_rate,
                    "n_common_voiced": f.n_common_voiced})
    except Exception as exc:                                  # noqa: BLE001
        out["f0_error"] = f"{type(exc).__name__}: {exc}"
    return out


def chance_level(references: list, syntheses: list, sr: int) -> dict:
    """MCD against MISMATCHED references: the score of getting it wrong.

    Without this an MCD figure cannot be read. The reference is rotated by one
    utterance, so each synthesis is compared against a recording of different
    words by the same speaker, through the same alignment and the same code.
    That is the number a system scores by having no relationship to the text at
    all, and every reported MCD should be read as a position between 0 and it.
    """
    import numpy as np

    from src.eval.mcd import mcd

    if len(references) < 2:
        return {"note": "needs at least 2 utterances"}
    vals = []
    for i in range(len(syntheses)):
        _, syn = syntheses[i]
        _, ref = references[(i + 1) % len(references)]      # deliberately wrong
        try:
            vals.append(mcd(ref, syn, sr).mcd_db)
        except Exception:                                   # noqa: BLE001
            continue
    if not vals:
        return {"note": "could not be computed"}
    a = np.array(vals, dtype="float64")
    return {"mcd_db_mean": float(a.mean()),
            "mcd_db_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(vals)}


def summarise(rows: list[dict]) -> dict:
    """Mean and spread per metric. Spread matters more than the mean here: a
    difference between two runs is only readable against it."""
    import numpy as np

    keys = ("mcd_db", "log_f0_rmse", "f0_rmse_hz", "vuv_error_rate",
            "length_ratio")
    out = {"n_utterances": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None
                and np.isfinite(r[k])]
        if not vals:
            continue
        a = np.array(vals, dtype="float64")
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                  "median": float(np.median(a)), "n": len(vals)}
    return out


def evaluate_bundle(bundle_dir: pathlib.Path, lang: str, split: str,
                    limit: int | None, out_dir: pathlib.Path) -> dict:
    import soundfile as sf

    from src.export.synthesize import load, read_manifest

    manifest = read_manifest(bundle_dir)
    rid = manifest["run_id"]
    record = {
        "run_id": rid, "bundle": bundle_dir.name,
        "architecture": manifest["architecture"],
        "language": manifest["language"],
        "input_repr": manifest["input_repr"],
        "seed": manifest.get("seed"),
        "step": manifest.get("step"),
        "config_hash": manifest.get("config_hash"),
        "sample_rate": manifest["sample_rate"],
        "split": split,
    }
    if manifest["language"] != lang:
        record["skipped"] = f"trained on {manifest['language']}, not {lang}"
        return record
    if manifest.get("needs_vocoder"):
        record["skipped"] = (
            "mel-only architecture: waveform metrics need a vocoder, and r06 "
            "and r17 have not trained. Not omitted, just not computable yet.")
        return record

    b = load(bundle_dir, device="cpu")
    sr = b.sample_rate
    col = reference_column(sr)
    rows_in = load_split(lang, split)
    if limit:
        rows_in = rows_in[:limit]

    per_utt, failures = [], []
    # kept so the chance level can be computed on exactly this material
    syntheses: list = []
    references: list = []
    for i, row in enumerate(rows_in, 1):
        ref_path = INTERIM / lang / row.get(col, "")
        if not row.get(col) or not ref_path.exists():
            failures.append({"id": row["id"], "error": f"no reference at {ref_path}"})
            continue
        try:
            ref, got = sf.read(str(ref_path), dtype="float32")
            if got != sr:
                failures.append({"id": row["id"],
                                 "error": f"reference is {got} Hz, run is {sr}"})
                continue
            sp = b.synthesize(row["text"])
            scored = score_one(ref, sp.waveform, sr)
            scored["id"] = row["id"]
            scored["tokens"] = len(sp.tokens)
            per_utt.append(scored)
            syntheses.append((row["id"], sp.waveform))
            references.append((row["id"], ref))
        except Exception as exc:                              # noqa: BLE001
            failures.append({"id": row["id"],
                             "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 2:
                traceback.print_exc()
        if i % 25 == 0:
            print(f"    {rid}: {i}/{len(rows_in)}", flush=True)

    record["summary"] = summarise(per_utt)
    record["chance"] = chance_level(references, syntheses, sr)
    record["failures"] = failures
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{rid}.json").write_text(
        json.dumps({**record, "utterances": per_utt}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--split", default="test",
                    help="test by default; dev selected the demo sentences "
                         "and is no longer neutral")
    ap.add_argument("--bundles", nargs="*", type=pathlib.Path)
    ap.add_argument("--limit", type=int,
                    help="first N utterances, for a quick pass")
    ap.add_argument("--threads", type=int, default=8,
                    help="cap BLAS threads: unbounded CPU work on the training "
                         "box has starved a trainer before")
    ap.add_argument("--out", type=pathlib.Path, default=TABLES)
    a = ap.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(a.threads)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    bundles = a.bundles or sorted(
        p for p in (HERE / "exports").glob("*") if (p / "manifest.json").exists())
    if not bundles:
        raise SystemExit("no bundles in exports/; run python -m src.export.bundle")

    print(f"scoring {len(bundles)} bundle(s) on {a.lang} {a.split}, "
          f"{a.threads} threads, CPU only")
    records = []
    for bd in bundles:
        print(f"  {bd.name}", flush=True)
        try:
            records.append(evaluate_bundle(bd, a.lang, a.split, a.limit, a.out))
        except SystemExit as exc:
            records.append({"bundle": bd.name, "skipped": str(exc)})
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            records.append({"bundle": bd.name,
                            "error": f"{type(exc).__name__}: {exc}"})

    combined = {
        "language": a.lang, "split": a.split,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds"),
        "limit": a.limit,
        "runs": records,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"eval_{a.lang}_{a.split}.json").write_text(
        json.dumps(combined, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'run':<6}{'arch':<13}{'input':<10}{'seed':>5}{'n':>5}"
          f"{'MCD dB':>9}{'sd':>7}{'chance':>9}{'%ofch':>7}"
          f"{'logF0':>8}{'V/UV':>7}{'len':>7}")
    for r in records:
        if "summary" not in r:
            print(f"{r.get('run_id', r.get('bundle','?')):<6}"
                  f"  -- {r.get('skipped') or r.get('error')}")
            continue
        s = r["summary"]
        g = lambda k, f="mean": s.get(k, {}).get(f)
        fmt = lambda v, w, p: (f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}")
        ch = (r.get("chance") or {}).get("mcd_db_mean")
        got = g("mcd_db")
        frac = (100.0 * got / ch) if (isinstance(got, float)
                                      and isinstance(ch, float) and ch) else None
        print(f"{r['run_id']:<6}{r['architecture']:<13}{r['input_repr']:<10}"
              f"{str(r.get('seed','-')):>5}{s['n_utterances']:>5}"
              f"{fmt(got,9,3)}{fmt(g('mcd_db','sd'),7,3)}"
              f"{fmt(ch,9,2)}{fmt(frac,7,1)}"
              f"{fmt(g('log_f0_rmse'),8,4)}{fmt(g('vuv_error_rate'),7,3)}"
              f"{fmt(g('length_ratio'),7,3)}")
    print(f"\nwrote {a.out / f'eval_{a.lang}_{a.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
