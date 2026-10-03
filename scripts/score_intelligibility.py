#!/usr/bin/env python3
"""Score intelligibility by transcribing the synthesis and comparing the text.

    python scripts/score_intelligibility.py --lang hindi
    python scripts/score_intelligibility.py --lang hindi --backend mms --limit 25

**It runs on the CPU and says so.** The card is training. This harness forces
CUDA_VISIBLE_DEVICES="" before torch is imported, so it cannot take memory from
a run in flight, and it caps the BLAS thread count for the same reason
scripts/evaluate.py does. Passing --device cuda overrides that deliberately and
prints a warning; nothing here needs a GPU.

**Both recognisers, reported separately.** See src/eval/asr.py for why they are
both CTC and why their disagreement is a result rather than noise to average.
Each backend writes its own table and its own JSON, and no number in this file
combines them.

**The headline is the partitioned rate, not the corpus rate.** A corpus CER
says how intelligible a system is; it does not say whether the errors are the
ones the schwa rule predicts. src/analysis/position_errors.py splits the
character errors by whether the reference word carries a rule-predicted
deletion, and `excess` is the deletion-site rate minus the rate on words with
no site. Positive means the errors concentrate where the rule fires, which is
the shape the graphemic arm should show if explicit grapheme-to-phoneme
conversion buys anything.

**A mel-only run cannot be scored here.** Unlike the duration measure, this one
needs a waveform, so FastSpeech 2 is unscorable until r06 exists. The row is
written with the reason rather than dropped.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

INTERIM = HERE / "data" / "interim"
TABLES = HERE / "results" / "tables"


def _cap_threads(n: int, device: str) -> None:
    """Before torch, numpy or onnxruntime are imported, or it has no effect."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    if device == "cpu":
        # The card is training. This is the only thing that reliably keeps a
        # library from allocating on it, and it has to happen before the import.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def score_bundle(bundle_dir: pathlib.Path, lang: str, backend_name: str,
                 split: str, limit: int | None, device: str,
                 out_dir: pathlib.Path) -> dict:
    from scripts.evaluate import load_split
    from src.analysis import position_errors as pe
    from src.eval import asr
    from src.export.synthesize import load, read_manifest
    from src.g2p import G2P

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
        "split": split, "asr": asr.model_card(backend_name)["repo"],
    }
    if manifest["language"] != lang:
        record["skipped"] = f"trained on {manifest['language']}, not {lang}"
        return record
    if manifest.get("needs_vocoder"):
        record["skipped"] = (
            "mel-only architecture: intelligibility needs a waveform and the "
            "vocoder has not trained. The duration measure in "
            "scripts/evaluate.py does not need one and covers this run.")
        return record

    engine = asr.backend(backend_name, lang, device=device)
    record["asr_detail"] = engine.describe()

    b = load(bundle_dir, device="cpu")
    sr = b.sample_rate
    g2p = G2P.for_language(lang)
    rows_in = load_split(lang, split)
    if limit:
        rows_in = rows_in[:limit]

    per_utt, parts, failures, pairs = [], [], [], []
    for i, row in enumerate(rows_in, 1):
        try:
            sp = b.synthesize(row["text"])
            if sp.waveform is None:
                failures.append({"id": row["id"],
                                 "error": "no waveform from this bundle"})
                continue
            hyp = engine.transcribe(sp.waveform, sr)
            part = pe.partition(g2p, row["text"], hyp)
            parts.append(part)
            pairs.append((row["text"], hyp))
            per_utt.append({
                "id": row["id"],
                "reference": row["text"],
                "hypothesis": hyp,
                "cer": asr.cer(row["text"], hyp),
                "cer_with_spaces": asr.cer(row["text"], hyp, drop_spaces=False),
                "wer": asr.wer(row["text"], hyp),
                "insertions": part["insertions"],
                **{f"{c}_edits": part["rates"][c].edits for c in pe.CLASSES},
                **{f"{c}_chars": part["rates"][c].chars for c in pe.CLASSES},
            })
        except Exception as exc:                              # noqa: BLE001
            failures.append({"id": row["id"],
                             "error": f"{type(exc).__name__}: {exc}"})
            if len(failures) <= 2:
                traceback.print_exc()
        if i % 10 == 0:
            print(f"    {rid}/{backend_name}: {i}/{len(rows_in)}", flush=True)

    record["failures"] = failures
    record["n"] = len(per_utt)
    if not per_utt:
        record["skipped"] = "nothing transcribed; see failures"
        return record

    pooled = pe.accumulate(parts)
    record["corpus_cer"] = asr.corpus_cer(pairs)
    record["corpus_cer_with_spaces"] = asr.corpus_cer(pairs, drop_spaces=False)
    record["classes"] = {
        c: {"edits": pooled["rates"][c].edits,
            "chars": pooled["rates"][c].chars,
            "words": pooled["rates"][c].words,
            "cer": pooled["rates"][c].cer if pooled["rates"][c].has_rate else None}
        for c in pe.CLASSES}
    record["contrast"] = pe.contrast(pooled)
    record["insertions"] = pooled["insertions"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"intel_{rid}_{backend_name}.json").write_text(
        json.dumps({**record, "utterances": per_utt}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    return record


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def table(records: list[dict]) -> str:
    head = (f"{'run':5s} {'arch':12s} {'repr':9s} {'asr':15s} {'n':>4s} "
            f"{'CER':>7s} {'final':>7s} {'medial':>7s} {'none':>7s} "
            f"{'excess':>8s} {'ins':>5s}")
    lines = [head, "-" * len(head)]
    for r in records:
        if r.get("skipped"):
            lines.append(f"{r['run_id']:5s} {r.get('architecture','?'):12s} "
                         f"{r.get('input_repr','?'):9s} "
                         f"{'':15s} {'-':>4s}   {r['skipped']}")
            continue
        cls = r["classes"]
        con = (r["contrast"] or {}).get("final")
        lines.append(
            f"{r['run_id']:5s} {r['architecture']:12s} {r['input_repr']:9s} "
            f"{r['asr'].split('/')[-1][:15]:15s} {r['n']:4d} "
            f"{_fmt(r['corpus_cer']):>7s} {_fmt(cls['final']['cer']):>7s} "
            f"{_fmt(cls['medial']['cer']):>7s} {_fmt(cls['neither']['cer']):>7s} "
            f"{_fmt(con['excess'] if con else None):>8s} {r['insertions']:5d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="both",
                    choices=["indicconformer", "mms", "both"])
    ap.add_argument("--exports", default=str(HERE / "exports"))
    ap.add_argument("--runs", nargs="*", default=None,
                    help="bundle directory names; default is all of them")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(TABLES))
    a = ap.parse_args(argv)

    if a.device != "cpu":
        print(f"WARNING: --device {a.device} will allocate on the card, which "
              f"may be training. Nothing here needs a GPU.", file=sys.stderr)
    _cap_threads(a.threads, a.device)

    names = ["indicconformer", "mms"] if a.backend == "both" else [a.backend]
    root = pathlib.Path(a.exports)
    if not root.exists():
        print(f"no exports at {root}", file=sys.stderr)
        return 1
    dirs = sorted(d for d in root.iterdir() if d.is_dir()
                  and (d / "manifest.json").exists())
    if a.runs:
        wanted = set(a.runs)
        dirs = [d for d in dirs if d.name in wanted]
    if not dirs:
        print(f"no bundles found under {root}", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(a.out)
    records = []
    for name in names:
        card = __import__("src.eval.asr", fromlist=["x"]).model_card(name)
        print(f"== {name}: {card['repo']}@{card['revision'][:12]} "
              f"({card['licence']}, {card['role']})", flush=True)
        for d in dirs:
            try:
                records.append(score_bundle(d, a.lang, name, a.split,
                                            a.limit, a.device, out_dir))
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                records.append({"run_id": d.name, "skipped":
                                f"{type(exc).__name__}: {exc}"})

    print()
    print(table(records))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"intelligibility_{a.lang}_{stamp}.txt").write_text(
        table(records) + "\n", encoding="utf-8")
    (out_dir / f"intelligibility_{a.lang}_{stamp}.json").write_text(
        json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
