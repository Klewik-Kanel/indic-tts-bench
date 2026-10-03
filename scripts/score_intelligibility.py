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

    # Kept on the record so the two arms of a pair can be resampled together.
    # Stripped again before the combined table is written, because the
    # per-run JSON below already holds them.
    record["utterances"] = per_utt
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"intel_{rid}_{backend_name}.json").write_text(
        json.dumps({**record, "utterances": per_utt}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    return record


def reference_floor(backend_name: str, lang: str, split: str,
                    limit: int | None, device: str) -> dict:
    """The recogniser's own error rate on the real recordings.

    Without this a CER on synthesis has no denominator. A recogniser that
    scores 0.18 on studio ground truth has told you that 0.18 of your
    synthesis CER was never about the synthesis, and the same partition run on
    real speech says whether the recogniser itself struggles at deletion sites,
    which would otherwise read as a property of the models. It is the
    counterpart of the chance level in scripts/evaluate.py.

    It also catches a broken load. MMS warns that its language-model head was
    "newly initialized because the shapes did not match", which is documented
    as expected because load_adapter replaces it afterwards. A head that
    actually stayed random would still emit fluent-looking Devanagari, and that
    garbage would read as unintelligible synthesis. Ground truth is the only
    input whose correct transcript is known in advance, so a floor near zero
    confirms the adapter loaded and a floor near one says it did not.

    Computed once per backend and attached to every run's record, because the
    references are the same for both arms of a pair. Always from wav16: the
    recognisers want 16 kHz, the corpus already has it, and a reference must
    never be resampled.
    """
    import soundfile as sf

    from scripts.evaluate import load_split
    from src.analysis import position_errors as pe
    from src.eval import asr
    from src.g2p import G2P

    engine = asr.backend(backend_name, lang, device=device)
    g2p = G2P.for_language(lang)
    rows = load_split(lang, split)
    if limit:
        rows = rows[:limit]

    pairs, parts, failures = [], [], []
    for row in rows:
        path = INTERIM / lang / row.get("wav16", "")
        if not row.get("wav16") or not path.exists():
            failures.append({"id": row["id"], "error": f"no wav16 at {path}"})
            continue
        try:
            wav, got = sf.read(str(path), dtype="float32")
            if got != asr.TARGET_SR:
                failures.append({"id": row["id"],
                                 "error": f"wav16 is {got} Hz, not "
                                          f"{asr.TARGET_SR}"})
                continue
            hyp = engine.transcribe(wav, got)
            pairs.append((row["text"], hyp))
            parts.append(pe.partition(g2p, row["text"], hyp))
        except Exception as exc:                              # noqa: BLE001
            failures.append({"id": row["id"],
                             "error": f"{type(exc).__name__}: {exc}"})
    out = {"backend": backend_name, "n": len(pairs), "failures": failures}
    if not pairs:
        out["error"] = "nothing transcribed from the references"
        return out
    pooled = pe.accumulate(parts)
    out["corpus_cer"] = asr.corpus_cer(pairs)
    out["classes"] = {
        c: (pooled["rates"][c].cer if pooled["rates"][c].has_rate else None)
        for c in pe.CLASSES}
    out["contrast"] = pe.contrast(pooled)
    out["examples"] = [{"reference": r, "hypothesis": h} for r, h in pairs[:3]]
    return out


def _select(dirs: list[pathlib.Path], wanted: list[str]) -> list[pathlib.Path]:
    """Bundles named by directory name OR by run_id.

    A bundle directory is `r02_step100000` while the run is `r02`, and the run
    id is what a person thinks in. Matching only the directory name turned
    `--runs r02 r05` into "no bundles found", which reads as a missing export
    rather than a filter that matched nothing. Both spellings work now, and an
    unmatched name is named along with what is actually on disk instead of
    being silently dropped.
    """
    from src.export.synthesize import read_manifest

    by_name: dict[str, pathlib.Path] = {}
    for d in dirs:
        by_name[d.name] = d
        try:
            rid = read_manifest(d)["run_id"]
        except SystemExit:
            continue
        # A directory name is never overwritten by a run id, so an exact
        # directory match always wins.
        by_name.setdefault(rid, d)

    out, missing = [], []
    for w in wanted:
        d = by_name.get(w)
        if d is None:
            missing.append(w)
        elif d not in out:
            out.append(d)
    if missing:
        print(f"no bundle for {missing}; on disk: "
              f"{sorted(d.name for d in dirs)}", file=sys.stderr)
    return out


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def floor_table(floors: list[dict]) -> str:
    """The recogniser's own rate on the real recordings, printed first.

    Printed above the runs rather than beside them, because it is a property
    of the recogniser and the corpus, not of any run. A synthesis CER is read
    against it.
    """
    head = (f"{'asr':16s} {'n':>4s} {'refCER':>7s} {'final':>7s} "
            f"{'medial':>7s} {'none':>7s} {'excess':>8s}")
    lines = ["GROUND TRUTH (the recogniser's own floor)", head, "-" * len(head)]
    for f in floors:
        if f.get("error"):
            lines.append(f"{f['backend'][:16]:16s} {f.get('n', 0):4d}   "
                         f"{f['error']}")
            continue
        con = (f.get("contrast") or {}).get("final")
        cls = f["classes"]
        lines.append(
            f"{f['backend'][:16]:16s} {f['n']:4d} {_fmt(f['corpus_cer']):>7s} "
            f"{_fmt(cls['final']):>7s} {_fmt(cls['medial']):>7s} "
            f"{_fmt(cls['neither']):>7s} "
            f"{_fmt(con['excess'] if con else None):>8s}")
    return "\n".join(lines)


def counts_table(records: list[dict]) -> str:
    """Edits over reference characters per class, which the rates alone hide.

    Added because reading 0.1667 and having to work out that it was 16 edits
    over 96 characters cost a round trip. A rate whose denominator is 96 is a
    different claim from one whose denominator is 4000, and the table should
    say which it is.
    """
    head = (f"{'run':5s} {'repr':9s} {'asr':16s} {'final':>12s} "
            f"{'medial':>12s} {'none':>12s} {'all':>12s}")
    lines = ["COUNTS (edits / reference characters)", head, "-" * len(head)]
    for r in records:
        if r.get("skipped"):
            continue
        cls = r["classes"]
        def cell(c):
            return f"{cls[c]['edits']}/{cls[c]['chars']}"
        total_e = sum(cls[c]["edits"] for c in ("final", "medial", "neither"))
        total_c = sum(cls[c]["chars"] for c in ("final", "medial", "neither"))
        lines.append(
            f"{r['run_id']:5s} {r['input_repr']:9s} "
            f"{r['asr'].split('/')[-1][:16]:16s} {cell('final'):>12s} "
            f"{cell('medial'):>12s} {cell('neither'):>12s} "
            f"{f'{total_e}/{total_c}':>12s}")
    return "\n".join(lines)


def pair_records(records: list[dict]) -> list[tuple[dict, dict]]:
    """Phonemic and graphemic records that differ only in input_repr.

    Keyed on everything that must match for the pair to be a comparison:
    recogniser, architecture, language, data rung and split. Two runs that
    differ in any of those are not two arms of one contrast.
    """
    by_cell: dict[tuple, dict] = {}
    for r in records:
        if r.get("skipped") or not r.get("utterances"):
            continue
        key = (r.get("asr"), r.get("architecture"), r.get("language"),
               r.get("step"), r.get("split"))
        by_cell.setdefault(key, {})[r.get("input_repr")] = r
    out = []
    for arms in by_cell.values():
        if "phoneme" in arms and "grapheme" in arms:
            out.append((arms["phoneme"], arms["grapheme"]))
    return out


def bootstrap_table(records: list[dict], n_boot: int, seed: int) -> str:
    """Paired resampling intervals on the grapheme-minus-phoneme excess.

    One index draw is applied to both arms, because the two arms are scored on
    the same utterances and the pairing removes the between-utterance variance
    that otherwise dominates the width.
    """
    from src.analysis import position_errors as pe

    pairs = pair_records(records)
    head = (f"{'asr':16s} {'site':7s} {'n':>4s} {'excess':>9s} "
            f"{'lo':>9s} {'hi':>9s} {'sign':>6s} {'clear of 0':>11s}")
    lines = [f"PAIRED BOOTSTRAP, grapheme minus phoneme, {n_boot} resamples, "
             f"seed {seed}", head, "-" * len(head)]
    if not pairs:
        lines.append("  no phonemic/graphemic pair among these runs")
        return "\n".join(lines)
    for ph, gr in pairs:
        asr = (ph.get("asr") or "").split("/")[-1][:16]
        for cls in (pe.FINAL, pe.MEDIAL):
            try:
                out = pe.bootstrap_difference(ph["utterances"], gr["utterances"],
                                              cls, n_boot=n_boot, seed=seed)
            except ValueError as exc:
                lines.append(f"{asr:16s} {cls:7s} {'-':>4s}   {exc}")
                continue
            if out.get("point") is None or "lo" not in out:
                lines.append(f"{asr:16s} {cls:7s} {out['n']:4d}   "
                             f"{out.get('reason', 'no interval')}")
                continue
            lines.append(
                f"{asr:16s} {cls:7s} {out['n']:4d} {out['point']:+9.4f} "
                f"{out['lo']:+9.4f} {out['hi']:+9.4f} "
                f"{out['same_sign']:6.2f} "
                f"{('yes' if out['excludes_zero'] else 'no'):>11s}")
    return "\n".join(lines)


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
                    help="run ids (r02) or bundle directory names "
                         "(r02_step100000); default is all of them")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(TABLES))
    ap.add_argument("--no-floor", action="store_true",
                    help="skip the ground-truth pass. The synthesis numbers "
                         "then have no denominator, so this is for debugging "
                         "only")
    ap.add_argument("--boot", type=int, default=2000,
                    help="paired resamples for the interval; 0 skips it")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--floor-only", action="store_true",
                    help="transcribe the real recordings and stop. The fastest "
                         "check that a recogniser loaded correctly")
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
    if not dirs:
        print(f"no bundles found under {root}", file=sys.stderr)
        return 1
    if a.runs:
        dirs = _select(dirs, a.runs)
        if not dirs:
            return 1

    out_dir = pathlib.Path(a.out)
    floors = []
    if not a.no_floor:
        for name in names:
            print(f"== {name}: ground truth", flush=True)
            try:
                floors.append(reference_floor(name, a.lang, a.split,
                                              a.limit, a.device))
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                floors.append({"backend": name, "n": 0,
                               "error": f"{type(exc).__name__}: {exc}"})
        print()
        print(floor_table(floors))
        for f in floors:
            if not f.get("error"):
                for ex in f.get("examples", []):
                    print(f"  {f['backend']} ref: {ex['reference']}")
                    print(f"  {f['backend']} hyp: {ex['hypothesis']}")
        print()
    if a.floor_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"asr_floor_{a.lang}.json").write_text(
            json.dumps(floors, indent=1, ensure_ascii=False), encoding="utf-8")
        return 0

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

    by_backend = {f["backend"]: f for f in floors}
    for r in records:
        detail = r.get("asr_detail") or {}
        f = by_backend.get(detail.get("backend"))
        if f and not f.get("error"):
            r["reference_floor"] = {"corpus_cer": f.get("corpus_cer"),
                                    "classes": f.get("classes"),
                                    "n": f.get("n")}

    print()
    if floors:
        print(floor_table(floors))
        print()
    print(table(records))
    print()
    print(counts_table(records))
    boot = ""
    if a.boot:
        boot = bootstrap_table(records, a.boot, a.boot_seed)
        print()
        print(boot)
    # The per-run JSON already holds them; keeping them here would put every
    # utterance of every run into the combined file twice.
    for r in records:
        r.pop("utterances", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"intelligibility_{a.lang}_{stamp}.txt").write_text(
        (floor_table(floors) + "\n\n" if floors else "")
        + table(records) + "\n\n" + counts_table(records)
        + (("\n\n" + boot) if boot else "") + "\n", encoding="utf-8")
    (out_dir / f"intelligibility_{a.lang}_{stamp}.json").write_text(
        json.dumps({"reference_floor": floors, "runs": records},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
