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
                 out_dir: pathlib.Path, draws: int = 1,
                 vocoder: pathlib.Path | None = None) -> dict:
    from scripts.evaluate import load_split
    from src.analysis import position_errors as pe
    from src.eval import asr
    from src.export.synthesize import draw_seed as synth_seed
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
    if manifest.get("needs_vocoder") and not vocoder:
        record["skipped"] = (
            "mel-only architecture: intelligibility needs a waveform and the "
            "vocoder has not trained. The duration measure in "
            "scripts/evaluate.py does not need one and covers this run.")
        return record

    engine = asr.backend(backend_name, lang, device=device)
    record["asr_detail"] = engine.describe()

    b = load(bundle_dir, device="cpu",
             vocoder=vocoder if manifest.get("needs_vocoder") else None)
    if b.vocoder is not None:
        record["vocoder"] = {
            "run_id": b.vocoder.manifest["run_id"],
            "step": b.vocoder.manifest.get("step"),
            "config_hash": b.vocoder.manifest.get("config_hash"),
        }
    sr = b.sample_rate
    g2p = G2P.for_language(lang)
    rows_in = load_split(lang, split)
    if limit:
        rows_in = rows_in[:limit]

    failures: list[dict] = []
    # One pass per draw. VITS samples at inference, so a single draw is
    # reproducible once seeded (see synthesize.draw_seed) but it is not the
    # system: the system is a distribution over draws. With --draws > 1 the
    # spread across draws is reported as its own number, which is the
    # inference-side counterpart of the seed-variance floor. Measured on
    # 3 October, two unseeded passes over the same ten utterances moved r05's
    # medial-site errors from 16 of 96 characters to 5 and reversed the sign
    # of the headline contrast, so this is not a refinement.
    passes = []
    for d in range(max(1, int(draws))):
        per_utt, parts, pairs = [], [], []
        for i, row in enumerate(rows_in, 1):
            try:
                sp = b.synthesize(row["text"], draw=d)
                if sp.waveform is None:
                    failures.append({"id": row["id"], "draw": d,
                                     "error": "no waveform from this bundle"})
                    continue
                hyp = engine.transcribe(sp.waveform, sr)
                part = pe.partition(g2p, row["text"], hyp)
                parts.append(part)
                pairs.append((row["text"], hyp))
                per_utt.append({
                    "id": row["id"],
                    "draw": d,
                    "seed": synth_seed(rid, row["text"], d),
                    "reference": row["text"],
                    "hypothesis": hyp,
                    "cer": asr.cer(row["text"], hyp),
                    "cer_with_spaces": asr.cer(row["text"], hyp,
                                               drop_spaces=False),
                    "wer": asr.wer(row["text"], hyp),
                    "insertions": part["insertions"],
                    **{f"{c}_edits": part["rates"][c].edits for c in pe.CLASSES},
                    **{f"{c}_chars": part["rates"][c].chars for c in pe.CLASSES},
                })
            except Exception as exc:                          # noqa: BLE001
                failures.append({"id": row["id"], "draw": d,
                                 "error": f"{type(exc).__name__}: {exc}"})
                if len(failures) <= 2:
                    traceback.print_exc()
            if i % 10 == 0:
                print(f"    {rid}/{backend_name} draw {d}: "
                      f"{i}/{len(rows_in)}", flush=True)
        if per_utt:
            passes.append({"draw": d, "per_utt": per_utt,
                           "parts": parts, "pairs": pairs})
        else:
            # Every utterance in this draw failed. Continuing would repeat the
            # same failure for every remaining draw, and the per-draw progress
            # counter would keep printing 50/50 while nothing was transcribed,
            # which is what happened on 4 October: one AttributeError was
            # reported twice and then 40 passes ran in silence. Stop here and
            # say so.
            print(f"    {rid}/{backend_name} draw {d}: ALL "
                  f"{len(rows_in)} utterances failed, abandoning the "
                  f"remaining draws", flush=True)
            break

    record["failures"] = failures
    record["draws"] = len(passes)
    if failures:
        # Distinct causes with counts, so a wall of identical tracebacks
        # becomes one line that names the bug.
        kinds: dict[str, int] = {}
        for f in failures:
            kinds[str(f.get("error"))] = kinds.get(str(f.get("error")), 0) + 1
        record["failure_kinds"] = kinds
        print(f"    {rid}/{backend_name}: {len(failures)} failures, "
              f"{len(kinds)} distinct:", flush=True)
        for msg, count in sorted(kinds.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {count:5d}x {msg[:110]}", flush=True)
    if not passes:
        record["n"] = 0
        record["skipped"] = (
            f"nothing transcribed: {len(failures)} failures. "
            + "; ".join(f"{c}x {m[:70]}"
                        for m, c in sorted(record.get("failure_kinds", {}).items(),
                                           key=lambda kv: -kv[1])[:2]))
        return record
    record["n"] = len(passes[0]["per_utt"])

    # Per draw first, so the spread is computable, then the mean across draws
    # is what the table prints. A mean with no spread beside it is what the
    # 3 October tables were, and they disagreed with each other.
    per_draw = []
    for pz in passes:
        pooled = pe.accumulate(pz["parts"])
        per_draw.append({
            "draw": pz["draw"],
            "corpus_cer": asr.corpus_cer(pz["pairs"]),
            "classes": {c: (pooled["rates"][c].cer
                            if pooled["rates"][c].has_rate else None)
                        for c in pe.CLASSES},
            "contrast": pe.contrast(pooled),
            "insertions": pooled["insertions"],
        })
    record["per_draw"] = per_draw

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _sd(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    # The pooled counts are taken over every draw, so the character
    # denominators in the counts table are draws x characters and the rate is
    # the pooled rate rather than a mean of rates.
    pooled_all = pe.accumulate([q for pz in passes for q in pz["parts"]])
    all_pairs = [q for pz in passes for q in pz["pairs"]]
    record["corpus_cer"] = asr.corpus_cer(all_pairs)
    record["corpus_cer_sd"] = _sd([d["corpus_cer"] for d in per_draw])
    record["corpus_cer_with_spaces"] = asr.corpus_cer(all_pairs,
                                                      drop_spaces=False)
    record["classes"] = {
        c: {"edits": pooled_all["rates"][c].edits,
            "chars": pooled_all["rates"][c].chars,
            "words": pooled_all["rates"][c].words,
            "cer": (pooled_all["rates"][c].cer
                    if pooled_all["rates"][c].has_rate else None),
            "cer_sd": _sd([d["classes"][c] for d in per_draw])}
        for c in pe.CLASSES}
    record["contrast"] = pe.contrast(pooled_all)
    for cls in (pe.FINAL, pe.MEDIAL):
        vals = [(d["contrast"] or {}).get(cls, {}) for d in per_draw]
        xs = [v.get("excess") for v in vals if v]
        if record["contrast"].get(cls) and xs:
            record["contrast"][cls]["excess_sd"] = _sd(xs)
            record["contrast"][cls]["excess_mean_over_draws"] = _mean(xs)
    record["insertions"] = pooled_all["insertions"]

    # Kept on the record so the two arms of a pair can be resampled together,
    # with the DRAWS MERGED per utterance.
    #
    # This was the first draw alone, on the reasoning that the bootstrap needs
    # one row per utterance and draws are not utterances. Both halves of that
    # are true and the conclusion was wrong: it left the interval computed from
    # a fifth of the data the table's class rates pool, so on 4 October the
    # table reported a final-site excess difference of +0.0263 while the
    # bootstrap beside it reported +0.0057. Two numbers in one table from two
    # different samples.
    #
    # Merging keeps the utterance as the sampling unit, which is what the
    # resampling is about, and gives each row the draws it actually has. A
    # resample then draws utterances and carries all of each one's draws, so
    # draw variance stays inside the row rather than becoming another thing
    # being resampled.
    record["utterances"] = _merge_draws(passes)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"intel_{rid}_{backend_name}.json").write_text(
        json.dumps({**record,
                    "utterances": [u for pz in passes for u in pz["per_utt"]]},
                   indent=1, ensure_ascii=False), encoding="utf-8")
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
    if failures:
        kinds: dict[str, int] = {}
        for f in failures:
            kinds[str(f.get("error"))] = kinds.get(str(f.get("error")), 0) + 1
        out["failure_kinds"] = kinds
        print(f"    floor {backend_name}: {len(failures)} failures, "
              f"{len(kinds)} distinct:", flush=True)
        for msg, count in sorted(kinds.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {count:5d}x {msg[:110]}", flush=True)
    if not pairs:
        top = sorted((out.get("failure_kinds") or {}).items(),
                     key=lambda kv: -kv[1])[:2]
        out["error"] = ("nothing transcribed from the references: "
                        + "; ".join(f"{c}x {m[:70]}" for m, c in top)
                        if top else "nothing transcribed from the references")
        return out
    pooled = pe.accumulate(parts)
    out["corpus_cer"] = asr.corpus_cer(pairs)
    out["classes"] = {
        c: (pooled["rates"][c].cer if pooled["rates"][c].has_rate else None)
        for c in pe.CLASSES}
    out["contrast"] = pe.contrast(pooled)
    out["examples"] = [{"reference": r, "hypothesis": h} for r, h in pairs[:3]]
    return out


def vocoder_ceiling(vocoder_dir: pathlib.Path, backend_name: str, lang: str,
                    split: str, limit: int | None, device: str) -> dict:
    """What the vocoder can reach on mels that are already correct.

    The recogniser floor says how well the recogniser reads real recordings.
    This says how well it reads the vocoder's rendering of the REAL mels. Any
    FastSpeech 2 score is bounded by it, because that arm's audio goes through
    the same vocoder, so it separates two suspects that are otherwise stacked:
    a bad mel from the acoustic model, and a bad vocoder.

    It was needed the moment r01 and r04 came back at a character error rate of
    0.9932 and 0.9924 with edits equal to reference characters, which is the
    "transcribed nothing" ceiling rather than a bad score. Attributing that to
    FastSpeech 2 without this measurement would be attributing it to whichever
    suspect was named first. r06's own held-out mel reconstruction error is
    0.2465 against 0.1434 for a Griffin-Lim round trip through the same
    quantity, so the vocoder is 1.72 times worse than an algorithm that needs
    no training, which is reason enough not to guess.

    The mel comes from features._spec_and_mel, the one implementation that
    produced every training mel, so the ceiling cannot be flattered by a second
    copy of the analysis.
    """
    import soundfile as sf

    from scripts.evaluate import load_split
    from src.analysis import position_errors as pe
    from src.eval import asr
    from src.export.synthesize import Bundle
    from src.g2p import G2P
    from src.train import features as F

    voc = Bundle(pathlib.Path(vocoder_dir), device=device)
    if not voc.is_vocoder:
        raise SystemExit(f"{vocoder_dir}: not a vocoder bundle")
    engine = asr.backend(backend_name, lang, device=device)
    g2p = G2P.for_language(lang)
    sr = voc.sample_rate
    col = "wav22" if sr == 22050 else "wav16"
    rows = load_split(lang, split)
    if limit:
        rows = rows[:limit]

    pairs, parts, failures = [], [], []
    for row in rows:
        path = INTERIM / lang / row.get(col, "")
        if not row.get(col) or not path.exists():
            failures.append({"id": row["id"], "error": f"no {col} at {path}"})
            continue
        try:
            wav, got = sf.read(str(path), dtype="float32")
            if got != sr:
                failures.append({"id": row["id"],
                                 "error": f"{col} is {got} Hz, vocoder is {sr}"})
                continue
            mel = F.mel_from_array(wav, sr)            # (frames, n_mels)
            hyp = engine.transcribe(voc.vocode(mel.T), sr)
            pairs.append((row["text"], hyp))
            parts.append(pe.partition(g2p, row["text"], hyp))
        except Exception as exc:                              # noqa: BLE001
            failures.append({"id": row["id"],
                             "error": f"{type(exc).__name__}: {exc}"})
    out = {"backend": backend_name, "n": len(pairs),
           "vocoder": voc.manifest["run_id"],
           "vocoder_step": voc.manifest.get("step"),
           "failures": failures}
    # Distinct causes with counts. The same lesson as score_bundle, which was
    # given it yesterday and this function was not: on 5 October the
    # IndicConformer ceiling failed on all 50 utterances and the table said
    # only "nothing transcribed", so the reason sat in a JSON file nobody had
    # a reason to open while MMS's row beside it looked fine.
    if failures:
        kinds: dict[str, int] = {}
        for f in failures:
            kinds[str(f.get("error"))] = kinds.get(str(f.get("error")), 0) + 1
        out["failure_kinds"] = kinds
        print(f"    ceiling {backend_name}: {len(failures)} failures, "
              f"{len(kinds)} distinct:", flush=True)
        for msg, count in sorted(kinds.items(), key=lambda kv: -kv[1])[:5]:
            print(f"      {count:5d}x {msg[:110]}", flush=True)
    if not pairs:
        top = sorted((out.get("failure_kinds") or {}).items(),
                     key=lambda kv: -kv[1])[:2]
        out["error"] = ("nothing transcribed from the vocoded ground truth: "
                        + "; ".join(f"{c}x {m[:70]}" for m, c in top)
                        if top else
                        "nothing transcribed from the vocoded ground truth")
        return out
    pooled = pe.accumulate(parts)
    out["corpus_cer"] = asr.corpus_cer(pairs)
    out["classes"] = {c: (pooled["rates"][c].cer
                          if pooled["rates"][c].has_rate else None)
                      for c in pe.CLASSES}
    out["examples"] = [{"reference": r, "hypothesis": h} for r, h in pairs[:3]]
    return out


def ceiling_table(ceilings: list[dict]) -> str:
    head = (f"{'asr':16s} {'vocoder':10s} {'n':>4s} {'CER':>7s} "
            f"{'final':>7s} {'medial':>7s} {'none':>7s}")
    lines = ["VOCODER CEILING (the real mels, through this vocoder)",
             head, "-" * len(head)]
    for c in ceilings:
        if c.get("error"):
            lines.append(f"{c['backend'][:16]:16s} "
                         f"{str(c.get('vocoder'))[:10]:10s} "
                         f"{c.get('n', 0):4d}   {c['error']}")
            continue
        cl = c["classes"]
        lines.append(f"{c['backend'][:16]:16s} {str(c['vocoder'])[:10]:10s} "
                     f"{c['n']:4d} {_fmt(c['corpus_cer']):>7s} "
                     f"{_fmt(cl['final']):>7s} {_fmt(cl['medial']):>7s} "
                     f"{_fmt(cl['neither']):>7s}")
    return "\n".join(lines)


def _merge_draws(passes: list[dict]) -> list[dict]:
    """One row per utterance, with every draw's counts summed into it.

    Order follows the first draw, so two arms scored on the same split come
    out in the same order and the paired bootstrap's order check passes for
    the right reason rather than by luck.
    """
    from src.analysis import position_errors as pe

    order = [u["id"] for u in passes[0]["per_utt"]]
    merged: dict[str, dict] = {}
    for pz in passes:
        for u in pz["per_utt"]:
            row = merged.setdefault(u["id"], {"id": u["id"], "draws": 0})
            row["draws"] += 1
            for c in pe.CLASSES:
                for field in (f"{c}_edits", f"{c}_chars"):
                    row[field] = row.get(field, 0) + int(u.get(field, 0))
            row["insertions"] = row.get("insertions", 0) + int(
                u.get("insertions", 0))
    # An utterance that failed in some draws keeps the draws it has; its
    # denominator is smaller, which is correct, and the row is not dropped.
    return [merged[i] for i in order if i in merged]


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
            f"{'dr':>3s} {'CER':>7s} {'+-':>7s} {'final':>7s} {'medial':>7s} "
            f"{'none':>7s} {'excess':>8s} {'exc+-':>7s} {'ins':>5s}")
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
            f"{r.get('draws', 1):3d} "
            f"{_fmt(r['corpus_cer']):>7s} {_fmt(r.get('corpus_cer_sd')):>7s} "
            f"{_fmt(cls['final']['cer']):>7s} "
            f"{_fmt(cls['medial']['cer']):>7s} {_fmt(cls['neither']['cer']):>7s} "
            f"{_fmt(con['excess'] if con else None):>8s} "
            f"{_fmt((con or {}).get('excess_sd')):>7s} {r['insertions']:5d}")
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
    ap.add_argument("--vocoder", type=pathlib.Path, default=None,
                    help="a vocoder BUNDLE directory for the mel-only arms, "
                         "e.g. exports/r06_step18000. Without it, FastSpeech 2 "
                         "runs are reported as unscorable rather than skipped.")
    ap.add_argument("--draws", type=int, default=1,
                    help="synthesis draws per utterance. VITS samples at "
                         "inference, so one seeded draw is reproducible but "
                         "is not the system; >1 reports the spread across "
                         "draws, which on 10 utterances reversed the sign of "
                         "the headline contrast on 3 October")
    ap.add_argument("--boot", type=int, default=2000,
                    help="paired resamples for the interval; 0 skips it")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--ceiling", action="store_true",
                    help="also transcribe the REAL mels rendered through "
                         "--vocoder. Any mel-only arm's score is bounded by "
                         "this, so it separates a bad acoustic mel from a bad "
                         "vocoder. Needs --vocoder.")
    ap.add_argument("--ceiling-only", action="store_true",
                    help="measure that ceiling and stop. Loads no acoustic "
                         "bundle.")
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
    ceilings = []
    if (a.ceiling or a.ceiling_only) and a.vocoder:
        for name in names:
            print(f"== {name}: vocoder ceiling via {a.vocoder}", flush=True)
            try:
                ceilings.append(vocoder_ceiling(a.vocoder, name, a.lang,
                                                a.split, a.limit, a.device))
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                ceilings.append({"backend": name, "n": 0,
                                 "error": f"{type(exc).__name__}: {exc}"})
        print()
        print(ceiling_table(ceilings))
        for c in ceilings:
            for ex in c.get("examples", []):
                print(f"  {c['backend']} ref: {ex['reference']}")
                print(f"  {c['backend']} hyp: {ex['hypothesis']}")
        print()
    elif (a.ceiling or a.ceiling_only) and not a.vocoder:
        print("--ceiling needs --vocoder", file=sys.stderr)
        return 1
    if a.ceiling_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"vocoder_ceiling_{a.lang}.json").write_text(
            json.dumps(ceilings, indent=1, ensure_ascii=False),
            encoding="utf-8")
        return 0

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
                                            a.limit, a.device, out_dir,
                                            draws=a.draws,
                                            vocoder=a.vocoder))
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
    if ceilings:
        print(ceiling_table(ceilings))
        print()
    print(table(records))
    print()
    print(counts_table(records))
    if a.draws < 2 and any(r.get("architecture") == "vits" for r in records):
        print()
        print("NOTE: --draws 1. VITS samples at inference, so these numbers "
              "are one reproducible draw and carry no spread. Two unseeded "
              "passes on 3 October reversed the sign of the medial contrast. "
              "Use --draws 5 or more before reporting any of this.")
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
        json.dumps({"reference_floor": floors,
                    "vocoder_ceiling": ceilings, "runs": records},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
