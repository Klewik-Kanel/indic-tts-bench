#!/usr/bin/env python3
"""Render the demo set through every bundle, into a static site's data.

A Static Space serves files and runs no Python, so everything a listener needs
has to exist as a file before it is pushed: the audio, the token sequences, and
the provenance of the weights that produced each clip. This writes all of it.

The natural recording goes in alongside the models, at the same sample rate as
the arm it sits next to. Without it a listener has nothing to calibrate
against, and "which of these two is better" is a much weaker question than
"how far is each from the speaker". It is also the reference a MOS or an AB
listening test needs, so this output is the listening-test material and not
only a demo.

Token sequences are written per arm because they are the point: `n ə m ə k`
beside `न म क` is the ablation in one line.

A mel-only architecture writes no audio unless --griffin-lim is passed. That
flag is explicit, never a default, because Griffin-Lim recovers phase by
iteration and has its own metallic signature: a listener told "this is the
model" would be hearing the algorithm. Clips produced that way carry
vocoder: "griffin-lim" in data.json and the page labels them.

    python scripts/render_demo.py --lang hindi
    python scripts/render_demo.py --lang hindi --bundles exports/r02_step100000
    python scripts/render_demo.py --lang hindi --griffin-lim
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.export import griffinlim                                 # noqa: E402
from src.export.synthesize import (END_TO_END,                     # noqa: E402
                                   VOCODER_ARCHITECTURES,
                                   load, read_manifest, write_wav)
from src.train.runner import load_config                           # noqa: E402

INTERIM = HERE / "data" / "interim"
TABLES = HERE / "results" / "tables"
CONFIGS = HERE / "configs"
DEFAULT_OUT = HERE / "space_static"


def reference_column(sample_rate: int) -> str:
    return "wav22" if sample_rate == 22050 else "wav16"


# Python's json writes Infinity, -Infinity and NaN, and reads them back, but
# none of the three is JSON. A browser's JSON.parse refuses the whole file, so
# one bad field in one run blanks the entire listening test. r08 trained with
# a non-finite config_hash on 2 October, that value reached its manifest, and
# the page showed "Could not load data.json" with every arm gone.
NOT_JSON = {float("inf"): "+Infinity", float("-inf"): "-Infinity"}


def json_safe(value, where="", seen=None):
    """`value` with every non-finite float replaced by a string saying so.

    The replacement is a string and not a null, because the field did hold
    something: a run whose recorded hash is not a hash. Silently dropping it
    would hide the defect that RESULTS.md documents, and keeping the float
    would make the file unreadable.
    """
    import math

    if isinstance(value, float) and not math.isfinite(value):
        name = NOT_JSON.get(value, "NaN")
        print(f"  {where or 'a field'} is {name}, which is not JSON. Writing "
              f'"not-recoverable" instead; see RESULTS.md on r08.')
        return "not-recoverable"
    if isinstance(value, dict):
        return {k: json_safe(v, f"{where}.{k}" if where else str(k))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def matrix_cell(man: dict) -> dict:
    """Which cell of the matrix a run is: its ladder rung and its seed.

    A page listing eight arms cannot group them without these two. Bundles
    exported before the fields entered the manifest do not carry them, and the
    config file does, but reading a config file is only safe against the hash:
    a config edited since the run was trained no longer hashes to the value
    recorded in the bundle. On a mismatch this reports nothing rather than
    something plausible and wrong, and says so, because a page that groups an
    arm under the wrong rung is worse than one that leaves it ungrouped.
    """
    cell = {"data": man.get("data") or "", "seed": man.get("seed")}
    if cell["data"] and cell["seed"] is not None:
        return cell

    rid = man["run_id"]
    cfg_path = CONFIGS / f"{rid}.yaml"
    if not cfg_path.exists():
        print(f"  {rid}: no configs/{rid}.yaml, so its rung and seed are "
              "unknown to the page")
        return cell

    recorded = str(man.get("config_hash") or "")
    cfg = load_config(cfg_path)
    on_disk = str(cfg.get("config_hash") or "")
    if not recorded or on_disk != recorded:
        print(f"  {rid}: bundle config_hash {recorded or '(none)'} but "
              f"configs/{rid}.yaml is {on_disk or '(none)'}, so the rung and "
              "seed are NOT read from it. Re-export the bundle.")
        return cell

    cell["data"] = cell["data"] or (cfg.get("data") or "")
    if cell["seed"] is None:
        cell["seed"] = cfg.get("seed")
    return cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--set", type=pathlib.Path,
                    help="demo set json; defaults to the one for --lang")
    ap.add_argument("--bundles", nargs="*", type=pathlib.Path,
                    help="bundle directories; defaults to every one in exports/")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vocoder", type=pathlib.Path, default=None,
                    help="a vocoder BUNDLE directory, e.g. exports/r06_step18000. "
                         "Its sample rate and language are checked against each "
                         "mel-only bundle before anything is rendered. With this, "
                         "--griffin-lim is unnecessary and the clips are real "
                         "system output rather than a labelled placeholder.")
    ap.add_argument("--griffin-lim", action="store_true",
                    help="make mel-only arms audible with a labelled "
                         "Griffin-Lim placeholder, not a vocoder")
    ap.add_argument("--gl-iters", type=int,
                    default=griffinlim.DEFAULT_ITERS,
                    help="Griffin-Lim iterations; past 60 buys almost nothing")
    a = ap.parse_args(argv)

    set_path = a.set or TABLES / f"demo_set_{a.lang}.json"
    if not set_path.exists():
        raise SystemExit(f"{set_path}: run scripts/build_demo_set.py first")
    demo = json.loads(set_path.read_text(encoding="utf-8"))

    bundles = a.bundles or sorted(
        p for p in (HERE / "exports").glob("*") if (p / "manifest.json").exists())
    if not bundles:
        raise SystemExit("no bundles; run python -m src.export.bundle first")

    audio_root = a.out / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    runs, rendered, failures = [], {}, []
    rates_needed: set[int] = set()

    for bundle_dir in bundles:
        # exports/ holds the vocoders beside the voices, and the default glob
        # takes everything in it. A vocoder has no text side, so it is not an
        # arm of a listening test: it is dropped here, by its manifest, before
        # any weights are loaded. Reaching synthesis with one aborts the whole
        # render, which is how a run that had already produced every clip
        # ended up writing no data.json at all.
        man = read_manifest(bundle_dir)
        if man["architecture"] in VOCODER_ARCHITECTURES:
            print(f"  skipping {bundle_dir.name}: "
                  f"{man['architecture']} is a vocoder, not a voice")
            continue
        if man["language"] != a.lang:
            print(f"  skipping {bundle_dir.name}: trained on "
                  f"{man['language']}, not {a.lang}")
            continue
        # The vocoder goes to the mel-only arms and to no others. Decided from
        # the manifest rather than by catching attach_vocoder's refusal, so a
        # real rate or language mismatch stays fatal instead of falling back
        # to a silent arm that looks like a result.
        needs = bool(man.get("needs_vocoder",
                             man["architecture"] not in END_TO_END))
        voc = a.vocoder if (a.vocoder and needs) else None
        b = load(bundle_dir, device=a.device, vocoder=voc)
        rid = b.manifest["run_id"]
        rates_needed.add(b.sample_rate)
        cell = matrix_cell(b.manifest)
        runs.append({
            "run_id": rid,
            "bundle": bundle_dir.name,
            "architecture": b.manifest["architecture"],
            "input_repr": b.manifest["input_repr"],
            "data": cell["data"],
            "seed": cell["seed"],
            "describes": b.describes,
            "step": b.manifest.get("step"),
            "config_hash": b.manifest.get("config_hash", ""),
            "git_commit": b.manifest.get("git_commit", ""),
            "sample_rate": b.sample_rate,
            "needs_vocoder": b.needs_vocoder,
        })
        out_dir = audio_root / rid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{rid}: {b.describes}")

        for s in demo["chosen"]:
            try:
                sp = b.synthesize(s["text"])
            except Exception as exc:                            # noqa: BLE001
                failures.append({"run_id": rid, "id": s["id"],
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED {s['id']}: {type(exc).__name__}: {exc}")
                continue
            entry = {"tokens": sp.tokens,
                     "seconds_elapsed": round(sp.seconds_elapsed, 3)}
            if sp.waveform is not None:
                rel = f"audio/{rid}/{s['id']}.wav"
                write_wav(a.out / rel, sp.waveform, sp.sample_rate)
                entry["audio"] = rel
                entry["audio_seconds"] = round(sp.audio_seconds, 3)
                # A vocoded mel is NOT end to end, and labelling it so would
                # hide which vocoder produced the audio in a page whose whole
                # job is provenance.
                vinfo = sp.extras.get("vocoder")
                if vinfo:
                    entry["vocoder"] = (f"hifigan:{vinfo['run_id']}"
                                        f"@{vinfo.get('step')}")
                    entry["vocoder_detail"] = vinfo
                    entry["mel_frames"] = int(sp.mel.shape[1])
                else:
                    entry["vocoder"] = "end-to-end"
            elif a.griffin_lim:
                # A labelled placeholder, recorded as one. The page shows the
                # label; an unlabelled vocoder-free rendering presented as the
                # system's output is a claim this project cannot make.
                wav = griffinlim.invert(sp.mel, sp.sample_rate,
                                        n_iter=a.gl_iters)
                rel = f"audio/{rid}/{s['id']}.wav"
                write_wav(a.out / rel, wav, sp.sample_rate)
                entry["audio"] = rel
                entry["audio_seconds"] = round(len(wav) / sp.sample_rate, 3)
                entry["mel_frames"] = int(sp.mel.shape[1])
                entry["vocoder"] = "griffin-lim"
                entry["gl_iters"] = int(a.gl_iters)
            else:
                entry["audio"] = None
                entry["mel_frames"] = int(sp.mel.shape[1])
                entry["vocoder"] = None
            rendered.setdefault(s["id"], {})[rid] = entry
            print(f"  {s['id']}  {len(sp.tokens):>3} tokens  "
                  f"{entry.get('audio') or 'mel only, silent'}"
                  f"{'  [griffin-lim placeholder]' if entry.get('vocoder') == 'griffin-lim' else ''}")

    # The natural recording, at each rate an arm uses.
    refs: dict[str, dict[str, str]] = {}
    for s in demo["chosen"]:
        per_rate = {}
        for rate in sorted(rates_needed):
            col = reference_column(rate)
            src = INTERIM / a.lang / s.get(col, "")
            if not s.get(col) or not src.exists():
                print(f"  reference missing for {s['id']} at {rate} Hz: {src}")
                continue
            rel = f"audio/reference_{rate}/{s['id']}.wav"
            (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, a.out / rel)
            per_rate[str(rate)] = rel
        refs[s["id"]] = per_rate

    sentences = []
    for s in demo["chosen"]:
        sentences.append({
            "id": s["id"], "text": s["text"], "seconds": s["seconds"],
            "words": s["words"], "medial": s["medial"], "final": s["final"],
            "medial_words": s["medial_words"],
            "reference": refs.get(s["id"], {}),
            "runs": rendered.get(s["id"], {}),
        })

    data = {
        "language": a.lang,
        "split": demo.get("split", "dev"),
        "criterion": demo.get("criterion", ""),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds"),
        "vocoder_bundle": str(a.vocoder) if a.vocoder else None,
        "griffin_lim": bool(a.griffin_lim),
        "gl_iters": int(a.gl_iters) if a.griffin_lim else None,
        "runs": runs,
        "sentences": sentences,
        "failures": failures,
    }
    # allow_nan=False is the guard, not the cleanup: json_safe should have
    # removed every non-finite value already, and if one survives this raises
    # here rather than publishing a page that cannot load.
    (a.out / "data.json").write_text(
        json.dumps(json_safe(data), indent=1, ensure_ascii=False,
                   allow_nan=False),
        encoding="utf-8")

    total = sum(1 for s in sentences for r in s["runs"].values() if r.get("audio"))
    mb = sum(p.stat().st_size for p in a.out.rglob("*.wav")) / 1e6
    print(f"\n{len(runs)} run(s), {len(sentences)} sentences, "
          f"{total} clips, {mb:.1f} MB of audio")
    if failures:
        print(f"{len(failures)} failure(s) recorded in data.json")
    print(f"wrote {a.out / 'data.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
