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
from src.export.synthesize import load, write_wav                  # noqa: E402

INTERIM = HERE / "data" / "interim"
TABLES = HERE / "results" / "tables"
DEFAULT_OUT = HERE / "space_static"


def reference_column(sample_rate: int) -> str:
    return "wav22" if sample_rate == 22050 else "wav16"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--set", type=pathlib.Path,
                    help="demo set json; defaults to the one for --lang")
    ap.add_argument("--bundles", nargs="*", type=pathlib.Path,
                    help="bundle directories; defaults to every one in exports/")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
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
        b = load(bundle_dir, device=a.device)
        if b.manifest["language"] != a.lang:
            print(f"  skipping {bundle_dir.name}: trained on "
                  f"{b.manifest['language']}, not {a.lang}")
            continue
        rid = b.manifest["run_id"]
        rates_needed.add(b.sample_rate)
        runs.append({
            "run_id": rid,
            "bundle": bundle_dir.name,
            "architecture": b.manifest["architecture"],
            "input_repr": b.manifest["input_repr"],
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
        "griffin_lim": bool(a.griffin_lim),
        "gl_iters": int(a.gl_iters) if a.griffin_lim else None,
        "runs": runs,
        "sentences": sentences,
        "failures": failures,
    }
    (a.out / "data.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

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
