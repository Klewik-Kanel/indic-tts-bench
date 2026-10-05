#!/usr/bin/env python3
"""Check the rendered demo, then publish it as a Hugging Face Static Space.

    python scripts/publish_space.py --dry-run
    python scripts/publish_space.py --repo-id Klewik/Indic-tts-demo

**Why the check is in here rather than in the command you type.** A Static
Space serves files and runs nothing, so whatever is uploaded is immediately
the public artefact. Three ways this page can look finished and be wrong, all
of them seen in this project:

  - data.json missing. The page fetches it at load, so the page renders an
    empty listening test. It looks like a working demo of nothing.
  - a clip named in data.json with no file on disk. The player shows controls
    that play silence, and a listener reads that as the model being silent.
  - a FastSpeech 2 arm with vocoder null. That arm emits a mel spectrogram and
    is genuinely silent without a vocoder, which is a rendering mistake rather
    than a result, and on 5 October it would have been published as one.

So this refuses rather than warns, and --dry-run performs every check and
uploads nothing.

**space_sdk must be "static".** The Space was first created with the Gradio
SDK, and `hf upload`'s implicit create then refused with 402 Payment Required
because Hugging Face restricted free cpu-basic in July 2026. A Static Space
needs no runtime and stays free, which is why the demo is pre-rendered audio
rather than live inference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

DEFAULT_DIR = HERE / "space_static"
DEFAULT_REPO = "Klewik/Indic-tts-demo"
# Architectures that emit a mel and are silent without a vocoder. A clip from
# one of these with no vocoder recorded is a rendering fault, not a finding.
MEL_ONLY = frozenset({"fastspeech2", "matcha"})

# A finished render writes every wav, then data.json. The gap between the two
# is seconds; this allows for a slow filesystem without allowing a day.
STALE_SLACK_S = 120.0


def audit(root: pathlib.Path) -> tuple[list[str], dict]:
    """(problems, summary). An empty problems list means it is safe to publish."""
    bad: list[str] = []
    page, data_path = root / "index.html", root / "data.json"
    if not page.exists():
        bad.append(f"{page} is missing")
    if not data_path.exists():
        bad.append(f"{data_path} is missing: the page fetches it at load and "
                   "renders an empty listening test without it")
        return bad, {}

    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception as exc:                                  # noqa: BLE001
        bad.append(f"{data_path} is not readable JSON: {exc}")
        return bad, {}

    arch = {r.get("run_id"): r.get("architecture") for r in data.get("runs", [])}
    sentences = data.get("sentences") or []
    if not sentences:
        bad.append("data.json lists no sentences")

    clips = 0
    missing: list[str] = []
    silent_mel: list[str] = []
    vocoders: dict[str, set] = {}
    for s in sentences:
        for rid, got in (s.get("runs") or {}).items():
            if not got:
                continue
            voc = got.get("vocoder")
            vocoders.setdefault(rid, set()).add(voc)
            rel = got.get("audio")
            if rel:
                clips += 1
                if not (root / rel).exists():
                    missing.append(rel)
            elif arch.get(rid) in MEL_ONLY:
                silent_mel.append(f"{rid}/{s.get('id')}")

    if missing:
        bad.append(f"{len(missing)} clips named in data.json have no file: "
                   + ", ".join(missing[:5])
                   + ("..." if len(missing) > 5 else ""))
    if silent_mel:
        bad.append(
            f"{len(silent_mel)} mel-only clips have no audio, so those arms "
            "would publish as silent: "
            + ", ".join(silent_mel[:5])
            + ("..." if len(silent_mel) > 5 else "")
            + ". Re-render with --vocoder.")
    for rid, vs in sorted(vocoders.items()):
        if arch.get(rid) in MEL_ONLY and (None in vs or "end-to-end" in vs):
            bad.append(f"{rid} is {arch.get(rid)} but its clips are labelled "
                       f"{sorted(str(v) for v in vs)}: a mel-only arm cannot "
                       "be end to end, so the label is wrong")
    if not clips:
        bad.append("no audio at all")

    # render_demo writes data.json last, after every clip. So a data.json
    # older than the newest wav means that render did not finish, and what is
    # about to be published is a mixture: the listing from one run and the
    # audio from another. The failure that produced this check wrote fresh
    # VITS clips, died on a vocoder bundle before writing the listing, and
    # left a page still labelling its FastSpeech 2 arms griffin-lim from the
    # day before. Every other check passed.
    wavs = sorted(root.rglob("*.wav"), key=lambda q: q.stat().st_mtime)
    if wavs:
        newest = wavs[-1]
        if newest.stat().st_mtime > data_path.stat().st_mtime + STALE_SLACK_S:
            lag = (newest.stat().st_mtime - data_path.stat().st_mtime) / 60.0
            bad.append(
                f"{data_path.name} is {lag:.0f} min older than "
                f"{newest.relative_to(root)}, so the last render did not "
                "finish writing it. The listing and the audio come from "
                "different runs. Re-render.")

    summary = {
        "language": data.get("language"),
        "split": data.get("split"),
        "sentences": len(sentences),
        "clips": clips,
        "vocoder_bundle": data.get("vocoder_bundle"),
        "generated_utc": data.get("generated_utc"),
        "arms": {rid: {"architecture": arch.get(rid),
                       "vocoder": sorted(str(v) for v in vs)}
                 for rid, vs in sorted(vocoders.items())},
        "failures": len(data.get("failures") or []),
    }
    return bad, summary


def describe(summary: dict) -> str:
    if not summary:
        return "(nothing to describe)"
    out = [f"{summary['language']} · {summary['split']} split · "
           f"{summary['sentences']} sentences · {summary['clips']} clips",
           f"vocoder bundle: {summary['vocoder_bundle']}",
           f"rendered: {summary['generated_utc']}",
           f"{'run':6s} {'architecture':13s} vocoder"]
    for rid, d in summary["arms"].items():
        out.append(f"{rid:6s} {str(d['architecture']):13s} "
                   + ", ".join(d["vocoder"]))
    if summary["failures"]:
        out.append(f"{summary['failures']} render failures recorded in data.json")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=pathlib.Path, default=DEFAULT_DIR)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check and upload nothing")
    ap.add_argument("--private", action="store_true",
                    help="create the Space private; the default is public "
                         "because a demo nobody can open is not a demo")
    a = ap.parse_args(argv)

    bad, summary = audit(a.dir)
    print(describe(summary))
    print()
    if bad:
        print("NOT PUBLISHING. Fix these first:", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        return 1
    print("checks pass")
    if a.dry_run:
        print("--dry-run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="space", space_sdk="static",
                    exist_ok=True, private=a.private)
    api.upload_folder(folder_path=str(a.dir), repo_id=a.repo_id,
                      repo_type="space")
    print(f"https://huggingface.co/spaces/{a.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
