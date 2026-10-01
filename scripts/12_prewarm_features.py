#!/usr/bin/env python3
"""Fill the feature cache in parallel before, or alongside, a training run.

Extracting pitch with YIN is the slowest part of preparing a batch, and done
lazily it lands inside the training loop, where it makes the first epoch
measure librosa rather than the GPU. This box has 256 cores and one GPU, so the
work is embarrassingly parallel and essentially free to do up front.

    python scripts/12_prewarm_features.py --lang hindi --sr 22050 --pitch
    python scripts/12_prewarm_features.py --lang hindi --sr 16000
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.train import batching                      # noqa: E402
from src.train import features as F                 # noqa: E402

INTERIM = HERE / "data" / "interim"


def one(args):
    wav, sr, root, pitch = args
    try:
        F.load_or_compute(wav, sr, root, want_pitch=pitch)
        return True, ""
    except Exception as e:                                   # noqa: BLE001
        return False, f"{wav.name}: {type(e).__name__}: {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["hindi", "marathi"])
    ap.add_argument("--sr", type=int, required=True, choices=[22050, 16000])
    ap.add_argument("--pitch", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args(argv)

    man = HERE / "data" / "processed" / a.lang / f"{a.split}.tsv"
    utts = batching.load_manifest(man, a.sr)
    root = HERE / "data" / "cache" / a.lang
    jobs = [(INTERIM / a.lang / u.wav, a.sr, root, a.pitch) for u in utts]

    print(f"{a.lang} {a.sr} Hz, pitch={a.pitch}: {len(jobs):,} utterances on "
          f"{a.workers} workers", flush=True)
    t0 = time.time()
    done = failed = 0
    errors = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for ok, msg in ex.map(one, jobs, chunksize=8):
            done += 1
            if not ok:
                failed += 1
                if len(errors) < 10:
                    errors.append(msg)
            if done % 250 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{len(jobs):,}  {done/el:.1f}/s  "
                      f"eta {(len(jobs)-done)/max(done/el,1e-9)/60:.1f} min",
                      flush=True)
    el = time.time() - t0
    print(f"done {done:,} in {el/60:.1f} min, {failed} failed")
    for e in errors:
        print("  ", e)
    # A failure here is a file that will fail again inside training, at step N,
    # hours later. Report it as a non-zero exit rather than a warning.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
