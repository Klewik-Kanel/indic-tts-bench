#!/usr/bin/env python3
"""Prove the training loop and the resume path before spending GPU hours.

The question this answers is not "does the model learn" but "does the
machinery work": does the budget hold, does the schedule move, does a killed
run come back at the right step with the right optimiser state, and does the
data order after a resume match the data order of a run that was never killed.

Those are exactly the failures that cost a whole Kaggle session, and none of
them need a GPU to find.

    python -m src.train.dryrun configs/r01.yaml --steps 50
    python -m src.train.dryrun configs/r01.yaml --steps 50 --kill-at 25

`--kill-at` runs to N, stops as if the session died, then starts again from the
checkpoint and compares the result against an uninterrupted reference run. The
comparison is on the weights themselves, not on the loss curve: a resume that
is nearly right produces a nearly right curve, which is exactly the kind of
defect that survives a glance and corrupts a result. Identical weights to the
last bit, or the resume path is broken.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from . import adapters, batching
from .runner import load_config, train


def _batch_ids(cfg: dict, utts, steps: int, start: int = 0) -> list[list[str]]:
    return [[u.uid for u in b] for _, b in batching.batch_stream(
        utts, int(cfg["batch_frames"]), int(cfg["seed"]), steps, start)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--kill-at", type=int, default=0)
    ap.add_argument("--adapter", default="toy",
                    help="toy by default; a real adapter needs its upstream package")
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/dryrun"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep", action="store_true", help="do not delete the output dir first")
    ap.add_argument("--no-weight-check", action="store_true",
                    help="skip the bit-exact comparison against a reference run")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    adapter = adapters.ADAPTERS[args.adapter]()
    out = args.out / f"{cfg['run_id']}_{args.adapter}"
    if out.exists() and not args.keep:
        shutil.rmtree(out)

    mpath = args.manifest or None
    utts = batching.load_manifest(
        mpath or (pathlib.Path("data/processed") / cfg["language"] /
                  ("train.tsv" if cfg["data"] == "train" else f"ladder/{cfg['data']}.tsv")),
        int(cfg["sample_rate"]))

    print(f"config {args.config}  adapter {args.adapter}  "
          f"{len(utts)} utterances  budget {cfg['batch_frames']} frames")

    # The data-order check needs no model and no GPU, so it runs first and
    # fails fast.
    ref = _batch_ids(cfg, utts, args.steps)
    if args.kill_at:
        after = _batch_ids(cfg, utts, args.steps, args.kill_at)
        if after != ref[args.kill_at:]:
            print("FAIL: the batch sequence after a resume differs from the "
                  "uninterrupted sequence; the data order is not reproducible")
            return 1
        print(f"data order: resume at {args.kill_at} reproduces the reference "
              f"sequence for the remaining {len(after)} steps")

    if not args.kill_at:
        r = train(cfg, adapter, out_dir=out, max_steps=args.steps,
                  log_every=max(1, args.steps // 5), ckpt_every=max(1, args.steps),
                  device=args.device, manifest=mpath)
        print(json.dumps(r.__dict__))
        return 0

    if not args.no_weight_check:
        import torch
        torch.manual_seed(int(cfg["seed"]))
    first = train(cfg, adapter, out_dir=out, max_steps=args.kill_at,
                  log_every=max(1, args.kill_at // 2), ckpt_every=args.kill_at,
                  device=args.device, manifest=mpath)
    print("killed at:", json.dumps(first.__dict__))

    second = train(cfg, adapter, out_dir=out, max_steps=args.steps,
                   log_every=max(1, (args.steps - args.kill_at) // 2),
                   ckpt_every=args.steps, device=args.device, manifest=mpath)
    print("resumed:  ", json.dumps(second.__dict__))

    if second.resumed_from != args.kill_at:
        print(f"FAIL: resumed from step {second.resumed_from}, expected {args.kill_at}")
        return 1

    if not args.no_weight_check:
        import torch
        ref = args.out / f"{cfg['run_id']}_{args.adapter}_reference"
        shutil.rmtree(ref, ignore_errors=True)
        torch.manual_seed(int(cfg["seed"]))
        train(cfg, adapters.ADAPTERS[args.adapter](), out_dir=ref,
              max_steps=args.steps, log_every=10 ** 9, ckpt_every=args.steps,
              device=args.device, manifest=mpath)
        a = torch.load(ref / f"checkpoints/step_{args.steps}/state.pt",
                       weights_only=False)["model"]
        b = torch.load(out / f"checkpoints/step_{args.steps}/state.pt",
                       weights_only=False)["model"]
        worst = max(float((a[k] - b[k]).abs().max()) for k in a)
        if worst != 0.0:
            print(f"FAIL: resumed weights differ from the uninterrupted run by "
                  f"{worst:g}; the resume path does not reproduce the run")
            return 1
        print(f"weights: resumed run is bit-identical to an uninterrupted "
              f"{args.steps}-step run across {len(a)} tensors")

    print(f"OK: resumed from step {args.kill_at}, finished at {second.steps_done}, "
          f"loss {first.final_loss:.4f} -> {second.final_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
