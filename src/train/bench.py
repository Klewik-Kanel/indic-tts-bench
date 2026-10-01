#!/usr/bin/env python3
"""Measure the step rate, then say what step budget fits in the time available.

Every schedule estimate in this project has so far been a guess carried over
from a plan written for a Kaggle T4. This replaces it with a measurement on the
hardware the runs will actually use.

What it reports, and why each part is needed:

**Steps per second, after warmup.** The first steps include CUDA context
creation, kernel autotuning and cache misses on the feature store. Including
them would understate the rate by a wide margin on a short benchmark and
overstate the time the real runs need. The first `--warmup` steps are therefore
timed and discarded.

**Peak GPU memory.** 12,000 mel frames was chosen on paper. If it does not fit
in 40 GB, the budget has to change before any run starts rather than after the
first one dies at step 300.

**The affordable step count.** Given a deadline and a number of runs, it divides
the remaining GPU time and prints what each run can afford. The answer is a
budget to put in the config, not a suggestion.

    python -m src.train.bench configs/r01.yaml --steps 60 --hours 60 --runs 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from . import adapters, batching
from .runner import load_config
from .text import TextEncoder


def bench(cfg: dict, adapter, steps: int, warmup: int, manifest: pathlib.Path | None):
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("WARNING: no GPU. A CPU rate says nothing about the A100 and must "
              "not be used to set a budget.")

    enc = TextEncoder.for_config(cfg["language"], cfg["input_repr"],
                                 merge_nukta=bool(cfg.get("merge_nukta", False)))
    mpath = manifest or (pathlib.Path("data/processed") / cfg["language"] /
                         ("train.tsv" if cfg["data"] == "train"
                          else f"ladder/{cfg['data']}.tsv"))
    utts = batching.load_manifest(mpath, int(cfg["sample_rate"]))

    model = adapter.build(len(enc.vocab), cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        str(cfg.get("precision", "")))
    use_amp = amp_dtype is not None and dev.type == "cuda"

    stream = batching.batch_stream(utts, int(cfg["batch_frames"]),
                                   int(cfg["seed"]), steps + warmup)
    t_start = None
    done = 0
    for i, (_, batch) in enumerate(stream):
        tensors = {k: (v.to(dev) if hasattr(v, "to") else v)
                   for k, v in adapter.collate(batch, enc, cfg).items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            loss = adapter.loss(model, tensors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()        # or we time the queue, not the work
        if i + 1 == warmup:
            t_start = time.time()           # start the clock after warmup
        elif t_start is not None:
            done += 1

    elapsed = time.time() - t_start if t_start else float("nan")
    rate = done / elapsed if elapsed and elapsed == elapsed else float("nan")
    peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else 0.0
    return {"architecture": cfg["architecture"], "run_id": cfg["run_id"],
            "params_m": round(n_params / 1e6, 2),
            "timed_steps": done, "elapsed_s": round(elapsed, 2),
            "steps_per_s": round(rate, 3),
            "peak_gpu_gib": round(peak, 2),
            "batch_frames": cfg["batch_frames"], "precision": cfg.get("precision")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=50, help="timed steps")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--hours", type=float, default=None,
                    help="GPU hours available, to print an affordable budget")
    ap.add_argument("--runs", type=int, default=1, help="runs sharing those hours")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    adapter = adapters.ADAPTERS[a.adapter or cfg["architecture"]]()
    r = bench(cfg, adapter, a.steps, a.warmup, a.manifest)
    print(json.dumps(r, indent=1))

    if a.hours and r["steps_per_s"] == r["steps_per_s"]:
        total = a.hours * 3600 * r["steps_per_s"]
        per_run = int(total / max(1, a.runs))
        print(f"\n  {a.hours} h * 3600 = {a.hours*3600:,.0f} GPU-seconds")
        print(f"  * {r['steps_per_s']} steps/s = {total:,.0f} steps")
        print(f"  / {a.runs} runs = {per_run:,} steps per run")
        print(f"  check: {per_run:,} / {r['steps_per_s']} / 3600 = "
              f"{per_run/r['steps_per_s']/3600:.2f} h per run, "
              f"{per_run/r['steps_per_s']/3600*a.runs:.1f} h total")
        if r["peak_gpu_gib"] > 36:
            print(f"\n  WARNING: peak {r['peak_gpu_gib']} GiB of 40. Reduce "
                  "batch_frames before launching, not after a run dies.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
