#!/usr/bin/env python3
"""Start, or restart, one training run.

    python -m src.train.launch configs/r01.yaml
    python -m src.train.launch configs/r01.yaml --max-steps 50000

Restarting is the same command. The run resumes from its last complete
checkpoint, and the data order is derived from (seed, step) rather than stored,
so a resumed run sees the batch an uninterrupted one would have seen. That was
verified bit-exact before any of this trained.

This box has no tmux, so a run goes under nohup and its progress is read from
the log. The reporter writes whole lines to a file and a rewriting line to a
terminal, so `tail -f` works during the run and `grep` works after it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from . import adapters
from .runner import RUNS, load_config, train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override the config's budget; for debugging only, "
                         "since the budget is the thing held constant")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--manifest", type=pathlib.Path, default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    adapters.assert_not_toy(cfg)
    adapter = adapters.for_config(cfg)
    out = a.out or RUNS / cfg["run_id"]

    print(f"{cfg['run_id']}: {cfg['architecture']} / {cfg['language']} / "
          f"{cfg['input_repr']} / {cfg['data']} / {cfg['precision']} / "
          f"hash {cfg.get('config_hash')}", flush=True)
    if a.max_steps:
        print(f"  BUDGET OVERRIDDEN to {a.max_steps:,} steps. This run is not "
              f"comparable with runs at {cfg['max_steps']:,}.", flush=True)

    r = train(cfg, adapter, out_dir=out, max_steps=a.max_steps,
              log_every=a.log_every, ckpt_every=a.ckpt_every,
              device=a.device, manifest=a.manifest)
    print(json.dumps(r.__dict__), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
