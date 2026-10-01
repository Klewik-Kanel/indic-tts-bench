#!/usr/bin/env python3
"""The training loop every architecture shares.

One loop, four architectures. The alternative, three upstream training scripts
with their own schedules, their own batching and their own checkpoint formats,
would make the fixed-budget claim unverifiable: the budget would be whatever
each repository happened to do. Here the budget is enforced in one place and
the architecture supplies only a model and a loss.

An architecture plugs in through `ModelAdapter`, which has three jobs and no
others: build a model given the vocabulary size, turn a list of utterances into
tensors, and return a scalar loss for a batch. Anything an architecture needs
beyond that is a declared deviation in its config, not a special case here.

torch is imported lazily so that this module, its vocabulary, its batching and
its checkpoint bookkeeping can be imported and tested in an environment with no
deep-learning stack, which is where most of this code was written.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Protocol

from . import batching, schedule
from .progress import Progress
from .checkpoint import CheckpointDir, RetentionPolicy
from .text import TextEncoder, Vocab

HERE = pathlib.Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"


class ModelAdapter(Protocol):
    """What an architecture must supply. Deliberately small."""

    name: str

    def build(self, vocab_size: int, cfg: dict): ...
    def collate(self, batch: list[batching.Utterance], enc: TextEncoder, cfg: dict) -> dict: ...
    def loss(self, model, tensors: dict) -> object: ...


@dataclass
class LoopResult:
    run_id: str
    steps_done: int
    final_loss: float
    seconds: float
    resumed_from: int


def _device(prefer: str = "auto"):
    import torch
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(cfg: dict, adapter: ModelAdapter, *, out_dir: pathlib.Path | None = None,
          max_steps: int | None = None, log_every: int = 50,
          ckpt_every: int = 5_000, device: str = "auto",
          manifest: pathlib.Path | None = None) -> LoopResult:
    """Run, or resume, one training run.

    `max_steps` overrides the config only for dry runs. A real run takes its
    step count from the config, because the step count is part of the budget
    the paper claims is identical.
    """
    import torch

    run_id = cfg["run_id"]
    out_dir = pathlib.Path(out_dir) if out_dir else RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_total = int(max_steps or cfg["max_steps"])

    # The run records what it was, beside its own checkpoints. An export months
    # from now reads this rather than trusting that configs/ still matches.
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=1, ensure_ascii=False, sort_keys=True),
        encoding="utf-8")

    enc = TextEncoder.for_config(cfg["language"], cfg["input_repr"],
                                 merge_nukta=bool(cfg.get("merge_nukta", False)))
    enc.vocab.save(out_dir / "vocab.json")

    mpath = manifest or (HERE / "data" / "processed" / cfg["language"] /
                         ("train.tsv" if cfg["data"] == "train"
                          else f"ladder/{cfg['data']}.tsv"))
    utts = batching.load_manifest(mpath, int(cfg["sample_rate"]))
    big = batching.oversized(utts, int(cfg["batch_frames"]))
    if big:
        print(f"{len(big)} utterances exceed the frame budget and will train "
              f"alone; longest {max(u.frames for u in big)} frames", flush=True)

    dev = _device(device)

    # Share the card. PyTorch's caching allocator keeps every block it has ever
    # used, so a run needing 4 GB sits on 37 of a 40 GB card and the next run
    # dies with an out-of-memory error that has nothing to do with its own size.
    # TRAIN_GPU_FRACTION caps what one process may reserve, so several runs fit.
    # It is a scheduling knob and not part of the budget: it changes what else
    # can run beside this one, never what this one computes.
    frac = os.environ.get("TRAIN_GPU_FRACTION")
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac))
        print(f"{run_id}: capped at {float(frac):.0%} of the GPU "
              f"({float(frac) * torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)",
              flush=True)

    model = adapter.build(len(enc.vocab), cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), betas=(0.9, 0.98))

    ck = CheckpointDir(out_dir / "checkpoints", RetentionPolicy())
    start = ck.resume_step()
    if start:
        blob = torch.load(ck.path_for(start) / "state.pt", map_location=dev,
                          weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        print(f"resumed {run_id} from step {start}", flush=True)

    # Precision. bf16 needs autocast but no gradient scaler: it has fp32's
    # exponent range, so there is nothing to scale away from. fp16 does need
    # one. Enabling a scaler under bf16 is not merely redundant, it reintroduces
    # the failure mode bf16 was chosen to remove, so the two are kept apart.
    prec = str(cfg.get("precision", "fp32"))
    on_cuda = dev.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    use_amp = amp_dtype is not None and on_cuda
    if prec == "bf16" and on_cuda and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            f"{run_id}: config asks for bf16 and this GPU does not support it. "
            "Change the budget deliberately rather than letting the run fall "
            "back to a precision the other runs did not use.")
    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16" and on_cuda))

    log_path = out_dir / "train_log.jsonl"
    prog = Progress(run_id, steps_total, start_step=start)
    t0 = time.time()
    loss_val = float("nan")
    step = start

    for step, batch in batching.batch_stream(
            utts, int(cfg["batch_frames"]), int(cfg["seed"]), steps_total, start):
        n = step + 1                                   # schedule steps are 1-based
        lr = schedule.lr_at(n, float(cfg["lr"]), int(cfg["warmup_steps"]))
        for g in opt.param_groups:
            g["lr"] = lr

        tensors = {k: (v.to(dev) if hasattr(v, "to") else v)
                   for k, v in adapter.collate(batch, enc, cfg).items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            loss = adapter.loss(model, tensors)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(cfg["grad_clip"]))
        scaler.step(opt)
        scaler.update()
        loss_val = float(loss.detach())

        prog.update(n, loss_val, lr,
                    gpu_mb=(torch.cuda.max_memory_allocated() / 1e6
                            if dev.type == "cuda" else None))

        if n % log_every == 0 or n == 1:
            rec = {"step": n, "loss": round(loss_val, 5), "lr": lr,
                   "grad_norm": round(float(gnorm), 4),
                   "batch": len(batch),
                   "frames": len(batch) * max(u.frames for u in batch),
                   "elapsed_s": round(time.time() - t0, 1)}
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        if n % ckpt_every == 0 or n == steps_total:
            ck.save(n, lambda dst: torch.save(
                {"model": model.state_dict(), "optimizer": opt.state_dict()},
                dst / "state.pt"),
                {"run_id": run_id, "config_hash": cfg.get("config_hash", ""),
                 "architecture": cfg["architecture"], "loss": loss_val})
            ck.prune()

    prog.close(loss_val)
    return LoopResult(run_id=run_id, steps_done=step + 1 if steps_total else 0,
                      final_loss=loss_val, seconds=round(time.time() - t0, 1),
                      resumed_from=start)


def load_config(path: pathlib.Path) -> dict:
    """Read a run YAML without a YAML dependency.

    The configs are generated by src/train/config.py and are flat scalars plus
    a list, so a 20-line reader is enough and the training environment needs no
    extra package. If the config schema ever grows nesting, this must be
    replaced by PyYAML rather than extended.
    """
    cfg: dict = {}
    key = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("  - "):
            if key is None:
                raise SystemExit(f"{path}: list item before any key")
            cfg.setdefault(key, []).append(raw[4:].strip())
            continue
        k, _, v = raw.partition(":")
        key = k.strip()
        v = v.strip()
        if v == "":
            cfg[key] = []
        elif v == "[]":
            cfg[key] = []
        elif v in ("true", "false"):
            cfg[key] = v == "true"
        elif v.startswith(("'", '"')):
            cfg[key] = v[1:-1]
        else:
            try:
                cfg[key] = int(v)
            except ValueError:
                try:
                    cfg[key] = float(v)
                except ValueError:
                    cfg[key] = v
    return cfg
