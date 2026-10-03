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
import re
import pathlib
import time
from dataclasses import dataclass
from typing import Protocol

from . import batching, schedule
from .progress import Progress
from .checkpoint import CheckpointDir, RetentionPolicy
from .devices import move_criterion
from .earlystop import Plateau
from .text import TextEncoder, Vocab

HERE = pathlib.Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"


class ModelAdapter(Protocol):
    """What an architecture must supply. Deliberately small.

    `n_optimizers`, `optimizers`, `param_groups` and `prepare` have defaults in
    `adapters.AdapterBase` and only an adversarial architecture overrides them.
    `loss` takes the index of the optimiser being stepped, which is 0 for every
    single-optimiser architecture and meaningful only for a GAN.
    """

    name: str
    n_optimizers: int

    def build(self, vocab_size: int, cfg: dict): ...
    def collate(self, batch: list[batching.Utterance], enc: TextEncoder, cfg: dict) -> dict: ...
    def optimizers(self, model, cfg: dict) -> list: ...
    def param_groups(self, model) -> list: ...
    def prepare(self, model, tensors: dict) -> dict: ...
    def loss(self, model, tensors: dict, optimizer_idx: int = 0) -> object: ...


@dataclass
class LoopResult:
    run_id: str
    steps_done: int
    final_loss: float
    seconds: float
    resumed_from: int


def _prefetch(items, depth: int = 2):
    """Yield from `items`, filling the next ones on a background thread.

    Collating a batch means reading one cache entry per utterance off disk and
    padding it, which is CPU and I/O work that the GPU cannot help with. Done
    inline it serialises against the step: the card finishes, then waits while
    the loop prepares the next batch. Measured on the A100, that showed up as
    66% utilisation and a step rate about a third under the benchmark.

    This changes when a batch is prepared, never which batch: the sequence is
    consumed in order, one thread produces, the loop consumes, and the data
    order still comes from (seed, step). An exception in the producer is
    re-raised in the consumer rather than hanging the loop, and the thread is a
    daemon so a killed run does not wait on it.

    TRAIN_PREFETCH=0 disables it, which is the first thing to try if a step ever
    looks non-deterministic.
    """
    import queue
    import threading

    if os.environ.get("TRAIN_PREFETCH") == "0":
        yield from items
        return

    q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
    DONE = object()

    def produce():
        try:
            for item in items:
                q.put(item)
        except BaseException as exc:                  # noqa: BLE001
            q.put(exc)
        else:
            q.put(DONE)

    threading.Thread(target=produce, daemon=True, name="collate").start()
    while True:
        item = q.get()
        if item is DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


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
          manifest: pathlib.Path | None = None,
          eval_every: int = 0, patience: int = 3,
          min_delta: float = 0.0) -> LoopResult:
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

    if bool(getattr(adapter, "needs_text", True)):
        enc = TextEncoder.for_config(cfg["language"], cfg["input_repr"],
                                     merge_nukta=bool(cfg.get("merge_nukta", False)))
        enc.vocab.save(out_dir / "vocab.json")
        vocab_size = len(enc.vocab)
    else:
        # The vocoder takes mel and waveform and no text. input_repr is "none"
        # for it, and TextEncoder.for_config raises on "none" because there is
        # genuinely no vocabulary to build. So no encoder, and no vocab.json:
        # a run with no vocabulary must not write a file claiming to have one.
        enc = None
        vocab_size = 0
        print(f"{run_id}: no text front end, architecture "
              f"{cfg['architecture']}", flush=True)

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
    # Threads. torch sizes its intra-op pool at one thread per core, per
    # process, and does not know another run is doing the same thing. On a
    # 256-core host two runs ask for 512 threads, the host load goes above the
    # core count, and the work that suffers is exactly the work the GPU is
    # waiting on: collating a batch, which is CPU and disk. Measured on this
    # box with no other tenant on the card: load average 309 on 256 cores and
    # GPU utilisation wandering between 75% and 95%.
    #
    # TRAIN_THREADS caps it. Like TRAIN_GPU_FRACTION this is a scheduling knob
    # and not part of the budget: it changes how this run shares a host, never
    # what it computes. The default leaves torch alone, so nothing changes for
    # a run that does not set it.
    threads = os.environ.get("TRAIN_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
        # Interop is the pool that runs separate graph branches; one is enough
        # when the per-op pool is already capped.
        try:
            torch.set_num_interop_threads(max(1, int(threads) // 4))
        except RuntimeError:
            pass          # already initialised, which is not worth failing over
        print(f"{run_id}: capped at {int(threads)} intra-op threads "
              f"(host has {os.cpu_count()} cores)", flush=True)

    frac = os.environ.get("TRAIN_GPU_FRACTION")
    if frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(frac))
        print(f"{run_id}: capped at {float(frac):.0%} of the GPU "
              f"({float(frac) * torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)",
              flush=True)

    model = adapter.build(vocab_size, cfg).to(dev)

    # The model is not the only thing with tensors in it. coqui's loss modules
    # hold an STFT window built on the CPU at construction, and nothing else
    # moves them, so the first spectral loss on the GPU fails with "stft input
    # and window must be on the same device". See src/train/devices.py.
    moved = move_criterion(adapter, dev)
    if moved:
        print(f"{run_id}: moved to {dev}: {', '.join(sorted(set(moved))[:6])}"
              f"{' and more' if len(set(moved)) > 6 else ''}", flush=True)

    # One optimiser for most architectures, two for an adversarial one. The
    # adapter decides, because the split is a property of the model: see
    # AdapterBase.optimizers. The loop below steps them in index order and hands
    # each one its own index, which is load-bearing for VITS.
    opts = adapter.optimizers(model, cfg)
    groups = adapter.param_groups(model)
    if len(groups) != len(opts):
        raise SystemExit(
            f"{run_id}: the adapter returned {len(opts)} optimisers and "
            f"{len(groups)} parameter groups; they must correspond")

    ck = CheckpointDir(out_dir / "checkpoints", RetentionPolicy())
    start = ck.resume_step()
    if start:
        blob = torch.load(ck.path_for(start) / "state.pt", map_location=dev,
                          weights_only=False)
        model.load_state_dict(blob["model"])
        # "optimizers" is a list, one state per optimiser. "optimizer" is the
        # single-optimiser key written before the two-optimiser path existed,
        # and r01 and r04 are checkpointed under it, so it is still read.
        if "optimizers" in blob:
            saved = blob["optimizers"]
            if len(saved) != len(opts):
                raise SystemExit(
                    f"{run_id}: checkpoint at step {start} holds "
                    f"{len(saved)} optimiser states and this run has "
                    f"{len(opts)}. Resuming would restart one optimiser's "
                    "moments from zero, which puts a transient in the loss "
                    "that cannot later be told from a real effect.")
            for o, st in zip(opts, saved):
                o.load_state_dict(st)
        elif "optimizer" in blob:
            if len(opts) != 1:
                raise SystemExit(
                    f"{run_id}: checkpoint at step {start} predates the "
                    "two-optimiser path and holds one state, but this run has "
                    f"{len(opts)} optimisers. Start this run fresh.")
            opts[0].load_state_dict(blob["optimizer"])

        # The RNG is state too, and leaving it out broke resume for any
        # architecture that draws during its training step. VITS trains its
        # decoder on a RANDOM waveform slice, so a resumed process took
        # different slices from an uninterrupted one and diverged. The effect
        # was easy to miss: during warmup the learning rate is around 1e-07, so
        # four steps moved the weights by 6e-07 in total and the divergence
        # looked like float noise. It is not noise. It is a different run.
        #
        # Dropout draws as well, so this applies to FastSpeech 2 and Matcha too.
        # The earlier bit-exact result used the toy adapter, which has no
        # randomness in its forward pass, and therefore never tested this.
        if "rng" in blob:
            torch.set_rng_state(blob["rng"]["cpu"].cpu().to(torch.uint8))
            cuda_states = blob["rng"].get("cuda") or []
            if cuda_states and dev.type == "cuda":
                torch.cuda.set_rng_state_all(
                    [s.cpu().to(torch.uint8) for s in cuda_states])
        else:
            print(f"WARNING {run_id}: checkpoint at step {start} predates RNG "
                  "capture, so this resume is NOT bit-exact for an "
                  "architecture that draws during its step. The run is still "
                  "valid; it is simply not the same run the uninterrupted one "
                  "would have been. Start fresh if that matters.", flush=True)
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

    def save_checkpoint(at: int) -> None:
        ck.save(at, lambda dst: torch.save(
            {"model": model.state_dict(),
             # A list, always, even for one optimiser: a checkpoint whose shape
             # depends on the architecture is a second thing that can disagree
             # with the config.
             "optimizers": [o.state_dict() for o in opts],
             # Captured at save time, so a resume continues the same random
             # sequence rather than starting a new one.
             "rng": {"cpu": torch.get_rng_state(),
                     "cuda": (torch.cuda.get_rng_state_all()
                              if dev.type == "cuda" else [])}},
            dst / "state.pt"),
            {"run_id": run_id, "config_hash": cfg.get("config_hash", ""),
             "architecture": cfg["architecture"], "loss": loss_val})
        ck.prune()

    # The stopping criterion, off unless a caller asks for it and the adapter
    # offers a validation pass. Only the vocoder uses it: every acoustic run
    # trains for exactly max_steps because the fixed-budget claim depends on
    # it, and assert_budget_matched enforces that. The vocoder is already
    # excluded from that assertion, so it stops when it stops improving and the
    # step it reached is reported rather than chosen.
    stopper = (Plateau(patience=patience, min_delta=min_delta)
               if eval_every and hasattr(adapter, "validate") else None)

    prog = Progress(run_id, steps_total, start_step=start)
    t0 = time.time()
    loss_val = float("nan")
    step = start

    # An adapter that crops a random segment needs the step number, so the crop
    # is derived from (seed, step) like the batch itself rather than from a
    # global RNG. Collate runs on the prefetch thread, which is ahead of the
    # main loop, so a draw from the global RNG there would be captured in a
    # checkpoint at a different position than the step it belongs to and the
    # bit-exact resume guarantee would quietly stop holding. Declared by the
    # adapter rather than passed to all of them, so the other three keep the
    # signature their tests were written against.
    wants_step = bool(getattr(adapter, "wants_step", False))

    def _collated():
        for s, b in batching.batch_stream(
                utts, int(cfg["batch_frames"]), int(cfg["seed"]), steps_total, start):
            yield s, b, (adapter.collate(b, enc, cfg, step=s) if wants_step
                         else adapter.collate(b, enc, cfg))

    for step, batch, cpu_tensors in _prefetch(_collated()):
        n = step + 1                                   # schedule steps are 1-based
        lr = schedule.lr_at(n, float(cfg["lr"]), int(cfg["warmup_steps"]))
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr

        # The move to the device stays on this thread; only the collate above
        # runs on the producer, so no CUDA call is made off the main thread.
        tensors = {k: (v.to(dev) if hasattr(v, "to") else v)
                   for k, v in cpu_tensors.items()}
        # Once per step, before any optimiser runs. For the coqui models this is
        # format_batch_on_device, which derives mel from the spectrogram; doing
        # it per optimiser would redo that work and, worse, rebuild the batch
        # between two steps that have to see the same one.
        tensors = adapter.prepare(model, tensors)

        losses = []
        gnorms = []
        for idx, (o, params) in enumerate(zip(opts, groups)):
            o.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = adapter.loss(model, tensors, idx)
            scaler.scale(loss).backward()
            scaler.unscale_(o)
            # Clip this optimiser's own parameters. One global norm over both
            # halves of a GAN would let the discriminator's gradients scale the
            # generator's clip, so grad_clip would not mean the same thing in a
            # VITS run as in a FastSpeech 2 one.
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(
                params, float(cfg["grad_clip"]))))
            scaler.step(o)
            scaler.update()
            losses.append(float(loss.detach()))

        # The headline loss is the last optimiser's: the generator's for VITS,
        # the only one for everything else. Both are logged, because one scalar
        # cannot show which half of a GAN is diverging.
        loss_val = losses[-1]
        gnorm = gnorms[-1]

        prog.update(n, loss_val, lr,
                    gpu_mb=(torch.cuda.max_memory_allocated() / 1e6
                            if dev.type == "cuda" else None))

        if n % log_every == 0 or n == 1:
            rec = {"step": n, "loss": round(loss_val, 5), "lr": lr,
                   "grad_norm": round(float(gnorm), 4),
                   "batch": len(batch),
                   **({"losses": [round(x, 5) for x in losses],
                       "grad_norms": [round(x, 4) for x in gnorms]}
                      if len(opts) > 1 else {}),
                   # The terms the headline scalar is made of. For FastSpeech 2
                   # the mel L1 is under one per cent of the sum and the rest is
                   # mean squared error on f0 in hertz, so a curve of `loss`
                   # alone is a pitch-error curve wearing a quality label.
                   **({"components": dict(adapter.last_components)}
                      if getattr(adapter, "last_components", None) else {}),
                   "frames": len(batch) * max(u.frames for u in batch),
                   "elapsed_s": round(time.time() - t0, 1)}
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        if n % ckpt_every == 0 or n == steps_total:
            save_checkpoint(n)

        if stopper is not None and n % eval_every == 0:
            val = adapter.validate(model, cfg)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": n, "val_mel_l1": val}) + "\n")
            if val is not None:
                print(f"{run_id}: step {n}, held-out mel L1 {val:.6f}", flush=True)
            if stopper.update(n, val):
                print(f"{run_id}: stopping at step {n}: {stopper.why()}",
                      flush=True)
                if n % ckpt_every != 0:
                    save_checkpoint(n)
                # Written beside the checkpoints so the reported step count has
                # the sequence it was decided from next to it.
                (out_dir / "stopped.json").write_text(json.dumps(
                    {"step": n, "reason": stopper.why(), "best": stopper.best,
                     "best_step": stopper.best_step,
                     "history": stopper.history}, indent=1), encoding="utf-8")
                break

    prog.close(loss_val)
    return LoopResult(run_id=run_id, steps_done=step + 1 if steps_total else 0,
                      final_loss=loss_val, seconds=round(time.time() - t0, 1),
                      resumed_from=start)


# Fields that are always text, whatever they look like. config_hash is the one
# that bit: r08's hash is 52e245223208, which Python's float() reads as a valid
# literal in scientific notation and returns inf for. The run then recorded
# `config_hash: Infinity` in its config.json and in every checkpoint's
# meta.json, which is not even valid strict JSON, and the one thing that was
# supposed to tie the weights back to the config they came from was gone. An
# all-digit hash would have become an int the same way.
STRING_KEYS = frozenset({
    "config_hash", "run_id", "architecture", "language", "input_repr", "data",
    "init_from", "precision", "lr_schedule", "aligner", "vocoder", "notes",
})

# Strict, so a hex string is never mistaken for a number. A float needs a
# decimal point here: every float in these configs has one (lr: 0.0002), and
# requiring it is what keeps 52e245223208 a string.
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$")


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
        elif key in STRING_KEYS:
            cfg[key] = v
        elif INT_RE.match(v):
            cfg[key] = int(v)
        elif FLOAT_RE.match(v):
            cfg[key] = float(v)
        else:
            cfg[key] = v
    return cfg
