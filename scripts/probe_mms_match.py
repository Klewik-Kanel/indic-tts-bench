#!/usr/bin/env python3
"""Could a warm start from MMS transfer anything to our VITS, and what?

configs/r02.yaml has declared `init_from: facebook/mms-tts-hin` since the matrix
was written, and nothing ever loaded it (RESULTS, 3 Oct). Before that is either
implemented or dropped, this says what a warm start could actually move.

Two reasons it is not a flag. MMS ships as a transformers `VitsModel` while our
runs use coqui's `TTS.tts.models.vits.Vits`, so parameter names come from
different codebases and will not correspond; the HiFi-GAN warm start hit exactly
this and matched zero tensors until `src/train/vocoder_init.py` was written. And
the text embedding cannot transfer whatever the names are, because MMS carries
its own vocabulary while ours is 78 phoneme symbols or 136 grapheme ones, so the
rows mean different things.

What could transfer is the decoder, the flow and the posterior encoder, which
are text-independent and are the expensive part to train. This reports the
structure of both models rather than assuming it, so the mapping can be designed
from the output instead of from memory.

Nothing is loaded into a training run. CPU only.

    python scripts/probe_mms_match.py --repo facebook/mms-tts-hin
"""
from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def summarise(name: str, state: dict, show: int = 3) -> dict:
    by_prefix: dict[str, list] = collections.defaultdict(list)
    for k, v in state.items():
        by_prefix[k.split(".")[0]].append((k, tuple(v.shape)))
    total = sum(int(v.numel()) for v in state.values())
    print(f"\n=== {name}: {len(state)} tensors, {total/1e6:.1f}M parameters, "
          f"{len(by_prefix)} top-level groups")
    for pre in sorted(by_prefix, key=lambda p: -len(by_prefix[p])):
        rows = by_prefix[pre]
        n = sum(int(state[k].numel()) for k, _ in rows)
        print(f"  {pre:<22} {len(rows):>4} tensors  {n/1e6:>7.2f}M")
        for k, shp in rows[:show]:
            print(f"      {k}  {shp}")
        if len(rows) > show:
            print(f"      ... {len(rows)-show} more")
    return by_prefix


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="facebook/mms-tts-hin")
    ap.add_argument("--vocab", type=int, default=78,
                    help="our phoneme inventory size, for building coqui's Vits")
    ap.add_argument("--sample-rate", type=int, default=16000)
    a = ap.parse_args(argv)

    import torch
    from transformers import VitsModel

    from src.train import adapters

    mms = VitsModel.from_pretrained(a.repo)
    mms_state = {k: v for k, v in mms.state_dict().items()}

    cfg = {"run_id": "probe", "architecture": "vits", "language": "hindi",
           "input_repr": "phoneme", "sample_rate": a.sample_rate,
           "lr": 2e-4, "merge_nukta": False}
    ours = adapters.for_config(cfg).build(a.vocab, cfg)
    our_state = {k: v for k, v in ours.state_dict().items()}

    summarise(f"MMS  {a.repo}", mms_state)
    summarise("ours  coqui Vits", our_state)

    print("\n=== name overlap")
    shared = set(mms_state) & set(our_state)
    print(f"  identical names: {len(shared)} of {len(our_state)}")

    print("\n=== shape overlap, as an upper bound on what any renaming could move")
    mms_shapes = collections.Counter(tuple(v.shape) for v in mms_state.values())
    our_shapes = collections.Counter(tuple(v.shape) for v in our_state.values())
    movable = sum((mms_shapes & our_shapes).values())
    params = sum(int(torch.empty(s).numel()) * n
                 for s, n in (mms_shapes & our_shapes).items())
    print(f"  tensors whose shape exists on both sides: {movable} of "
          f"{len(our_state)} ({100*movable/len(our_state):.1f}%), "
          f"{params/1e6:.1f}M parameters")
    print("  NOTE: a shared shape is not a shared role. This is a ceiling, not "
          "a result; the mapping still has to be written per module.")

    print("\n=== the text side, which cannot transfer")
    for label, st in (("MMS", mms_state), ("ours", our_state)):
        for k, v in st.items():
            if "emb" in k.lower() and v.dim() == 2 and min(v.shape) > 8:
                print(f"  {label:<5} {k}  {tuple(v.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
