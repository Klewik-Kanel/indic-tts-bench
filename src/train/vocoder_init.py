#!/usr/bin/env python3
"""Rename a published HiFi-GAN generator's tensors onto the names this model uses.

A warm start that matches nothing is a cold start wearing a checkpoint's name,
and that is exactly what a straight `load_state_dict` gives here. Three
independent naming differences stand between a downloaded generator and coqui's
`HifiganGenerator`, and none of them is visible without comparing key lists:

**The weight-norm API changed in torch.** `weight_norm` used to store a tensor
as `X.weight_g` and `X.weight_v`. Since the move to `parametrizations` it stores
`X.parametrizations.weight.original0` and `.original1`. Every published
checkpoint predates that; a modern torch builds the new names. The structure and
the shapes are identical, only the keys differ, so this is a rename and nothing
more. Which direction to go is decided by looking at the TARGET's own keys, not
by checking a torch version, so this keeps working on either.

**Some lineages wrap each convolution in a module of its own.** speechbrain
writes `conv_pre.conv.weight_g` where coqui writes `conv_pre.weight_g`. One
segment, dropped.

**The generator may sit under a prefix on one side or the other.** A checkpoint
may carry `module.`, `generator.` or `model_g.`; coqui's `GAN` holds the
generator as `model_g`, so the target keys are prefixed and the source's are not.

Verified against speechbrain/tts-hifigan-ljspeech and coqui's HifiganGenerator
built with HifiganConfig's own generator_model_params: 234 of 234 tensors map,
no shape disagreements, `load_state_dict(strict=True)` accepts the result, and
the loaded generator turns 32 mel frames into 8,192 samples.

This module knows nothing about torch. It takes a mapping of names to tensors
and a list of target names, and returns a renamed mapping, so the rules are
tested without a GPU or an upstream package.
"""

from __future__ import annotations

import re

# Prefixes a checkpoint may carry around its generator.
SOURCE_PREFIXES = ("module.", "generator.", "model_g.", "model.generator.")

# The leaf names weight_norm and a plain Conv1d produce.
LEAF = r"(bias|weight|weight_g|weight_v)"

NEW_G = ".parametrizations.weight.original0"
NEW_V = ".parametrizations.weight.original1"


def strip_source_prefix(key: str) -> str:
    for p in SOURCE_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def drop_wrapper_module(key: str) -> str:
    """speechbrain's extra `.conv` level: conv_pre.conv.weight_g -> conv_pre.weight_g."""
    return re.sub(rf"\.conv\.{LEAF}$", r".\1", key)


def to_target_naming(key: str, parametrized: bool) -> str:
    """Convert between the two weight-norm spellings, whichever the target uses."""
    if parametrized:
        key = re.sub(r"\.weight_g$", NEW_G, key)
        key = re.sub(r"\.weight_v$", NEW_V, key)
    else:
        key = key.replace(NEW_G, ".weight_g").replace(NEW_V, ".weight_v")
    return key


def target_prefix(target_keys) -> str:
    """The prefix the target hangs its generator under, usually `model_g.`."""
    for k in target_keys:
        if k.startswith("model_g."):
            return "model_g."
    return ""


def remap_generator_state(source: dict, target_keys) -> tuple[dict, dict]:
    """Rename `source` onto `target_keys`.

    Returns (renamed, report). The report carries `matched`, `unmatched_source`
    (names that landed nowhere, with what they were tried as) and
    `missing_target` (target names nothing supplied), because a caller has to be
    able to say how complete the warm start was rather than assume it worked.
    """
    tset = set(target_keys)
    parametrized = any(NEW_G in k for k in tset)
    prefix = target_prefix(tset)

    renamed: dict = {}
    unmatched: list[tuple[str, str]] = []
    for key, value in source.items():
        candidate = to_target_naming(
            drop_wrapper_module(strip_source_prefix(key)), parametrized)
        for attempt in (prefix + candidate, candidate):
            if attempt in tset:
                renamed[attempt] = value
                break
        else:
            unmatched.append((key, prefix + candidate))

    report = {
        "matched": len(renamed),
        "target_total": len(tset),
        "unmatched_source": unmatched,
        "missing_target": sorted(tset - set(renamed)),
        "target_naming": "parametrizations" if parametrized else "weight_g",
        "target_prefix": prefix,
    }
    return renamed, report


def describe(report: dict, limit: int = 4) -> str:
    lines = [f"  mapped {report['matched']} of {report['target_total']} target "
             f"tensors (target naming: {report['target_naming']}, "
             f"prefix: {report['target_prefix'] or 'none'})"]
    if report["unmatched_source"]:
        lines.append(f"  {len(report['unmatched_source'])} source tensor(s) "
                     "landed nowhere:")
        for src, tried in report["unmatched_source"][:limit]:
            lines.append(f"    {src}  ->  {tried}")
    if report["missing_target"]:
        lines.append(f"  {len(report['missing_target'])} target tensor(s) had no "
                     "source, left at their initial values:")
        for k in report["missing_target"][:limit]:
            lines.append(f"    {k}")
    return "\n".join(lines)
