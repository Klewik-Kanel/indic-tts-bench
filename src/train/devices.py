#!/usr/bin/env python3
"""Move the things a run holds that the model does not own.

The loop moves the model to the device and stops there, which is correct for
weights and wrong for everything else a run keeps. coqui's `get_criterion()`
returns loss modules, and its spectral losses hold a window tensor inside them:
`TorchSTFT` builds `hann_window(win_length)` at construction, on the CPU,
because it has no idea where the run will live. Nothing moved it, so the first
spectral loss computed on the GPU raised

    stft input and window must be on the same device
    but got self on cuda:0 and window on cpu

VITS never hit this because its losses do not take an STFT of the waveform;
HiFi-GAN's L1 spectrogram term does, and it is on by default.

`module.to(device)` moves parameters and registered buffers. It does NOT move a
plain tensor kept as an attribute, and different versions of coqui store that
window differently: current source makes it an `nn.Parameter`, older code a bare
tensor. So this walks attributes as well, which costs nothing and removes the
need to know which version is installed.

No project imports, so it is testable on its own.
"""

from __future__ import annotations


def move_module(obj, device, _seen=None, _depth=0):
    """Move a module and any stray tensor attributes it keeps. Returns names moved.

    Recurses through attributes and through child modules, because the window
    lives two levels down: GeneratorLoss holds L1SpecLoss holds TorchSTFT holds
    the window.
    """
    import torch
    import torch.nn as nn

    if obj is None or _depth > 6:
        return []
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return []
    _seen.add(id(obj))

    moved: list[str] = []
    if isinstance(obj, (list, tuple)):
        for item in obj:
            moved += move_module(item, device, _seen, _depth + 1)
        return moved

    if isinstance(obj, nn.Module):
        obj.to(device)
        moved.append(type(obj).__name__)
        for child in obj.children():
            moved += move_module(child, device, _seen, _depth + 1)
        # Attributes that are tensors but neither parameters nor buffers: `.to`
        # leaves these exactly where they were, which is the whole defect.
        own = dict(obj.named_parameters(recurse=False))
        own_buffers = dict(obj.named_buffers(recurse=False))
        for name, value in list(vars(obj).items()):
            if name.startswith("_") or name in own or name in own_buffers:
                continue
            if isinstance(value, torch.Tensor) and value.device != torch.device(device):
                setattr(obj, name, value.to(device))
                moved.append(f"{type(obj).__name__}.{name}")
            elif isinstance(value, (list, tuple)):
                moved += move_module(value, device, _seen, _depth + 1)
    return moved


def move_criterion(adapter, device) -> list[str]:
    """Move whatever the adapter stashed in `_criterion`, in place."""
    crit = getattr(adapter, "_criterion", None)
    if crit is None:
        return []
    moved = move_module(crit, device)
    return moved
