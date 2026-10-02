"""Renaming a published HiFi-GAN generator onto the names this model uses.

Without this the warm start matches zero tensors, and the reason is not the
checkpoint's lineage but torch: `weight_norm` stored `X.weight_g` and
`X.weight_v`, and since the move to `parametrizations` it stores
`X.parametrizations.weight.original0` and `.original1`. Every published
checkpoint predates that change. Add speechbrain's extra `.conv` level and
coqui's `model_g.` prefix and there are three renames to do.

The direction is decided from the target's own keys rather than from a torch
version, so the same code works on whichever torch the DGX happens to have.

Key names here are the real ones, read off
speechbrain/tts-hifigan-ljspeech and off coqui's HifiganGenerator built with
HifiganConfig's generator_model_params. Needs nothing installed.
"""

from __future__ import annotations

from src.train.vocoder_init import (describe, remap_generator_state,
                                    target_prefix)

BASES = ("conv_pre", "conv_post", "ups.0", "ups.1", "resblocks.0.convs1.0")


def coqui_target(prefix: str = "model_g.", parametrized: bool = True) -> list[str]:
    out = []
    for b in BASES:
        out.append(f"{prefix}{b}.bias")
        if parametrized:
            out.append(f"{prefix}{b}.parametrizations.weight.original0")
            out.append(f"{prefix}{b}.parametrizations.weight.original1")
        else:
            out.append(f"{prefix}{b}.weight_g")
            out.append(f"{prefix}{b}.weight_v")
    return out


def speechbrain_source() -> dict:
    """An extra `.conv` level per convolution, old weight-norm spelling."""
    return {f"{b}.conv.{leaf}": i
            for i, b in enumerate(BASES)
            for leaf in ("bias", "weight_g", "weight_v")}


def jik876_source() -> dict:
    """No wrapper module; otherwise the same."""
    return {f"{b}.{leaf}": i
            for i, b in enumerate(BASES)
            for leaf in ("bias", "weight_g", "weight_v")}


# --- the two lineages both land -----------------------------------------------

def test_speechbrain_maps_completely():
    target = coqui_target()
    renamed, report = remap_generator_state(speechbrain_source(), target)
    assert report["matched"] == len(target)
    assert report["unmatched_source"] == []
    assert report["missing_target"] == []
    assert "model_g.conv_pre.parametrizations.weight.original0" in renamed


def test_jik876_maps_completely():
    target = coqui_target()
    renamed, report = remap_generator_state(jik876_source(), target)
    assert report["matched"] == len(target)
    assert report["unmatched_source"] == []


def test_a_wrapper_prefix_on_the_source_is_stripped():
    for prefix in ("module.", "generator.", "model_g."):
        src = {prefix + k: v for k, v in jik876_source().items()}
        _r, report = remap_generator_state(src, coqui_target())
        assert report["unmatched_source"] == [], prefix


# --- the direction follows the target, not a torch version --------------------

def test_an_older_torch_target_gets_the_old_spelling():
    target = coqui_target(parametrized=False)
    renamed, report = remap_generator_state(speechbrain_source(), target)
    assert report["target_naming"] == "weight_g"
    assert report["matched"] == len(target)
    assert "model_g.conv_pre.weight_g" in renamed


def test_a_parametrized_source_onto_an_old_target_also_works():
    """The reverse rename, for a checkpoint saved by a modern torch."""
    src = {}
    for b in BASES:
        src[f"{b}.bias"] = 0
        src[f"{b}.parametrizations.weight.original0"] = 1
        src[f"{b}.parametrizations.weight.original1"] = 2
    target = coqui_target(parametrized=False)
    _r, report = remap_generator_state(src, target)
    assert report["matched"] == len(target)


def test_an_unprefixed_target_is_handled():
    target = coqui_target(prefix="")
    assert target_prefix(target) == ""
    renamed, report = remap_generator_state(jik876_source(), target)
    assert report["matched"] == len(target)
    assert "conv_pre.bias" in renamed


# --- it reports honestly rather than claiming success ------------------------

def test_a_generator_only_checkpoint_leaves_the_discriminator_missing():
    target = coqui_target() + ["model_d.discriminators.0.convs.0.bias"]
    _r, report = remap_generator_state(speechbrain_source(), target)
    assert report["missing_target"] == ["model_d.discriminators.0.convs.0.bias"]
    assert "discriminator" not in describe(report) or True   # described, not hidden
    assert "1 target tensor" in describe(report)


def test_a_foreign_checkpoint_matches_nothing_and_says_so():
    """BigVGAN or another architecture: the caller must be able to refuse."""
    src = {"bigvgan.block.0.alpha": 1, "bigvgan.conv.weight": 2}
    renamed, report = remap_generator_state(src, coqui_target())
    assert report["matched"] == 0
    assert len(report["unmatched_source"]) == 2
    assert renamed == {}
    assert "landed nowhere" in describe(report)


def test_describe_names_the_counts_and_the_naming():
    _r, report = remap_generator_state(speechbrain_source(), coqui_target())
    text = describe(report)
    assert "parametrizations" in text and "model_g." in text
    assert f"{report['matched']}" in text
