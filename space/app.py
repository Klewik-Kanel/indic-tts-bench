#!/usr/bin/env python3
"""Demo for the Indic TTS benchmark: Hindi text in, speech out, arms side by side.

This Space exists to make one contrast audible. Devanagari orthography does not
mark schwa deletion, so the phonemic arm is handed the deletion by the front end
while the graphemic arm has to infer it from characters. Both arms are otherwise
the same architecture, the same seed, the same nine hours of speech and the same
step budget, so what differs between the two players below is the input
representation and nothing else.

It installs the project from git at a pinned commit and calls
`src.export.synthesize`, the same path the listening test uses, rather than
reimplementing anything. The front end here is byte-for-byte the front end that
trained the weights; that is the whole reason the Space is Python and not
JavaScript.

The token panel is not decoration. Seeing `n ə m ə k` against `न म क` is what
makes the comparison legible to someone who has not read the dissertation.
"""

from __future__ import annotations

import os
import pathlib
import traceback

import gradio as gr
import numpy as np
from src.export import hub
from src.export.synthesize import Bundle

WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", "Klewik/indic-tts-bench")
CACHE = pathlib.Path(os.environ.get("BUNDLE_CACHE", "/tmp/bundles"))

EXAMPLES = [
    "नमक कमल",
    "आज सुबह बहुत ठंड थी।",
    "वह हर रोज़ सुबह पाँच बजे उठता है।",
    "कमल के फूल पानी में खिलते हैं।",
]

_loaded: dict[str, Bundle] = {}


def load(name: str) -> Bundle:
    """Download the weights on first use and keep the model in memory.

    Lazy, so startup costs only the manifests and the Space is interactive
    before a 330 MB file has moved.
    """
    if name not in _loaded:
        root = hub.fetch_bundle(WEIGHTS_REPO, name, CACHE)
        _loaded[name] = Bundle(root, device="cpu")
    return _loaded[name]


BUNDLES = hub.discover(WEIGHTS_REPO, CACHE)


def label(name: str) -> str:
    m = BUNDLES[name]
    return (f"{m['run_id']} — {m['architecture']}, {m['language']}, "
            f"{m['input_repr']}" + ("  (mel only)" if m.get("needs_vocoder") else ""))


def mel_image(mel: np.ndarray) -> np.ndarray:
    """A mel as a viewable image, low frequencies at the bottom.

    No matplotlib: one less dependency in an image that already carries torch.
    """
    m = np.asarray(mel, dtype="float32")
    lo, hi = float(m.min()), float(m.max())
    norm = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    return (np.flipud(norm) * 255).astype("uint8")


def provenance(m: dict) -> str:
    return (f"**{m['run_id']}** · step {m.get('step', '?')} · "
            f"config `{m.get('config_hash', '?')}` · "
            f"commit `{(m.get('git_commit') or '?')[:10]}` · "
            f"{m['sample_rate']} Hz · trained {m.get('trained_precision', '?')}, "
            f"stored {m.get('stored_precision', '?')}")


def speak(text: str, name: str):
    """One arm. Returns audio, mel image, the token sequence, and provenance."""
    if not name:
        return None, None, "", "Pick a run."
    if not text or not text.strip():
        return None, None, "", "Type some Devanagari text."
    try:
        b = load(name)
        sp = b.synthesize(text)
    except Exception as exc:                                  # noqa: BLE001
        traceback.print_exc()
        return None, None, "", f"**Failed:** {type(exc).__name__}: {exc}"

    toks = " ".join(sp.tokens)
    repr_ = BUNDLES[name]["input_repr"]
    panel = (f"**{len(sp.tokens)} {repr_} tokens**\n\n`{toks}`")

    note = provenance(BUNDLES[name])
    if sp.waveform is not None:
        note += (f"\n\n{sp.audio_seconds:.2f} s of audio in "
                 f"{sp.seconds_elapsed:.2f} s ({sp.realtime_factor:.2f}× realtime "
                 "on a shared CPU)")
        return (sp.sample_rate, sp.waveform), None, panel, note

    note += ("\n\n**Silent.** This architecture produces a mel spectrogram and "
             "needs a vocoder, which is still training. The mel is shown "
             "instead; it is the model's real output, not a placeholder.")
    return None, mel_image(sp.mel), panel, note


def build() -> gr.Blocks:
    names = list(BUNDLES)
    e2e = [n for n in names if not BUNDLES[n].get("needs_vocoder")]
    default_a = e2e[0] if e2e else (names[0] if names else None)
    default_b = e2e[1] if len(e2e) > 1 else default_a

    with gr.Blocks(title="Indic TTS benchmark") as demo:
        gr.Markdown(
            "# Indic TTS benchmark\n"
            "Phonemic against graphemic input for Devanagari. Both arms share "
            "architecture, seed, corpus and step budget, so the only thing that "
            "differs is what the model is given to read.\n\n"
            "Devanagari does not write schwa deletion: *नमक* is pronounced "
            "*namak*, not *namaka*. The phonemic arm receives the deletion from "
            "the front end. The graphemic arm has to work it out."
        )
        if not names:
            gr.Markdown(
                f"**No bundles found in `{WEIGHTS_REPO}` under `{hub.PREFIX}`.** "
                "Export one with `python -m src.export.bundle` and upload it.")
            return demo

        text = gr.Textbox(label="Devanagari text", value=EXAMPLES[0], lines=2)
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text])
        go = gr.Button("Synthesize", variant="primary")

        outs = []
        with gr.Row():
            for side, default in (("A", default_a), ("B", default_b)):
                with gr.Column():
                    pick = gr.Dropdown(choices=[(label(n), n) for n in names],
                                       value=default, label=f"Run {side}")
                    audio = gr.Audio(label="Speech", type="numpy")
                    mel = gr.Image(label="Mel (no vocoder yet)", visible=True)
                    toks = gr.Markdown()
                    prov = gr.Markdown()
                    outs.append((pick, audio, mel, toks, prov))

        for pick, audio, mel, toks, prov in outs:
            ev = dict(fn=speak, inputs=[text, pick],
                      outputs=[audio, mel, toks, prov])
            go.click(**ev)
            pick.change(**ev)

        gr.Markdown(
            "---\nNine hours of single-speaker speech per arm, 100,000 steps, "
            "an identical frame budget. At that scale neither arm is a product; "
            "the point is the difference between them. Runs marked *mel only* "
            "are waiting on the vocoder and show their spectrogram instead."
        )
    return demo


if __name__ == "__main__":
    build().launch()
