---
title: Indic TTS Benchmark
emoji: 🗣️
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
short_description: Phonemic vs graphemic input for Hindi TTS, side by side
---

# Indic TTS benchmark

Hindi text in, speech out, with the two arms of one ablation side by side.

Devanagari does not write schwa deletion. *नमक* is pronounced *namak*, not
*namaka*, and nothing in the spelling says so. The phonemic arm is handed the
deletion by a rule-based front end. The graphemic arm reads the characters and
has to infer it. Everything else about the two runs is identical: the same
architecture, the same seed, the same nine hours of single-speaker speech, the
same 100,000 steps and the same per-step frame budget. So the difference you
hear between the two players is the input representation.

At nine hours neither arm is a product. The comparison is the point.

## How this Space is built

It installs the project from git at a pinned commit and calls
`src.export.synthesize`, which is the same code path the listening test uses.
The text front end here is the front end that trained the weights, down to the
normaliser and the schwa configuration. That is why this is Python rather than
in-browser ONNX: a reimplemented front end could differ from the trained one in
exactly the place the study is measuring, and a schwa-deletion bug in a port
would be invisible.

Weights are fp32 export bundles pulled from the model repo at first use, so
startup costs only the manifests and the first synthesis pays for the download.

Runs whose architecture emits mel spectrograms are silent until the vocoder
finishes training. They show their spectrogram instead, which is the model's
real output rather than a stand-in.

## Provenance

Each result shows the run id, the training step, the config hash and the git
commit the weights came from, because a demo that cannot say which version it
is was the thing this project set out to avoid.

## Configuration

| variable | default | purpose |
|---|---|---|
| `WEIGHTS_REPO` | `Klewik/indic-tts-bench` | model repo holding `exports/` |
| `BUNDLE_CACHE` | `/tmp/bundles` | where downloaded bundles land |
