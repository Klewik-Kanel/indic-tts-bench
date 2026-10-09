# Codebase map

Repository root on the Mac: `/Users/klewik/indic-tts-bench`. On the DGX:
`/workspace/indic-tts-bench`. Same git repository, same contents.

---

## 1. Layout

```
configs/        one YAML per run, r01..r23. The unit of reproducibility.
data/           raw, interim, processed. Mostly gitignored; splits are not.
handoff/        this folder
jobs/           Kaggle job directories (historical; Kaggle was abandoned for the DGX)
listening_test/ empty — phase 8 never started
logs/           download and build logs
paper/          LaTeX sources, the progress report, the Springer draft, slides
results/        tables/ (JSON), figures/, raw/
scripts/        operator entry points: shell for the user, Python for the work
space/          the Gradio app (kept for live free text from a notebook)
space_static/   the Static Space: index.html, README.md, data.json, audio/
src/            the library
stresstests/    the contested-schwa set and the elicitation sheets
tests/          38 files, 614 test functions
.github/workflows/pages.yml
RESULTS.md      the lab notebook — 110 KB, 40 sections (39 of them dated)
```

---

## 2. `src/` module by module

### `src/g2p/` — the front end

| Module | Job |
|---|---|
| `normalize.py` | Devanagari normalisation, shared by **both** arms and every language |
| `devanagari.py` | graphemes → phones, inherent schwas intact |
| `schwa.py` | Hindi schwa deletion, isolated as a switchable stage |
| `phoneset.py` | shared IPA inventory for Devanagari languages |
| `numbers.py` | ASCII digits → Devanagari number words |
| `score.py` | scores the front end against the stress-test gold forms |
| `elicit.py` | builds a blind elicitation sheet, merges filled sheets |

### `src/data/` — corpora

| Module | Job |
|---|---|
| `prepare.py` | standardise raw corpora into training masters |
| `profile.py` | per-speaker hours, length filters, QA checks |
| `splits.py` | freeze the evaluation splits, build the nested ladder |
| `g2p_coverage.py` | run G2P over every transcript, report what it cannot handle |

### `src/train/` — the harness

| Module | Job |
|---|---|
| `config.py` | the run config dataclass, `config_hash`, YAML emission |
| `runner.py` | **the training loop every architecture shares**; also `load_config` |
| `adapters.py` | one adapter per architecture plus a toy adapter that depends on nothing |
| `batching.py` | frame-budget batching, deterministic and identical across architectures |
| `features.py` | acoustic features, computed once per utterance and cached |
| `text.py` | text → token ids, and the vocabulary that pins them down |
| `schedule.py` | learning-rate schedule, shared |
| `checkpoint.py` | what is saved, what is kept, what resume means |
| `earlystop.py` | stopping criterion, for the one run type with no step budget (the vocoder) |
| `devices.py` | move the things a run holds that the model does not own |
| `dryrun.py` | prove the loop and the resume path before spending GPU hours |
| `launch.py` | start, or restart, one run |
| `progress.py` | live progress, with and without a terminal |
| `bench.py` | measure the step rate, say what budget fits the time available |
| `melstats.py` | mel mean and sd over a corpus (Matcha needs it) |
| `vocoder_init.py` | rename a published HiFi-GAN generator's tensors onto our names |

### `src/eval/` and `src/analysis/`

| Module | Job |
|---|---|
| `eval/mcd.py` | mel-cepstral distortion with DTW alignment |
| `eval/f0.py` | F0 RMSE and voiced/unvoiced error |
| `eval/asr.py` | the intelligibility proxy: model registry, CER/WER, normalisation |
| `analysis/schwa_sites.py` | where deletion actually happens, per word and per sentence |
| `analysis/position_errors.py` | where the recogniser's errors fall relative to sites; the paired bootstrap |
| `analysis/duration_bias.py` | milliseconds of excess per deletion site, no ASR needed |

### `src/export/`

| Module | Job |
|---|---|
| `bundle.py` | package a finished run as a self-contained bundle |
| `synthesize.py` | text in, speech out, from a bundle and nothing else |
| `vocoder.py` | the pretrained vocoder's mel front end, checked before it is trusted |
| `griffinlim.py` | make a mel audible without a vocoder, as a **labelled** placeholder |
| `hub.py` | find and fetch export bundles from a Hugging Face model repo |

---

## 3. The invariants that matter

These are the things that break quietly if you change them without reading.

### The bundle format

A bundle is a **directory**, not a file: `manifest.json` + `model.pt`, plus
`vocab.json` unless `input_repr == "none"` (a vocoder has no text side).
`BUNDLE_VERSION = 3`. A bundle written by a newer build is refused in
milliseconds rather than after a multi-second torch import, which is why
`read_manifest()` is separate from `Bundle`.

`manifest["audio"]` holds the mel parameters. `n_mels` is **not** top level —
it is inside `audio`. The manifest also carries `run_id`, `architecture`,
`language`, `input_repr`, `sample_rate`, `data`, `seed`, `needs_vocoder`,
`config_hash`, `git_commit`, `describes`, and a per-file sha256.

`model.pt` loads with `strict=True`. A silently partial load is how a demo ends
up running a half-initialised model and nobody can say why it sounds wrong.

### `config_hash`

SHA-256 over all config fields **minus** `COSMETIC = {notes, created, run_id,
corrections}`, truncated to 12 hex characters. Changing any hashed field makes
it a different run. `corrections` is outside the hash by design, so a claim
later found false can be recorded without moving a finished run's provenance.

### The mel

`log(max(mel_fb @ |stft|, 1e-5))`, N_FFT 1024, hop 256, window 1024, 80 mels,
fmin 0, fmax sr/2. **Magnitude** mel, with librosa framing: `center=True`,
reflect pad 512.

A mel handed to a vocoder built on different analysis parameters produces audio
that is recognisably speech and quietly wrong. `RESULTS.md` names the three axes
on which published families differ from this one — **band fmax, amplitude
convention, and framing** — and `src/export/vocoder.py` checks them. The
comparison table is in `handoff/01-PROJECT-AND-METHOD.md` §4.

### Tensor layouts differ by architecture

FastSpeech 2 wants mel as `[B, frames, mels]`. VITS wants the spectrogram as
`[B, freq, frames]`. Pitch, energy and waveform are all `[B, 1, T]`. A wrong
layout gives an unreadable CUDA device-side assert that names no tensor.
`CUDA_LAUNCH_BLOCKING=1` moves the error back to the Python frame that caused
it, and is the first thing to reach for.

### Things coqui does that this project has to work around

This project uses coqui's **model classes** but **not** its Trainer and not its
dataset pipeline. Three consequences:

1. **`ForwardTTSLoss` is unnormalised.** `RESULTS.md` describes it as a sum of
   **five terms**, of which the two largest are mean squared errors in
   **physical units** — f0 in hertz and the L2 norm of a linear spectrogram
   frame, **both at α = 0.1**. coqui's own dataset pipeline z-scores both; this
   project bypasses that pipeline, so they are raw. **This is why FastSpeech 2
   produces noise.** See `04` and `05`. `[Unverified]` Longer per-term
   breakdowns of this loss circulate in the conversation history; only the five
   terms and the two α = 0.1 weights are on the record. Read
   `ForwardTTSLoss` in the installed coqui-tts before quoting any other
   decomposition in the paper.
2. **`GAN.on_train_step_start` is never called.** It sets `train_disc` from
   `trainer.total_steps_done`, and `binary_loss_weight` ramps with
   `trainer.epochs_done`. Neither happens here, so the harness does it.
3. **`init_from_config` overwrites `num_chars`** with coqui's own 67-symbol
   character set, so a 78-symbol Devanagari inventory indexes past the end of
   the embedding. Symptom: a CUDA device-side assert naming no tensor.
   `assert_embedding_fits` catches it on the CPU now.

Also: `format_batch_on_device` **consumes** `waveform_rel_lens`, it does not
compute it. Supply it as the **frame** ratio, not the sample ratio.

### Determinism

The data order comes from `(seed, step)`, not from the checkpoint, so a
checkpoint cannot disagree with its manifest and a resumed run sees the batch an
uninterrupted one would have seen. Verified bit-exact across 11 tensors on a
20-step dry run, and again across **all 404 tensors** on the vocoder path.

Synthesis is seeded per `(run_id, text, draw)` with a salted SHA-256,
`SEED_SALT = "indic-tts-bench/synthesis/v1"`, shifted right one bit for a 63-bit
value. `torch.manual_seed` is called before `inference`, unconditionally — it
costs nothing for a deterministic architecture and removes a difference between
architectures that is otherwise silent.

---

## 4. `scripts/` — the operator surface

Shell scripts numbered `01`–`12` are historical setup, run in the user's own
Terminal. The working set:

| Script | What it does |
|---|---|
| `status.sh` | one screen of live training status, written for `watch` (`-w`) |
| `gpu_free.sh` | is the card free to launch on |
| `queue_all.sh` | queue every remaining pair so the box needs no touching |
| `verify_queue.py` | is every run accounted for, and can each one actually start |
| `offload.py` | copy weights, results, splits and the demo off the DGX |
| `offload_loop.sh` | run the offload on a timer |
| `offload_status.py` | what has actually reached Hugging Face, asked of the Hub |
| `12_prewarm_features.py` | fill the feature cache in parallel before a run |
| `evaluate.py` | score finished runs against the frozen test split, write the table |
| `score_intelligibility.py` | CER with the recogniser's floor, site partition, bootstrap |
| `build_demo_set.py` | choose the demo sentences on a stated rule |
| `render_demo.py` | render the demo set through every bundle into `space_static/` |
| `publish_space.py` | audit the rendered demo, then publish it as a Static Space |
| `repair_config_hash.py` | repair provenance the old config loader corrupted |
| `check_text_coverage.py` | is every symbol the front end emits in the vocabulary |
| `phone_stats.py` | phone frequency and position |
| `probe_mms_match.py` | could an MMS warm start transfer anything to our VITS |
| `_status_line.py` | one formatted line for one run, from its last step record |

---

## 5. Tests

38 files holding **614 `def test_` functions**, counted directly from the
source, which `parametrize` expands to more cases than that.

`[Unverified]` On 5 October a run of the whole suite reported **641 passed and
11 failed**, giving 652 cases, with all 11 failures being `librosa` missing on
the Mac. That was observed in a working session and **was never written into
`RESULTS.md`**, whose last recorded test state is "202 across the torch-free
suites" on 3 October. The 614 figure is verifiable from the repository; 641,
652 and 11 are not. Re-run the suite before quoting any of them.

The per-file numbers below count functions, not expanded cases, so they sum to
614.

**There is currently no way to run the suite on the Mac.** It has no pytest and
no `librosa`, and the shim that stood in — `/tmp/shim_run.py`, whose `fixture`
was a pass-through decorator and whose `MonkeyPatch` had no `delenv` — was never
in the repository and is gone, because `/tmp` was cleared. Install pytest and
librosa there, or run the suite on the DGX.

```
test_score_intelligibility.py   61   the harness's bookkeeping, no model, no audio
test_position_errors.py         43   errors attributed to the right word class
test_asr.py                     41   the proxy's arithmetic and its pins
test_train_loop.py              33   training parts that can be wrong without a GPU
test_static_page_grouping.py    24   what the page RENDERS, run under node
test_g2p.py                     22   front end
test_vocode.py                  22   attaching a vocoder, with the checks that matter
test_publish_space.py           20   the checks between a render and a public URL
test_hifigan_adapter.py         19   the vocoder adapter
test_matcha_adapter.py          19   the Matcha adapter and what it refuses to guess
test_eval_metrics.py            18   known-answer tests for the objective metrics
test_vocoder_mel_match.py       18   the vocoder's mel front end must match
test_synthesis_seed.py          16   synthesis must be reproducible
test_json_is_json.py            15   data.json must be readable by a BROWSER
test_pages_workflow.py          14   the Pages gate, executed rather than read
test_two_optimisers.py          14   the VITS path must not change anything else
test_offload_plan.py            13   what must be carried for the box to be disposable
test_duration_bias.py           12   excess duration against deletion-site count
test_init_from_honesty.py       12   a config may not claim a warm start it never performs
test_own_vocoder.py             12   the export path must accept OUR vocoder
test_corrections.py             10   a false deviation is corrected, not rewritten
test_evaluate.py                10   evaluation bookkeeping
test_griffinlim.py              10   the placeholder's guards
test_loss_components.py         10   the step record must say what the loss is made of
test_render_demo_bundles.py     10   which bundles become arms, and which get a vocoder
test_seed_floor.py              10   the seed runs differ from r01 in the seed and nothing else
test_synthesize.py              10   synthesis parts that do not need torch
test_status_line.py              9   the formatter does arithmetic and runs for days
test_vocoder_init.py             9   renaming a published generator onto our names
test_adapter_parity.py           8   two arms must see the same data
test_best_checkpoint.py          8   a stopping criterion must keep the weights it stopped for
test_earlystop.py                8   numbers in, a decision out, no GPU
test_config_loader.py            7   the reader must not turn a hash into a number
test_devices.py                  7   moving tensors the model does not own
test_static_page.py              7   the page's claims about how each clip was made
test_hub.py                     11   bundle discovery and its path arithmetic
test_grapheme_number_parity.py  11   both arms handed the same string
test_schwa_sites.py             11   deletion sites and the demo-set statistics
```

**The testing standard here is "execute it", not "grep it".** Two real bugs were
found only by running code that a source grep had passed: the Pages gate ran
under `set -u` without `-e` and would have deployed a broken site, and the
listening page's grouping put the unintelligible architecture first. Both are
now covered by tests that run the thing under `node` or `bash` and read what it
produced.

---

## 6. Dependencies

`coqui-tts 0.27.5`, `matcha-tts 0.0.7.2`, `torch 2.14.1+cu126`, Python 3.12 on
the DGX. `librosa`, `soundfile`, `huggingface_hub`. The Mac clone has an old
`.venv` on Python 3.9 that is not used for anything current.

**On the DGX, always `/workspace/venv/bin/python`.** Plain `python` is the
container's system install with a mismatched ABI, and `import torchaudio` fails
with an `undefined symbol` error. This has been hit more than once.
