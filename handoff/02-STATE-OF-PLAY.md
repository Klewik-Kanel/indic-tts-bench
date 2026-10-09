# State of play

**Accurate as of 5 October 2026.** Ten runs have an unknown or in-flight
state; §1 lists them and gives the commands to resolve them. Nothing here is guesswork: where the state is unknown it
says so.

---

## 1. The matrix: 23 runs

Every one has a config at `configs/<id>.yaml` whose `config_hash` is the run's
identity. `9h` is the full training split; the ladder rungs are nested subsets.

| run | arch | lang | input | data | seed | rate | config_hash | state |
|---|---|---|---|---|---|---|---|---|
| r01 | fastspeech2 | hindi | phoneme | 9h | 0 | 22050 | `e3bbebc98825` | **done** |
| r02 | vits | hindi | phoneme | 9h | 0 | 16000 | `d3873342a10d` | **done** |
| r03 | matcha | hindi | phoneme | 9h | 0 | 22050 | `1d85771a0e91` | never run — future work |
| r04 | fastspeech2 | hindi | grapheme | 9h | 0 | 22050 | `6aeae2e66a19` | **done** |
| r05 | vits | hindi | grapheme | 9h | 0 | 16000 | `05eaea2642e8` | **done** |
| r06 | hifigan | hindi | none | 9h | 0 | 22050 | `3120211db4c7` | **done**, best step 12000, exported at 18000 |
| r07 | fastspeech2 | hindi | phoneme | 5h | 0 | 22050 | `c1b1edce3d2d` | **done** |
| r08 | fastspeech2 | hindi | phoneme | 1h | 0 | 22050 | `52e245223208` | **done** — hash was corrupted, see below |
| r09 | fastspeech2 | hindi | phoneme | 30min | 0 | 22050 | `ed76b539d2bc` | unknown |
| r10 | fastspeech2 | hindi | phoneme | 10min | 0 | 22050 | `1c378a7604cf` | unknown |
| r11 | vits | hindi | phoneme | 5h | 0 | 16000 | `93bae73db260` | unknown |
| r12 | vits | hindi | phoneme | 1h | 0 | 16000 | `414a33d2879d` | unknown |
| r13 | vits | hindi | phoneme | 30min | 0 | 16000 | `40e2939875a1` | unknown |
| r14 | vits | hindi | phoneme | 10min | 0 | 16000 | `348e61158e4b` | unknown |
| r15 | fastspeech2 | marathi | phoneme | 9h | 0 | 22050 | `d72a3563cbfc` | **done** |
| r16 | vits | marathi | phoneme | 9h | 0 | 16000 | `983453d32f43` | was running 5 Oct |
| r17 | hifigan | marathi | none | 9h | 0 | 22050 | `99a3999ca13e` | **done** |
| r18 | fastspeech2 | marathi | grapheme | 9h | 0 | 22050 | `3cccd1725aeb` | **done** |
| r19 | vits | marathi | grapheme | 9h | 0 | 16000 | `c2ba36578a66` | was running 5 Oct |
| r20 | fastspeech2 | hindi | phoneme | 9h | 1 | 22050 | `0476f129360d` | was running 3 Oct |
| r21 | fastspeech2 | hindi | phoneme | 9h | 2 | 22050 | `ce7dc788bc86` | was running 3 Oct |
| r22 | vits | hindi | phoneme | 9h | 1 | 16000 | `500e5e3eae82` | **done** |
| r23 | vits | hindi | phoneme | 9h | 2 | 16000 | `72f4c9aa7494` | **done** |

**r08's config hash was recorded as `Infinity`.** On 2 October the config loader
parsed the all-digit hash as a float and `cfg["config_hash"] = inf` was written
into `runs/r08/config.json` and all five of its checkpoints. The real hash
cannot be regenerated. `configs/r08.yaml` carries the correct
`52e245223208`; the run directory does not. Only r08 of the whole matrix is
affected, because only its hash happens to parse as a number. The weights are
unaffected. `scripts/repair_config_hash.py` exists for this. On 5 October that
`Infinity` reached `space_static/data.json` and broke the public listening page
— see `05-DEFECTS-AND-TRAPS.md`.

### How to resolve the unknowns

```bash
cd /workspace/indic-tts-bench && bash scripts/status.sh
ls /workspace/runs/
for r in r09 r10 r11 r12 r13 r14 r16 r19 r20 r21; do
  echo -n "$r: "; cat /workspace/runs/$r/stopped.json 2>/dev/null || echo "(no stopped.json)"
done
```

---

## 2. What is genuinely finished

**Front end.** Built, split so schwa deletion is a switchable stage, shared
normaliser across both arms, scored at 39/47 = 83.0% against speaker judgement
with **five** caveats recorded, all of which have to travel with the number.
See `01-PROJECT-AND-METHOD.md` §2.

**Corpora.** Both downloaded, standardised, profiled, split, frozen.
`SPLITS.lock` written, `--verify` re-hashes against it, four structural checks
pass. G2P coverage asserted over the full corpora, zero unknown symbols over
11,044 transcripts. `[Unverified]` `project-plan-v3.md` puts the token count at
870,547; that figure is not in `RESULTS.md`. `[Unverified]` An
earlier handoff claimed the 16 split files were checked byte-identical across
two machines; `RESULTS.md` records no cross-machine check.

**Training harness.** Shared config schema, frame-budget batching, LR schedule,
checkpointing. Resume proven **bit-exact** — a killed run restarted from its
checkpoint gives weights identical to one never interrupted, because the data
order comes from `(seed, step)` rather than from the checkpoint. First verified
across 11 tensors on a 20-step dry run, then again across **all 404 tensors** on
the vocoder path. Two-optimiser path for VITS with per-optimiser clipping and both
states checkpointed. Adapter parity: the two input arms see identical batches.

**Vocoder path, end to end.** r06 and r17 trained on ground-truth mels, warm
started from a published HiFi-GAN generator whose tensors are renamed onto this
model's names. The mel front end is verified rather than assumed, on three axes.
A bundle refuses a vocoder whose sample rate or language does not match.

**Export and demo.** `src/export/bundle.py` writes a self-contained bundle
(manifest + weights + vocabulary) with a provenance record. One synthesis path
shared by the demo and the listening test. Demo sentence set chosen from the
held-out split by a stated, deterministic rule. Static listening page, light and
dark, that explains itself.

**Evaluation harness.** MCD with chance level, F0 RMSE, duration-bias slope,
character error rate with the recogniser's floor, site partition, paired
bootstrap over utterances, seeded synthesis with multiple draws.

**The headline result.** See `04-RESULTS-AND-OBSERVATIONS.md`.

**614 `def test_` functions**, counted from the source. `[Unverified]` A run on
5 October reported 641 passed and 11 failed, the 11 being `librosa` missing on
the Mac; that is a session observation and is not in `RESULTS.md`. Re-run the
suite before quoting it.

---

## 3. What is not finished

Phase 6 (evaluation) stood at 8 of 13 on 3 October. Remaining, none of it
needing a GPU:

1. **Stress-test error rates per architecture.** The contested-schwa set exists
   at `stresstests/hindi_schwa_set_validated.tsv`; the per-architecture pass was
   never run.
2. **MOS proxies.** UTMOSv2 and NISQA are planned. They must be either validated
   on Devanagari speech or labelled English-trained in the paper. Neither has
   happened.
3. **macOS real-time factor on the M5.** Never measured.
4. **The gradient-clipping diagnostic** (blocker b16, open). 25 minutes, outside
   the matrix: 5,000 steps on the 10-minute rung at `grad_clip 1.0` against
   `clip 100`. Gradient norms of 97 to 1139 against a clip of 1.0 mean
   essentially every step is scaled down by two to three orders of magnitude.
   If clipping is the limiter it is the most plausible single explanation for
   every run flattening at step 50,000. Reportable either way without
   retraining. The note that AdamW largely cancels a uniform rescale is marked
   `[Inference]` because it was an argument, not a measurement.
5. **The listening test** (phase 8). Nothing done but the stimuli selection.
   The user says no consent or ethics process is required. That is his
   statement, not an institutional record; confirm it before recruiting.
6. **Blocker b19**, a wording question: the paper's threats section must say the
   resource claim and the front-end accuracy figure are Hindi-specific. The user
   has been asked to confirm and has not.

**Paper.** `paper/springer_draft.tex` exists, 763 lines, LNCS class, first
person, with the progress report's bibliography reformatted to LNCS. Sections:
Introduction, Related Work, Testbed, Evaluation, Preliminary Results, one
section titled "Two Measurement Defects" holding **three** subsections
(`sec:loss`, `sec:punct`, `sec:seed`), Threats to Validity, Conclusions. **The
heading says "Two" over three subsections — fix that.** It predates the
4–5 October results and needs them folded in.

---

## 4. Publishing state, 5 October

**GitHub.** `github.com/Klewik-Kanel/indic-tts-bench`, public, everything
pushed. `.gitignore` excludes `*.pt`, `*.wav`, `checkpoints/` and the data
directories, with deliberate exceptions: `data/processed/**/*.tsv`,
`SPLITS.lock`, `data/interim/**/manifest.tsv` and `space_static/audio/**/*.wav`
are un-ignored because they are the reproducibility record and the deliverable.

**GitHub Pages.** `.github/workflows/pages.yml` publishes `space_static/`
directly, with no `docs/` copy. It **skips** cleanly when nothing has been
rendered yet and fails only on a half-committed site. It was failing on every
push until 5 October because `space_static/data.json` and the wavs exist only on
the DGX and have never been committed.

**Hugging Face.** `Klewik/indic-tts-bench` (private) holds the offloaded
weights and results. `Klewik/Indic-tts-demo` is the Static Space for the
listening page.

**The last known demo state:** rendered on the DGX with the trained vocoder
attached — 8 arms, 12 sentences, 96 clips, 17.8 MB, all four FastSpeech 2 arms
carrying `hifigan:r06@18000`. It was then published with a `data.json` holding
the token `Infinity`, which no browser can parse, and the page showed "Could not
load data.json". Three commits fixed that chain. **Whether the re-render and
re-publish were run is unknown — confirm it first.**

---

## 5. Live state the card last recorded

The tracking artifact's own state is frozen at **3 October, 18:15 IST**. Its
run table and phase checklist are in `ARTIFACT-EXPORT.md`. Four days of work
followed it and were never written back. Where it disagrees with this file or
with `RESULTS.md`, those are newer.
