# Start here: Indic TTS benchmark, handoff to a fresh session

> **SUPERSEDED.** This is the 1 October handoff, kept because sections 4 and 5
> hold traps and corrections that are still live. For the current position read
> `handoff/00-READ-ME-FIRST.md`. Sections 3 and 7 below are historical: the
> vocoder question was decided on 2 October, VITS completes steps, and both
> arms have audio.

Paste this file, or attach it, as the first message of a new chat. It is written
to be self-contained: nothing important lives in the conversation that produced
it.

**Project.** M.Tech dissertation, Kaustubh Pandey (2025PA17326), NSUT New Delhi,
supervisor Dr. Shobha Bhatt. The question: does explicit Devanagari
grapheme-to-phoneme conversion still buy anything for neural TTS, or do modern
architectures absorb Hindi schwa deletion from data alone? Hindi is the testbed,
Marathi the same-script control, and a data ladder varies resource level.

---

## 1. Read these four, in this order

1. `claude/project-plan-v3.md` — the design and the phase table
2. `PLAN-full-matrix.md` — the 19-run schedule and what blocks each block
3. `RESULTS.md` in the repo — the dated log; every number carries its config
4. The artifact "Schwa Deletion Benchmark" — live status, 49 log entries

Do not re-derive anything that is already in those. They are current.

---

## 2. The machine, and how to reach it

| | |
|---|---|
| GPU | NSUT DGX, one A100-SXM4-40GB, idle apart from this work |
| Access | JupyterLab at `http://192.168.33.8:9014/lab`, campus VPN, Terminal tile |
| Driving it | Claude in Chrome, against the user's own browser tab |
| Repo on DGX | `/workspace/indic-tts-bench`, auto-pulls every 30 s |
| Python | `/workspace/venv/bin/python` — **never** the container's system Python |
| Runs | `/workspace/runs/<id>/`, logs `/workspace/runs/<id>.log` |
| Status | `bash /workspace/status.sh` (`-w` to watch) |
| Repo | `github.com/Klewik-Kanel/indic-tts-bench`, public |
| Mac clone | `~/indic-tts-bench`; Claude commits, **only the user pushes** |

The user runs `git push` from their Mac. Claude cannot: the credentials are
theirs and Claude will not handle them. The DGX pulls on its own.

---

## 3. Where the work actually stands *(as of 1 October — historical)*

**Done and verified**

- Both corpora downloaded, standardised and split on the DGX. All 16 split
  files match `SPLITS.lock` byte for byte, on a different OS and filesystem, so
  reproducibility is demonstrated rather than asserted.
- Front end scores **39/47 = 83.0%** against native-speaker judgement, with four
  caveats recorded in plan v3 section 2. This replaced a meaningless 49/49.
- Resume proven **bit-exact**: a killed run restarted from its checkpoint gives
  weights identical to one never interrupted.
- **r04 complete**: FastSpeech 2, Hindi, graphemic. 100,000 steps, final loss
  302.46, checkpoint `step_100000`.
- **r01** was at 87.6% and finishing. Together these are the Hindi ablation
  pair, which is the dissertation's core comparison.

**Blocked, and in priority order** *(all four since resolved)*

1. **No HiFi-GAN adapter exists.** FastSpeech 2 emits mel spectrograms and is
   silent without a vocoder, so r01 and r04 cannot be heard, evaluated on audio,
   or demonstrated. This outranks further training. Either write the adapter and
   train r06, or use a pretrained HiFi-GAN and put the deviation in the table.
2. **VITS does not complete a step.** Four glue bugs fixed so far; six of the
   nineteen runs depend on it.
3. Marathi feature cache not pre-warmed.
4. `src/eval/` has MCD and log-F0 only. ASR-WER, predicted MOS and RTF are not
   written. `src/analysis/` is empty. `listening_test/` is empty.

---

## 4. Traps already paid for

**The container image.** Four faults, each documented with its diagnosis in
`scripts/11_dgx_env.sh`: a dead NVIDIA package index in four `pip.conf` files;
`PIP_CONFIG_FILE=/dev/null` not covering the site-level config; pip 24.3.1
crashing in its own version parser; PyPI's torchaudio failing against NVIDIA's
torch with an ABI error. Training therefore runs from `/workspace/venv` with a
matched upstream stack.

**coqui's `init_from_config` overwrites `num_chars`** with its own 67-symbol
character set, so a 78-symbol Devanagari inventory indexes past the end of the
embedding. Symptom: a CUDA device-side assert naming no tensor.
`assert_embedding_fits` catches it on the CPU now.

**Tensor layouts differ by architecture.** FastSpeech 2 wants mel as
`[B, frames, mels]`; VITS wants the spectrogram as `[B, freq, frames]`; pitch,
energy and waveform are all `[B, 1, T]`. Wrong layouts give the same unreadable
CUDA assert. `CUDA_LAUNCH_BLOCKING=1` moves those errors back to the Python
frame that caused them, and is the first thing to reach for.

**`format_batch_on_device` consumes `waveform_rel_lens`; it does not compute
it.** Supply it as the frame ratio, not the sample ratio.

**PyTorch's caching allocator** holds every block it ever used: a run needing
4.2 GB sat on 37.5 GB of a 40 GB card. `TRAIN_GPU_FRACTION` caps it.

---

## 5. Corrections owed to the written report

- FastSpeech 2 is **37.99 M** parameters, not the ~27 M printed. The difference
  is `use_pitch`, `use_energy` and the internal aligner, all required. VITS at
  83.05 M matches.
- The matrix is **19 runs**, not 17. The Marathi control was encoded with only
  phonemic arms and could not measure an effect; r18 and r19 were added.
  *(It is now 23: r20–r23 are the seed floors.)*
- Precision is **bf16**, not fp16. The A100 has it; the T4 the plan was written
  for did not.
- Two stale cross-references flagged earlier and left alone on instruction:
  section 2.1.4 cites "Section 6.2" for StyleTTS 2 (it is 6.3), and section 3.9
  points at "Section 5.2" for run identifiers (now 5.1.2).

---

## 6. How to work cheaply in the new session

Terminal screenshots and long conversation history are what cost money, not the
training. So:

- The user runs `bash /workspace/status.sh` and pastes the output when a
  decision is needed. One paste is far cheaper than one screenshot.
- Runs take about 6.8 h each. Check two or three times a day.
- Several instructions per message, not one word per message.
- Start a fresh session per phase and re-read the four documents in section 1.

---

## 7. The first thing to do in the new session *(historical — decided 2 Oct)*

Decide the vocoder question, because it blocks the demo and the evaluation both:

> FastSpeech 2 is silent without HiFi-GAN. Write the adapter and spend about
> 7 GPU-hours on r06, or synthesise with a pretrained HiFi-GAN and declare the
> deviation?

Then pre-warm Marathi features and launch the r15 + r18 pair, which is the next
block that is not blocked on anything.

**Decided:** fit our own vocoder by fine-tuning a published HiFi-GAN generator
on this project's own mels, one per language — r06 for Hindi, r17 for Marathi.
Both trained. See `handoff/01-PROJECT-AND-METHOD.md` §4.
