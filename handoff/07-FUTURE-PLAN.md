# What remains

Priority order, with costs. **Agree the order with the user before doing any of
it** — the deadline is close and the choices trade against each other.

---

## 1. The deadline

The user set it on 3 October: **result gathering and paper writing must start by
Sunday 11 October 2026.** This file was written on 9 October. Treat training as
no longer having room unless the user says the GPU access has been extended.

GPU access was described as running "a few more days" on 3 October. **Confirm
whether the DGX is still available before planning anything that needs it.**

---

## 2. Immediate, and none of it needs a GPU

### 2.1 Confirm the current state

```bash
cd /workspace/indic-tts-bench && bash scripts/status.sh
ls /workspace/runs/
for r in r09 r10 r11 r12 r13 r14 r16 r19 r20 r21; do
  echo -n "$r: "; cat /workspace/runs/$r/stopped.json 2>/dev/null || echo "(none)"
done
```

Ten runs have an unknown state in `02-STATE-OF-PLAY.md`. Several of them are the
VITS ladder, which is the one remaining block that could still add a result to
the paper without retraining anything.

### 2.2 Confirm the demo published

Open `huggingface.co/spaces/Klewik/Indic-tts-demo`. It should show eight arms
grouped, with audio on all of them. If it says "Could not load data.json", the
re-render after the `Infinity` fix never happened — the commands are in
`06-RUNBOOK.md` §4.

### 2.3 Offload everything

The GPU access is temporary and the splits were only carried from 5 October
onward. `scripts/offload.py --all-checkpoints`. Until this has run, losing the
box loses the work.

### 2.4 The VITS ladder result

If r11–r14 finished, the ladder is scoreable **today** with no retraining,
because VITS is end to end and needs no vocoder. That turns a secondary claim —
"does the benefit depend on resource level" — from absent into measured. This is
the highest-value remaining item that costs nothing.

```bash
$V scripts/score_intelligibility.py --lang hindi --draws 5 --boot 2000 \
   --runs r02 r05 r11 r12 r13 r14
```

### 2.5 The cheap evaluation items

Each is hours, not days, and each closes a line in the phase table:

- **Stress-test error rates per architecture.** The set is at
  `stresstests/hindi_schwa_set_validated.tsv`.
- **MOS proxies.** UTMOSv2 and NISQA. They must be either validated on
  Devanagari speech or **labelled English-trained in the paper**. Do not report
  them unlabelled.
- **macOS real-time factor on the M5.** Runs on the Mac, not the DGX.
- **The clipping diagnostic**, 25 minutes on the 10-minute rung at `grad_clip
  1.0` against `clip 100`. Outside the matrix, so nothing's comparability is
  touched, and reportable either way. It is the most plausible single
  explanation for every run flattening at step 50,000.
- **The full 300-utterance test split at 5 draws**, replacing the 50-utterance
  pass. CPU only, hours rather than minutes.

### 2.6 Settle blocker b19, and the consent question

The paper's threats section must say that the resource claim and the 83.0%
front-end accuracy figure are **Hindi-specific** — the stress-test set is Hindi
only and the ladder r07–r14 is all Hindi. The user was asked to confirm the
wording and has not. **Ask again; it is one sentence and it blocks the section.**

While asking, settle the other thing taken on his word: that no consent or
ethics process is required for the listening test. There is no institutional
record of that. A study with twenty recruited participants is the kind of thing
a department asks about afterwards.

---

## 3. The FastSpeech 2 question

This is the one real decision left, and it has a GPU cost.

FastSpeech 2 produces unintelligible speech because `ForwardTTSLoss` leaves
pitch and energy as mean squared errors in physical units, so the mel decoder
gets under one per cent of the gradient (`04-RESULTS-AND-OBSERVATIONS.md` §2).
The fix is to z-score f0 and energy in `src/train/features.py` — three lines —
and it changes the `config_hash` of every FastSpeech 2 run.

```
mel share, as trained:   1.814 / 314.60              = 0.005766
after z-scoring:         1.814 / (1.814 + 2*0.1)     = 0.900695
fold change:             0.900695 / 0.005766         = 156.21
second route:            314.60 / 2.014              = 156.21
```

**Cost, and neither available rate is trustworthy.** Two figures for a
FastSpeech 2 pair exist in the repository: **6.24 h** at
`scripts/queue_all.sh:74`, which is the one the code quotes, and **6.76 h** from
`PLAN-full-matrix.md`, which `RESULTS.md` retired after a foreign tenant slowed
the ladder rungs from 4.11 it/s to about 1.3 — roughly a factor of 3. Both
descend from an idle card. Both are computed below. **Read either as a floor and
get a contention-corrected rate before committing.** See
`05-DEFECTS-AND-TRAPS.md` §4.

```
all 10 FastSpeech 2 runs    10 / 2 = 5.0 pairs
  at 6.24 h/pair            5.0 * 6.24 = 31.20 h   second route: 10 * 3.12 = 31.20 h
  at 6.76 h/pair            5.0 * 6.76 = 33.80 h   second route: 10 * 3.38 = 33.80 h

r01 + r04 only              2 / 2 = 1.0 pair
  at 6.24 h/pair            1.0 * 6.24 =  6.24 h   second route:  2 * 3.12 =  6.24 h
  at 6.76 h/pair            1.0 * 6.76 =  6.76 h   second route:  2 * 3.38 =  6.76 h

r01, r04 + the FS2 ladder   6 / 2 = 3.0 pairs
  at 6.24 h/pair            3.0 * 6.24 = 18.72 h   second route:  6 * 3.12 = 18.72 h
  at 6.76 h/pair            3.0 * 6.76 = 20.28 h   second route:  6 * 3.38 = 20.28 h
```

The two rates disagree by 8.3%. I am not picking one: use 6.24 if you want the
figure the code itself quotes, and treat the 3× contention factor as the real
uncertainty either way.

### The three options

**A. Retrain r01 and r04 only, 6.24 to 6.76 h, as a labelled objective
variant.**
Gives the paper a working FastSpeech 2 row for the headline contrast. The other
eight FastSpeech 2 runs — r07–r10, r15, r18, r20, r21 — keep the as-trained
objective, and the appendix carries the loss decomposition, which it already
does. *This was the recommendation on 5 October.*

**B. Retrain nothing.** Report FastSpeech 2 as a measurement defect — which it
genuinely is, and an instructive one — and make the headline contrast VITS-only.
The project already has a defensible result without FastSpeech 2.

**C. Retrain all ten, 31.20 to 33.80 h at the idle-card rates.** Only if the
GPU is still available and the deadline has moved. It makes the ladder
comparable across both architectures. At the contended rate this could be three
times longer, which almost certainly puts it out of reach.

`[Inference]` Normalising should make the mel decoder trainable and I expect
audible speech from r01 at the nine hour rung. That is expected, not
guaranteed. The 90% figure is the share of the objective, not a prediction of
intelligibility, and nothing has been trained with the normalised loss yet.

---

## 4. The paper

`paper/springer_draft.tex`, 763 lines, LNCS class, first person, bibliography
reformatted from the progress report. Sections: Introduction, Related Work,
Testbed, Evaluation (MCD with chance level, duration bias, intelligibility),
Preliminary Results, three sections on measurement defects (`sec:loss`,
`sec:punct`, `sec:seed`), Threats to Validity, Conclusions.

**It predates the 4–5 October results**, and its section heading reads "Two
Measurement Defects" over three subsections, which is its own defect. What has
to be folded in:

- the seed floor under the arm gap, with the three-variance ordering
- the paired bootstrap intervals and the **withdrawal** of the medial-site claim
- the vocoder transparency ceiling on both recognisers
- the FastSpeech 2 verdict and its two independent routes
- whichever of §2.5 gets done

The user's instruction on voice: *"use the language similar to what i used for
my progress report ... no need to match the synopsis, we can deviate from that,
but not too much from the progress report."*

### Corrections owed to the written report

- FastSpeech 2 is **37.99 M** parameters, not the ~27 M printed. The difference
  is `use_pitch`, `use_energy` and the internal aligner, all required. VITS at
  83.05 M matches.
- The matrix is **23 runs** (19 in the earlier count, plus the four seed-floor
  runs r20–r23). The original 17 missed the Marathi graphemic arms: the control
  was encoded with only phonemic arms and could not measure an effect, so r18
  and r19 were added.
- Precision is **bf16**, not fp16. The A100 has it; the T4 the plan was written
  for did not.
- Two stale cross-references, flagged and left alone on instruction: §2.1.4
  cites "Section 6.2" for StyleTTS 2 (it is 6.3), and §3.9 points at
  "Section 5.2" for run identifiers (now 5.1.2).

---

## 5. Explicitly deferred

- **Marathi**, to dissertation part 2. r15, r17, r18 trained; r16 and r19 were
  on the card. Not in the paper.
- **Matcha-TTS** (r03), to future work. The adapter is written and kept.
- **The human listening test** (phase 8). No consent process required,
  confirmed. The stimuli and the page exist; nothing else does.
- **The remaining open blockers** b4, b6, b8, b14, b16, b19, b20, b22, b23 —
  full text in `ARTIFACT-EXPORT.md`. Several are already overtaken by events;
  read them against `02-STATE-OF-PLAY.md` before acting on any.

---

## 6. If someone asks "what is the one thing to do next"

Confirm the run states, then score the VITS ladder if it finished. It is the
only remaining item that adds a result to the paper at zero GPU cost, and it
closes the one secondary claim the project set out to make and has not yet
measured.
