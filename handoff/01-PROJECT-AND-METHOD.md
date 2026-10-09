# The project and its method

Everything here was decided deliberately and most of it was argued out in
`RESULTS.md` on a dated line. Where a decision had a cost, the cost is stated.

---

## 1. The question

Devanagari is an abugida: every consonant letter carries an inherent schwa
unless a diacritic cancels it. Hindi then deletes many of those schwas in
speech, and the orthography does not mark the deletion. क म ल is written
*kamala* and said *kamal*. The deletion is conditioned — word-final after a
single consonant is near-categorical, medial deletion depends on syllable
structure and morphology — so it is a genuine rule, not noise.

Classical TTS pipelines therefore put a grapheme-to-phoneme front end in front
of the acoustic model. Modern end-to-end neural systems are often trained on
characters directly, on the argument that the model learns the mapping from
data. **Nobody had measured, for Devanagari, under matched compute, whether
that argument holds.** That is the gap.

The contrast is therefore:

- **phonemic arm** — the rule-based front end runs, schwa deletion applied, the
  model receives phones
- **graphemic arm** — no front end, the model receives normalised characters and
  must infer the deletion from audio

Everything else is identical by construction: architecture, random seed, corpus,
step budget, batching, learning-rate schedule, precision.

### Secondary questions

- **Does it depend on resource level?** A nested data ladder, 9 h → 5 h → 1 h →
  30 min → 10 min, same step budget at every rung.
- **Is the effect about schwa, or about script?** Marathi is written in the same
  script and deletes schwa far less. A Marathi control with both arms separates
  the two. *(Deferred — see §7.)*
- **Does it depend on architecture?** FastSpeech 2 (non-autoregressive, mel
  predictor plus vocoder) against VITS (end-to-end, produces waveform).

---

## 2. The front end

`src/g2p/` holds it, split so that schwa deletion is a switchable stage rather
than being baked in:

| Module | Job |
|---|---|
| `normalize.py` | Devanagari normalisation, shared by **both** arms and every language. Combining marks, nukta, ZWJ/ZWNJ, digit handling. |
| `devanagari.py` | Grapheme sequence → phone sequence with inherent schwas **intact**. |
| `schwa.py` | Hindi schwa deletion as its own stage, switched on for the phonemic arm only. |
| `phoneset.py` | Shared IPA inventory across Devanagari languages, so Hindi and Marathi phones are comparable. |
| `numbers.py` | ASCII digits → Devanagari number words. |
| `score.py` | Scores the front end against gold forms. |
| `elicit.py` | Builds a blind elicitation sheet and merges filled sheets. |

**Both arms share the same normaliser.** That is deliberate: if the graphemic
arm saw unnormalised text the comparison would confound normalisation with
phonemisation.

### The front end's accuracy is 83.0%, and the number has a history

The first figure was **49/49**. It was discarded as meaningless: the rule and
the answer key had been written by the same author in the same sitting, so it
measured self-consistency, not accuracy.

A contested-schwa stress-test set was then built and scored against native
speaker judgement, blind: **39/47 = 83.0%**.

**Five caveats are recorded with it in `RESULTS.md`, and all five have to
travel with the number.** The first is the one most easily lost:

1. **One sheet, not three.** Three speakers' responses were averaged into a
   single sheet *before* scoring. So inter-speaker agreement cannot be computed,
   the zero variation rate is an artefact of the averaging rather than a
   finding, and wherever speakers genuinely disagreed the split was resolved by
   whoever did the averaging. None of it can be repaired from the averaged file.
   **The per-speaker sheets are the thing to keep if this is ever rerun.**
2. Seven of the eight mismatches are the **speakers keeping** a schwa the rule
   deletes, and only one runs the other way. That is the signature of
   citation-form reading: a list of 49 isolated words is close to the worst
   possible condition for eliciting a conversational pace. So 83.0% is a **lower
   bound under conditions that favour retention**, not an estimate of the rule
   in running speech.
3. Four of the eight sit in one phonological environment — राष्ट्र, कृष्ण, धर्म
   and सत्य, all word-final schwa after a consonant cluster, all
   Sanskrit-derived. `SchwaConfig.block_before` exists for exactly that
   environment and is **deliberately empty**: populating it from these four
   words would fit the rule to its own test set, and the fix needs a held-out
   set of tatsama words. With those four counted correct the accuracy would be
   43/47 = 91.5%, which is what fixing it is worth if it generalises.
4. **Two are ordinary citation-form retention.** आदमी came back as /aːd̪əmiː/
   and फ़ैसला as /fɛːsəlaː/; both delete in normal speech. कल came back as
   /kələ/, which is the same effect on a word short enough that it should not
   have happened.
5. **One answer is impossible and is still in the denominator.** न was answered
   as having no vowel, which yields a gold form of /n/ — a consonant with no
   vowel, not a pronounceable Hindi word. It is a filling error, not a
   judgement. Excluding it gives 39/46 = 84.8%. It was left in and flagged,
   because dropping rows that disagree with the rule is how an accuracy figure
   stops being one.

**Status of the number**, from `RESULTS.md`: reportable as the front end's
accuracy against elicited judgement, with the averaging and the citation-form
bias stated. **Not** reportable as a measure of the rule in connected speech,
and **not** comparable to published G2P accuracies, which are scored against
dictionaries rather than speakers.

Separately, and not one of the five: the stress-test set is **Hindi only**,
which constrains how the paper may word the claim. That is blocker b19.

A matching discovery: the worked examples in the elicitation sheet were leaking
test answers, and the sheet was redesigned after a pilot misread. Both are in
`RESULTS.md` under 14 September.

---

## 3. The corpora

**IndicTTS**, Hindi and Marathi. One male speaker per language, speaker 1, hours
matched.

| | utterances | hours |
|---|---|---|
| Hindi | 5,085 train / 100 dev / 300 test | 10.289 h after trimming (5.5% loss) |
| Marathi | 5,159 train / 100 dev / 300 test | 9.884 h after trimming (8.4% loss) |

Standardisation: resample, trim silence, loudness normalise to −23 LUFS.

**The splits are frozen and reproducible.** Membership comes from a salted
SHA-256 of the utterance id, so a rebuild reproduces `test.tsv` byte for byte,
and `SPLITS.lock` holds the checksums that `--verify` re-hashes against.
`RESULTS.md` lists four checks under "Structural checks, all passing": strict
superset, pairwise disjoint, rungs inside train, and the byte-for-byte rebuild.
(Some documents say "three structural assertions", counting the rebuild as
reproducibility rather than structure.)

`[Unverified]` The 1 October handoff also claimed all 16 split files were
verified identical on the DGX against the Mac — a different OS and filesystem.
**`RESULTS.md` records no such cross-machine check**, only single-machine
re-hashing. Treat the cross-machine claim as unconfirmed until someone re-runs
`--verify` on both and records it.

**The ladder's top rung is 9 h, not 10**: it is the largest round figure both
languages clear after trimming. The rungs are **nested subsets**: 1 h ⊂ 5 h ⊂
9 h. That matters for interpreting the training loss (see `04`).

**G2P coverage was asserted over the full corpora before any training**, per
language and per arm, with zero unknown symbols. `scripts/check_text_coverage.py`
checks that every symbol the front end can emit is in the vocabulary that trains.

---

## 4. The training design

### Fixed budget

Every acoustic run gets **100,000 steps at 12,000 mel frames per batch, bf16**.
The budget is the thing being held constant; no run gets a different one.

**A frame is not a fixed amount of audio**, and this bit. 12,000 frames × hop
256 = 3,072,000 samples per step. At 22.05 kHz that is 139.319728 s of audio;
at 16 kHz it is 192.00 s. Ratio **1.3781**. FastSpeech 2 runs at 22.05 kHz and
VITS at 16 kHz, so the VITS runs see 37.8% more audio for the same nominal
budget. The alternative — matching audio seconds — would have made the 16 kHz
budget 8,707 frames and invalidated every finished VITS run.

`[Unverified]` **The intent was to declare this in the methods table, and no
entry records the choice actually being taken.** `RESULTS.md` sets out three
routes and says the decision is Kaustubh's. The 1.3781 figure does appear in the
`corrections` field of every VITS config, so the substance is on record; the
methods table is not written yet. **Confirm the wording with the user before the
paper claims it.**

Epochs are inverse to rung size, since the step count is fixed:

| rung | steps/epoch | epochs |
|---|---|---|
| 9 h | 232.56 | 430 |
| 5 h | 129.20 | 774 |
| 1 h | 25.84 | 3,870 |
| 30 min | 12.92 | 7,740 |
| 10 min | 4.31 | 23,220 |

### No forced alignment

MFA was dropped. It would hand FastSpeech 2 externally supervised durations that
VITS never sees, inside a comparison whose subject is the architecture. It is
also asymmetric across languages — MFA publishes a Hindi model and no Marathi
one. **Alignment is learned inside every model** (FastSpeech 2 uses its internal
MAS aligner).

### No warm start for VITS, and a correction that was recorded rather than hidden

The VITS configs originally declared `init_from` an MMS checkpoint, and the
16 kHz sample rate was justified by that warm start. **The warm start never
happened** — the adapter never read the field. Discovered 3 October.

The decision (blocker b12, closed): **do not change the trained weights.** No
MMS warm start, no extra steps, no rate change. Either lever would have imported
or added training compute that the other architectures do not get, and the
equal-compute claim is worth more than better audio. The false `init_from` and
the 16 kHz justification were recorded in a **new non-hashed `corrections`
field** on every VITS run, so no finished run's provenance moved. Ten configs
carry three corrections each today — r02, r05, r11–r14, r16, r19, r22, r23.
`RESULTS.md` says "all eight" because it was written on 3 October, before r22
and r23 existed.

A startup assertion, `adapters.assert_init_from_is_honest`, refuses to run
when a config declares an `init_from` its adapter never reads **and no
`corrections` entry mentions `init_from`**. The VITS configs still declare
`init_from: facebook/mms-tts-hin` and still launch, because their corrections
field says the warm start never happened. That is intended, and it surprises
people, so it is worth saying plainly.

### The vocoder

FastSpeech 2 emits mel spectrograms and is silent without a vocoder. The chain
of decisions:

1. **2 Oct:** use a pretrained HiFi-GAN and declare the deviation. Reason: the
   acoustic runs alone need 54.08 h of wall clock (8 pairs at 6.76 h), and the
   vocoder pair would add 7.0 h — about the margin by which the full matrix
   overran the deadline.
2. **Then: no published HiFi-GAN matches this project's mel analysis.**
   `RESULTS.md` corrects an earlier entry that had blamed the frequency band
   alone, and names the three axes, with no family matching on all three:

   | axis | this project | jik876 / Matcha / BigVGAN | coqui default |
   |---|---|---|---|
   | band fmax | sr/2 = 11025 | 8000 | `None`, so sr/2 = 11025 |
   | amplitude | `log(max(mel, 1e-5))` | `log(clamp(mel, 1e-5))` — **identical to ours** | `20*log10` then normalise |
   | framing | librosa `center=True`, reflect pad 512 | `center=False`, reflect pad 384 | coqui's own |

   So the amplitude convention is already shared exactly with the jik876
   family, and coqui's band already matches ours. The framing difference is
   512 against 384 samples of reflect padding — 128 samples, 5.8 ms, half of
   one 11.61 ms hop. `[Unverified]` whether coqui's *released* `hifigan_v2`
   keeps `mel_fmax: None` in its shipped config; the default says it should,
   and `RESULTS.md` gives a two-minute command on the DGX that settles it.

   The general point stands whichever family you pick: a mel handed to a
   vocoder built on different analysis parameters produces audio that is
   recognisably speech and quietly wrong.
3. **Reversed, same day:** fit the vocoder to our mels by fine-tuning, keeping
   every trained parameter, rather than bending our mels to a vocoder. r06
   (Hindi) and r17 (Marathi) entered the matrix.
4. **One vocoder per language**, not one shared. The speaker differs by language,
   and a Hindi-fitted vocoder on Marathi audio would put a speaker mismatch
   inside the control that exists to isolate language.

The vocoder's mel front end is **checked, not assumed**: `src/export/vocoder.py`
verifies the analysis parameters against this project's and refuses on a
mismatch. `src/export/bundle.py` repeats the check when attaching a vocoder to a
bundle, on sample rate and language.

This project's mel: `log(max(mel_fb @ |stft|, 1e-5))`, N_FFT 1024, hop 256,
window 1024, 80 mels, fmin 0, fmax sr/2. **Magnitude** mel, not power.

### Seed variance floors

**No difference smaller than the measured seed variance is reported as a
finding.** That is a standing rule, and it required runs of its own: r20 and
r21 (FastSpeech 2 at seeds 1 and 2) and r22 and r23 (VITS at seeds 1 and 2).
A floor does not transfer between architectures, so each needs its own.

The VITS pair was sequenced first because VITS is end to end and can be scored
with no vocoder, which made the phonemic-versus-graphemic contrast the first
obtainable result rather than the last.

---

## 5. The evaluation design

Three layers of variance have to be separated, and the claim only stands if they
order correctly:

```
draw sd   (inference sampling, same weights)
  <  seed sd   (training randomness, same config)
      <  arm gap   (the effect)
```

### Intelligibility is the primary measure

Measured as **character error rate** on synthesised held-out sentences,
transcribed by automatic speech recognition.

**Two CTC recognisers, decoded greedily**, chosen deliberately:

- `ai4bharat/indic-conformer-600m-multilingual` (language arg `hi`)
- `facebook/mms-1b-all` (language arg `hin`)

**Why CTC and not a seq2seq model:** a seq2seq decoder carries an implicit
language model that repairs exactly the slurred-schwa errors being counted. A
weaker CTC model decoded greedily is the better instrument here — it is a more
faithful transducer of what was actually said.

**The recogniser's own floor is measured** on the real recordings, so the
synthesis is scored against the instrument's error rather than against zero.
Both arms sit 4.06 to 5.93 times the floor, which is the regime where the
recogniser has room to discriminate.

**Punctuation is stripped in one shared normalisation** before scoring and
before the site partition. This was a real defect: a trailing comma added a
segment and reclassified a word-final site as medial in 6 of 6 sampled words.

**Synthesis is seeded per (run, text, draw)** with a salted SHA-256, and the
harness takes `--draws`. Before that every number rested on one unseeded draw,
and one finding reversed sign on re-run.

**A paired bootstrap over utterances** gives the interval: one index draw
applied to both arms, same utterance order enforced, draws merged per utterance
first so the interval has the same characters the rate does.

### The other measures, and what each is worth

| Measure | Status | Note |
|---|---|---|
| Character error rate, site-partitioned, with floor and bootstrap | **primary, done for VITS** | `scripts/score_intelligibility.py` |
| Duration bias: ms of excess per deletion site | built | needs no vocoder and cannot saturate |
| MCD with DTW, against a chance level | built, **cannot resolve this contrast** | see below |
| F0 RMSE, voiced/unvoiced error | built | log-F0 is scale invariant, which is why it is log |
| MOS proxies (UTMOSv2, NISQA) | **not done** | must be validated on Devanagari or labelled English-trained |
| Stress-test error rates per architecture | **not done** | |
| macOS real-time factor on the M5 | **not done** | |
| Human listening test | **not done** | the user confirmed no consent or ethics process is needed; that is his statement, not an institutional record |

**MCD was 34× too large and cannot resolve this comparison anyway.** It is
reported against a chance level because the raw figure cannot be read alone.
This is the same lesson as the vocoder episode in `05`: a measure that cannot
resolve a contrast is not evidence about that contrast.

---

## 6. Standing rules

These were agreed and have not bent:

- Every acoustic run is 100,000 steps at 12,000 mel frames in bf16.
- `TRAIN_GPU_FRACTION` is a scheduling knob, never part of the budget.
- **No number reaches the paper that cannot be regenerated from a config file.**
- `RESULTS.md` is appended to, never rewritten.
- No difference smaller than the measured seed variance is reported.
- **Never pair the two halves of a comparison against different neighbours.**
  r01 ran with r04, r15 with r18, r16 with r19. The budget is steps rather than
  time so it would not matter, but the answer is easier to give when the
  schedule was symmetric by construction.
- The run is the unit of reproducibility: edit the config file, not the code.
- `config_hash` is a SHA-256 over all config fields minus
  `{notes, created, run_id, corrections}`. Changing a hashed field makes it a
  different run.

---

## 7. Scope, as it actually stands

**3 October: Hindi only for the paper. Marathi deferred to dissertation part 2.**
The user's words: *"lets do hindi only for now, as that will help me with the
research paper, and we can add marathi later for the dissertation part 2."*

The Marathi runs r15, r17 and r18 did train and their weights exist. They are
not in the paper.

**Matcha-TTS moved to future work on 2 October.** The adapter is written and
kept (`src/train/adapters.py`) and r03's config is on disk. It was never
trained. The import is lazy and **`matcha-tts` is not declared anywhere** — not
in `pyproject.toml`, not in `space/requirements.txt`. The version `0.0.7.2`
appears only in a comment in `adapters.py`. Anyone reviving r03 installs it by
hand.

**Consequence for the paper's threats section, still open as blocker b19:** the
resource claim and the front-end accuracy figure are Hindi-specific, because the
contested-schwa stress-test set is Hindi only and the ladder r07–r14 is all
Hindi. The user has been asked to confirm the wording and has not yet.

The user's constraint on scope, verbatim:

> *"what i want is a clear result set on what was promised initially, can make
> slight adjustments, but not a lot."*
