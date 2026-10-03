# Results log

Append only. Every number carries the commit that produced it. Nothing here is
a paper result until it is marked as such.

## 2026-09-13 — front end, provisional

Command: `python -m src.g2p.score stresstests/hindi_schwa_set.tsv --include-unvalidated`

    49/49 = 100.0%
    applies 18/18, blocked 5/5, cluster 4/4, final 6/6, loan 6/6, mono 4/4, nasal 6/6

**Not a result.** The rule implementation and the gold forms were written by
the same author in the same sitting, so this figure measures self-consistency,
not front-end quality. The scorer refuses to report a validated number because
zero of the 49 items have been checked by an independent speaker. The real
figure comes after that pass, scored without `--include-unvalidated`.

Unit tests: 26 passing, cloud container and device.

## 2026-09-13 — objective metrics, known-answer verification

MCD and F0 implemented and verified against answers derivable on paper rather
than against real audio. 13 metric tests, 39 in total, all passing.

The two that matter:

- **MCD is amplitude invariant.** Scaling a waveform by 0.25 changes MCD by
  less than 1e-6 dB. Scaling multiplies the power spectrum by a constant, which
  is a constant dB offset across mel bins, which lands entirely in the zeroth
  DCT coefficient. Dropping c0 therefore makes the measure blind to gain. If
  this ever fails, MCD is partly measuring loudness.
- **log-F0 RMSE is scale invariant.** A 10% pitch error reads as ln(1.1) at
  both 100 Hz and 300 Hz, exact to 1e-12. A linear RMSE would report 10 Hz and
  30 Hz for the same perceptual error and would let a low-pitched speaker
  dominate any average.

Two guards against silent bias, both tested: `compare` raises rather than
truncating mismatched F0 tracks, since truncation would favour systems that
predict shorter durations; and a comparison with no commonly-voiced frames
returns NaN rather than 0.0, so an empty comparison cannot look like a perfect
score.

## 2026-09-14 — access verified, corpora sized

Kaggle: `kaggle.json` present at mode 600, live `kernels list` authenticated as
`klewikkanel`. Hugging Face: authenticated as `Klewik`. GitHub:
`Klewik-Kanel/indic-tts-bench`, `main` tracking `origin`.

Corpus trial passes, 50 utterances each, zero decode failures:

| corpus | repo | rows | mean | estimated total | rate |
|---|---|---|---|---|---|
| Hindi | SPRINGLab/IndicTTS-Hindi | 11,825 | 6.70 s | 21.99 h | 48 kHz |
| Marathi | SPRINGLab/IndicTTS_Marathi | 10,939 | 7.56 s | 22.97 h | 48 kHz |

Hours estimated two independent ways from the 50-utterance sample; both agree.
The estimate assumes the first 50 rows are representative, which is exactly the
assumption the full export will test, so these figures are provisional.

Both corpora carry only `audio`, `text` and `gender`. There is no speaker id,
so "single speaker" can only mean "single gender", and the control arm needs
Hindi and Marathi matched on it. The 50-row samples show gender `0` throughout,
which tells us nothing about the distribution.

Source is 48 kHz, not the 22.05 kHz assumed in plan v2. Sampling-rate policy is
an open decision.

## 2026-09-14 — decisions and the validation instrument

**Sampling rate: 22.05 kHz master**, per the synopsis. Consequence recorded
here so it is not rediscovered later: MMS-VITS is natively a 16 kHz model, so
condition B is trained at 16 kHz and its output carries no energy above 8 kHz.
Full-band MCD would therefore penalise it for missing bandwidth rather than for
worse acoustic modelling. Mitigation, applied throughout: MCD is reported in
two columns, full band and 8 kHz band-limited, with the band-limited column as
the headline comparison; listening-test stimuli are low-passed to 8 kHz for
every system so raters hear equal bandwidth; ASR-WER is unaffected because the
recogniser resamples to 16 kHz regardless.

**Kaggle phone verification confirmed.** Accelerators available.

**Elicitation instrument built.** `src/g2p/elicit.py` generates a blind sheet
that shows each word with numbered empty slots where an unwritten schwa may sit
and never shows the rule's prediction. Speakers answer Y/N/V per slot rather
than writing IPA, because IPA transcription by non-phoneticians adds noise
unrelated to the question. The merge step turns their answers into gold forms
through the same converter, so the symbols come from the inventory and only the
judgement is human. Round-trip verified on two simulated sheets: 46 resolved,
1 flagged variable, 2 with no inherent schwa to elicit, 1 slot-level
disagreement detected.

## 2026-09-14 — full export and corpus profile

Command: `python -m src.data.profile`, output `results/tables/corpus_profile.json`.

| | utts | hours | speakers | suspicious transcripts |
|---|---|---|---|---|
| Hindi | 11,825 | 26.23 | 2 | 0 |
| Marathi | 10,939 | 22.37 | 2 | 0 |

**The 50-row extrapolation was wrong for Hindi.** Estimated 21.99 h, actual
26.23 h, 19% low. Marathi estimated 22.97 h against 22.37 h actual, 2.6% high.
The first 50 rows were not representative in Hindi. Recorded because it is the
reason those figures were labelled provisional, and the reason a full pass runs
before any number is used.

**Speaker identity recovered.** The corpora label speakers 0 and 1 with no key.
Median F0 over 25 sampled files each separates them cleanly and identically in
both languages:

| | speaker 0 | speaker 1 |
|---|---|---|
| Hindi | 198.9 Hz, likely female | 108.8 Hz, likely male |
| Marathi | 237.2 Hz, likely female | 125.7 Hz, likely male |

Inferred from pitch, not from a corpus label, and recorded as such.

**Length filter: 1 to 15 s, not 1 to 12 s.** At 1-12 s no single Hindi speaker
reaches the 10 h top rung (9.05 h and 8.96 h). At 1-15 s every candidate
speaker clears it:

| | 1-12s | 1-15s | 1-20s |
|---|---|---|---|
| Hindi female | 9.05 | 10.73 | 11.84 |
| Hindi male | 8.96 | 10.89 | 11.93 |
| Marathi female | 10.09 | 11.19 | 11.48 |
| Marathi male | 10.18 | 10.80 | 10.87 |

15 s at 22.05 kHz with hop 256 is 1,292 mel frames, comfortable for all three
architectures. The Hindi tail is long (max 127.7 s) but legitimate: the 127.7 s
item carries 1,600 characters, which is 12.5 characters per second, the corpus
median. Long reads, not broken rows.

**Transcript QA: clean.** Characters per second is tightly distributed in both
corpora (Hindi median 12.52, p1 9.16, p99 15.77; Marathi median 9.10, p1 6.64,
p99 11.70) and nothing falls outside 0.4x to 2.0x the median. Zero transcripts
look misaligned with their audio in either language.

**Speaker choice: male, speaker 1, in both languages.** Hours are equivalent
either way (10.89 and 10.80 male, 10.73 and 11.19 female). The male speakers
sit 17 Hz apart across the two languages against 38 Hz for the female pair, so
matching on male leaves less acoustic difference between the Hindi and Marathi
arms that has nothing to do with schwa deletion. Provisional pending review.

## 2026-09-14 — elicitation sheet redesigned after a pilot misread

The first reader of the sheet answered the wrong question, and did so in a way
that looked like a valid answer. On `नमकीन  n[1]m[2]kiin[3]` they read the
numbered blanks as asking whether a vowel exists anywhere in that syllable,
so slot 3 was nearly marked Y because of the written ii, and slot 1 was marked
N despite the word being pronounced namkeen with an audible a.

Both errors are silent: nothing in the data would have revealed them, and with
two or three speakers reading it the same way the resulting gold set would have
been confidently wrong.

Fixes: the candidate vowel is now spelled out in place as a numbered
parenthesised (a), so `n(a1)m(a2)kiin(a3)` asks a concrete question about a
specific sound; the instructions state that every plainly written vowel is not
in question; and three worked examples with answers sit at the top of both the
HTML and the TSV, including the exact नमकीन case that was misread.

Recorded because it is a reminder that an instrument needs piloting on a real
reader before it collects data, not just unit tests.

**Speaker choice confirmed: male, speaker 1, in both languages.**

## 2026-09-14 — worked examples were leaking test answers

The three worked examples on the sheet were घर, नमकीन and कमल. All three are
items in the 49. The sheet was therefore handing every participant the correct
answers to three of the words it was about to test them on, which would have
made those items unscoreable, and nothing in the merge would have flagged it.

Replaced with कपड़ा, बर्तन and पानी, none of which appear in the set. The three
now cover a deletion, a retention, and a word with no slots at all.
`check_examples_are_outside` runs at sheet-generation time and refuses to write
a sheet whose examples appear in the set, so this cannot recur silently.

Second point of procedure: the project owner has now seen worked answers for
नमकीन and समझना while learning to read the sheet, so they cannot serve as one
of the scored speakers. Coordinating rather than participating is the cleaner
role in any case.

## 2026-09-14 — standardisation and frozen splits

`src/data/prepare.py`, male speaker only, both languages:

| | utterances | hours | peak-limited | gain range |
|---|---|---|---|---|
| Hindi | 5,485 | 10.289 | 1 | -8.7 to +2.1 dB |
| Marathi | 5,559 | 9.884 | 0 | -9.2 to +2.8 dB |

Verified on a sample: both rates land at exactly -23.00 LUFS, the 22.05 kHz and
16 kHz copies agree on duration to under a millisecond, peaks stay under 0.995,
and the 50 ms edge padding is present. The peak guard fired once in Hindi, so
it is not dead code.

**Silence trimming cost more than plan v2 assumed.** Hindi went from 10.887 h
to 10.289 h, a 5.5% loss; Marathi from 10.795 h to 9.884 h, 8.4%. Marathi
carried noticeably more leading and trailing silence.

**The ladder's top rung is 9 h, not 10 h.** After trimming and after the frozen
test and dev sets, training audio is 9.53 h in Hindi and 9.16 h in Marathi. 9 h
is the largest round figure both clear. Matching the rungs across languages
matters more than a round 10: an unmatched top rung would put a data-quantity
difference inside the control comparison, which is the one place it must not be.

`src/data/splits.py`, frozen and checksummed:

| | test | dev | train | 9h | 5h | 1h | 30min | 10min |
|---|---|---|---|---|---|---|---|---|
| Hindi | 300 (0.579 h) | 100 | 5,085 (9.525 h) | 4,793 | 2,668 | 541 | 265 | 86 |
| Marathi | 300 (0.542 h) | 100 | 5,159 (9.160 h) | 5,071 | 2,810 | 562 | 279 | 90 |

Structural checks, all passing: every ladder rung is a strict superset of the
one below it; test, dev and train are pairwise disjoint; every rung lies inside
train; and a rebuild reproduces `test.tsv` byte for byte, because membership
comes from a salted SHA-256 of the utterance id rather than a seeded shuffle of
whatever order the manifest happened to be written in.

`SPLITS.lock` records the salt and a checksum per file, and `--verify` re-hashes
them. Rebuilding without `--force` is refused.

## 2026-09-14 — G2P coverage over the full corpora

`python -m src.data.g2p_coverage`, all 11,044 transcripts. Both languages now
reach **zero unknown symbols**. The first pass did not: 4.1% of Hindi phone
tokens and 2.7% of Marathi had no mapping.

**The serious find was a bug I introduced.** Keeping punctuation as a prosody
token left it attached to the word through the schwa stage, which broke the
rule twice over. A trailing comma means the word-final schwa is no longer
word-final, so it survives; and its survival supplies the right-hand vowel that
lets the schwa to its left delete. नमक is /nəmək/ but नमक, came out /nəmkə/, a
different word. 16.5% of Hindi word tokens and 14.9% of Marathi carry
punctuation, and the corruption would have landed on the phoneme arm alone,
so the phoneme-versus-grapheme comparison would have been measuring the bug.
Punctuation is now peeled off the edges before the schwa stage and restored
after, with a parametrised invariance test over nine words and four marks in
both languages.

Other fixes, each found by running over real text rather than the 49-word set:

- 9 Hindi tokens carried a nukta on a consonant with no nukta form (व़क्त for
  वक़्त). The stray mark also broke the parse of the following character.
- ऱ RRA, 199 Marathi tokens, only ever in the rya cluster; folds to र.
- आॅ and अॉ, keyboard slips for ऑ.
- A colon inside a Devanagari word is a typed visarga (स्वत:च for स्वतःच).
- ॐ is a ligature for a syllable, not a letter; expands to ओम्.
- Southern short vowels ऒ ऎ and the vocalic-RR matra, 6 tokens, fold onto
  their long counterparts.
- Orphan matras with no consonant to attach to are dropped and noted.
- Digits: 15 Marathi utterances, none in Hindi. `src/g2p/numbers.py` expands
  0-20, exact tens, 21-25, and round hundreds and thousands. Anything outside
  that raises rather than guessing: a wrong number word would be a
  pronunciation error baked into training data and invisible afterwards.

Punctuation is kept rather than stripped, folded to four marks (, . ? !). It is
the only prosodic signal the text carries and it covers roughly one token in
six.

**Inventory.** 75 declared symbols; Hindi uses 72, Marathi 67. Marathi's
inventory is a strict subset of Hindi's, missing rĩ x ɔː̃ ɣ ɽ, all Perso-Arabic
or marginal. Nothing appears in Marathi that is absent from Hindi, so the
control comparison sits in one symbol space with no Marathi-only embedding rows.

Rare phones are recorded as a known weakness: Marathi has ɽʱ once, z once and
q twice, which are embedding rows with almost no gradient. Setting
`merge_nukta=True` for Marathi is the obvious mitigation and is left as a
decision for the training configuration rather than being applied silently here.

## 2026-09-14 — run configs, and forced alignment dropped

`src/train/config.py` generates every run as a YAML file in `configs/`. The run
is the unit of reproducibility: a results row carries its run id and config
hash, so any number in the paper traces back to exact settings.

Two assertions run at generation time. `assert_budget_matched` requires every
acoustic-model run to share max_steps, batch_frames, lr, schedule, warmup,
precision and grad clip; if it ever fails the comparison is between
architectures AND budgets and the results table means nothing. A second check
refuses two runs whose settings hash identically under different ids.

**That second check immediately paid for itself.** The plan had 19 runs. The
ladder's 9 h rung was a second copy of the main run: r01 and r07 trained the
same architecture on the same data under the same budget, and would have cost
10 GPU-hours to produce a number we already had. The duplication was invisible
while run_id was inside the hash, because that made two identical runs look
distinct. run_id is now excluded, the ladder starts at 5 h, and the plan is
**17 runs and 136 GPU-hours**, down from 19 and 146.

**Forced alignment is dropped. Alignment is learned inside every model.**

Three reasons, in order of importance.

It was an uncontrolled asymmetry. VITS and Matcha learn alignment internally
through monotonic alignment search; giving FastSpeech 2 externally supervised
MFA durations would hand one architecture information the others never see,
inside a comparison whose entire subject is the architecture.

It was asymmetric across languages as well. MFA publishes a pretrained Hindi
acoustic model but none for Marathi, so the Marathi arm would have used an
aligner trained on 9 h of its own data while Hindi used one trained on far
more. That difference would have sat directly inside the control comparison.

And it cannot run here regardless: `montreal-forced-aligner` pip-installs, but
its Kaldi bindings (`_kalpy`) have no aarch64 Linux wheel and fail to build,
since MFA ships through conda. Verified, not assumed.

FastSpeech 2 therefore uses an internal unsupervised aligner of the kind in
Badlani et al., "One TTS Alignment to Rule Them All" (ICASSP 2022), which is
what current implementations do anyway. Limitation to report rather than hide:
an internal aligner is generally weaker than MFA at very small data sizes, so
the lowest ladder rungs may be affected more than the top ones. That is a
property of the ladder to measure, not a reason to reintroduce the asymmetry.

## 2026-09-14 — Kaggle smoke test built

`src/kaggle/job.py smoke` generates a job directory; `scripts/05_kaggle.sh`
pushes, polls and fetches it from a native terminal. Claude cannot reach
kaggle.com, so it builds the job and reads the result, and the round trip runs
on the user's machine.

The smoke test exists because every assumption behind the 136 GPU-hour plan is
still unverified. It checks, for a few minutes of quota:

- **A GPU is actually attached.** An account without phone verification runs
  CPU-only and says nothing. On a 12-hour run that failure looks identical to a
  job that never finishes.
- **Which GPU, and how fast.** A fixed fp16 matmul, so T4 and P100 sessions are
  comparable and the step budget comes from a measurement rather than a guess.
- **Internet and Hugging Face authentication** from inside the notebook.
- **A checkpoint survives a round trip.** Push 1 MiB to the Hub, delete it
  locally, pull it back, compare SHA-256. Resume-from-checkpoint is what caps
  the cost of a killed session at N steps instead of a whole run, and an
  untested resume path is not a safety net.

Each stage reports independently and the result is written to
`smoke_result.json` in the kernel output, so a partial failure says which part
failed rather than just failing.

One manual step: the notebook reads `HF_TOKEN` from Kaggle Secrets, and secrets
are attached through the web UI, not the API. First push, attach, push again.

## 2026-10-01 — front end scored against speaker judgement

Commands:

    python -m src.g2p.elicit merge stresstests/hindi_schwa_set.tsv \
        "stresstests/filled/*.tsv"
    python -m src.g2p.score stresstests/hindi_schwa_set_validated.tsv

    front-end accuracy (validated subset)
      39/47 = 83.0%

      applies 17/18  94.4%      blocked 2/5  40.0%
      cluster  3/4   75.0%      final   6/6 100.0%
      loan     5/6   83.3%      mono    2/4  50.0%
      nasal    4/4  100.0%

This replaces the 49/49 self-consistency figure of 13 September. It is the
first number in the project that was not produced by checking the author's rule
against the author's own answer key. Two of the 49 words have no inherent schwa
to elicit, so the denominator is 47.

**One sheet, not three.** The responses of three speakers were averaged into a
single sheet before I saw them, so the merge ran with one input and printed its
warning about agreement. Three consequences, all of which weaken the claim and
none of which can be repaired from the averaged file. Inter-speaker agreement
cannot be computed. The variation rate is reported as zero, which is an
artefact of the averaging rather than a finding, since no slot came through
marked V. And any slot where the three genuinely split was silently resolved by
whoever did the averaging. The per-speaker sheets are the thing to keep if this
is ever rerun.

**The errors point one way.** Seven of the eight mismatches are the speakers
keeping a schwa that the rule deletes. Only one runs the other way. That is the
signature of citation-form reading: a word read aloud on its own, slowly, keeps
vowels that the same speaker drops in connected speech. The sheet asks for a
normal conversational pace, but a list of 49 isolated words is close to the
worst possible condition for getting one. So 83.0% is a lower bound under
reading conditions that favour retention, not an estimate of the rule's
accuracy in running speech.

**Four of the eight are one phonological environment.** राष्ट्र, कृष्ण, धर्म and
सत्य all keep a word-final schwa after a consonant cluster, and all four are
Sanskrit-derived. Hindi is known to retain final schwa in exactly that
environment. `SchwaConfig.block_before` exists for this and is deliberately
empty, so the baseline rule is reported unmodified. Populating it from these
four words would fit the model to the test set and the resulting figure would
mean nothing. The fix needs a held-out set of tatsama words that this scoring
run has never seen. With those four counted correct the accuracy would be
43/47 = 91.5%, which is what the fix is worth if it generalises.

**Two are ordinary citation-form retention.** आदमी came back as /aːd̪əmiː/ and
फ़ैसला as /fɛːsəlaː/. Both delete in normal speech. कल came back as /kələ/,
which is the same effect on a word short enough that it should not have
happened.

**One answer is impossible and should not be in the denominator.** न was
answered N, which yields a gold form of /n/: a consonant with no vowel, not a
pronounceable Hindi word. It is a filling error, not a judgement. Excluding it
gives 39/46 = 84.8%. I have left it in the reported figure and flagged it here
rather than quietly dropping the row, because dropping rows that disagree with
the rule is how an accuracy figure stops being one.

Tests: 71 passing. The 7 failures are `ModuleNotFoundError: librosa` in the
device VM, which has no audio stack; the 65 non-audio tests pass there and the
metric tests pass in the cloud container.

**Status of this number.** Reportable as the front end's accuracy against
elicited judgement, with the averaging and the citation-form bias stated. Not
reportable as a measure of the rule in connected speech, and not comparable to
published G2P accuracies, which are scored against dictionaries rather than
speakers.

## 2026-10-01 — training loop built, resume proven bit-exact

Phase 3's first half. No GPU was involved and none was needed: every failure
this looks for is a failure of the machinery, not of the model, and all of them
cost a Kaggle session if found late.

Six modules, one per thing that can be wrong on its own.

`src/train/text.py` builds the vocabulary from the declared inventory rather
than from the corpus, so every ladder rung shares one embedding table and the
10-minute run is not quietly a different model from the 9-hour run. 78 phoneme
symbols, 136 grapheme symbols. An unknown symbol raises instead of becoming
`<unk>`, because the coverage run already proved there are none and a silent
substitution would hide a regression in the front end.

`src/train/batching.py` batches to a frame budget, not an utterance count. The
budget is charged as `len(batch) * max_frames`, which is what the GPU actually
computes, so one long utterance in a batch of twenty cannot blow past the
memory the budget is meant to bound. On the Hindi 9 h set at 12,000 frames:
265 batches an epoch, median 16 utterances, largest 53, no batch over budget,
no utterance too long to batch.

`src/train/schedule.py` implements the one learning-rate schedule all four
architectures share, instead of inheriting three upstream implementations of
it. Continuity at the warmup join is the property worth testing, and it is:
both branches give exactly 2e-4 at step 4,000.

`src/train/checkpoint.py` keeps the last two checkpoints and every 25,000th,
writes to a temporary directory and renames, and marks completion only after
the rename. A directory without its marker is visibly unfinished and resume
skips it. Pruning moves into `_trash` rather than unlinking, because deletion
inside a connected folder fails and an operation that silently frees no space
is worse than one that moves it.

`src/train/runner.py` is the loop. torch is imported lazily, so the vocabulary,
the batching and the checkpoint logic all import and test in an environment
with no deep-learning stack, which is where they were written.

`src/train/adapters.py` holds one adapter per architecture plus a toy adapter
with real parameters and a real masked loss. `assert_not_toy` refuses to let a
config that names it produce a number.

**The resume result.** `python -m src.train.dryrun configs/r01.yaml --steps 20
--kill-at 10` runs to step 10, stops as if the session was killed, restarts
from the checkpoint, and then compares against a reference run that was never
interrupted. The comparison is on the weights, not the loss curve: a nearly
correct resume produces a nearly correct curve, which is exactly the defect
that survives a glance.

    data order: resume at 10 reproduces the reference sequence for 10 steps
    weights:    bit-identical to an uninterrupted 20-step run across 11 tensors

Bit-identical, not close. The data order is derived from (seed, step) rather
than stored in the checkpoint, so a checkpoint cannot disagree with the
manifest it was trained on, and the optimiser state travels with the weights,
so Adam's moments do not restart from zero and leave a transient in the loss
that would later be indistinguishable from a real effect.

Running the same dry run against `configs/r02.yaml` stopped with a missing
`wav16` file, which is the correct behaviour: r02 is the 16 kHz MMS-initialised
VITS run, the manifest loader selected the 16 kHz column from the config's
sample rate, and only the 22.05 kHz copies had been staged. The rate selection
is in one place and no training script has to know about it.

Tests: 25 new, 90 passing without an audio stack, 96 with one.

**What this does not show.** The three real adapters raise `NotImplementedError`
by design. FastSpeech 2, VITS and Matcha-TTS are constructed from upstream
packages inside the Kaggle job, where those packages and their pretrained
checkpoints are reachable; vendoring them into this repository to satisfy a
local import would be pretending to a verification that has not happened. The
loop, the budget, the schedule, the data order and the resume path are proven.
The models are not yet built.

## 2026-10-02 — vocoder decision: pretrained HiFi-GAN

**Decided: synthesise with a pretrained HiFi-GAN. r06 and r17 leave the
matrix.** The deviation goes in the methods table rather than being absorbed
quietly: the vocoder in this study is not trained on the corpus and is not
matched to the speaker.

Reasoning kept here so it is not re-argued later. The acoustic runs alone need
54.08 h of wall clock (8 pairs at 6.76 h); the vocoder pair would add 7.0 h, and
7.08 h is exactly the margin by which the full matrix overran the Sunday
deadline. The vocoder is also shared by both arms of every comparison in the
matrix, so it cancels in the phonemic-versus-graphemic contrast: it shifts
absolute MCD, not the measured effect.

That last point is an expectation, not a measurement. It has not been tested
here, and a pretrained vocoder's speaker and language mismatch could interact
with the two arms differently. It is checked, not assumed, once r01 and r04 have
audio: the same mels through the same vocoder, with MCD reported for both arms.

**Open and blocking, before any audio from it is scored.** The checkpoint's mel
configuration must match this project's: 22.05 kHz, hop 256, FFT 1024, window
1024, 80 mels, fmin 0, fmax sr/2. A mismatch on any of these produces audio that
sounds plausible and scores wrongly, and nothing downstream would reveal it.
`src/export/bundle.py` already asserts the hop against `batching.HOP_LENGTH`;
the rest of the parameters need the same treatment before the first synthesis.

**Matcha-TTS stays in the matrix.** r03 is kept, so `MatchaAdapter` is still to
be written. The architecture count in the paper stays at three.

## 2026-10-02 — the DGX is shared, and the step rate is not ours to set

Diagnosing why the ladder rungs slowed from 4.11 it/s to about 1.3:

    GPU            99% utilisation, 1410 of 1410 MHz, 58 C, 283 of 400 W
    throttling     none: clocks_event_reasons.active 0x0
    on the card    three processes: 8976, 10098 and 11192 MiB
    host           load average 293.82 on 256 cores
    /workspace     Lustre over tcp, 52T of 56T used, 95% full

**A third process is on the GPU and it is not ours.** Two runs were launched,
each capped by `TRAIN_GPU_FRACTION=0.25`, which on this card is 10,084 MiB. The
third process holds 11,192 MiB, which is above that cap, so it cannot be one of
ours. The card was at 99% utilisation with no throttling while our runs crawled,
which is what sharing looks like from inside a container: `nvidia-smi` reports
every process as `[Not Found]` because the names are in another PID namespace,
and `ps` sees only our own.

The load average tells the same story. Our two trainers accounted for about 8
cores of a 293.82 load on 256 cores, so roughly 285 cores of work belong to
someone else. The handoff's claim that the DGX is "idle apart from this work" was
true when it was written and is not true now.

**Consequences for the schedule, which matter more than the cause.** Every
wall-clock figure in `PLAN-full-matrix.md` descends from 4.11 it/s measured on an
idle card. That number is not a property of the run; it is a property of the card
at the time, and it is now unreliable in both directions. The step budget is
unaffected: 100,000 steps is 100,000 steps whoever else is on the machine, and
nothing about the comparison between runs changes. Only the calendar does.

Recorded rather than fixed, because no change on our side can reclaim another
tenant's share. What our side can do is read less: the feature-cache change of
`ffeb64d` reads mel, pitch and energy instead of whole entries including the
spectrogram and the waveform, which matters more on a 95%-full shared Lustre
mount than it did on the local assumption it was written against.

**Open question for the lab, not for the code.** Whether this DGX is scheduled or
first-come. If runs are going to share it routinely, pairing two of our own runs
on one card is the wrong default, and the deadline arithmetic needs a measured
contention factor rather than an idle-card rate.

## 2026-10-02 — the fixed budget is fixed in frames, and a frame is not a fixed amount of audio

Found while writing the adapter parity check that `PLAN-phase3.md` §4 asks for.
Not a code defect: every line involved does what it says. It is a confound in
what the budget means.

    batch_frames        12,000       identical in all 19 configs
    assert_budget_matched            enforces that the NUMBER is identical
    frames_for(s, sr)   ceil(s * sr / 256)

A frame is 256 samples at the run's own sample rate. FastSpeech 2 and Matcha run
at 22,050 Hz, and every VITS run is 16,000 Hz, which is a declared deviation for
the MMS initialisation. So the same 12,000 frames is:

    22,050 Hz     86.133 frames/s     12,000 frames = 139.32 s of audio per step
    16,000 Hz     62.500 frames/s     12,000 frames = 192.00 s of audio per step

    ratio 192.00 / 139.32 = 1.3781, which is exactly 22,050 / 16,000

**Every VITS run sees 37.8% more audio per step than its FastSpeech 2
counterpart, at a budget the code certifies as identical.** Over 100,000 steps
that is 3,870 h of audio against 5,333 h. `assert_budget_matched` passes because
it compares the integer 12,000, and the integer is the same.

**What this does not touch.** The dissertation's question is phonemic versus
graphemic input, and that contrast is measured within one architecture: r01
against r04, r02 against r05, and each ladder pair. Those arms share a sample
rate, so they share batch composition exactly, which the new parity test now
asserts. The central claim is unaffected.

**What it does touch.** Any statement that compares FastSpeech 2 with VITS
directly, including the two ladders against each other, since the VITS ladder
rungs are trained on more audio per step at every rung.

**Three routes, and this is Kaustubh's to decide, not mine.**

1. Declare it. The budget is "12,000 frames at the run's native rate", the
   asymmetry goes in the deviations table with the 1.3781 figure, and
   cross-architecture comparisons are reported with it stated. Costs nothing,
   no reruns, and the limitation is explicit.
2. Equalise the audio. A 16 kHz run would need `batch_frames = 8,707` for the
   same 139.32 s per step. This changes the budget definition from frames to
   seconds, so `assert_budget_matched` has to compare audio seconds rather than
   the frame integer, and every VITS run already finished would need redoing.
3. Equalise nothing and drop the cross-architecture claim, leaving architecture
   as a factor the study holds constant rather than compares.

Route 1 is the only one that costs no GPU time, and r01 and r04 are already
finished under the current definition.

Recorded now because the number is needed before any cross-architecture result
is written down, not after. `tests/test_adapter_parity.py` pins 139.32, 192.00
and 1.3781, so whichever route is taken, changing the design fails a test rather
than silently changing what the paper compares.

## 2026-10-02 — the vocoder's mel front end is checked, not assumed

Closes the item the vocoder decision left open and blocking. `--vocoder` took a
bare string, recorded it in the manifest, and read nothing.

`src/export/vocoder.py` reads a vocoder config from a json file, a directory, or
a checkpoint's embedded config, which is the only copy certainly matching the
weights, and compares seven parameters against `features.py`:

    sample_rate  22050      n_fft 1024    win_length 1024    hop_length 256
    n_mels 80               fmin 0.0      fmax 11025.0 (sr/2)

Both published key conventions are read: coqui's `audio.fft_size`, `num_mels`,
`mel_fmin`, `mel_fmax`, and jik876's `n_fft`, `hop_size`, `win_size`, `fmin`,
`fmax`. A null `mel_fmax` is honoured as sr/2, which is what coqui means by it.
Absence of any other parameter is a mismatch rather than a pass: a config that
does not state its hop cannot be checked, and assuming the convenient answer is
the failure this file exists to prevent.

`bundle.py --vocoder` now takes a path and raises on a mismatch, so a bundle
with a mismatched vocoder is never written. A bare name is still accepted and
recorded as `mel_verified: false`, so a later reader can tell "checked and
matching" from "nobody looked". Manifest version 2.

20 assertions pass in `tests/test_vocoder_mel_match.py`, including the one worth
naming: [Inference] HiFi-GANs published for LJSpeech commonly cap the mel filter
bank at 8 kHz while this project builds to 11,025 Hz. Sample rate, hop, FFT and
window all agree in that case and only the top of the spectrum is analysed
differently, which is the hardest version to hear. Not verified against a real
checkpoint, because none is in hand yet.

Two incidental guards came with it. `project_mel` asserts `features.HOP ==
batching.HOP_LENGTH`, two constants that must hold the same number: one computes
the frames and the other is what `src/eval/mcd.py` aligns against. And a test
asserts that `features.FMIN_HZ, FMAX_HZ = 60.0, 600.0` never become mel band
edges: they are the pitch tracker's search range for one adult speaker, and
wiring them into the filter bank would make every mel in the project wrong
rather than only the vocoder's.

Still to do: obtain a checkpoint and run `python -m src.export.vocoder <path>`.
Nothing is a result until that passes and the checkpoint's identifier is
recorded here.

## 2026-10-02 — no published HiFi-GAN matches this project's mel band

The 8 kHz warning of earlier today was labelled an inference. It is now
measured, and it is not one checkpoint's quirk.

    source                                      sr     n_fft  hop  win   mels  fmin  fmax
    this project (features.py)                   22050  1024   256  1024  80    0     11025
    jik876/hifi-gan config_v1.json               22050  1024   256  1024  80    0     8000
    jik876/hifi-gan config_v2.json               22050  1024   256  1024  80    0     8000
    jik876/hifi-gan config_v3.json               22050  1024   256  1024  80    0     8000
    Matcha-TTS's bundled HiFi-GAN (hifigan/config.py)   same shape          0     8000
    Matcha-TTS's own training mel (data/ljspeech.yaml)  same shape          0     8000

Every parameter agrees except the top of the mel filter bank. This project
builds to sr/2; everything published builds to 8 kHz. About 22 of the 80 bands
lie above 8 kHz, so a vocoder trained on the 8 kHz convention has never seen
roughly a quarter of what our acoustic models predict, and band k means a
different frequency in the two conventions.

**Why this happened, which matters more than the number.** `fmax = sr/2` is a
defensible choice when you train your own vocoder, and r06 and r17 were exactly
that. Dropping them on 2 October to save 7.0 h removed the only component that
was going to be fitted to this mel convention. The saving created an
incompatibility with every off-the-shelf vocoder, and that was not visible at
the time the trade was made.

**Scope.** 9 of the 17 runs are mel-based and need a vocoder: r01, r03, r04,
r07, r08, r09, r10, r15, r18. The 8 VITS runs are end to end and are not
affected at all. Of the 9, four have already trained on sr/2 mels: r01 and r04
are complete, r07 and r08 are mid-flight.

**Routes, with costs.**

1. Move the project to `fmax = 8000` and recompute the feature cache. Every
   published vocoder then matches, and Matcha's LJSpeech initialisation becomes
   legitimate rather than a representation mismatch. Costs retraining the four
   runs already on sr/2 mels: 2 pairs, 39.2 h at the contended rate or 13.5 h
   on an idle card, plus a cache rebuild whose cost is unmeasured.
2. Keep sr/2 and train a vocoder after all, which reinstates r06 and r17 and
   the 7.0 h that were saved by dropping them.
3. Keep sr/2 and convert at synthesis time: map our 80 bands to the vocoder's
   convention through the linear spectrogram with a pseudo-inverse of our
   filter bank. No retraining, but it is a lossy step inside the audio path of
   every mel-based run and it would have to be measured and declared.
4. Keep sr/2 and report only VITS audio, with FastSpeech 2 and Matcha compared
   on training-time metrics alone. Cheapest, and it gives up the architecture
   comparison the vocoder decision was meant to protect.

Route 1 is the only one that ends with every number computed through the
convention everything else in the field uses. It is also the one that admits
the 2 October saving was not a saving.

Not decided here. `tests/test_vocoder_mel_match.py` and the new Matcha
statistics guard both pin the current value, so whichever route is taken, the
change is deliberate and breaks a test rather than passing silently.

## 2026-10-02 — Matcha-TTS adapter written

r03's adapter exists, so `NotImplementedError` is gone from the matrix. Written
against matcha-tts 0.0.7.2's actual API rather than against its documentation:

- `MatchaTTS.get_losses(batch)` takes `x, x_lengths, y, y_lengths, spks,
  durations` and returns duration, prior and flow-matching losses separately.
  Upstream's `training_step` optimises their unweighted sum, so the adapter
  returns `sum(...)`. A weighting would be a modelling decision and would
  belong in the config and the deviations table, not in Python.
- `y` is `[B, n_feats, frames]`, channels-first like VITS's spectrogram and
  unlike FastSpeech 2's `[B, frames, mels]`.
- `encoder` and `cfm` are read by attribute, because upstream passes OmegaConf
  nodes, while `decoder` is splatted as `Decoder(..., **decoder_params)`. Those
  two shapes are not interchangeable, and the difference is visible only by
  reading `flow_matching.py`. A dict for the first or a namespace for the second
  fails at construction.
- Hyperparameters are copied from upstream's shipped configs, not chosen here:
  RoPE encoder, 192 channels, 6 layers, 2 heads; decoder channels [256, 256],
  num_heads 2, act_fn snakebeta. `Decoder`'s own code defaults are num_heads 4
  and act_fn "snake", and the shipped config overrides both, so published
  Matcha is the config rather than the defaults.

Two preconditions it refuses to guess at.

**Corpus mel statistics.** Matcha normalises its mel targets by a scalar mean
and standard deviation of the corpus, held as buffers. Upstream ships
LJSpeech's (-5.536622, 2.116101) and defaults to (0.0, 1.0) when none are
given. Either would train happily against the wrong centre and scale.
`src/train/melstats.py` computes ours from this project's own cache in one
pass, records the mel parameters it used, and `MatchaAdapter.mel_stats` refuses
a file computed under different parameters, because a log-mel's mean is a
property of the filter bank as much as of the audio.

**Initialisation.** `r03.yaml` says `init_from: matcha_ljspeech`, which names
nothing on disk, so the adapter stops and says so rather than silently training
from scratch, which would be a budget change. And even with a real checkpoint
it refuses while the mel bands differ: those weights were fitted to an 8 kHz
bank, so starting from them under sr/2 is a different model, not a warm start.
That is the mel-band decision above, reaching r03 from a second direction.

18 assertions pass without torch, the matcha package or audio; the collate test
needs all three and runs on the DGX. **Unverified:** that `matcha-tts` installs
beside `coqui-tts` 0.27.5 in the DGX venv. It pulls lightning, torchvision,
torchmetrics and torchaudio, and its monotonic-alignment search is a Cython
extension that builds at install. Nothing here has been run against the real
package.

## 2026-10-02 — correction: the mel mismatch has three axes, not one

The route list in the entry above treated the mel band as the only difference.
Reading the actual analysis code of each family shows three axes, and no family
matches on all three. This supersedes that list.

    axis          this project            jik876 / Matcha / BigVGAN   coqui (default cfg)
    band fmax     sr/2 = 11025            8000                        None, so sr/2 = 11025
    amplitude     log(max(mel, 1e-5))     log(clamp(mel, 1e-5))       20*log10 then normalise
                                          IDENTICAL to ours           by ref_level_db etc
    framing       librosa center=True,    center=False, reflect pad   coqui's own
                  reflect pad 512         384

Verified: `features.py` against `matcha/utils/audio.py` (which is jik876's code
copied), `TTS/vocoder/configs/hifigan_config.py` (mel_fmax None at two places),
and `TTS/utils/audio/numpy_transforms.py`, whose `build_mel_basis` passes
`fmax=mel_fmax` straight to librosa, where None means sr/2. BigVGAN's
`bigvgan_22khz_80band` config was read from Hugging Face: fmax 8000.

So the amplitude convention we already share with the jik876 family, exactly,
and coqui's band already matches ours. The framing difference is 512 against
384 samples of reflect padding, which is 128 samples, 5.8 ms, half of one
11.61 ms hop.

**[Unverified]** Whether coqui's *released* `vocoder_models/en/ljspeech/
hifigan_v2` keeps `mel_fmax: None` in its shipped config. The default says it
should. The release gateway is not reachable from here, and coqui's own
downloader on the DGX settles it in about two minutes:

    python -c "from TTS.utils.manage import ModelManager as M; \
      print(M().download_model('vocoder_models/en/ljspeech/hifigan_v2'))"
    python -m src.export.vocoder <the config.json that prints>

**Timing, which is the part that decays.** Four runs are committed to the sr/2
convention: r01 and r04 complete, r07 and r08 running with about 14.8 h left
each. Five mel-based runs have not started. Deciding before r07 and r08 finish
costs nothing extra; letting them finish and then changing the convention
spends that time for nothing. The eight VITS runs are unaffected whatever is
decided.

## 2026-10-02 — fit the vocoder to our mels, rather than our mels to a vocoder

Constraint stated: the trained parameters are not to be discarded, GPU access is
finite, and the study's credibility comes first. Those three pick one route, and
it is the one that was dropped this morning, warm-started.

**Reinstate r06 and r17 as fine-tuning, not as training from scratch.**

    already in the bank, preserved     r01 6.83 h + r04 6.73 h complete,
                                       r07 and r08 3.7 h each in flight = 21.0 h
    retrain to an 8 kHz convention     39.2 h contended, and discards those 21.0 h
    fit the vocoder to our convention  19.6 h contended at full length, discards nothing
                                       7.0 h on an idle card, the figure used on 2 Oct

Half the GPU cost of the retraining route, and nothing already computed is
thrown away.

**Why this is the stronger position scientifically, not a compromise.** The
three-axis mismatch disappears rather than being worked around: a vocoder
fitted to our own ground-truth mels is fitted to our band, our amplitude
convention and our STFT framing at once, whatever they happen to be. The
pretrained route needed all three to coincide. This one needs none of them to.

It also deletes a deviation instead of adding one. "Vocoder not trained on the
corpus and not matched to the speaker" was recorded this morning as the price of
the decision. Fine-tuning on the corpus removes that sentence from the methods
table, and a corpus-matched vocoder is the ordinary expectation in the TTS
literature rather than something needing defence. The alternative that costs no
GPU, converting between mel conventions with a pseudo-inverse at synthesis
time, would have put a lossy step that nobody else uses inside the audio path of
nine runs, and would have needed its own error analysis to be believed.

**The warm start is genuinely close**, which is why this is cheaper than the
7.0 h the original pair was budgeted at. Our amplitude convention is already
byte-for-byte what jik876, BigVGAN and Matcha use: `log(clamp(mel, 1e-5))`, no
dB, no normalisation. Only the band and 128 samples of padding differ, so a
published checkpoint starts near the right answer and has to move in two
parameters' worth of ways, not in all of them.

**Step count is to be measured, not assumed.** [Inference] Warm-started HiFi-GAN
fine-tuning on one speaker converges in thousands of steps rather than the
100,000 a fresh run takes, but that is the literature's general behaviour and not
a measurement from this corpus. So the run gets a stopping criterion rather than
a step budget: mel reconstruction error on held-out ground-truth audio, stopped
when it plateaus, and the step count it reached is reported. This is the one
place where the fixed-budget rule does not apply, because the vocoder is not an
object of comparison in the study. That exemption gets stated in the methods
table rather than left implicit.

**What unblocks immediately, before any vocoder exists.** The mel-domain
metrics need no vocoder at all: predicted against ground-truth mel distortion
with DTW alignment, and duration or alignment accuracy. r01 and r04 can be
scored on those today. What waits for the vocoder is anything computed from a
waveform: ASR-WER, the UTMOSv2 and NISQA proxies, the stress-test error rates
and the listening test.

**Open, and smaller than before.** One vocoder per language or one for both.
r06 and r17 existed separately because the speaker differs, and using a
Hindi-fitted vocoder on Marathi audio would put a speaker mismatch inside the
control that exists to isolate language. Two seems right for that reason, and it
is the difference between 19.6 h and roughly half of it.

## 2026-10-02 — Matcha to future work, and the rest of the matrix queued

**Matcha-TTS (r03) leaves the matrix.** Commented out in `plan_runs` rather than
deleted, with the adapter, its 18 tests and `configs/r03.yaml` all kept, so this
is reversible by uncommenting three lines. It joins StyleTTS 2 in a named
`FUTURE_WORK` dict in `config.py`, so the paper's future-work section is
generated from the repository rather than remembered.

Matrix after the change: 18 runs, 8 FastSpeech 2, 8 VITS, 2 vocoders, and
`assert_budget_matched` still passes. The planning estimate drops from 152 to
140 GPU-h, which both arithmetic routes agree on.

The reason is compute, stated plainly: Matcha needs its package installed beside
coqui-tts, a corpus mel-statistics pass, and 100,000 steps from scratch, because
its published LJSpeech weights were fitted to a mel bank reaching 8 kHz while
this project builds to sr/2. The dissertation's question is phonemic versus
graphemic input, and two architectures answer it twice, in two languages, across
the whole ladder. A third strengthens the generalisation; it does not carry the
claim.

**The remaining 14 runs are queued as 7 pairs**, in `scripts/queue_all.sh`, in
priority order rather than by run number:

    r02 r05    Hindi VITS, the phonemic-versus-graphemic ablation
    r15 r18    Marathi FastSpeech 2, both control arms
    r16 r19    Marathi VITS, both control arms
    r11 r12    VITS ladder, 5 h and 1 h
    r13 r14    VITS ladder, 30 min and 10 min
    r09 r10    FastSpeech 2 ladder, 30 min and 10 min
    r06 r17    the two vocoders, fine-tuned on our own mels

The ladder is last among the acoustic runs on purpose. It is the shock absorber
in plan v3, so if the card is withdrawn mid-queue what is lost is the weakest
claim rather than the central one.

Four properties that matter for something left running for days. It is
idempotent: a run with `step_100000`, or a vocoder with `stopped.json`, is
skipped, so re-running it after any interruption resumes rather than restarts.
A failing pair is logged and the queue moves on, because one broken config must
not cost the remaining runs. Between pairs it waits for our own runs to exit
rather than for the card to empty, since a foreign tenant may never leave. And
it waits a bounded hour for a tenant to go, then proceeds with a memory fraction
computed from whatever is actually free: waiting indefinitely for someone else's
job is how a queue silently does nothing.

It cannot stop another tenant arriving. Nothing available here can. What it does
is keep our share occupied continuously, which is the only lever we have.

**Weights are copied off the machine on a timer**, since the GPU access is
temporary and a 95%-full Lustre mount is not a backup. `scripts/offload.py`
uploads to a Hugging Face model repository, using the token installed in phase 1
and closing plan v3's phase 3 task t5.

Every small file goes every time: run config, vocabulary, training log, stopping
record, the run matrix and this file. A checkpoint without those is a blob of
numbers. Then one checkpoint per run, the final one if it exists and otherwise
the most recent, because a 5,000-step cadence across 18 runs would upload the
same architecture dozens of times to no purpose, and the point is to survive
losing the machine rather than to keep a complete history. Export bundles are
included once they exist, since the front end consumes those rather than raw
checkpoints. A local state file of sizes and modification times means a
twenty-minute cadence costs almost nothing after the first pass, and an upload
is recorded as done only after it returns, so a failure is retried next cycle.

Verified against a synthetic run tree: newest checkpoint for an in-flight run,
final for a finished one, and a half-written `.tmp_step_*` save ignored.

## 2026-10-02 — r08's config hash was recorded as Infinity

Found by writing `scripts/verify_queue.py`, which recomputes each config's hash
from `plan_runs` and compares it against the YAML. One run disagreed, and the
reason was not the YAML.

`load_config` tried `int()` then `float()` on every unquoted value. A 12-hex
config hash is sometimes a valid float literal:

    r08's hash   52e245223208
    float(...)   inf

So r08 trained with `cfg["config_hash"] = inf`, and wrote `config_hash:
Infinity` into `runs/r08/config.json` and into all five of its checkpoints'
`meta.json`. `Infinity` is not valid strict JSON, so a conforming parser refuses
those files outright. The standing rule is that no number reaches the paper that
cannot be regenerated from a config file; for r08 the field that connects the
two said infinity.

Only r08 of the nineteen is affected, because only its hash happens to parse.
An all-digit hash would have become an int by the same route, silently and
without even looking odd.

**Fixed in the loader.** `STRING_KEYS` names the fields that are always text,
and the numeric patterns are now strict: an int must be all digits, and a float
must contain a decimal point. Every float in these configs has one, so nothing
that was a number stops being one, which `tests/test_config_loader.py` checks
over all nineteen real configs alongside the r08 case, the all-digit case and
`inf`, `nan` and `1e999`.

**Already-written metadata is repaired separately**, by
`scripts/repair_config_hash.py`, which rewrites only a `config_hash` that is not
a string, takes the replacement from `configs/<run>.yaml`, touches no tensors,
and reports before it writes. It dumps with `allow_nan=False` so an invalid
value cannot be written back out.

r08's weights are unaffected. What was lost was the record of which config
produced them, and that is recoverable because the config file itself was always
correct.

## 2026-10-02 — the vocoder warm start would have matched zero tensors

Preparing r06 and r17 turned up two blockers. Neither is about the checkpoint
being missing, which is what the symptom looked like.

**First: r06 died before `init_from` was ever read.** `runner.train` builds
`TextEncoder.for_config(language, input_repr)` unconditionally, `config.py`
enforces `input_repr: none` for `hifigan`, and there is no vocabulary for
`"none"`, so the encoder raises `ValueError`. Adapters now declare `needs_text`;
the loop builds an encoder only when one is wanted and writes no `vocab.json`
for a run that has no vocabulary. `bundle.py` stopped copying `vocab.json`
unconditionally, which would have taken a vocoder export down later.

**Second, and the real find: a straight `load_state_dict` matches nothing**, and
the reason is torch rather than the checkpoint. `weight_norm` used to store a
tensor as `X.weight_g` and `X.weight_v`; since the move to `parametrizations` it
stores `X.parametrizations.weight.original0` and `.original1`. Every published
checkpoint predates that, and a modern torch builds the new names:

    coqui HifiganGenerator, torch 2.14   conv_pre.parametrizations.weight.original0
    speechbrain LJSpeech checkpoint      conv_pre.conv.weight_g
    jik876 lineage                       conv_pre.weight_g

Same 234 tensors, same shapes, three differences in spelling: the weight-norm
API, speechbrain's extra `.conv` level, and coqui's `model_g.` prefix.
`src/train/vocoder_init.py` does those renames, deciding the direction from the
target model's own keys rather than from a torch version, so it works on
whichever torch the DGX has.

**Verified rather than argued**, by building coqui's generator from
`HifiganConfig`'s own `generator_model_params` and loading the real
speechbrain/tts-hifigan-ljspeech checkpoint through the shipped module:

    mapped                     234 of 235 target tensors
    the one miss               the discriminator, which a generator-only file has none of
    shape disagreements        0
    load_state_dict(strict)    accepted
    weights changed            yes
    synthesis                  (1, 80, 32) -> (1, 1, 8192), finite, within [-1, 1]
    samples per mel frame      256, which is the hop

The architecture matches without adjustment: speechbrain's `hyperparams.yaml`
declares upsample factors [8, 8, 2, 2], kernels [16, 16, 4, 4], 512 initial
channels, resblock type 1 with kernels [3, 7, 11], which is exactly
`HifiganConfig`'s default generator.

**[Unverified]** What mel convention those weights expect. speechbrain's
published `hyperparams.yaml` states only the architecture, not the analysis, so
the claim that its amplitude convention is closer to ours than coqui's is not
something this file can support. It matters less than it would elsewhere,
because the whole point of fine-tuning is that the vocoder is refitted to our
mels, and the stopping criterion measures when that has happened rather than
assuming a step count.

`VOCODER_INIT` in `config.py` now names the checkpoint and the configs were
regenerated, so r06 is `3120211db4c7` and r17 is `99a3999ca13e`. Setting it there
rather than editing the YAML keeps `config_hash` honest, which is what
`verify_queue.py` checks.

## 2026-10-02 — r06 verified end to end, and the vocoder path is closed

Dry-run output, on the real model with the real checkpoint:

    r06: no text front end, architecture hifigan
    r06: warm start from /workspace/vocoders/hifigan_lj_generator.ckpt
      mapped 234 of 404 target tensors (target naming: parametrizations, prefix: model_g.)
      234 generator tensor(s) loaded
    r06: moved to cuda: DiscriminatorLoss, GeneratorLoss, L1Loss, L1SpecLoss, MSEDLoss, MSEGLoss
    r06  step 1/2  loss 41.7214
    r06  step 2/2  loss 41.3006

    weights: resumed run is bit-identical to an uninterrupted 4-step run across 404 tensors
    OK: resumed from step 2, finished at 4, loss 41.7208 -> 39.8516

234 of 404 is the expected split: 234 generator tensors and 170 in the MPD and
MSD discriminators, which a generator-only checkpoint cannot supply and which
therefore start fresh. That is stated in the run's own output rather than
inferred later.

The resume check is the one that matters for this adapter. Its segment crop is
hashed from (seed, step, utterance id) rather than drawn from an RNG, and the
only way to know that reproduces across a restart is to kill a run and compare
weights. Bit-identical across all 404 tensors, so it does.

Three defects were fixed getting here, and each would have been silent or
misattributed:

1. `TextEncoder.for_config(language, "none")` raises, and the loop called it
   before any adapter was built, so r06 died before `init_from` was read.
2. A straight `load_state_dict` matched zero tensors, because torch renamed
   weight-norm storage from `X.weight_g` to
   `X.parametrizations.weight.original0` and every published checkpoint
   predates that.
3. coqui's loss modules hold an STFT window built on the CPU at construction,
   and the loop moved only the model, so the first spectral loss on the GPU
   failed on a device mismatch.

The resume check ran on the CPU, because the GPU had 2.75 GiB free beside r02
and r05 and `--kill-at` runs three trainings. Checkpointing, RNG restoration
and the crop are device-independent, so the proof stands. When the queue reaches
pair 7 the card is free and the fraction will be 0.45, which is 17.73 GiB
against the 6.34 this used.

r06 and r17 now need nothing. `verify_queue.py` reports no blockers.

## 2026-10-02 — gradient clipping is active on nearly every step, in every run

Checked because r02 and r05 logged pre-clip norms of 188 and 643 against
`grad_clip: 1.0`, and the worry was that a budget field held identical across
architectures was doing something very different to each of them, the way
`batch_frames` does.

It is not that. `clip_grad_norm_` reports the norm before clipping, and the
medians over every logged step are:

    run  arch         arm           median        max     clipped by, at the median
    r01  fastspeech2  phoneme 9h    368.44     184,919    368x
    r04  fastspeech2  grapheme 9h   106.86     110,145    107x
    r07  fastspeech2  phoneme 5h    210.68     493,259    211x
    r08  fastspeech2  phoneme 1h    122.13     149,720    122x
    r02  vits         phoneme 9h   1138.87       4,103    1139x
    r05  vits         grapheme 9h    97.31       1,815     97x

FastSpeech 2 medians run 107 to 368 and VITS 97 to 1139. The ranges overlap, so
this is not an architecture asymmetry: every run is clipped hard on essentially
every step. The within-pair spread is 3.4x for the FastSpeech 2 arms and 11.7x
for the VITS arms, which is a property of the vocabularies and sequence lengths
the arms differ in rather than of the budget.

**[Inference] Why this matters less than the ratios suggest.** The optimiser is
AdamW, which divides by the square root of the second-moment estimate. Scaling
every gradient by one constant scales the first moment by that constant and the
second by its square, so the update is largely unchanged. The cancellation is
not exact, because the factor varies from step to step while the moments are
running averages, and because of epsilon. That argument is from how Adam is
defined, not from an experiment here, and no ablation on this corpus supports it.

**And the clip is earning its place.** The largest single norm recorded is
493,259, on r07. An unclipped step of that size would destroy a run, and spikes
of 10^5 appear in all four FastSpeech 2 logs.

No change. It applies identically to both arms of every comparison, so no
contrast is affected, and changing it now would break comparability with the
ladder runs still to come. Recorded because a reader finding 1139x in a log
later deserves to know it was looked at.

## 2026-10-03 — the grapheme arm never expanded numbers

r02 and r05 finished and the queue did not advance. Two separate bugs, found
one behind the other.

**The chain stalled on a shell idiom.** `ours_running()` was

    pgrep -cf "src\.train\.launch" 2>/dev/null || echo 0

`pgrep -c` prints its count on stdout *and* exits 1 when the count is zero. So
on no match the variable held two lines, `0\n0`, and the `-eq 0` guard in
`wait_for_our_runs` errored instead of returning true. The condition could
never be satisfied, so the loop would have spun for its full 30 h budget and
then given up. It only appeared now because while r02 and r05 were alive
`pgrep -c` printed `2` and exited 0, so the `echo 0` never fired: the guard
inverted the moment the pair finished. `scripts/gpu_free.sh` had the same line,
where the two-line value reached `$(( ))`. Both are `|| true` now.

**Then r18 died at step 1.**

    KeyError: symbol '8' is not in the 136-symbol vocabulary;
              the front end and the vocabulary have diverged

`normalize` runs `devanagari_digits_to_ascii`, turning १८ into `1`, `8`.
`grapheme_inventory` is the U+0900 block plus punctuation and the word
boundary, and it does contain Devanagari ०–९ at U+0966–U+096F, but ASCII
digits are outside the block. `phonemize` expanded numbers before tokenising;
`graphemes` did not. One missing call.

The crash is the smaller half. For every utterance containing a numeral the
phoneme arm received a spoken number and the grapheme arm received a digit
glyph, so the phoneme-versus-grapheme contrast was absorbing a front-end
difference. That is exactly what `src/train/text.py` says it is built to
prevent: *"If the two arms built their vocabularies differently [...] the
comparison would measure bookkeeping rather than linguistics."*

**Scope, measured over the full manifests.**

    language   utterances   digit runs (by length)   grapheme OOV   phoneme OOV
    hindi           5,485   none                     0              0
    marathi         5,559   19  (9x1, 6x2, 3x3, 1x4) 27 utt. max    0

Hindi contains no digits at all, so r01, r04, r07 and r08 cannot have been
affected and their provenance stands. In Marathi the affected utterances are at
most 27 of 5,559, which is 0.486%. The phoneme arm is clean on the real data,
so r15 was correct as it ran and did not need restarting.

**The fix.** `graphemes` now calls `_expand_numbers` exactly as `phonemize`
does. `normalize` and `phonemize` are deliberately untouched, which makes the
phoneme path byte-identical rather than merely verified-equal, and keeps the
vocabulary at 136 symbols so r18 requeues with the embedding table r04 already
trained. Widening the inventory to admit ASCII digits was the obvious
alternative and was rejected: r04 is a finished Hindi grapheme run on 136
symbols, and putting Marathi grapheme on a larger table would place a table-size
difference on the language axis.

`scripts/check_text_coverage.py` now asserts that front-end output lies inside
the vocabulary, for each (language, input_repr) pair, from the manifests. It is
four checks for eighteen runs because coverage depends on nothing else. The
point is where it fails: `Vocab.encode` raises rather than substituting `<unk>`,
which is the right choice, but it raises inside `collate` on the prefetch thread
at step 1, so the cost of finding out was a spent pair slot. Now it is seconds.

    hindi/phoneme     5485 utterances,  78-symbol vocabulary, 0 OOV
    hindi/grapheme    5485 utterances, 136-symbol vocabulary, 0 OOV
    marathi/phoneme   5559 utterances,  78-symbol vocabulary, 0 OOV
    marathi/grapheme  5559 utterances, 1 distinct OOV symbol

**One OOV symbol was left, and the recording settled it.** `ma_007005` reads

    सध्या या सर्व गोष्टीzबरोबरच फायबर ग्लासचाही मोठ्या प्रमाणावर उपयोग केला जातो.

A Latin `z` sits inside a Devanagari word. The grammar suggests गोष्टींबरोबरच,
which would make the `z` a slip for anusvara U+0902, but grammar is not
evidence about what was said, so the audio was checked instead. The speaker
nasalises. The repair is therefore from the recording.

`latin_z_to_anusvara` is scoped to the Devanagari-flanked case,
`(?<=[\u0900-\u097F])z(?=[\u0900-\u097F])`, mirroring `colon_to_visarga`
rather than going into `TYPO_MAP`. A bare `z` → anusvara entry there would
rewrite any Latin token that ever reaches the chain; an unflanked `z` now stays
put and fails the coverage check loudly instead of being silently nasalised.
One occurrence across both corpora.

The phoneme arm never raised on this character because the phonemiser mapped it
to /z/, which is in the phone inventory for ज़. So the defect was invisible on
that side while being fatal on the other, and the arms disagreed about this
utterance for a second reason entirely separate from the digits. With the
repair, anusvara before the labial /b/ assimilates and the phone is /m/:

    before:  oː ʂ ʈ iː z b ə r oː
    after:   oː ʂ ʈ iː m b ə r oː

All four combinations are now clean:

    hindi/phoneme     5485 utterances,  78-symbol vocabulary, 0 OOV
    hindi/grapheme    5485 utterances, 136-symbol vocabulary, 0 OOV
    marathi/phoneme   5559 utterances,  78-symbol vocabulary, 0 OOV
    marathi/grapheme  5559 utterances, 136-symbol vocabulary, 0 OOV

**This changes one utterance for the phoneme arm, which has a consequence for
r15.** r15 was launched before the repair and is training on /z/ at that
position. r18 will train on /m/. The pair differs in one utterance of 5,559,
0.0180% of the corpus and one phone within it, which is the kind of asymmetry
this project has otherwise refused to carry. Hindi is unaffected, so r01, r04,
r07 and r08 are untouched; `normalize` is shared, but no Hindi transcript
contains a Devanagari-flanked `z`, which is also why hindi/grapheme reported 0
OOV before the repair existed.

**The queue now checks coverage for every run in a pair, not just the first.**
`queue_all.sh` dry-ran only `FIRST`, which is precisely how r18 reached the GPU:
r15 dry-ran clean and r18 died at step 1 with the slot already spent. The gate
sits before the memory fractions are computed and skips a vocoder, whose
`input_repr` is `none` and which has no vocabulary to check.

## 2026-10-03 — the demo runs on CPU, not on the GPU

The demo was first framed as a choice between live inference hosted on the DGX
and a static page of pre-rendered audio, on the grounds that the card is
temporary. That framing was wrong. Inference is a CPU workload; the GPU is for
training. The real choice is only where the CPU lives, and the DGX was simply
where the checkpoints and the environment already were.

**Decision: a Hugging Face Space.** The weights are already in
`Klewik/indic-tts-bench`, so a Space loads a bundle from the repo it is
offloaded to and runs the same Python front end that trained the model. Nothing
is ported, so nothing can diverge. It outlives the GPU access, takes free text
rather than a fixed sentence list, and an examiner can reach it from a link.

The two alternatives and why not. A static page with pre-rendered audio cannot
take a sentence nobody anticipated, though it remains the right shape for the
listening test, where a fixed set is a requirement. In-browser ONNX needs
`normalize.py`, `devanagari.py`, `schwa.py` and `numbers.py` reimplemented in
JavaScript, and a schwa-deletion bug in that port would silently change the
thing the dissertation is about. That is the one component where a
reimplementation is not acceptable.

**r02 and r05 can be demonstrated now.** VITS is end to end, so those two need
no vocoder at all, and they are the pair the central phonemic-versus-graphemic
ablation rests on. The four FastSpeech 2 runs emit mels and stay silent until
r06 and r17 finish; Griffin-Lim would make them audible immediately and would
match the project's mel definition exactly, but it cannot be presented as the
system's quality and would have to be labelled as a placeholder in the
interface.

**coqui's inference surface, measured rather than assumed.**

    ForwardTTS.inference(x, aux_input={'d_vectors', 'speaker_ids'})
      model_outputs  [B, T_frames, n_mels]      mel, silent on its own
    Vits.inference(x, aux_input={'x_lengths', 'd_vectors', ...})
      model_outputs  [B, 1, N]                  waveform

A 24-token input gave 48 frames and 12,288 samples, which is hop 256 confirmed
from the other direction. Note the mel orientation: coqui returns frames first,
while the HiFi-GAN generator takes `[B, n_mels, T]`. `synthesize.py` transposes
once, on the way out, so mels leave it as `[n_mels, T]` and agree with both the
vocoder and `features.compute`. A mel handed over the wrong way round produces
audio that is recognisably speech and subtly wrong, which is the failure nobody
catches in a demo room.

**`src/export/synthesize.py`** is the single synthesis path, used by both the
Space and the listening test. It rebuilds the model through the adapter that
trained it, loads the bundle's own `vocab.json` rather than rebuilding one,
encodes through the same `TextEncoder`, and loads weights with `strict=True`.
It returns a mel and says so for the mel architectures instead of inventing a
waveform.

`BUNDLE_VERSION` is 3: the manifest now records `lr`, because
`VitsAdapter.build` reads it to set coqui's `lr_disc` and `lr_gen`. Neither
enters the inference graph, but carrying the real value is better than having
the synthesis module invent a number that then reads as a hyperparameter.
Nothing needed re-exporting, since no bundle had been produced yet.

## 2026-10-03 — the demo set is selected by rule, not by ear

Hugging Face restricted free `cpu-basic` Spaces in July 2026: Gradio and Docker
Spaces on the free tier now require PRO, Static Spaces stay free. So the
permanent artifact is a Static Space with pre-rendered audio, and the Gradio app
stays in the repo for a notebook when free text is wanted live. The static route
is not a consolation: a fixed, pre-registered sentence set is what a listening
test requires, so this is work the dissertation needed anyway.

**The selection rule.** Two properties pull against each other. The sentences
must come from the held-out dev split, so nobody can say the demo was picked to
flatter a model; and they must actually exercise schwa deletion, or the
demonstration shows nothing about the claim. Hand-written sentences would get the
second and lose the first.

So: dev split only, filtered to 1.5–6.0 s, banded by **medial** schwa deletion
count (none / one / few=2–3 / many=4+), taken round-robin from the richest band
down, sorted by utterance id inside each band. Deterministic given the split.
The rule, the band sizes and every per-sentence count are written to
`results/tables/demo_set_hindi.json` so the choice can be checked rather than
trusted.

Word-final deletion is counted but never selected on. It is near-categorical in
Hindi and both arms learn it, so a set chosen on total deletion sites would be a
set chosen on the easy case. Medial deletion is the conditioned one, and the
place the arms should come apart if the phonemic front end is earning its place.

**What Hindi dev yields.** 100 utterances, 47 inside the length window:

    band            in window
    many (4+)               5
    few  (2-3)             20
    one                    13
    none                    9

The twelve chosen span 6 medial sites down to 0. The three with no medial
deletion are the control: if both arms handle those and diverge on the heavy
ones, that is the claim showing up in audio rather than in a table. If they
diverge everywhere, the gap is about something else.

**Deletion sites came from the training path, not a reimplementation.**
`delete_schwas` has always returned the indices it deleted, and `phonemize_word`
discarded them. That function is refactored into
`_phonemize_word_with_sites`, with `phonemize_word` and a new `deletion_sites`
both calling it, so there is no second copy of the punctuation-and-lexicon
handling to drift away from what trained six runs. All 65 existing G2P tests
pass unchanged, which is the point of doing it that way; 176 assertions pass in
total across the torch-free suites.

A lexicon hit reports zero sites, because no rule ran — the true answer, not a
missing one. Sites index the core word's pre-deletion segments, so a trailing
comma cannot shift them, which matters given `phonemize_word`'s own docstring
about punctuation changing schwa context.

**The natural recording ships alongside the arms**, at each arm's sample rate.
Without it the question is only "which of these two do you prefer", which is
much weaker than "how far is each from the speaker", and it is the reference an
AB or MOS test needs. Mel-only architectures write no audio at all rather than a
Griffin-Lim stand-in, and the page says they are silent and why.

The page marks the words where medial deletion applies, so a listener knows
where to attend, and prints each clip's run id, step, config hash and git commit
beside it.

## 2026-10-03 — FastSpeech 2 made audible by a labelled placeholder

The FastSpeech 2 arms emit mel spectrograms and r06 and r17 have not trained,
so the choice was between leaving half the matrix silent in the demo or
inverting the mel without a vocoder. Griffin-Lim, labelled everywhere it plays.

**Why it is labelled and not a default.** Griffin-Lim recovers phase by
iteration from magnitudes alone and carries its own metallic, smeared
signature whatever the mel is worth. A listener told "this is the model" would
be judging the algorithm. So `--griffin-lim` is explicit, every clip it
produces records `vocoder: "griffin-lim"` in `data.json`, the player prints the
caveat beside that clip, and the page repeats it once at the top. What the
placeholder is good for is checking that the words are there, that the front
end and the weights line up, and that the interface works before a vocoder
exists. None of those is a quality claim.

**It inverts this project's mel, not a generic one.** `_spec_and_mel` computes
`log(max(mel_fb @ |stft|, 1e-5))` with the bank built `fmin=0, fmax=sr//2`.
Three details each break the inversion without raising: the forward mel is
MAGNITUDE, so librosa's default `power=2.0` would square-root the spectrum and
throw away roughly half the dynamic range; the bank must be built to `sr//2`
rather than to an 8 kHz convention, which is the same trap recorded on 2 Oct;
and the log has to be undone with `exp` first.

**Verified by round-trip rather than by ear.** One real corpus utterance,
`hi_005009`, through our forward mel, inverted, then re-analysed through the
identical forward path. That is the same signal `HiFiGanAdapter.validate` uses,
so the numbers are comparable to what the vocoder will report.

    n_iter   22.05 kHz    16 kHz
         1     0.2980    0.3487
        16     0.1615    0.2046
        32     0.1492    0.1888
        60     0.1434    0.1810
       120     0.1388    0.1787
    shuffled   2.1473    2.2866   <- the same mel, frames permuted

So 15.0x and 12.6x closer than chance at 60 iterations, which is the default:
32 to 60 buys 0.0058 and 60 to 120 buys 0.0046. Measured with librosa 0.11.0 in
the container, while the DGX runs 1.0.0, so the figures are indicative and the
numbers that matter are the ones the DGX produces.

Griffin-Lim returns a whole number of hops and lands up to one frame short, 45
samples or 2.0 ms at 22.05 kHz. Left unpadded: a placeholder should not quietly
invent samples.

**The guards fire before librosa is imported.** The first version validated
after it, which meant a transposed mel could not be caught without the audio
stack installed and failed only after a multi-second import. coqui returns
`[B, T_frames, n_mels]` and this wants `[n_mels, frames]`, so a 184-frame mel
handed over untransposed would be read as 184 mel bands; that now raises by
name. 12 assertions on the guards and on agreement with `features.mel_params`,
188 across the torch-free suites.

## 2026-10-03 — where the phoneme arm is exposed: every word-final consonant

A listening impression on r02 was that /l/ and the "gh" sounds break more than
the rest. Measured on the 9 h Hindi training split rather than left as an
impression: 4,793 utterances, 87,316 words, 350,391 phone tokens, 72 of the
declared phones actually in use.

    phone   rank    count   per 1k   word-final   of those, by deletion
    l         14     8873    25.32         1517                    1517
    ɦ          9    13906    39.69         1357                    1357
    bʱ        31     2311     6.60           69                      69
    d̪ʱ       39     1277     3.64          170                     170
    dʒʱ       50      517     1.48           56                      56
    ɡʱ        49      524     1.50           10                      10
    ɖʱ        60      202     0.58           28                      28

**The breathy-voiced series is a resource problem and nothing more exotic.**
/ɡʱ/ at 1.50 per thousand is about 17 times rarer than /l/, and /ɖʱ/ at 0.58
rarer again. Nine hours does not contain many of them. That is the kind of
limitation a fixed-budget study should report rather than discover late, and it
is a prediction for the ladder: these should degrade first as the rungs shrink.

**/l/ is common, so the explanation is positional, and it is structural.** For
every phone checked, the word-final count and the deletion-derived count are the
same number. Devanagari writes a word-final consonant with an inherent schwa, so
in the phonemic arm no word ends in a consonant unless the rule deletes that
schwa. Every word-final consonant is a context the front end created. Only /ɡ/
differs at all, by one.

And the exposure is large: **27,912 of 87,316 words, 32.0%, end in a consonant
because a final schwa was deleted.** Commonest are /r/ 6,758, /n/ 2,643,
/k/ 2,095, /t̪/ 2,048, /s/ 1,886, /l/ 1,517, /ɦ/ 1,357, /m/ 1,234. So one Hindi
word in three has its last phone where the rule put it. This also names the
earlier impression precisely: कमल is /kəməl/ with the final schwa gone, so the
/l/ that was heard to break is a deletion-derived word-final /l/, sixth most
common of that class.

**The testable prediction.** If deletion-derived position is the cause, the
phonemic arm's errors should concentrate on word-final consonants rather than on
those consonants generally, and the graphemic arm should not show that pattern
because it never creates the context. That is a per-position error analysis
against the stress-test scoring, and it needs a vocoder first.

What is deliberately NOT computed: 32.0% times the 83.0% front-end agreement.
That 83.0% came from a deliberately contested schwa set, so it does not transfer
to ordinary corpus text, and the rule's corpus-wide error rate remains unknown.

Two incidental checks while looking: /h/ at 59 tokens is the visarga from ः and
is correctly distinct from ह as /ɦ/, not a front-end inconsistency; /ɭ/ at 2
tokens is retroflex L from ळ in a Marathi proper noun inside Hindi text, and
has an effectively untrained embedding, which is the outcome
`src/train/text.py` documents as intended. 278 words keep a word-final schwa,
none of them by deletion, which is the `min_vowels` guard declining to strip a
monosyllable.

`scripts/phone_stats.py` produces this for any language and rung.

## 2026-10-03 — Griffin-Lim is exonerated; the mel is over-smoothed; every run converged by 50k

The FastSpeech 2 arms came out of the demo as noise rather than as metallic
speech, which is not what a correct inversion of a trained mel sounds like. The
cause is now separated into three findings, in the order they were established.

**The inverter is not at fault, and the argument is clean.** Griffin-Lim
reproduced r01's predicted mel at L1 0.0754 against 0.1431 for the ground-truth
mel of the same utterance, so it was **1.90x more faithful** on the input that
sounds worse. An inverter more accurate on the worse-sounding input cannot be
what makes it worse. librosa 1.0.0 on the DGX also round-trips ground truth at
0.1431 against the container's 0.1434 with librosa 0.11.0, so the version
difference is closed as a concern.

**The mel is over-smoothed along time.**

                                               gt     r01   r01 as % of gt
    temporal variation (frames shuffled)   2.1477  1.1029            51.4%
    per-band std over time                  2.071   1.529            73.8%
    per-frame std across bands              1.704   1.848           108.5%

Spectral shape within a frame is intact, marginally richer than ground truth.
What is missing is frame-to-frame structure: half of it. That is the standard
fixed point of a non-autoregressive mel predictor under an L1 loss, which
regresses toward the mean of the plausible mels. Griffin-Lim is the worst
possible partner for it, and not coincidentally: it carries no prior and must
recover phase from magnitudes by iteration, so a blurred magnitude spectrum
gives it nothing to lock onto. A neural vocoder has a learned prior and
tolerates smoothing far better, which is what r06 is for.

The competing hypothesis is dead. A failed internal aligner would collapse
durations toward uniform; r01 predicted 160 frames for about 30 tokens, 5.3
frames or 62 ms per phone, at 87.0% of the reference utterance's length. The
aligner that replaced MFA is working.

**Every run converged by about step 50,000, and the second half bought nothing.**

    run   arch  data      start     ~50k   80-90k  90-100k   2nd half / 1st
    r01   fs2   9h      1665.72   309.30   307.74   314.60          -0.0039
    r04   fs2   9h      1672.91   301.41   295.75   300.33           0.0008
    r07   fs2   5h      1998.41   303.50   306.37   299.85           0.0022
    r08   fs2   1h      1705.62    16.03     9.80    10.04           0.0035
    r02   vits  9h        58.44    27.85    27.01    27.03           0.0268
    r05   vits  9h        60.10    27.30    26.62    26.73           0.0174

For FastSpeech 2 this is informative, the loss being a reconstruction loss: the
over-smoothing is the objective's fixed point and no further training reduces
it. For VITS it is not. Its loss is a sum including adversarial terms, and a
flat generator loss is weakly related to perceptual quality, so r02's plateau is
NOT evidence that r02 is as good as it gets. [Inference, from how the objective
is composed rather than from an experiment here.]

Consequence for the budget question in b4, and a finding in its own right: for
this corpus and these architectures the training objective converges at roughly
half the declared budget. The remaining runs must still take 100,000 steps,
because comparability across the matrix is the point, but a future study on this
data would not need to.

Consequence for b12: since more steps are wasted either way, the VITS re-run
decision is not about compute. An MMS warm start changes the starting basin,
which is the kind of change that can help where more steps cannot, and is the
only option that might improve how VITS sounds. 22.05 kHz from scratch makes the
cross-architecture comparison defensible and does nothing for quality. The two
serve different goals.

**One anomaly, open, and it gates the ladder.** r01 at 9 h ends on 314.60 and
r07 at 5 h on 299.85, a ratio of 0.953 and effectively flat. r07 to r08 at 1 h
is a 29.9x drop for five times less data. An overfitting curve does not have a
cliff in it. Either the 1 h rung is memorised outright, roughly 530 utterances
over about 4,000 epochs, or the logged loss is not comparable across rungs,
which would make the ladder's loss trend meaningless as plotted. To check:
whether the logged value is a mean over frames or a sum, and what the per
component terms do at the same step.

**The placeholder is not being shipped for FastSpeech 2.** A silent arm with an
explanation is honest; a noise arm labelled "placeholder" invites a listener to
conclude the model is broken when what they are hearing is the inversion method
meeting a smoothed mel. `src/export/griffinlim.py` stays, having earned its
place on the ground-truth path and as a debugging tool, but `--griffin-lim` is
not passed when rendering the page. Phase 7 task 9 waits for r06.

## 2026-10-03 — VITS stays as it is, and the false claim is corrected rather than rewritten

Decided: no MMS warm start, no extra steps, no change of sample rate, and r02
and r05 keep their trained weights. A warm start would import training compute
the fixed budget does not count, and extra steps for one architecture would end
the symmetry outright. The equal-compute claim is the study's main
methodological asset and is worth more than better audio. The artefacts stay and
are reported; the material for that is already here, in the phone frequencies,
the over-smoothing measurement and the step-50,000 convergence.

Marathi keeps its own vocoder. r17 stays paired with r06 on the original
argument: the speaker differs by language, so a Hindi-fitted vocoder used on
Marathi audio would put a speaker mismatch inside the control that exists to
isolate language.

**The provenance problem that follows, and how it is handled.**
`configs/r02.yaml` declares `init_from: facebook/mms-tts-hin` and a deviation
saying the 16 kHz rate is "inherited from the MMS checkpoint". Both are false,
and `deviations` and `init_from` are both inside `config_hash`. Correcting
either in place would change the hash of eight VITS runs, two of which have
finished, orphaning r02's checkpoint and its export bundle from their own
provenance record. Rewriting the declaration would also hide that a false claim
was ever made, which is the opposite of what a deviations table is for.

So a `corrections` field was added to `RunConfig`, placed in `COSMETIC` and
therefore outside the hash, emitted into the YAML beside the original claim. The
hash now records what was declared and trained; the record says what was later
found untrue and when. All eight VITS runs carry three corrections: that no warm
start was ever implemented and every VITS run trained from random
initialisation; that the 16 kHz rate is consequently unmotivated rather than
inherited, and leaves the VITS runs with a narrower band and 192.00 s of audio
per step against 139.32 s at an identical 12,000-frame budget, which any
FastSpeech 2 against VITS comparison inherits; and that a warm start was
considered and declined, with the reason.

Verified rather than assumed: all 18 configs regenerated and every one of the 18
hashes is byte-identical to what was on disk before, including r02's
`d3873342a10d`, which is the hash its trained checkpoint and its export bundle
both carry. `runner.load_config` reads the new list without tripping the string
hardening that turned r08's hash into `inf` on 2 October.

**The vocoders are now next in the queue.** They were last while they were an
upper bound on 10 GPU-h each with nothing depending on them; neither half of
that holds any more, since the FastSpeech 2 arms are mute without one and their
mels need a learned prior, and they stop on a plateau criterion rather than
spending a budget. Moving them ahead of the Marathi VITS pair also means that
pair cannot launch on an unsettled configuration.

14 assertions on the corrections mechanism, including that adding one cannot
move a hash while editing a deviation can, and that no non-VITS run carries one.
202 across the torch-free suites.

## 2026-10-03 — the seed variance floor is queued, as r20 and r21

Plan v3's standing rule says no difference between two runs is reportable until
the variance floor exists. It still did not, with about 60 GPU-h spent and 90
planned on eighteen runs whose differences nobody could interpret. Every run
used seed 0, so every number in the results table is a single sample with no
scale against it.

r20 and r21 are r01's cell at seeds 1 and 2: FastSpeech 2, Hindi, phonemic, 9 h.
The spread across r01, r20 and r21 is then the smallest difference this setup
can resolve, because nothing else about the three runs differs. Verified rather
than asserted: the only hash-bearing field that differs is `seed`, with
`run_id` and `notes` cosmetic and outside the hash, and all three hashes differ
while every budget field is identical. All 20 configs regenerated, and every
pre-existing hash is unchanged, r01's `e3bbebc98825` included.

r01's cell because that is where the central phonemic-versus-graphemic claim
lives, and FastSpeech 2 because it is the cheaper architecture and carries eight
of the runs. 4.2 h for the pair on a free card at the rate r15 measured.

**What the number will mean, in both directions.** If the floor is small against
the phonemic-versus-graphemic gap, it licenses every other comparison in the
dissertation, which is what four hours is being spent on. If it is comparable to
the gap, the headline claim weakens and has to be stated with that spread
attached — and "at nine hours and this budget, seed variance is of the same
order as the input-representation effect" is then a finding in its own right,
and an uncomfortable one for a literature in which single-seed TTS comparisons
are common. Either way the floor is already inside the results; measuring it
only decides whether that is known.

The timing is what makes a wide floor survivable. With 90 GPU-h still unspent
the response can be more seeds per cell and intervals rather than point
estimates. Found after the write-up, there is no response.

Queued behind the vocoders and ahead of the ladder, which matters: if the floor
is wide, the smallest ladder rungs are where it is widest and the claims
thinnest, so the result should be known before 30 GPU-h goes into them.

## 2026-10-03 — a VITS seed floor first, and a check that would have caught the MMS gap

**r22 and r23: r02's cell at seeds 1 and 2, queued ahead of the vocoders.** A
noise floor is architecture-specific, so the FastSpeech 2 pair says nothing
about how far apart two VITS runs land, and VITS is the half that can be scored
first: it is end to end, so r02 and r05 need no vocoder. That makes the
phonemic-versus-graphemic contrast the first result obtainable rather than the
last, and reporting it without a scale underneath was the thing the seed work
exists to prevent. Sequenced first, the chain is r15 and r18 finishing at 1.5 h
then this pair at 9.7 h, so the central comparison is complete and interpretable
about eleven hours from now instead of on Wednesday. 9.7 h moves the slack
before 11 October from 5.2 days to 4.8. The matrix is 22 runs.

**`assert_init_from_is_honest`, at `adapters.for_config`.** A config may not
claim a warm start its adapter never performs. r02 declared
`init_from: facebook/mms-tts-hin`, `VitsAdapter.build` never read it, and the
config, the deviations table and the class docstring all stated the 16 kHz rate
was inherited from that checkpoint. Nothing failed and nothing warned. Eight
runs carried wrong provenance until the audio was listened to and the cause
traced backwards. `for_config` is the one chokepoint where a config meets an
adapter, so the check sits there and covers training, dry runs and evaluation
alike.

The decided case has to pass or the queued VITS runs could not start, so the
rule is that a mismatch is allowed exactly when a `corrections` entry mentions
`init_from` — that is, when somebody has written down what is wrong and why.
`reads_init_from` defaults to False on `AdapterBase`, which makes a new adapter
that forgets the flag refuse rather than silently claim to honour a warm start.
A correction about something else does not excuse it, because a check that any
correction satisfies is decoration.

Verified against the real matrix: all 22 runs pass, so nothing queued is
blocked. 12 assertions, half on the cases that must refuse, including that the
refusal message names all three remedies — implement the warm start, clear
`init_from`, or record a correction. A refusal that does not name the remedy
gets worked around.

**The 500-step dry run per architecture is removed from the plan, not done.**
Eight runs reaching 100,000 steps is far stronger evidence that every
architecture works than a 500-step job would be, and the queue already dry-runs
two steps before each pair. Performing it now would tick a box and tell us
nothing.

## 2026-10-03 — MCD was 34x too large, and it cannot resolve this comparison anyway

The harness's first run reported MCD 275.103 for r02 and 279.089 for r05.
Published TTS figures run 3 to 8 dB. Two separate problems came out of chasing
that, and the second matters more than the first.

**The bug.** `mel_cepstrum` called `librosa.feature.mfcc`, which applies
`power_to_db`, that is `10*log10`. `MCD_CONSTANT = 10*sqrt(2)/ln(10)` exists to
convert NATURAL-log cepstral coefficients into decibels, so the conversion was
happening twice. Predicted inflation 4.34x; measured 3.71x on a Griffin-Lim
reconstruction and 4.19x between unrelated recordings. `mel_cepstrum` now takes
its own DCT of `log(mel_power)` so the coefficients match the constant by
construction.

Every existing MCD test passed throughout, because all of them checked relative
behaviour — zero against itself, larger for more different spectra — and none
checked magnitude. Five tests added, including a plausible-range assertion and
a source check that `librosa.feature.mfcc` is not used.

**The scale, measured rather than assumed.** An MCD figure alone is
uninterpretable, so the baselines were established on real corpus audio:

    current formula          fixed
       0.000                 0.000    identical signal
      70.744                19.082    a severe but faithful reconstruction
                                      (Griffin-Lim of the real mel)
     275.103                   --     r02
     279.089                   --     r05
     447.1                 106.7      CHANCE: two unrelated real recordings

**The finding that matters.** r02 and r05 sit 54.3% and 55.4% of the way from
a faithful reconstruction to completely unrelated audio. The gap between them
is 3.986, which is 0.89% of the chance level and 0.31 standard errors on 25
utterances. Rescaling by the bug factor preserves every one of those ratios
exactly.

So MCD cannot resolve the phonemic-versus-graphemic contrast at this quality
level. When both systems are this far from the reference, the measure is
dominated by how far they both are, and the difference between them disappears
into it. The full 300-utterance test split reduces the standard error by a
factor of 3.5, which is not enough to rescue a 0.31-sigma gap.

**What follows for the evaluation plan.** The acoustic-distance metrics will
describe how bad both systems are, accurately, and will not answer the
dissertation's question. The metrics that can are the ones that measure the
phenomenon rather than the distance: stress-test error rates per architecture,
and the per-position analysis splitting deletion-derived word-final consonants
from the rest, where 32.0% of Hindi words sit. Those do not saturate, because a
schwa is either deleted or it is not.

`scripts/evaluate.py` now computes the chance level for every run by rotating
the reference by one utterance, so each synthesis is scored against a recording
of different words by the same speaker through the same alignment. The table
prints it and the percentage of it. No MCD figure leaves this project without
the number it should be read against.

## 2026-10-03 — phase 6 starts with the measure that answers the question

MCD cannot resolve the phonemic-versus-graphemic contrast at this quality
level, and no amount of care with the acoustic metrics changes that: both arms
sit about 55% of the way from a faithful reconstruction to unrelated audio, and
the gap between them is 0.89% of chance, 0.31 standard errors. So phase 6 is
being built in the reverse of the planned order, with the phenomenon measured
first and acoustic distance kept as supporting description.

**The measure: a schwa takes time.** The rule deletes a word-final schwa in
32.0% of Hindi words. The phonemic arm is handed that deletion; the graphemic
arm has to infer it. If it fails to, its output is longer than the reference by
roughly one vowel per missed site. So, per run,

    syn_seconds - ref_seconds  =  intercept + slope * n_deletion_sites

and `slope` is milliseconds of excess per deletion site. A model deleting
correctly has a slope near zero; one keeping the schwas has a slope near the
duration of a short vowel. The intercept is fitted rather than assumed zero, so
a model that merely speaks 8% slow does not read as one keeping schwas — tested
directly.

Three properties make it worth more than its simplicity suggests. It needs no
vocoder, because a mel frame count is a duration, so FastSpeech 2 is measurable
while r06 is still training. It cannot saturate, because however rough the
audio a schwa is either there or not. And it is falsifiable in the right
direction: if both arms show the same slope, the front end is not buying
deletion behaviour, which is a real answer rather than a null dressed up.

**Discrimination, on synthetic data where the truth is known:**

    a model keeping every schwa   slope  67.8 ms   se 1.5   t = 45.1   r = 0.99
    a model deleting correctly    slope   0.5 ms   se 1.4   t =  0.4   r = 0.05

Against MCD's t = 0.31 for the same contrast. The gap between the two fitted
slopes is over 5 standard errors at realistic noise.

The standard error is reported with every slope, and the fit refuses rather
than guessing in the two cases where a regression would otherwise invent a
finding: fewer than three usable points, and no variation in site count across
utterances, where no slope is identifiable at all.

**The harness no longer skips mel-only runs.** Only their waveform metrics are
marked unavailable, with the reason; the duration measure runs on them now. A
row is never silently dropped.

`deletion_sites` is computed from the TEXT, so both arms of a pair get the same
count for the same utterance. That is the design: one arm receives the deletion
and the other must work it out, and the regression asks which happened.

12 assertions on the fit, 10 on the harness bookkeeping, 316 across the
torch-free suites.
