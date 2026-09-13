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
