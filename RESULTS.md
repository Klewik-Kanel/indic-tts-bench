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
