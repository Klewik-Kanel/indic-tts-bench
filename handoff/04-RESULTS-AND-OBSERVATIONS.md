# Results and observations

Every number here traces to a dated section of `RESULTS.md`. Where a claim is an
inference rather than a measurement it is labelled. Where a claim was withdrawn,
the withdrawal is recorded rather than the claim being deleted.

---

## 1. The headline result — VITS, held-out intelligibility

**4 October.** 50 test utterances, 5 synthesis draws each, two CTC recognisers,
17,770 reference characters per run (6,835 word-final site, 2,370 medial site,
8,565 no site).

```
recogniser        run  arm        CER      draw sd
IndicConformer    r02  phoneme   0.1237    0.0081
                  r22  phoneme   0.1078    0.0072
                  r23  phoneme   0.1051    0.0077
                  r05  grapheme  0.1536    0.0088
MMS-1B-all        r02  phoneme   0.2111    0.0038
                  r22  phoneme   0.1880    0.0042
                  r23  phoneme   0.1854    0.0066
                  r05  grapheme  0.2443    0.0018
```

**The three variances order correctly, on both recognisers.** This is the
structure the claim needs and the first time all three existed at once:

```
IndicConformer   draw 0.00795  <  seed 0.01005  <  arm gap 0.02990
MMS-1B-all       draw 0.00410  <  seed 0.01415  <  arm gap 0.03320
```

The arm gap is **2.98 standard deviations** of the seed spread on
IndicConformer and **2.35** on MMS, or 1.61 and 1.29 times the full seed range.
**The graphemic arm is worse than all three phonemic seeds on both
recognisers.** No sign change anywhere.

Both arms sit **4.06 to 5.93 times** the recogniser's own floor, so this is
measured where the recogniser has room to discriminate rather than near its own
error.

### What may be claimed, and what may not

**May be claimed:** explicit grapheme-to-phoneme conversion improves
intelligibility, globally, under matched compute, for VITS on Hindi at the nine
hour rung, by a margin larger than the training-seed spread on two independent
recognisers.

**May not be claimed:** that the improvement is attributable to schwa deletion
specifically. The paired bootstrap over utterances, 2,000 resamples:

```
IndicConformer  final   +0.0057  [-0.0506, +0.0597]   not clear of zero
                medial  -0.0068  [-0.0944, +0.0799]   not clear of zero
MMS-1B-all      final   +0.0200  [-0.0553, +0.0951]   not clear of zero
                medial  -0.0069  [-0.1027, +0.0859]   not clear of zero
```

The graphemic arm is worse in **every** class. On IndicConformer: final +0.0515,
medial +0.0536, no site +0.0252. The site classes degrade about twice as much as
the no-site class, which is the shape the hypothesis predicts, **but the
difference of excesses is not separable from zero at this sample size.**

**Read those intervals with the caveat `RESULTS.md` attaches to them.** The
bootstrap rows and the class rates above them were computed from *different
samples*: `record["utterances"]` held draw 0 alone while the class rates pooled
all five. That is why the same quantity reads **+0.0263** from the class rates
(0.0515 − 0.0252) and **+0.0057** from the bootstrap. The defect is fixed in the
code — draws are merged per utterance before resampling — but **the table above
is the pre-fix version**, kept as printed, and `RESULTS.md` says the run is to
be repeated. The intervals are therefore the wide version. **Re-run before any
of this reaches the paper.**

`[Inference]` A broadly distributed benefit is consistent with the phonemic
representation helping *as a representation*: r05 carries 136 symbols against
r02's 78, so the graphemic arm fits a larger embedding table from the same nine
hours. That confound is intrinsic to the contrast rather than a defect in it.

### A claim that was withdrawn in full

**The medial-site claim of 3 October is withdrawn.** It rested on one unseeded
draw. With seeding, five draws, 50 utterances and an interval, its point
estimate is −0.0068 and −0.0069. It had been described in conversation as "the
best available evidence"; it was one draw.

---

## 2. FastSpeech 2 produces unintelligible speech, and the cause is the objective

r01 and r04 came back at CER **0.9932** and **0.9924**, with the number of edits
equal to the number of reference characters. That is the "transcribed nothing"
ceiling, not a bad score.

Two suspects were stacked behind it: the acoustic model's mels, and the vocoder.
A **ceiling measurement** — the real recordings' own mels pushed through r06 —
separates them.

```
recogniser        floor    ceiling   vocoder cost   r01
IndicConformer    0.0259   0.0264    +0.0005        0.9932
MMS-1B-all        0.0442   0.0487    +0.0045        0.9789
```

**The vocoder explains 0.05% of r01's error above the floor on IndicConformer
and 0.48% on MMS.** The acoustic model explains the other 99.95% and 99.52%, a
ratio of 206.7 to 1 on MMS. The vocoded ground-truth transcripts are accurate
Hindi. Two recognisers of different architecture, trained on different data,
agree that r06 is very nearly invisible to a listener reading words.

By class on IndicConformer, ceiling minus floor: final +0.0037, medial −0.0042,
no site +0.0012. **The vocoder does not favour or penalise deletion sites**, so
it cannot be producing the site-partitioned numbers.

### The cause, measured independently

**3 October.** r07 at the 5 h rung ends on loss 299.85; r08 at the 1 h rung ends
on 10.04, a factor of **29.9** lower on **five times less data** — the epochs
table gives 774 against 3,870, which is the same 5×. An overfitting curve does
not have a cliff in it. (`RESULTS.md` says "nine times less data" at this point,
comparing against the 9 h rung rather than against r07. The 29.9 is r07 against
r08.)

It is not a cliff and not a quality difference. `ForwardTTSLoss` sums five terms
and the two largest are mean squared errors in **physical units**: f0 in hertz
and the L2 norm of each linear spectrogram frame, both at α = 0.1, neither
normalised anywhere. `features.compute` produces f0 in hertz with 0 for unvoiced
frames and energy as `np.linalg.norm(spec)`. coqui's own dataset pipeline would
z-score both; this project bypasses that pipeline.

Measured on 12 utterances of the 1 h rung and 12 in the 9 h rung but not the
1 h one, 22.05 kHz, through `src/train/features.py` itself:

```
term                             1 h rung    9 h only
1.0 * mel L1 vs mean predictor      1.814       1.972
0.1 * var(f0 in Hz)               569.131     653.995
0.1 * var(frame energy)            45.381      45.647
sum                               616.326     701.614

f0:      mean  98.0 Hz, sd 75.4     mean 100.8 Hz, sd 80.9, 15.2% unvoiced
energy:  mean  25.7,    sd 21.3     mean  24.1,    sd 21.4
```

A mean predictor scores about 617 to 702. r01 at 314.60 implies an f0 RMS error
of 51.7 Hz against a speaker standard deviation of 80.9 Hz, so the pitch
predictor has recovered about a third of the variance. r08 at 10.04 implies
10.0 Hz — 541 utterances seen 3,870 times have their f0 memorised. **That is the
whole 29.9.**

**The consequence.** The mel term is the only one that measures spectral quality
and it is **0.6 per cent of r01's logged loss** and 18 per cent of r08's. The
mel decoder is optimised against under one per cent of the gradient while an
unnormalised pitch head takes most of the rest.

```
mel share of r01's achieved loss:  1.814 / 314.60            = 0.005766
against the mean-predictor sum:    1 - 614.512 / 616.326     = 0.002943
after z-scoring f0 and energy:     1.814 / (1.814 + 2*0.1)   = 0.900695
fold change:                       0.900695 / 0.005766       = 156.21
second route:                      314.60 / 2.014            = 156.21
```

The shares are written to six places on purpose. Rounded to 0.0058 and 0.9007
the first line evaluates to 155.29, not 156.21, and the two routes would appear
to disagree when they do not.

(The first two are different quantities, not two routes to one: 0.58% of what
the model reached, 0.29% of what a mean predictor would.)

**The prediction and the measurement agree by two independent routes.** The loss
decomposition of 3 October predicted unintelligible FastSpeech 2; the ceiling
measurement of 5 October confirmed it and exonerated the vocoder.

**The fix is three lines in `features.py`** — z-score f0 and energy — and it
changes the `config_hash` of every FastSpeech 2 run: r01, r04, r07, r08, r09,
r10, r15, r18, r20, r21. Ten runs. See `07-FUTURE-PLAN.md` for the cost.

**The objective was not changed mid-matrix.** Instead `_record` on
`AdapterBase` keeps the scalar terms of each step's loss dict and the step
record carries them under `components`, so every run from r09 onward can be read
directly rather than reconstructed from a filter bank.

---

## 3. The other standing observations

### Every run flattened by step 50,000

Across the matrix. The most plausible single explanation is gradient clipping:
gradient norms of 97 to 1139 against `grad_clip 1.0` mean essentially every step
is scaled down by two to three orders of magnitude. `[Inference]` The note that
AdamW largely cancels a uniform rescale is an argument, not a measurement.

A 25-minute diagnostic outside the matrix would settle it — 5,000 steps on the
10-minute rung at clip 1.0 against clip 100 — and is reportable either way
without retraining anything. **Not yet run** (blocker b16).

### The grapheme arm never expanded numbers

**3 October.** A parity defect: both arms must be handed the same string.
`test_grapheme_number_parity.py` now has 11 assertions on it.

### MCD cannot resolve this comparison

It was also 34× too large at one point. It is reported against a chance level
because the raw figure cannot be read alone. The lesson is general and is made
twice in this project in opposite directions: **a measure that cannot resolve a
contrast is not evidence about that contrast, and a measure that ranks
confidently is not thereby measuring what you want.**

### Griffin-Lim was exonerated; the mel is over-smoothed

**3 October.** The metallic quality of the placeholder clips was blamed on
Griffin-Lim. It was the mel. Every run had converged by 50k.

### Where the phoneme arm is exposed

**3 October.** Every word-final consonant. Word-final deletion is
near-categorical in Hindi and both arms learn it, which is why the demo set is
banded on **medial** deletion and word-final is reported but not selected on.

---

## 4. The demo and the listening page

Rendered on the DGX: 8 arms, 12 sentences, 96 clips, all four FastSpeech 2 arms
carrying `hifigan:r06@18000`, the four VITS arms end to end. `[Unverified]` Those
figures and a total of 17.8 MB come from a render observed in a working session
on 5 October and are not in `RESULTS.md`. 96 = 8 × 12 and `build_demo_set.py`
defaults to `--count 12`, so the shape is consistent; the byte total is not
corroborated anywhere.

The page groups arms by what they share — the comparison, the same arm at other
seeds, the same arm with less data — and names the training noise explicitly so
a listener knows what scale a difference has to beat. It states outright that
the FastSpeech 2 arms are not intelligible and why, naming **r01 at 0.993 and
r04 at 0.992** as the two that were measured, and **r07 and r08 as not
measured**. Lending those two a neighbour's number would be a fabricated
measurement on a public page.

The demo sentence set is chosen by a stated, deterministic rule from the
held-out dev split: filtered to a listening length, banded by medial deletion
count, taken from the richest band down, sorted by utterance id. Sentences with
no medial deletion are kept as a control.

---

## 5. Measurements that are cheap and have not been made

- Stress-test error rates per architecture
- MOS proxies on Devanagari, or labelled English-trained
- macOS real-time factor on the M5
- The clipping diagnostic
- The full 300-utterance test split at 5 draws (everything above is 50)

None needs a GPU.
