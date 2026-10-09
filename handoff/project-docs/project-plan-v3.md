# Project Plan v3: Does Explicit Devanagari G2P Still Matter for Neural TTS?

Version 3, 1 October 2026. Supersedes v2 of 13 September. The research question,
the contributions and the repository layout are unchanged from v2 and are not
restated here; read v2 for those. This version exists to record what has
actually been built, what the plan got wrong, and where the critical path now
runs.

---

## 0. What changed from v2

**Forced alignment is out.** v2 made FastSpeech 2 depend on Montreal Forced
Aligner, and made Marathi depend on training an MFA acoustic model from the
rule-based dictionary. Both are gone, for three reasons in order of weight.
MFA's Hindi model is trained on a different corpus and a different phone set,
so its durations would carry an unmeasured mismatch into exactly the arm the
study is testing. Training a Marathi model from our own dictionary would make
the alignment a function of the schwa rule, which is the variable under test.
And FastSpeech 2 can take durations from an internal unsupervised aligner
(Badlani et al., ICASSP 2022), which keeps the duration source identical across
the phonemic and graphemic arms. The 3-day MFA risk line in v2 is closed.

**The matrix is 17 runs, not 19.** Two of v2's runs were the same run. The
ladder's 9-hour rung duplicated the Hindi main run, and the duplication hid
inside the config hash because the run identifier was part of it. 146 GPU-hours
becomes 136.

**StyleTTS 2 stays out, now on the record.** v2 dropped it in passing. It is
now explicit future work, excluded for compute rather than for quality.

**The ladder starts at 9 hours, not 10.** Matching the rungs across the two
languages matters more than a round number.

**The front end has a real accuracy figure.** v2 promised one. See section 2.

---

## 1. Phase table, with status

| Phase | Work | Status |
|---|---|---|
| 0 | Repo scaffold, G2P front end, stress-test set, unit tests | **Done.** Zero unknown symbols over 870,547 phone tokens on 11,044 transcripts |
| 1 | Folder connected, egress mapped, secrets installed, Kaggle round trip proven | **Done.** Egress confirmed denied for Kaggle and Hugging Face from both of Claude's environments; every network step runs from a native terminal |
| 2 | Download, profiling, standardisation, frozen splits, ladder subsets | **Done.** 22,764 raw utterances, 22,088 standardised. Splits frozen by salted SHA-256, `SPLITS.lock` written, three structural assertions pass. Ladder built at 9 h, 5 h, 1 h, 30 min, 10 min, nested |
| 2b | Schwa rule validated against native-speaker judgement | **Done, 1 Oct.** 39/47 = 83.0%. See section 2 |
| 3a | Shared training loop, batching, schedule, checkpointing, resume | **Done, 1 Oct.** Resume proven bit-identical to an uninterrupted run. See section 3 |
| 3b | The three real adapters, built inside the Kaggle job | **Open. This is the critical path.** FastSpeech 2, VITS and Matcha-TTS raise `NotImplementedError` by design until they are constructed where their upstream packages and checkpoints are reachable |
| 4 | Main runs, grapheme ablation, vocoder | Blocked on 3b |
| 5 | Ladder and Marathi control | Blocked on 4 |
| 6 | Objective evaluation, RTF on the M5, tables and figures | **Part built.** MCD and log-F0 done and verified against analytic answers. ASR-WER, predicted MOS and RTF not written. `src/analysis/` is empty |
| 7 | Listening test: build, pilot, recruit, run, analyse | **Not started.** `listening_test/` is empty. Gated on real audio from phase 4 |
| 8 | Paper draft, testbed release, checkpoints published | Phase-I progress report and defence deck delivered. Paper itself blocked on 6 and 7 |

Tests: 25 new in `tests/test_train_loop.py`, 90 passing without an audio stack,
96 with one.

---

## 2. The front-end accuracy figure

39 of 47 words, 83.0%, scored against gold forms built from native-speaker
yes/no judgements rather than from the author's own answer key. This replaces
the 49/49 self-consistency figure, which was never a result. Full error
analysis is in `RESULTS.md` under 1 October. Four things the paper must carry
with the number:

1. Three speakers' responses were averaged into one sheet before the merge ran,
   so inter-speaker agreement cannot be computed and the variation rate reads as
   zero, which is an artefact of the averaging and not a finding.
2. Seven of the eight errors are speakers keeping a schwa that the rule deletes.
   That is citation-form reading. 83.0% is a lower bound under conditions that
   favour retention.
3. Four of the eight are a single environment: word-final schwa after a
   consonant cluster in Sanskrit-derived words. `SchwaConfig.block_before`
   exists for this and stays empty, because populating it from these four words
   would fit the rule to its own test set. The fix needs held-out tatsama words.
4. One answer is impossible (न answered as having no vowel) and is left in the
   denominator rather than quietly dropped.

---

## 3. What phase 3a settles

One training loop serves all four architectures, so the fixed-budget claim is
enforced in a single place instead of inherited from three upstream training
scripts that would each do something slightly different.

**Batching is by mel frames, charged as `len(batch) * max_frames`.** That is
what the GPU computes, so the budget bounds real memory rather than a sum that
ignores padding. On the Hindi 9 h set at 12,000 frames: 265 batches an epoch,
median 16 utterances, largest 53, nothing over budget.

**Resume is proven, not assumed.** `python -m src.train.dryrun configs/r01.yaml
--steps 20 --kill-at 10` kills a run, restarts it from its checkpoint, and
compares the weights against a run that was never interrupted. Bit-identical
across all tensors. The data order is derived from (seed, step) rather than
stored, so a checkpoint cannot disagree with its manifest, and the optimiser
state travels with the weights, so Adam's moments do not silently restart.

**The vocabulary comes from the inventory, not the corpus.** Every ladder rung
shares one embedding table, so the 10-minute run is not quietly a different
model from the 9-hour run.

A toy adapter with real parameters exercises the loop on CPU in seconds.
`assert_not_toy` refuses to let a config that names it produce a number.

---

## 4. Critical path from here

```
phase 3b adapters ──> main runs ──> ladder and control
                                          │
         phase 6 eval code (writable now) ─┤
         phase 7 listening test (writable now) ───┘
```

Phase 6's missing evaluation code and phase 7's listening-test infrastructure do
not depend on any trained model and can be written while GPU runs are queued.
Writing them after phase 4 would serialise work that does not have to be.

---

## 5. Open items carried from v2

Unchanged and still open: the free-tier budget, where the 5-hour ladder rung is
the shock absorber and seed replication is never cut; the ASR floor on
ground-truth audio, which goes in every WER table; the 16 kHz bandwidth cap on
MMS-initialised VITS, handled with dual-column MCD and 8 kHz low-passed
listening stimuli.

Resolved since v2: the MFA dependency, the duplicated run, the front end's
unknown error rate, and the untested resume path.

New and open: the elicitation needs a rerun with per-speaker sheets kept
separate if the agreement and variation numbers are to be reported at all.

---

**Note added 9 October 2026 during the account handover.** Several things in
this document have since been overtaken:

- Kaggle was abandoned for the NSUT DGX. The adapters were built there, not
  inside a Kaggle job. Phase 3b is done.
- The matrix is now **23 runs**, not 17: the Marathi control needed graphemic
  arms (r18, r19) and the seed-variance floor needed four runs of its own
  (r20–r23).
- The 16 kHz justification for VITS is **false** — the MMS warm start it rested
  on never happened. See `handoff/01-PROJECT-AND-METHOD.md` §4.
- `src/analysis/` and `src/eval/` are no longer thin. Phase 6 stood at 8 of 13
  on 3 October.
- Matcha-TTS (r03) moved to future work on 2 October.

Read `handoff/02-STATE-OF-PLAY.md` for the current position.
