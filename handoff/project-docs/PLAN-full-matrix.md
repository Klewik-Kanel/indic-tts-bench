# Plan: train the full 19-run matrix

Written 1 October 2026. Supersedes the 5-run "Saturday scope" cut, which was
made when the step rate was unmeasured and the adapters unbuilt. Both of those
are now settled, and the full matrix is reachable.

> **Note added 9 October 2026, during the account handover. Read this before
> section 1.**
>
> **Every wall-clock figure in this document is retired.** `RESULTS.md` on
> 2 October: *"Every wall-clock figure in `PLAN-full-matrix.md` descends from
> 4.11 it/s measured on an idle card. That number is not a property of the run;
> it is a property of the card at the time, and it is now unreliable in both
> directions."* A foreign tenant appeared and the ladder rungs slowed from
> 4.11 it/s to about 1.3. Section 4's arithmetic and the deadline table are
> therefore void, and any schedule needs a measured contention factor that
> nobody has measured. The step budget is unaffected.
>
> Section 1's three rates are also not mutually consistent: 3.54 it/s alone
> against 4.11 it/s *each* while sharing is a 2.32× total speedup, not the
> 1.38× claimed. That has never been resolved.
>
> "Four at once was tested against and rejected" means **argued against**, not
> run. There is no measurement behind it.
>
> The matrix also grew to 23 runs after this was written: r18 and r19 because
> the Marathi control had only phonemic arms and could not measure an effect,
> and r20–r23 as seed-variance floors. Matcha (r03) moved to future work.
>
> What is still worth having here: the **pairing rule** in section 3, the
> build-before-each-block list in section 5, the launch commands in section 6,
> and the standing rules in section 7.

---

## 1. Why the scope can be restored

The cut was made on a guess of 3.06 steps/s inherited from a plan written for a
Kaggle T4. The A100 measures better, and two runs share it:

```
one run alone        3.54 it/s
two runs sharing     4.11 it/s each   (speedup 1.38x total)
100,000 steps        6.76 h per run
two in parallel      6.76 h per PAIR
```

Memory is not the constraint. A run needs 4.2 GB of a 40 GB card;
`TRAIN_GPU_FRACTION=0.25` stops PyTorch's caching allocator from squatting on
the rest. Four at once was tested against and rejected: the gain from two is
already sublinear, and four would make each run crawl while quadrupling the
blast radius of an out-of-memory error.

---

## 2. The matrix, 19 runs

| Block | Runs | Purpose |
|---|---|---|
| Hindi main | r01 FS2-ph, r02 VITS-ph, r03 Matcha-ph | the three architectures on phonemic input |
| Hindi ablation | r04 FS2-gr, r05 VITS-gr | the same architectures on raw graphemes |
| Hindi vocoder | r06 HiFi-GAN | makes FS2 and Matcha audible |
| Ladder, FS2 | r07 5h, r08 1h, r09 30min, r10 10min | how the effect changes with data |
| Ladder, VITS | r11 5h, r12 1h, r13 30min, r14 10min | the same, second architecture |
| Marathi control | r15 FS2-ph, r16 VITS-ph, r18 FS2-gr, r19 VITS-gr | separates schwa from script |
| Marathi vocoder | r17 HiFi-GAN | makes the Marathi FS2 arm audible |

The 9-hour ladder rung is r01 and r02, not a separate run. Listing it again
would train the same thing twice.

---

## 3. Schedule, in pairs

Each line is one pair sharing the GPU, 6.76 h wall clock.

| Slot | Pair | Why these two together |
|---|---|---|
| done | **r04** | finished, 100,000 steps, loss 302.46 |
| now | **r01** | finishing, ~88% at time of writing |
| 1 | r02 + r05 | VITS Hindi, both arms. Blocked until the VITS adapter completes a step |
| 2 | r06 + r17 | both vocoders together; neither needs a text front end |
| 3 | r15 + r18 | Marathi FS2, both arms: the control's FS2 half |
| 4 | r16 + r19 | Marathi VITS, both arms: the control's VITS half |
| 5 | r07 + r11 | ladder 5 h, both architectures |
| 6 | r08 + r12 | ladder 1 h |
| 7 | r09 + r13 | ladder 30 min |
| 8 | r10 + r14 | ladder 10 min |
| 9 | r03 | Matcha, alone or beside whatever is left |

Pairing rule: **never pair the two halves of a comparison against different
neighbours.** r01 and r04 ran together, r15 with r18, r16 with r19. If one arm
of an ablation ran alone and the other shared the card, the two would differ in
wall clock, and someone will eventually ask whether that mattered. It does not,
because the budget is steps rather than time, but the answer is much easier to
give when the schedule was symmetric by construction.

---

## 4. Arithmetic

```
remaining acoustic runs      15
pairs                        ceil(15/2) = 8
acoustic wall clock          8 * 6.76 h = 54.1 h
vocoders (one pair)          7.0 h          [Unverified: not yet measured]
TOTAL                        61.1 h = 2.54 days
```

Against the clock, from the evening of 1 October:

| Deadline | Hours available | Verdict |
|---|---|---|
| Sun 4 Oct, midnight | 54 | does not fit, short by about 7 h |
| Mon 5 Oct, midnight | 78 | fits, 17 h of slack |
| Tue 6 Oct, midnight | 102 | fits comfortably |

**So the full matrix needs the GPU until Monday night, not Sunday.** If Sunday
is immovable, the ladder is what gets cut: it is eight runs and 27 hours, and it
is the one block whose absence weakens a secondary claim rather than the main
one. Cutting the Marathi control instead would remove the ability to separate
schwa deletion from script, which is the dissertation's second contribution.

The vocoder figure is the weak number here. It is a plan estimate that has never
been measured, and HiFi-GAN has no adapter yet, so treat 7 h as a placeholder
until one run has been timed.

---

## 5. What has to be built before each block

**Before slot 1 (VITS).** The VITS adapter must complete a training step. Four
glue bugs have been fixed and it is not yet past the loss call. Everything about
VITS is blocked on this, which is six of the nineteen runs.

**Before slot 2 (vocoders).** A HiFi-GAN adapter does not exist. `ADAPTERS` has
fastspeech2, vits, matcha and toy. Until this is written, FastSpeech 2 produces
mel spectrograms and nothing can be listened to, evaluated on audio, or
demonstrated. **This is the most important missing piece in the project right
now**, ahead of any further training.

**Before slots 3 and 4 (Marathi).** The Marathi feature cache must be
pre-warmed, exactly as Hindi was:

```
/workspace/venv/bin/python scripts/12_prewarm_features.py --lang marathi --sr 22050 --pitch
/workspace/venv/bin/python scripts/12_prewarm_features.py --lang marathi --sr 16000
```

Hindi took 1.7 minutes for 5,085 utterances across 48 workers. Marathi is 5,160,
so expect the same. Skipping this makes the first epoch measure librosa rather
than the GPU.

**Before slots 5 to 8 (ladder).** Nothing. The rungs are subsets of the Hindi
training set and their features are already cached.

---

## 6. Launching a pair

```
cd /workspace/indic-tts-bench
TRAIN_GPU_FRACTION=0.25 nohup /workspace/venv/bin/python -m src.train.launch \
  configs/r15.yaml --out /workspace/runs/r15 --ckpt-every 5000 --log-every 100 \
  > /workspace/runs/r15.log 2>&1 &
TRAIN_GPU_FRACTION=0.25 nohup /workspace/venv/bin/python -m src.train.launch \
  configs/r18.yaml --out /workspace/runs/r18 --ckpt-every 5000 --log-every 100 \
  > /workspace/runs/r18.log 2>&1 &
```

Check with `bash /workspace/status.sh`. Restarting after a kill is the same
command: the run resumes from its last complete checkpoint, and the data order
is derived from (seed, step) rather than stored, so a resumed run sees the batch
an uninterrupted one would have seen. That was verified bit-exact before any of
this trained.

Export a finished run so it can be demonstrated off the DGX:

```
/workspace/venv/bin/python -m src.export.bundle /workspace/runs/r04 \
  --out /workspace/exports --vocoder r06_step100000
```

---

## 7. Standing rules that do not bend for the schedule

- Every acoustic run is 100,000 steps at 12,000 mel frames in bf16. The budget
  is the thing being held constant and no run gets a different one.
- `TRAIN_GPU_FRACTION` is a scheduling knob, not part of the budget. It changes
  what can run beside a job, never what that job computes.
- No number reaches the paper that cannot be regenerated from a config file.
- `RESULTS.md` is appended to, never rewritten.
- No difference smaller than the measured seed variance is reported as a
  finding, and the variance floor has not been measured yet.

*(The last clause is now out of date: the floor was measured on 4 October. See
`handoff/04-RESULTS-AND-OBSERVATIONS.md`.)*
