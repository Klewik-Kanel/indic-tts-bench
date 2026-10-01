# Plan: next steps from the evening of 1 October

Written 1 October 2026, 23:30 IST. Companion to `PLAN-full-matrix.md`, which holds
the schedule. This file holds the decision that gates it and the terminal blocks
that carry it out. Live board: the "Indic TTS Run Board" artifact.

---

## 1. The dilemma, stated once

FastSpeech 2 and Matcha emit mel spectrograms. Without a vocoder, r01 and r04 —
the Hindi ablation pair, which is the dissertation's core comparison — cannot be
heard, cannot be scored on audio, and cannot be demonstrated. `hifigan` is a
declared architecture in `src/train/config.py` and has configs at r06 and r17,
but there is no `hifigan` entry in `ADAPTERS`. Verified on the Mac clone, 1 Oct.

**Option 1. Write the adapter, train r06 + r17.**
Every number comes from this project's own training, the vocoder is matched to
the speaker and to the 22.05 kHz master, and there is no deviation to declare.
Against it: no adapter exists, HiFi-GAN is the first of the four with a
discriminator, and the 7.0 h figure is a plan estimate that has never been
measured. It pushes the full matrix past Sunday.

**Option 2. Synthesise with a pretrained HiFi-GAN.**
Audio exists today; evaluation and the listening test unblock immediately. It
frees exactly the hours by which the full matrix overruns Sunday. Against it: a
deviation to declare in the methods table, an absolute-MCD shift from speaker
and language mismatch that is untested here, and r06 and r17 drop out.

### The arithmetic behind "exactly"

Inputs read from `PLAN-full-matrix.md` section 4: 15 remaining acoustic runs,
6.76 h per pair sharing the card, 7.0 h for the vocoder pair, 54 h available to
Sunday 4 October midnight.

    pairs                  ceil(15 / 2) = 8
    acoustic wall clock    8 * 6.76 = 54.08 h
    total with vocoders    54.08 + 7.0 = 61.08 h
    overrun, acoustic only 54.08 - 54 = 0.08 h
    overrun, with vocoders 61.08 - 54 = 7.08 h

Checked a second way: 15 * 6.76 / 2 = 50.70 h. The two routes disagree, 54.08
against 50.70, because the odd run (r03) occupies a full 6.76 h slot alone.
54.08 is the schedulable figure; 50.70 is the figure if r03 could share with
something, and nothing is left for it to share with. Both are recorded rather
than one being chosen silently.

### Recommendation, and what it rests on

Option 2. The 7.08 h shortfall against Sunday is the vocoder pair almost exactly.

[Inference] The vocoder is shared by both arms of every comparison in the
matrix, so it cancels in the phonemic-versus-graphemic contrast: it shifts
absolute MCD, not the measured effect. Expected, not guaranteed. The speaker and
language mismatch of a pretrained vocoder has not been tested in this project,
and the claim should be checked on r01 and r04 before it is relied on in the
report.

[Unverified] No pretrained HiFi-GAN checkpoint has yet been confirmed reachable
from the DGX, or confirmed to match the 22.05 kHz master and hop 256. Block 1
below tests reachability. Matching the mel configuration is a separate check and
must happen before any audio from it is scored.

If Sunday is not the real deadline, Option 1 is the better dissertation and
Monday night fits it with slack. That is the question to answer first.

---

## 2. Sequence

1. Settle the vocoder question. Everything on audio waits on it.
2. Pre-warm the Marathi feature cache, both sample rates. CPU only, safe beside
   a running job. Skipping it makes the first epoch measure librosa.
3. Confirm r01 reached step 100,000.
4. Launch r15 + r18, the Marathi FS2 pair. Gated on 2 and 3.
5. Get VITS past one training step. Six runs depend on it.
6. Write ASR-WER, predicted MOS and RTF into `src/eval/`. Gated on 1.
7. Measure the seed variance floor: two runs, same config, different seed. Until
   that number exists, no difference can be reported as a finding.
8. Build the listening test. Stimuli low-passed to 8 kHz for every system.
9. Fix the four corrections owed to the written report: FastSpeech 2 is 37.99 M
   parameters not ~27 M, precision is bf16 not fp16, the matrix is 19 configs,
   and the two stale cross-references in sections 2.1.4 and 3.9.

---

## 3. Terminal blocks

One paste each, each printing one report. The blocks are clubbed on purpose: a
screenshot costs far more than pasted text, and a round trip per command costs
more than either.

### Block 1 — assess, and pre-warm Marathi

Answers four questions at once (did r01 finish, what is on the card, is the
Marathi cache warm, can the DGX reach the Hub) and then warms the cache.

```
cd /workspace/indic-tts-bench && git log --oneline -1 && echo "=== status" && bash /workspace/status.sh && echo "=== r01 / r04 tail" && tail -n 3 /workspace/runs/r01.log /workspace/runs/r04.log && echo "=== checkpoints" && ls -1 /workspace/runs/r01 /workspace/runs/r04 2>&1 | tail -n 20 && echo "=== gpu" && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv && nvidia-smi --query-compute-apps=pid,used_memory --format=csv && echo "=== marathi cache before" && du -sh /workspace/features/marathi* 2>&1 | tail -n 5 && echo "=== hub reachable?" && /workspace/venv/bin/python -c "import urllib.request,json; r=urllib.request.urlopen('https://huggingface.co/api/models?search=hifigan&limit=5',timeout=20); print([m['modelId'] for m in json.load(r)])" 2>&1 | tail -n 3 && echo "=== prewarm 22050 (pitch)" && /workspace/venv/bin/python scripts/12_prewarm_features.py --lang marathi --sr 22050 --pitch 2>&1 | tail -n 6 && echo "=== prewarm 16000" && /workspace/venv/bin/python scripts/12_prewarm_features.py --lang marathi --sr 16000 2>&1 | tail -n 6 && echo "=== marathi cache after" && du -sh /workspace/features/marathi* 2>&1 | tail -n 5 && echo "=== BLOCK 1 DONE"
```

The feature-cache path is a guess at `/workspace/features/`; if `du` reports no
such file, the `--help` of `scripts/12_prewarm_features.py` names the real one
and the prewarm itself still runs.

### Block 2 — launch r15 + r18

Run only once Block 1 shows r01 at step 100,000 and the cache warm. Both arms of
one ablation share the card, so neither arm gets a wall clock the other did not.

```
cd /workspace/indic-tts-bench && for R in r15 r18; do TRAIN_GPU_FRACTION=0.25 nohup /workspace/venv/bin/python -m src.train.launch configs/$R.yaml --out /workspace/runs/$R --ckpt-every 5000 --log-every 100 > /workspace/runs/$R.log 2>&1 & echo "launched $R pid $!"; done; sleep 90 && echo "=== status" && bash /workspace/status.sh && echo "=== first lines" && tail -n 12 /workspace/runs/r15.log /workspace/runs/r18.log && echo "=== gpu" && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv && echo "=== BLOCK 2 DONE"
```

### Block 3 — reproduce the VITS failure readably

`CUDA_LAUNCH_BLOCKING=1` moves a device-side assert back to the Python frame
that caused it; without it the traceback names no tensor. Paste the whole
traceback, not the last line.

```
cd /workspace/indic-tts-bench && echo "=== adapters registered" && /workspace/venv/bin/python -c "from src.train import adapters; print(sorted(adapters.ADAPTERS))" && echo "=== r02 dryrun, 2 steps" && CUDA_LAUNCH_BLOCKING=1 /workspace/venv/bin/python -m src.train.dryrun configs/r02.yaml --steps 2 2>&1 | tail -n 60 && echo "=== r05 dryrun, 2 steps" && CUDA_LAUNCH_BLOCKING=1 /workspace/venv/bin/python -m src.train.dryrun configs/r05.yaml --steps 2 2>&1 | tail -n 40 && echo "=== tests" && /workspace/venv/bin/python -m pytest -q 2>&1 | tail -n 15 && echo "=== BLOCK 3 DONE"
```

r02 is the 16 kHz MMS-initialised run and previously stopped on a missing
`wav16` file, which was correct behaviour rather than a bug. If it stops there
again, the 16 kHz copies have not been staged on the DGX and that is the fix,
not the adapter.

---

## 4. Open inconsistency in the written material

The roll number appears as 2025PA17326 in `HANDOFF-start-here.md` and in a
different form in the synopsis PDF. One of the two is wrong and it belongs on a
title page, so it needs checking against the enrolment record rather than
against either document.
