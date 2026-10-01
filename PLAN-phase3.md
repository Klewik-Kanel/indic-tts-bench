# Plan: finish phase 3

Written 1 October 2026, 23:45 IST. Phase 3 is the critical path in
`project-plan-v3.md`: nothing in phases 4, 5 or 7 can start until the adapters
train. This file is the step-by-step for closing it.

---

## 0. A correction owed to plan v3 first

Plan v3's phase table says of 3b: "FastSpeech 2, VITS and Matcha-TTS raise
`NotImplementedError` by design". That is no longer true and has not been true
since the DGX work. Read from `src/train/adapters.py` on the Mac clone today:

| Adapter | State in the code | Evidence |
|---|---|---|
| `toy` | complete, real parameters, real masked loss | `assert_not_toy` keeps it out of results |
| `fastspeech2` | complete: `build`, `collate`, `loss` all written | r04 trained 100,000 steps, final loss 302.46 |
| `vits` | complete code, fails at runtime | has not passed one step |
| `matcha` | stub, all three methods raise `NotImplementedError` | deliberate: "deferred past the current deadline" |
| `hifigan` | **no adapter class at all** | declared in `ARCHITECTURES`, absent from `ADAPTERS` |

So phase 3b is one architecture working, one architecture broken, one stub, and
one missing. 3a is done and its resume proof stands.

---

## 1. Step 1 — VITS past one step

Six of the nineteen runs are VITS. Four glue bugs have already been paid for and
are recorded in the handoff; this is the fifth and it is a different kind of bug,
because reading the code today turns up two problems that are not layout
problems.

### 1a. The runner has one optimiser, not two

`src/train/runner.py` line 118 builds a single `AdamW` over
`model.parameters()`. There is no second optimiser anywhere in the file.
`VitsAdapter.loss` carries the comment "the discriminator step is handled by the
runner's two-optimiser path, which is the declared deviation" — that path does
not exist. The adapter is written against a runner feature that was planned and
not built.

Consequence even if the step runs: one optimiser over all parameters means the
generator and the discriminator are updated by the same step against a single
scalar loss, so the adversarial training is not adversarial. VITS would produce
a number, and the number would not be VITS.

### 1b. `optimizer_idx=0` is probably the discriminator, not the generator

`VitsAdapter.loss` calls `model.train_step(t, self._criterion, optimizer_idx=0)`
with the comment "optimizer_idx 0 is the generator".

[Inference] In coqui's VITS the order is the other way round: `get_optimizer`
returns the discriminator optimiser first and the generator second, and
`train_step` with `optimizer_idx=0` runs the discriminator branch, which caches
its forward outputs for the generator branch to reuse at `optimizer_idx=1`.
If that is right, the current call asks for the discriminator loss while the
code around it expects a generator loss, and the generator branch is never
reached. This is expected from how that class is written, not verified here:
coqui is not installed in the Mac `.venv`, so it could not be checked locally.

**Verify it from the installed source on the DGX before changing a line.** This
is cheap and decisive, and it replaces a guess with the actual signature:

```
cd /workspace/indic-tts-bench && /workspace/venv/bin/python - <<'PY'
import inspect
from TTS.tts.models.vits import Vits
for fn in ("train_step", "get_optimizer", "get_criterion", "get_lr"):
    f = getattr(Vits, fn, None)
    print("="*30, fn, "="*30)
    print("absent" if f is None else inspect.getsource(f)[:3500])
PY
```

Read, in the printed source: which `optimizer_idx` takes the discriminator
branch, what `get_criterion` returns and in what order, whether `train_step`
reads a cache populated by the other branch, and what keys it expects in the
batch. Then fix the adapter to match what the source says rather than what the
comment says.

### 1c. Then build the two-optimiser path

In `runner.py`, when the adapter declares two optimisers:

- Build them from the model's own `get_optimizer()` so the parameter groups are
  split the way upstream splits them, instead of guessing which modules are the
  discriminator.
- Step them in the order the source requires, discriminator branch first, and
  `zero_grad` each separately.
- Clip each optimiser's own parameter group, not `model.parameters()` wholesale.
- Checkpoint both optimiser states. `checkpoint.py` currently stores
  `{"model", "optimizer"}`; a VITS checkpoint needs both, or resume silently
  restarts the discriminator's Adam moments from zero and the bit-exact resume
  guarantee stops holding for six of the runs.
- Log both losses. One scalar hides which half is diverging.

**Done when:** `configs/r02.yaml` and `configs/r05.yaml` both complete two steps
under `src/train/dryrun`, the log shows a generator and a discriminator loss
moving independently, and a kill-and-resume dry run on r02 reproduces weights
bit-identically the way r01 already does. The resume check is not optional here:
it is the only thing that proves the second optimiser state survives a restart.

Block 3 in `PLAN-next-steps.md` reproduces the current failure with
`CUDA_LAUNCH_BLOCKING=1`. Run it before editing, so the fix is aimed at the real
traceback.

Note on r02: it previously stopped on a missing `wav16` file. That was correct
behaviour, not a bug — r02 is the 16 kHz MMS-initialised run. If it stops there
again, the 16 kHz copies are not staged on the DGX, and that is the fix.

---

## 2. Step 2 — the HiFi-GAN adapter, or the declared deviation

Gated on the vocoder decision in `PLAN-next-steps.md`. The two branches are
different amounts of work and only one of them is phase-3 work at all.

**If option 2 wins (pretrained):** no adapter is written. Instead, phase 3 closes
with the vocoder marked as an external dependency, and the work moves to the
export path: confirm the checkpoint's sample rate and hop length match the
22.05 kHz master and hop 256, then make `src/export/bundle.py --vocoder` accept
an external checkpoint. A mel configuration mismatch here is silent and ruins
every audio metric downstream, so it is checked before anything is synthesised,
not after.

**If option 1 wins (train it):** this is a fourth adapter and the first with a
discriminator of its own, so it reuses the two-optimiser path from step 1. That
is the argument for doing VITS first regardless of this decision: the
machinery HiFi-GAN needs is the machinery VITS needs.

A HiFi-GAN adapter also differs from the other three in its inputs: it takes mel
and waveform and no text at all. `config.py` already enforces that
(`input_repr == "none"` for `hifigan` and never for anything else), so
`collate` supplies no `tokens` and the `TextEncoder` is not involved.

**Done when:** either r06 completes two steps under the dry run with both losses
moving, or the deviation is written into the methods table and
`RESULTS.md`, with the pretrained checkpoint's identifier, sample rate and hop
length recorded.

---

## 3. Step 3 — Matcha-TTS

One run, r03, one slot, last in the schedule. The stub says "deferred past the
current deadline" and keeping it in the matrix is deliberate so the plan does not
quietly lose it.

Decide explicitly rather than by drift: either write the adapter in the same
shape as the other two, or move r03 to future work in the paper alongside
StyleTTS 2, which is already excluded on compute. Three architectures is the
stronger claim; two is a defensible one, and the dissertation's question is about
phonemic versus graphemic input, which two architectures already answer twice.

**Done when:** r03 trains, or plan v3 and the paper both say Matcha is future
work for the same stated reason as StyleTTS 2.

---

## 4. Step 4 — close the phase properly

Phase 3 is not finished when the code runs. Three things make it finished.

**The adapter parity check.** `assert_budget_matched` already refuses runs that
differ in budget. Nothing yet checks that two adapters saw the same data. Add a
test that one batch, passed through both the FastSpeech 2 and the VITS
`collate`, carries the same utterance ids in the same order and the same frame
counts. A silent divergence there would put a data difference inside an
architecture comparison, which is the one place the study cannot afford it.

**The seed variance floor.** Two runs, same config, different seed. Until that
number exists, no difference between any two runs can be reported as a finding.
It is listed in plan v3's standing rules and it is still unmeasured. It is two
runs, so it belongs in a pair slot, and it should go in before the ladder rather
than after, because the ladder's smallest rungs are exactly where the variance
is widest and the claims are weakest.

**A `RESULTS.md` entry per adapter.** With the command, the commit, and the
step count. The rule holds: nothing is a result until it is marked as one, and
no number reaches the paper that cannot be regenerated from a config file.

---

## 5. Order, and why this order

```
1. verify coqui's train_step signature on the DGX     minutes, no GPU
2. fix VitsAdapter.loss to match it                   minutes
3. build the two-optimiser path in runner.py          the real work
4. prove r02 and r05 dry-run, then resume bit-exact   minutes
5. vocoder branch: external checkpoint, or r06        decision-gated
6. Matcha: write it, or move it to future work        a decision, not code
7. parity test, seed variance pair, RESULTS entries   closes the phase
```

Steps 1 through 4 need no decision from anyone and unblock six runs. Step 5 needs
the vocoder answer. Step 6 needs a scope answer. So the order above puts every
item that is blocked on a person after every item that is not.

---

## 6. What is being claimed here, and what is not

Verified by reading the repository today: the adapter inventory in section 0,
the single optimiser in `runner.py`, the `hifigan` gap between `ARCHITECTURES`
and `ADAPTERS`, and the Matcha stub's text.

[Inference] The `optimizer_idx` diagnosis in 1b. It follows from how coqui's VITS
is written, it is the most likely cause of a run that never passes the loss call,
and it is not verified in this session. The command in 1b verifies it.

[Unverified] That the two-optimiser path is the only thing between VITS and a
completed step. Four glue bugs were already fixed before this one, and the
traceback from Block 3 may name a fifth that has nothing to do with optimisers.
