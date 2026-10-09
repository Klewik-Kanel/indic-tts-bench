# Defects found, and traps already paid for

Two kinds of thing here. **Defects** are bugs in this project's own code that
produced wrong numbers or would have. **Traps** are properties of the
environment that cost days and will cost them again.

The reason this file is long is that most of these were invisible: the code ran,
the output looked reasonable, and the number was wrong.

---

## 1. Defects that corrupted a measurement

### Punctuation moved words between error classes

A trailing comma survived `normalize()` and added a segment, so `word_stats`
reclassified a word-final site as medial — in **6 of 6 sampled words**. The
punctuation share by class was 0.0000 / 0.0690 / 0.0561, so the three classes
were not even equally affected. Fixed with one shared `normalise_for_scoring`
used by both the scorer and the partition.

### Synthesis was unseeded, and a finding reversed sign

Every intelligibility number before 4 October rested on **one unseeded draw**.
The medial-site finding reversed sign on re-run. Fixed with `draw_seed(run_id,
text, draw)` — a salted SHA-256 — and a `--draws` flag, and the claim was
withdrawn in full rather than re-stated.

### The bootstrap and the table were computed from different samples

`record["utterances"]` held draw 0 alone while the class rates pooled all five
draws. Same quantity, two answers: +0.0057 against +0.0263. Fixed with
`_merge_draws`, which merges per utterance before resampling so the interval has
the same characters the rate does and the utterance stays the sampling unit.

### `Bundle.device` is a string, and 250 synthesis calls raised in silence

The code read `self.device.type`. `device` is a `str`, so every call raised, and
the harness printed `50/50` progress while transcribing nothing. Fixed to
`str(self.device).startswith("cuda")`, plus abort-on-total-failure and
per-cause failure summaries.

### The vocoder's best weights were thrown away

r06's best validation loss was **0.246545 at step 12000**; what got exported was
step 18000 at **0.262312** — 6.40% worse, 2.25σ. Evaluation ran every 2,000
steps and checkpoints every 5,000, and r06's saved checkpoints were 15,000 and
18,000, so **no checkpoint was ever written at 12,000** and the best weights
could not be recovered. `tests/test_best_checkpoint.py` states the rule that
follows: `eval_every` has to divide `ckpt_every`.

**Half of this is fixed and half is deliberately not.** The discarded weights
are fixed: the runner now saves on a new best and records it in `best.json`.

`min_delta` is **not** fixed. It was 1e-4 against a noise level of 0.012466,
**125× too small** to ever fire, and `Plateau.min_delta` still defaults to
`0.0`. `tests/test_best_checkpoint.py` records why: changing it changes what the
two finished vocoders would have done, and they are not being re-run. It is a
measurement on the record, not an outstanding bug.

### A truncated error message invented a repository name

A 110-character truncation turned
`ai4bharat/indic-conformer-600m-multilingual` into `ai4bh`, and that fake name
was reported to the user as the repository that failed. **The user caught it.**
Fixed with `_first_full_message` at 400 characters. A truncation that invents a
plausible wrong answer is worse than a long line.

### `Infinity` in `data.json` blanked the public page

r08's corrupted `config_hash` (a float `inf`) travelled through its config, its
checkpoints, its bundle manifest and into `data.json` as the bare token
`Infinity`. Python's `json` **writes** it and **reads** it back; a browser's
`JSON.parse` refuses the whole document on one such token. Every arm vanished
and the page said "Could not load data.json".

**Two holes, because Python's json is more permissive than JSON.parse in both
directions.** `render_demo` emitted it, and `publish_space` audited the file
with the permissive reader and reported "checks pass" on a page that could not
load. Fixed: `json_safe` replaces every non-finite value with the string
`"not-recoverable"` and names the field; `allow_nan=False` as a backstop; the
audit parses with a `parse_constant` that raises; `bundle.py` keeps a config
hash as text.

### The render died on a vocoder bundle and nobody noticed

`exports/` holds the vocoders beside the voices and the default glob takes
everything. Asking a vocoder to speak raises `SystemExit`, which is **not** an
`Exception`, so it walked past the per-sentence handler and killed the render.
`data.json` is written last, so a run that had already produced every clip of
every real arm wrote nothing, and the page kept serving the previous day's file
with its Griffin-Lim labels. Fixed: vocoders are dropped by manifest before any
weights load, and the vocoder is attached by `needs_vocoder` rather than by
catching `attach_vocoder`'s refusal — which had also been swallowing genuine
rate and language mismatches and turning them into silent arms.

A second guard followed: `publish_space` now refuses a `data.json` older than
the newest wav by more than 120 seconds, because `render_demo` writes it last.

### The Pages gate would have deployed a broken site

`.github/workflows/pages.yml` ran under `set -u` and not `set -e`. The strict
JSON check printed its error, exited 1, and the next line wrote `ready=true`
anyway. **Found by executing the step, not by reading it.**

### The offload would have backed up nothing

The `offload` step **in `scripts/queue_all.sh`** guarded on `HF_REPO` being
non-empty and skipped the call entirely without it, while `offload.py` itself
defaults to `Klewik/indic-tts-bench`. A chain armed from a shell that never
exported `HF_REPO` would have uploaded nothing. Fixed at `queue_all.sh:43` with
`HF_REPO=${HF_REPO-Klewik/indic-tts-bench}` — no colon, so an explicit empty
still disables backups, which is the only way to turn them off deliberately.
`offload_loop.sh` does not set it and expects the caller to.

### The offload did not carry the splits

Weights trained on an unrecorded split are not reproducible by anyone. The audio
is a public corpus; which utterance landed in train, dev or test is not, and
`SPLITS.lock` is the only record. It lived only on the DGX, was never committed,
and the offload did not carry it. Fixed 5 October.

### A duplicate `plan()` call walked the filesystem twice

An edit shadowed the original definition. Caught by reading the diff.

---

## 2. One inference that measurement reversed

**4 October:** "r06 is not a working vocoder." Inferred from its held-out mel L1
of 0.2465 losing to a Griffin-Lim round trip at 0.1434, a factor of 1.72.

**It was wrong.** The ceiling measurement puts the vocoder at 0.0005 CER on
IndicConformer and 0.0045 on MMS — very nearly transparent. **A spectral
distance and an intelligibility proxy rank these two systems in opposite
orders.**

This is the same lesson the project makes against MCD, turned against its own
reasoning. It is the single most instructive error in the record and is worth
re-reading before any confident inference about a measure.

---

## 3. My own test false positives, which are their own category

Several tests passed while testing nothing. All found and fixed, listed because
the failure modes recur:

- A `mean(` substring check caught the across-draws average instead of the
  intended function.
- A source slice from `reference_floor` to `_select` swept in `vocoder_ceiling`'s
  legitimate `wav22`.
- A determinism test built a **uniform** effect, so every seed agreed and the
  test could not fail.
- `2 + 9` asserted for an 8-character Devanagari string.
- `pytest.approx` applied to arrays.
- Source-grep tests that passed while the code path they described was
  unreachable — twice, and both times a real bug was behind them.

**The rule this produced: write tests that execute the thing.** The page's
grouping is tested by running its script under `node` and reading the markup.
The Pages gate is tested by running it under `bash` in five states and reading
the exit code and the outputs it set.

---

## 4. Environment traps

### The DGX container image

Four faults, each diagnosed and recorded in `scripts/11_dgx_env.sh`:

1. a dead NVIDIA package index in four separate `pip.conf` files
2. `PIP_CONFIG_FILE=/dev/null` does not cover the site-level config
3. pip 24.3.1 crashes in its own version parser
4. PyPI's torchaudio fails against NVIDIA's torch with an ABI error

Training therefore runs from `/workspace/venv` with a matched upstream stack.

### Always `/workspace/venv/bin/python`

Plain `python` is the container's system install. `import torchaudio` fails with
`undefined symbol`. Hit more than once, including by me after writing the rule
down.

### PyTorch's caching allocator squats

It holds every block it ever used, so a run can sit on far more of the card than
it needs. `TRAIN_GPU_FRACTION` caps it; at 0.25 on this card that resolves to
10,084 MiB. It is a scheduling knob and never part of the budget.

`[Unverified]` The 1 October handoff quoted "a run needing 4.2 GB sat on 37.5 GB
of a 40 GB card". **Those two figures are not in `RESULTS.md`** and I could not
find their source. The behaviour is real and the cap works; the numbers are
unconfirmed.

### The DGX is shared, and the step rate is not ours to set

**Treat every wall-clock figure in this project as unreliable.** `RESULTS.md`
says so directly on 2 October, and the handoff documents that quote 6.76 h per
pair all descend from the number it retired.

What was measured, on an **idle** card, and written into
`handoff/project-docs/PLAN-full-matrix.md`: one run alone 3.54 it/s; two runs
sharing 4.11 it/s each; 100,000 steps = 6.76 h per run and per pair. *(Those
three are not mutually consistent as written — 3.54 alone against 4.11 each
while sharing is a 2.32× total speedup, not the 1.38× the document claims. The
inconsistency is in the source and has never been resolved.)*

What happened next: a **third process appeared on the card and was not ours**,
holding 11,192 MiB, above the 10,084 MiB that `TRAIN_GPU_FRACTION=0.25` allows.
The card sat at 99% utilisation with no throttling while our runs crawled, and
the host load average was 293.82 on 256 cores, of which our two trainers
accounted for about 8. **The ladder rungs slowed from 4.11 it/s to about 1.3.**

`RESULTS.md`'s own verdict: *"Every wall-clock figure in `PLAN-full-matrix.md`
descends from 4.11 it/s measured on an idle card. That number is not a property
of the run; it is a property of the card at the time, and it is now unreliable
in both directions."*

**The step budget is unaffected** — 100,000 steps is 100,000 steps whoever else
is on the machine, and nothing about the comparison between runs changes. Only
the calendar does. Any schedule from here needs a **measured contention
factor**, not an idle-card rate, and nobody has measured one.

Inside a container `nvidia-smi` reports every foreign process as `[Not Found]`,
because the names are in another PID namespace, and `ps` sees only our own. That
is why `status.sh` lists every process on the card rather than only ours.

**Open question for the lab, not for the code:** whether this DGX is scheduled
or first-come. If runs share it routinely, pairing two of our own on one card is
the wrong default.

### Git on the Mac through the device bridge

Every git write left a `.lock` file that could not be removed, and the next git
command then failed with "Another git process seems to be running". Deletion had
to be granted for the folder before this stopped. If it recurs: move the lock
aside, or request delete permission for `/Users/klewik/indic-tts-bench`.

Git on the bridge also needs `GIT_AUTHOR_NAME` / `GIT_COMMITTER_NAME` and the
emails as environment variables, or set them in the clone's config.

### Placeholders in shell commands — twice

`<the repo id>` on 3 October and `<your token>` on 5 October. Bash parses `<` as
a redirect and the command fails with a syntax error. The second occurrence was
a repeat of a correction already made. **Never hand over a command containing
angle-bracket placeholders.**

### Commits that were never pushed

Three times a flag was "unrecognised" on the DGX because the commit adding it
was sitting unpushed on the Mac. The division of labour means the assistant must
say *"push, then run"* every single time, not assume.

---

## 5. Two framing errors worth recording

**The loss cliff was half misframed.** The initial claim was "either the 1 h
rung is memorised outright or the logged loss is not comparable". **The first
disjunct turned out to be right** — `RESULTS.md` concludes "541 utterances seen
3870 times have their f0 memorised. That is the whole 29.9." The second was
wrong: the rungs are nested subsets of one speaker, so the units are identical
and the loss *is* comparable in that narrow sense. What the framing missed is
which term dominates — the mel term is 0.6% of r01's loss. The section is titled
"resolved" rather than "correction", so do not go looking for a retraction.

**Pair-cost estimates were stated as measured when they were guesses.** A
comment claimed 4.2 h against 9.7 h; the values put in their place were 6.24 h
and 18.58 h, now at `queue_all.sh:74` and `:79`. Note that those too descend
from an idle-card rate, so the correction fixed the provenance of the numbers
rather than making them trustworthy.
