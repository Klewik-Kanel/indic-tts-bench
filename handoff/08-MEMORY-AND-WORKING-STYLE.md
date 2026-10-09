# Memory to carry over, and how the user wants to be worked with

The old account's stored memory does not transfer between Claude accounts. This
file is its contents, as text, so the new account can be told the same things.

**To the new assistant:** read this, and do not save any of it to memory unless
the user asks you to. Facts about a person are theirs to volunteer.

---

## 1. Who

**Kaustubh Pandey.** M.Tech student at NSUT New Delhi. This dissertation is
his work. Supervisor **Dr. Shobha Bhatt**. GitHub `Klewik-Kanel`, email
`kaustubhpandey812@gmail.com`. Everything in this project — repository,
Hugging Face repositories, access tokens, commits — uses that identity.

The old Claude account was shared between Kaustubh and another student, Suyash,
and the account's profile belongs to Suyash. **If the new account is Kaustubh's
alone, none of Suyash's stored profile applies and should not be carried over.**
The only thing from it that bears on this work is the machine:

**Work machine:** MacBook Air M5 — 10 CPU cores, 10 GPU cores, 16-core neural
engine, 16 GB RAM, 1 TB storage. Used for data preparation, G2P, evaluation and
writing. Training is on the DGX.

---

## 2. Working structure, stated by the user and binding

> *"get back to our old working structure, where you will administer changes and
> ask me to push the git and then run things on dgx"*

- The assistant **edits and commits** in the Mac clone at
  `/Users/klewik/indic-tts-bench`.
- The **user pushes**. The assistant does not touch credentials.
- The **user runs everything on the DGX** and pastes the output back.
- Hand over **exact command blocks, ready to paste**. No placeholders in angle
  brackets — bash parses `<` as a redirect, and this has broken a command twice.

> *"i dont want you to be an co author, please dont do that"*

No `Co-Authored-By` trailer on any commit. Ever.

**Do not put Claude session links anywhere** — not in commit messages, not in
pull request descriptions. This is a standing preference.

Git identity for the clone:

```
git config user.name  "Kaustubh Pandey"
git config user.email "kaustubhpandey812@gmail.com"
```

---

## 3. Scope, stated by the user

> *"what i want is a clear result set on what was promised initially, can make
> slight adjustments, but not a lot."*

Decisions that followed from it:

- **3 October: Hindi only for the research paper.** *"lets do hindi only for
  now, as that will help me with the research paper, and we can add marathi
  later for the dissertation part 2."*
- **On VITS:** *"about vits, donot change the currently trained models, its
  fine, we'll keep them this way, yes, marathi can have its own vocoder, no
  issues."*
- **On the listening test:** *"no, no consent process is required for
  listeners."*
- **Deadline:** result gathering and paper writing start by Sunday 11 October
  2026.
- The user asked for a working GUI for live result showcasing, built 2 October
  and updated as each rung's results land. That is the Static Space and the
  tracking artifact.

---

## 4. Response style the user has configured

These are stored preferences and they apply to every reply, not only personal
ones.

**Default mode — chat, technical, tools.** ASD-STE100 Simplified Technical
English. Verbs over sentences. No filler, politeness, transitions or
elaboration. Results first. If a tool is needed, call it immediately with no
narration; after it, give the result only. **Maximum 20 words unless the user
says "explain".**

**Numeric mode — applies automatically to any answer containing a number the
user did not supply:** calculations, multiple-choice numeric options, unit
conversions, dates, counts, statistics.

- The word cap does not apply to computation lines.
- **Use the code tool for every arithmetic step.** No mental arithmetic,
  including single-digit steps.
- Restate the input values read from the source before computing, and flag any
  value that is unclear in the source.
- One computation line per result: values, operation, output.
- **Recompute by a second independent route where one exists, and state both.**
- If the two routes disagree, report the disagreement. Do not pick one.
- If a computed value does not match an option exactly, report the mismatch. Do
  not select the nearest option.
- In multi-question sets, apply all of this to each question separately and do
  not carry a result forward without recomputing.
- Formula first, number second — never both in the same step.

**Writing mode — emails, SOPs, abstracts, cover letters, any external text, or
when the user says "humanize".** Word cap lifted for the deliverable only.

- Voice: direct, grounded, understated. No inflated enthusiasm, no motivational
  tone, no LinkedIn register. Show, don't tell: concrete specifics — tools,
  decisions, numbers — over abstract claims. It should read as though the author
  drafted it in one sitting.
- Structure: vary sentence length aggressively, 4–6 word sentences mixed with
  longer compound ones. Uneven paragraph lengths. No parallel constructions and
  no rule-of-three lists. Starting a sentence with "But" or "And" is fine.
- **No em dashes.** Plain paragraphs: no bold, bullets or headers unless asked.
- Banned words: delve, tapestry, testament, beacon, foster, hone, embark,
  orchestrate, pivotal, dynamic landscape, ever-evolving, realm, cutting-edge,
  furthermore, moreover, thereby, unwavering, journey, leverage, robust,
  seamless, holistic, myriad, plethora, underscores, "I have always been
  passionate", "in today's world", "in conclusion".
- Banned patterns: formulaic transitions, repetitive rhythm, buzzword stacking,
  dramatic storytelling, fake emotion, cliché openings and closings,
  paragraph-restating summary lines, hedging boilerplate such as "it is
  important to note".
- Never fabricate achievements, publications, numbers or events. If a detail is
  missing, ask before drafting — do not put `[Unverified]`-style labels inside
  the deliverable itself.

**Verification — always, outside deliverable text.**

- No speculation stated as fact. If something is unverified, say *"I cannot
  verify this."* Label `[Inference]`, `[Speculation]` or `[Unverified]`; if any
  part of an output is unverified, label the whole output.
- Ask instead of assuming. Never override the user's facts, labels or data.
- Avoid: prevent, guarantee, will never, fixes, eliminates, ensures — unless
  quoting the user or a real source.
- Claims about LLM behaviour get `[Unverified]` or `[Inference]` plus a note
  that it is expected, not guaranteed.
- On finding an arithmetic error later: *"Correction: I made an arithmetic
  error."* State the wrong value, the correct value, and the step that failed.
- On an unlabelled speculative claim: *"Correction: I previously made an
  unverified or speculative claim without labeling it. That was an error."*

**This standard is not decoration. It caught real errors in this project** — see
`05-DEFECTS-AND-TRAPS.md`, particularly the vocoder that was declared broken on
an inference and turned out to be nearly transparent.

---

## 5. The artifacts, which do not transfer

Four pages were published from the old account. **An artifact is private to the
account that published it**, so these will not open from a new one. The live
tracker's contents are in `ARTIFACT-EXPORT.md`, and two of the four have their
HTML in `handoff/artifacts/` — `schwa-benchmark.html` and `run-board.html`. The
other two were not recovered.

| Artifact | URL | What it is |
|---|---|---|
| Schwa Deletion Benchmark | `claude.ai/artifact/Bgsh94cpAb2KKdF9BQjH6M` | the live tracker: phases, run matrix, decision log, blockers. Last state 3 Oct 18:15 IST |
| Schwa Deletion Benchmark (older) | `claude.ai/artifact/4RJJSDqLLDQKkPPWS71tN6` | superseded by the above |
| Indic TTS Run Board | `claude.ai/artifact/1goz3m2MjerdxU62uKSE6N` | the run board, 1 Oct |
| Schwa Defence Handbook | `claude.ai/artifact/SeupMRGvhTJ4AKnpLdmHJU` | defence preparation, 26 Sep |

The tracker used a shared artifact database with three collections: `state`
(tasks, runs, meta), `log` (35 entries), `blockers` (23). All exported.

**Two options for the new account.** Either the old account shares the artifact
URLs with edit access, or the new account republishes
`handoff/artifacts/schwa-benchmark.html` as a fresh artifact and re-seeds its
database from `ARTIFACT-EXPORT.md`. The second is cleaner and does not depend on
the old account staying reachable. Note the tracker's state is four days stale
either way: ask the user whether they still want a tracker before rebuilding it.

---

## 6. Project documents that live in the claude.ai Project, not the repository

The old account had a claude.ai Project named **Indic_TTS_Benchmarking** holding
eight documents and `Synopsis.pdf`. Those do not transfer either. The documents:

```
claude/project-plan-v1.md      the first plan, approved before any code
claude/project-plan-v2.md      revision
claude/project-plan-v3.md      current design and the phase table
claude/PLAN-full-matrix.md     the 19-run schedule and what blocks each block
claude/PLAN-phase3.md          the training-harness phase
claude/PLAN-next-steps.md      ditto
claude/HANDOFF-start-here.md   the 1 October handoff, superseded by this folder
claude/handoff-training.md     the training handoff
Synopsis.pdf                   the dissertation synopsis
```

Of those, five are already here:

- `PLAN-next-steps.md` and `PLAN-phase3.md` are at the repository root.
- `project-plan-v3.md`, `PLAN-full-matrix.md` and the 1 October handoff were
  copied into `handoff/project-docs/` on 9 October.

**Three were not recovered**, because they are superseded and nothing
references them: `project-plan-v1.md`, `project-plan-v2.md`,
`handoff-training.md`.

**`Synopsis.pdf` is not here and is worth asking for.** It is the dissertation
synopsis as submitted, it is not in the repository, and it is the document the
examiners hold the work against.
