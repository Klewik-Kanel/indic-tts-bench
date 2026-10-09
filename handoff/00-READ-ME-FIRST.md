# Read me first

**Written 9 October 2026, for a transfer of this project to a different Claude
account on the same MacBook Air.**

This folder is the complete handover. Nothing load-bearing is left in the chat
that produced it, in a Claude artifact, or in the old account's memory. If you
are the new assistant: read this file, then the eight numbered files after it, and
do not re-derive anything they already settle.

---

## 1. What this project is, in one paragraph

An M.Tech dissertation for **Kaustubh Pandey**, NSUT New Delhi, supervisor
**Dr. Shobha Bhatt**. The research question: *does explicit Devanagari
grapheme-to-phoneme conversion still buy anything for neural text-to-speech, or
do modern architectures absorb Hindi schwa deletion from data alone?* Hindi is
the testbed, Marathi was planned as a same-script control, and a nested data
ladder from 9 hours down to 10 minutes varies the resource level. The immediate
deliverable is a Springer-format conference paper; the dissertation follows.

Devanagari does not write schwa deletion. न म क is read *namak*, not *namaka*,
and nothing in the spelling says so. One arm of every comparison is handed the
deletion by a rule-based front end. The other reads raw characters. Everything
else is held identical: architecture, seed, corpus, step budget.

---

## 2. Read these, in this order

| # | File | What it holds |
|---|---|---|
| 01 | `01-PROJECT-AND-METHOD.md` | the question, the design, every methodological decision and why it was taken |
| 02 | `02-STATE-OF-PLAY.md` | the 23-run matrix with each run's state, what is finished, what is not |
| 03 | `03-CODEBASE-MAP.md` | every module and script, the invariants, how a run actually executes |
| 04 | `04-RESULTS-AND-OBSERVATIONS.md` | the findings with their numbers, and what may and may not be claimed from them |
| 05 | `05-DEFECTS-AND-TRAPS.md` | every bug found, its symptom, its fix, and the traps that cost days |
| 06 | `06-RUNBOOK.md` | the DGX, Hugging Face, GitHub, the exact commands, the credentials policy |
| 07 | `07-FUTURE-PLAN.md` | what remains, in priority order, against the deadline |
| 08 | `08-MEMORY-AND-WORKING-STYLE.md` | what the old account remembered; how the user wants to be worked with |

Two more, and they are source material rather than narrative:

- `ARTIFACT-EXPORT.md` — the live tracking artifact's contents: a 35-entry
  decision log, 23 blockers, the phase checklist. The artifact itself does not
  transfer between accounts; this is its text.
- `artifacts/*.html` — the two tracking pages themselves, kept so they can be
  republished from the new account if wanted.

And one that is not in this folder because it is the repository's own spine:

- **`RESULTS.md` at the repository root, 110 KB, 40 sections, 39 of them
  dated.** It is the
  lab notebook. Every number in the paper traces to it. It is appended to and
  never rewritten. The handoff files summarise it; they do not replace it.

---

## 3. The division of labour, which does not change

The user stated this and it is binding:

> *"get back to our old working structure, where you will administer changes and
> ask me to push the git and then run things on dgx"*

So: **the assistant edits and commits in the Mac clone. The user pushes. The
user runs everything on the DGX and pastes the output back.** The assistant
never handles the user's credentials and cannot reach the DGX. Hand over exact
command blocks, ready to paste.

Two more standing rules from the user:

- **No `Co-Authored-By` trailer on commits.** The assistant is not a co-author.
- **No Claude session links anywhere** — not in commit messages, not in pull
  request descriptions.

Git identity for this repository is Kaustubh's:

```
git config user.name  "Kaustubh Pandey"
git config user.email "kaustubhpandey812@gmail.com"
```

GitHub account `Klewik-Kanel`, which is that same email. Everything — repository,
Hugging Face repos, access tokens — lives under that identity.

---

## 4. Where everything lives

| Thing | Where |
|---|---|
| Mac clone | `/Users/klewik/indic-tts-bench` — this folder |
| GitHub | `github.com/Klewik-Kanel/indic-tts-bench`, public |
| DGX repo | `/workspace/indic-tts-bench`, pulled by the user |
| DGX runs | `/workspace/runs/<id>/`, logs at `/workspace/runs/<id>.log` |
| DGX Python | `/workspace/venv/bin/python` — **never** the container's system Python |
| Weights backup | Hugging Face `Klewik/indic-tts-bench`, private |
| Listening demo | Hugging Face Space `Klewik/Indic-tts-demo`, Static SDK |
| Pages mirror | GitHub Pages, from `space_static/` via `.github/workflows/pages.yml` |

---

## 5. The single most important fact about the current results

**The VITS result is real and defensible. The FastSpeech 2 result is not a
result at all — those models produce unintelligible speech, for a reason that is
measured and fixable.**

VITS, phonemic against graphemic, on held-out intelligibility, 50 utterances
and 5 synthesis draws, two independent CTC recognisers: the arm gap is larger
than the seed spread, which is larger than the draw spread, on both recognisers,
with no sign change. That ordering is what makes the finding reportable.

FastSpeech 2 came back at character error rate 0.9932 and 0.9924, which is the
"transcribed nothing" ceiling. The cause was isolated to the training objective,
not the ablation and not the vocoder: `ForwardTTSLoss` sums a mel term against
pitch and energy terms left in physical units, so the mel decoder was optimised
against roughly 0.6 per cent of the gradient. See `04` and `05`. Fixing it means
retraining; see `07`.

Do not let anyone read the FastSpeech 2 rows as evidence about phonemic versus
graphemic input. They are a record of what was run.

---

## 6. First three things to do in the new session

1. **Confirm the current run states.** The last state recorded here is from
   5 October and some runs were still on the card. Ask the user to paste:

   ```bash
   cd /workspace/indic-tts-bench && bash scripts/status.sh
   ls /workspace/runs/
   ```

2. **Confirm the demo published.** The Static Space was being fixed on
   5 October after a `data.json` containing the token `Infinity` broke it. Ask
   the user to open `huggingface.co/spaces/Klewik/Indic-tts-demo` and say
   whether the page loads with audio.

3. **Read `07-FUTURE-PLAN.md` and agree the order of the remaining work** with
   the user before doing any of it. The deadline for starting result gathering
   and paper writing was Sunday 11 October, and that is now imminent.

---

## 7. How to be useful here, concretely

This project has an unusually high standard of evidence and the user enforces
it. The patterns that have worked:

- **Measure rather than infer.** Several of the worst errors in this project
  were confident inferences that measurement reversed. One is recorded in `05`
  where a vocoder was declared broken on a spectral distance and turned out to
  cost 0.0045 character error rate.
- **Label unverified claims.** The user requires `[Inference]`, `[Speculation]`
  and `[Unverified]` tags on anything not measured, and an explicit
  *"Correction: ..."* when something earlier was wrong. `RESULTS.md` does this
  throughout; keep doing it.
- **Use a code tool for arithmetic, every step, and check by a second route.**
  This is a standing user preference. State both results. If they disagree,
  report the disagreement rather than picking one.
- **Write tests that execute the thing, not tests that grep the source.** Two
  real bugs in this repository were found only by running code that a source
  grep had passed. The suite is 614 test functions across 38 files; keep the
  bar.
- **Terse by default.** Verbs over sentences. Results first. The user asks for
  "explain" when they want more.
