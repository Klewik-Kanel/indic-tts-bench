# Runbook

Everything operational. Commands here are the exact text to hand the user — they
run on the DGX, the assistant does not.

---

## 1. Identities and credentials

| | |
|---|---|
| GitHub account | `Klewik-Kanel` |
| Email | `kaustubhpandey812@gmail.com` |
| Repository | `github.com/Klewik-Kanel/indic-tts-bench`, public |
| Hugging Face user | `Klewik` |
| Weights repo | `Klewik/indic-tts-bench`, private |
| Demo Space | `Klewik/Indic-tts-demo`, Static SDK |

**The assistant never handles credentials.** The user pushes, the user runs.
A token that appears in a chat should be rotated afterwards; say so once, then
drop it.

**GitHub no longer accepts account passwords for git.** A personal access token
is required: github.com → Settings → Developer settings → Personal access tokens
→ Tokens (classic) → scope `repo`. On the DGX, store it where a container
rebuild cannot wipe it:

```bash
git config --global user.name  "Kaustubh Pandey"
git config --global user.email "kaustubhpandey812@gmail.com"
git config --global credential.helper 'store --file=/workspace/.git-credentials'
cd /workspace/indic-tts-bench && git pull
```

Username at the prompt is `Klewik-Kanel`; the password is the `ghp_` token.

**A container rebuild wipes `$HOME`** including `~/.cache/huggingface`, which is
how a gated-model failure appeared out of nowhere. Keep both the Hugging Face
token and its cache in `/workspace`:

```bash
echo 'export HF_TOKEN=hf_...'        >> /workspace/env.sh
echo 'export HF_HOME=/workspace/.hf' >> /workspace/env.sh
source /workspace/env.sh
```

`IndicConformer` is a **gated** repository: the licence must be accepted on the
Hugging Face website with the same account before the model will download.

---

## 2. The machine

| | |
|---|---|
| GPU | NSUT DGX, one A100-SXM4-40GB, shared with other tenants |
| Access | JupyterLab at `http://192.168.33.8:9014/lab`, campus VPN, Terminal tile |
| Repo | `/workspace/indic-tts-bench` |
| Python | `/workspace/venv/bin/python` — **never** plain `python` |
| Runs | `/workspace/runs/<id>/`, logs `/workspace/runs/<id>.log` |
| Env file | `/workspace/env.sh` |

**Do not plan a schedule from the figures below.** One run alone 3.54 it/s,
two sharing 4.11 it/s each, 100,000 steps = 6.76 h per run and per pair — all
measured on an **idle** card, before a foreign tenant appeared and the ladder
rungs slowed to about 1.3 it/s. `RESULTS.md` retired them: *"unreliable in both
directions."* A schedule needs a measured contention factor and nobody has
measured one. The step budget is unaffected; only the calendar is. See
`05-DEFECTS-AND-TRAPS.md` §4.

---

## 3. Daily commands

### Status

```bash
cd /workspace/indic-tts-bench && bash scripts/status.sh      # one screen
cd /workspace/indic-tts-bench && bash scripts/status.sh -w   # watch
```

Earlier notes say `bash /workspace/status.sh`. The script's first act is
`cd "$(dirname "$0")/.."`, so it sets its own working directory and can be
invoked from anywhere — **but only at its real path inside the repository**. A
copy placed at `/workspace/status.sh` would `cd` to `/` and break. If one exists
there it was put there by hand; nothing in the repository creates it.

Read-only and cheap by design: three `nvidia-smi` queries, one `pgrep`, one line
tailed per running log. It lists **every** process on the card, not only ours —
a foreign tenant is the commonest reason the step rate changes and it is
invisible from our own logs.

**Pasting that output costs far less than a screenshot.** Ask for pastes.

### Launch a pair

```bash
cd /workspace/indic-tts-bench
TRAIN_GPU_FRACTION=0.25 nohup /workspace/venv/bin/python -m src.train.launch \
  configs/r15.yaml --out /workspace/runs/r15 --ckpt-every 5000 --log-every 100 \
  > /workspace/runs/r15.log 2>&1 &
TRAIN_GPU_FRACTION=0.25 nohup /workspace/venv/bin/python -m src.train.launch \
  configs/r18.yaml --out /workspace/runs/r18 --ckpt-every 5000 --log-every 100 \
  > /workspace/runs/r18.log 2>&1 &
```

Restarting after a kill is the **same command**: the run resumes from its last
complete checkpoint, and the data order comes from `(seed, step)` rather than
being stored.

### The whole queue, unattended

```bash
bash scripts/queue_all.sh
```

It is idempotent (a run with `step_100000`, or a vocoder with `stopped.json`, is
skipped), a failing pair does not stall the queue, it waits for **our** runs to
exit rather than for the card to empty, it waits a bounded time for a foreign
tenant and then proceeds with a memory fraction computed from what is actually
free, it offloads after every pair, and it checks every run's front-end output
against its vocabulary before launching — because a vocabulary miss raises
inside collate at step 1 and would otherwise take the whole slot down.

### Pre-warm features before a new language

```bash
/workspace/venv/bin/python scripts/12_prewarm_features.py --lang marathi --sr 22050 --pitch
/workspace/venv/bin/python scripts/12_prewarm_features.py --lang marathi --sr 16000
```

`--workers` defaults to 48. Skipping this step makes the first epoch measure
`librosa` rather than the GPU. `[Unverified]` `PLAN-full-matrix.md` says Hindi
took 1.7 minutes for 5,085 utterances; that figure is not in `RESULTS.md` and I
could not source it further.

---

## 4. Export, evaluate, publish

### Export bundles

```bash
V=/workspace/venv/bin/python
for r in r01 r02 r04 r05 r06 r07 r08 r22 r23; do $V -m src.export.bundle runs/$r; done
```

r22 and r23 are the VITS seed replicas and the listening page groups by seed, so
leaving them out gives a six-arm page instead of eight. Add r11–r14 once the
VITS ladder is confirmed finished, and r15/r17/r18 for a Marathi page.

Cheap — it only rewrites manifests and copies weights. Re-export after any
change to the manifest schema. The manifest carries `data` and `seed`, which the
listening page groups by.

### Score intelligibility

```bash
$V scripts/score_intelligibility.py --lang hindi --draws 5 --boot 2000
```

Flags: `--draws`, `--boot`, `--boot-seed`, `--vocoder`, `--ceiling`,
`--ceiling-only`, `--floor-only`, `--no-floor`, `--runs`, `--limit`.
`--runs` accepts run ids **or** exact bundle directory names.

### Render and publish the demo

```bash
cd /workspace/indic-tts-bench && git pull
V=/workspace/venv/bin/python
$V scripts/render_demo.py --lang hindi --device cuda --vocoder exports/r06_step18000
$V scripts/publish_space.py --dry-run
source /workspace/env.sh
$V scripts/publish_space.py --repo-id Klewik/Indic-tts-demo
```

**Read the dry run before publishing.** It prints one line per arm with its
architecture and vocoder; `r01 fastspeech2 hifigan:r06@18000` confirms the
vocoder attached. It refuses on: missing `index.html`, missing `data.json`,
JSON a browser cannot parse, no sentences, a clip named with no file, a mel-only
arm with no audio, a mel-only arm labelled end-to-end, no audio at all, or a
`data.json` more than 120 s older than the newest wav.

Without `--vocoder` the FastSpeech 2 arms write no audio and publish silent.
Each language has its own vocoder — Marathi uses r17.

### Offload everything off the DGX

```bash
source /workspace/env.sh
$V scripts/offload.py --dry-run --all-checkpoints   # read the total first
$V scripts/offload.py --all-checkpoints             # or without, for final checkpoints only
```

It carries: `RESULTS.md`, `README.md`, every `configs/*.yaml`, per run its
`config.json` / `vocab.json` / `train_log.jsonl` / `stopped.json`, the
checkpoints, every export bundle, `results/tables/`, `space_static/`, and the
frozen splits (`data/processed/**/*.tsv`, `SPLITS.lock`,
`data/interim/**/manifest.tsv`, `data/raw/dataset_profile.json`). It skips what
has not changed, so re-running costs almost nothing.

```bash
$V scripts/offload_status.py     # what has actually reached the Hub
bash scripts/offload_loop.sh     # run it on a timer
```

---

## 5. GitHub Pages

`.github/workflows/pages.yml` publishes `space_static/` directly — no `docs/`
copy, because two copies of the wav files are two things to keep in step and the
second one drifts.

It **skips** cleanly when nothing has been rendered (normal: the site is
produced on the DGX and committed from there) and **fails** on a half-committed
site: wavs without `data.json`, `data.json` without wavs, a clip named in
`data.json` that is not in the commit, or `data.json` holding `Infinity`.

To actually publish: enable Pages in Settings → Pages → Source → GitHub Actions,
then commit `space_static/data.json` and `space_static/audio/` from the DGX.
`.gitignore` already un-ignores `space_static/audio/**/*.wav`.

---

## 6. Running the tests

On the DGX, with pytest available:

```bash
cd /workspace/indic-tts-bench && /workspace/venv/bin/python -m pytest -q
```

On the Mac there is no pytest and no `librosa`, so a handful of cases fail for
that reason alone. A shim was used at `/tmp/shim_run.py`; it is not in the
repository and **it is already gone** — `/tmp` was cleared, so there is
currently no way to run the suite on the Mac at all. Either install pytest and
librosa there, or run the suite on the DGX. Until one of those happens, treat
every claim about test state as of 5 October.

---

## 7. Things that will bite

- **Always `/workspace/venv/bin/python`.** Plain `python` fails on
  `import torchaudio` with `undefined symbol`.
- **Push before telling the user to run.** Three separate "unrecognised flag"
  reports were commits sitting unpushed on the Mac.
- **Never put `<angle bracket>` placeholders in a shell command.** Bash parses
  `<` as a redirect. This happened twice.
- **The DGX is shared.** Check `status.sh` for foreign tenants before concluding
  anything from a step rate.
- **A container rebuild wipes `$HOME`.** Anything that must survive goes in
  `/workspace`.
