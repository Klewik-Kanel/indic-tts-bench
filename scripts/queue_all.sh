#!/bin/bash
# Queue every remaining pair, so the box does not need touching again.
#
# Order is by what the dissertation needs, not by run number: the Hindi VITS
# ablation arms first, then the four Marathi control arms, then the ladder,
# then the vocoders. The ladder is the shock absorber, so it is last among the
# acoustic runs: if the card is taken away mid-queue, what is lost is the
# weakest claim rather than the central one.
#
# Properties that matter for something left running for days:
#   - idempotent. A run with step_100000, or a vocoder with stopped.json, is
#     skipped, so this can be re-run after any interruption.
#   - a failing pair does not stall the queue. It is logged and the next pair
#     starts. One broken config must not cost the remaining runs.
#   - between pairs it waits for OUR runs to exit, not for the card to empty,
#     because another tenant may never leave.
#   - it waits a bounded time for a foreign tenant to go, then proceeds with a
#     memory fraction computed from whatever is actually free. Waiting forever
#     for someone else's job to finish is how a queue silently does nothing.
#   - it offloads weights after every pair, because the GPU access is temporary.
#   - before launching, every run in the pair has its front-end output checked
#     against its vocabulary. The dry run covers only the first of a pair, and
#     a vocabulary miss raises inside collate at step 1, so without this a
#     second-position run can take the whole slot down with it.
#
#   setsid nohup bash scripts/queue_all.sh > /workspace/runs/queue.log 2>&1 &
set -uo pipefail

REPO=${REPO:-/workspace/indic-tts-bench}
RUNS=${RUNS:-/workspace/runs}
PY=${PY:-/workspace/venv/bin/python}
TENANT_WAIT_MIN=${TENANT_WAIT_MIN:-60}     # how long to wait for a tenant to go
OURS_WAIT_H=${OURS_WAIT_H:-30}             # how long to wait for our own runs
OFFLOAD=${OFFLOAD:-1}                      # back up after each pair

# Pairs, in priority order. r03 is absent on purpose: Matcha is future work as
# of 2 October, see FUTURE_WORK in src/train/config.py.
# Reordered on 3 Oct. The vocoders came last while they were an upper bound on
# 10 GPU-h each with nothing depending on them. Both of those changed: the
# FastSpeech 2 arms are mute without one, which blocks the demo, the listening
# test and every waveform metric, and the mel-domain evidence on 3 Oct showed
# the FastSpeech 2 mels are over-smoothed in a way only a vocoder with a
# learned prior can carry. They also stop on a plateau criterion rather than
# spending a fixed budget, so they are the cheapest item here.
#
# The VITS pairs sit behind them deliberately: their configuration is still
# open (see RESULTS, 3 Oct, init_from) and every VITS run has to share whatever
# it settles on, the four ladder rungs included.
PAIRS=(
  "r02 r05"      # Hindi VITS: done at 100k, pending a decision on re-running
  "r06 r17"      # the two vocoders, fine-tuned on our own mels
  "r20 r21"      # seed variance floor: r01's cell at seeds 1 and 2
  "r15 r18"      # Marathi FastSpeech 2: the control's two arms
  "r16 r19"      # Marathi VITS: the control's two arms
  "r11 r12"      # VITS ladder, 5 h and 1 h
  "r13 r14"      # VITS ladder, 30 min and 10 min
  "r09 r10"      # FastSpeech 2 ladder, 30 min and 10 min
)

say() { echo "[$(date -u +%m-%d\ %H:%M:%S)] $*"; }

arch_of() { grep -m1 '^architecture:' "$REPO/configs/$1.yaml" | awk '{print $2}'; }
lang_of() { grep -m1 '^language:'     "$REPO/configs/$1.yaml" | awk '{print $2}'; }
repr_of() { grep -m1 '^input_repr:'   "$REPO/configs/$1.yaml" | awk '{print $2}'; }

finished() {
  [ -d "$RUNS/$1/checkpoints/step_100000" ] && return 0
  [ -f "$RUNS/$1/stopped.json" ] && return 0
  return 1
}

ours_running() { pgrep -cf "src\.train\.launch" 2>/dev/null || true; }

on_card() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true
}

wait_for_our_runs() {
  local tries=$(( OURS_WAIT_H * 30 ))          # a 120 s poll
  for _ in $(seq 1 "$tries"); do
    [ "$(ours_running)" -eq 0 ] && return 0
    sleep 120
  done
  say "GIVING UP: our own runs still going after ${OURS_WAIT_H} h"
  return 1
}

wait_for_tenant() {
  local tries=$(( TENANT_WAIT_MIN / 2 ))
  for _ in $(seq 1 "$tries"); do
    [ "$(on_card)" -eq 0 ] && { say "the card is clear"; return 0; }
    sleep 120
  done
  say "a tenant is still on the card after ${TENANT_WAIT_MIN} min; proceeding with what is free"
  return 0
}

fractions() {   # echoes "PAIR SOLO", computed from free memory right now
  local tot used free
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  free=$(( tot - used ))
  awk -v f="$free" -v t="$tot" 'BEGIN{
    p=(f*0.90/2)/t; if(p>0.45)p=0.45; if(p<0.05)p=0.05;
    s=(f*0.90)/t;   if(s>0.90)s=0.90; if(s<0.05)s=0.05;
    printf "%.2f %.2f", p, s}'
}

# Sets LAST_PID rather than echoing it. A process started inside $( ) belongs
# to that subshell, so the parent cannot `wait` on it: bash refuses with "not a
# child of this shell" and the queue would run every pair at once.
LAST_PID=""
launch_one() {   # run_id fraction -> sets LAST_PID
  local r=$1 frac=$2 extra=()
  if [ "$(arch_of "$r")" = "hifigan" ]; then
    # The vocoder has no step budget: it is excluded from
    # assert_budget_matched because it is not an object of comparison. It stops
    # when held-out mel reconstruction error plateaus and reports the step it
    # reached. These are flags, not config fields, so they stay out of
    # config_hash.
    extra=(--eval-every 2000 --patience 3 --min-delta 1e-4)
  fi
  TRAIN_GPU_FRACTION=$frac TRAIN_THREADS=${TRAIN_THREADS:-16} \
    OMP_NUM_THREADS=${TRAIN_THREADS:-16} MKL_NUM_THREADS=${TRAIN_THREADS:-16} \
    nohup "$PY" -m src.train.launch "$REPO/configs/$r.yaml" \
      --out "$RUNS/$r" --ckpt-every 5000 --log-every 100 "${extra[@]}" \
      > "$RUNS/$r.log" 2>&1 &
  LAST_PID=$!
}

offload() {
  [ "$OFFLOAD" = "1" ] || return 0
  if [ -n "${HF_REPO:-}" ]; then
    say "offloading to ${HF_REPO}"
    "$PY" "$REPO/scripts/offload.py" >> "$RUNS/offload.log" 2>&1 \
      && say "offload ok" || say "OFFLOAD FAILED, see $RUNS/offload.log"
  else
    say "HF_REPO is not set, so nothing is being backed up off this machine"
  fi
}

cd "$REPO" || exit 1
mkdir -p "$RUNS"
say "queue armed with ${#PAIRS[@]} pairs"
say "order: ${PAIRS[*]}"

for pair in "${PAIRS[@]}"; do
  set -- $pair
  A=$1; B=$2
  todo=()
  for r in "$A" "$B"; do
    if finished "$r"; then say "$r is already finished, skipping"; else todo+=("$r"); fi
  done
  if [ ${#todo[@]} -eq 0 ]; then say "pair $A $B is done, next"; continue; fi

  say "=== next pair: ${todo[*]}"
  wait_for_our_runs || exit 1
  wait_for_tenant

  # Re-check after waiting. What we waited for may have been these very runs,
  # launched by hand: deciding before the wait and acting after it would relaunch
  # a run that finished while we waited.
  still=()
  for r in "${todo[@]}"; do
    if finished "$r"; then say "$r finished while we waited, skipping"; else still+=("$r"); fi
  done
  todo=("${still[@]}")
  [ ${#todo[@]} -eq 0 ] && { say "pair $A $B is done, next"; continue; }
  git -C "$REPO" pull --ff-only 2>&1 | tail -2

  # The vocoders need a checkpoint to warm-start from, or they would train from
  # scratch and give back the saving that put them back in the matrix.
  skip_pair=0
  for r in "${todo[@]}"; do
    if [ "$(arch_of "$r")" = "hifigan" ]; then
      init=$(grep -m1 '^init_from:' "$REPO/configs/$r.yaml" | sed "s/^init_from:[[:space:]]*//; s/'//g")
      if [ -z "$init" ] || [ ! -e "$init" ]; then
        say "SKIPPING $r: init_from is ${init:-empty} and that is not a checkpoint on this machine."
        say "  Set init_from in configs/$r.yaml to a downloaded HiFi-GAN and re-run this script."
        skip_pair=1
      fi
    fi
  done
  [ "$skip_pair" = "1" ] && { say "pair skipped, continuing with the rest"; continue; }

  # Text coverage, for EVERY run in the pair. The dry run below exercises
  # only FIRST, which is exactly how r18 was launched to its death: r15 dry-ran
  # clean, then r18 raised KeyError on an out-of-inventory symbol inside
  # collate, on the prefetch thread, at step 1, having burned the pair slot.
  # This check is text only and costs seconds. input_repr "none" is a vocoder,
  # which has no vocabulary and nothing to check.
  skip_pair=0
  for r in "${todo[@]}"; do
    rep=$(repr_of "$r")
    [ "$rep" = "none" ] && continue
    lang=$(lang_of "$r")
    if ! "$PY" "$REPO/scripts/check_text_coverage.py" \
          --lang "$lang" --input-repr "$rep" > "$RUNS/coverage_$r.log" 2>&1; then
      say "SKIPPING $r: front end emits symbols outside its vocabulary."
      say "  It would raise in collate at step 1. See $RUNS/coverage_$r.log"
      tail -n 8 "$RUNS/coverage_$r.log"
      skip_pair=1
    fi
  done
  [ "$skip_pair" = "1" ] && { say "pair skipped, continuing with the rest"; continue; }

  read -r PAIRFRAC SOLOFRAC <<<"$(fractions)"
  say "memory fractions: paired ${PAIRFRAC} each, solo ${SOLOFRAC}"

  FIRST=${todo[0]}
  ADAPTER=$(arch_of "$FIRST")
  say "dry run: $FIRST as $ADAPTER at ${PAIRFRAC}"
  if ! TRAIN_GPU_FRACTION=$PAIRFRAC "$PY" -m src.train.dryrun \
        "$REPO/configs/$FIRST.yaml" --adapter "$ADAPTER" --steps 2 \
        > "$RUNS/dryrun_$FIRST.log" 2>&1; then
    say "DRY RUN FAILED for $FIRST, see $RUNS/dryrun_$FIRST.log; skipping this pair"
    tail -n 12 "$RUNS/dryrun_$FIRST.log"
    continue
  fi

  pids=()
  if [ ${#todo[@]} -eq 2 ]; then
    for r in "${todo[@]}"; do
      launch_one "$r" "$PAIRFRAC"
      pids+=("$LAST_PID")
      say "launched $r pid $LAST_PID at ${PAIRFRAC}"
    done
  else
    launch_one "${todo[0]}" "$SOLOFRAC"
    pids+=("$LAST_PID")
    say "launched ${todo[0]} alone pid $LAST_PID at ${SOLOFRAC}"
  fi

  say "waiting for ${pids[*]}"
  for p in "${pids[@]}"; do wait "$p" || say "pid $p exited non-zero"; done

  for r in "${todo[@]}"; do
    if finished "$r"; then
      say "$r finished"
    else
      last=$(ls -1 "$RUNS/$r/checkpoints" 2>/dev/null | grep -c '^step_' || true)
      say "ATTENTION: $r did NOT reach the end; $last checkpoint(s) on disk. Re-running this script resumes it."
    fi
  done
  offload
done

say "queue complete"
offload
