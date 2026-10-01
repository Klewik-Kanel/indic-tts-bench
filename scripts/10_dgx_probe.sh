#!/usr/bin/env bash
# Survey the DGX before planning a single training run.
#
# Paste this into the JupyterLab terminal. It only reads: no installs, no
# writes outside a scratch file, nothing that can disturb another user's job.
#
# It answers the five questions that decide how the 136-GPU-hour plan is
# actually executed, and gets them wrong loudly rather than quietly:
#   1. What GPUs are attached, how many, and how much memory each
#   2. Whether a scheduler stands in front of them
#   3. Whether anyone else is using them right now
#   4. Where data can live, and how much room there is
#   5. Whether the session survives a closed laptop
#
# Output goes to dgx_probe.txt as well as the screen. Send that file back.

OUT="${HOME}/dgx_probe.txt"
exec > >(tee "$OUT") 2>&1

line() { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }

line "identity"
echo "host:      $(hostname)"
echo "user:      $(whoami)   uid $(id -u)   groups: $(id -Gn)"
echo "home:      $HOME"
echo "date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "kernel:    $(uname -srm)"
[ -r /etc/os-release ] && . /etc/os-release && echo "os:        $PRETTY_NAME"

line "gpus"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,persistence_mode \
             --format=csv
  echo
  echo "driver/cuda: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
       "/ $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1)"
  echo
  echo "-- who is using them right now --"
  # An A100 with memory in use and no visible process is someone else's job in
  # another container. Treat the memory column as the truth, not the pid list.
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  echo
  echo "MIG (a DGX is often sliced; a slice is not a whole A100):"
  nvidia-smi -L
else
  echo "nvidia-smi NOT FOUND."
  echo "Either this terminal is on a login node with no GPU, or the GPU is"
  echo "reached only through a scheduler. Check the scheduler section below."
fi

line "scheduler"
found=0
for c in sbatch squeue sinfo srun scontrol qsub qstat bsub condor_submit pbsnodes; do
  if command -v "$c" >/dev/null 2>&1; then echo "found: $c"; found=1; fi
done
if [ "$found" -eq 1 ]; then
  echo
  command -v sinfo >/dev/null 2>&1 && { echo "-- partitions and time limits --"; sinfo -o '%20P %5a %10l %6D %N' 2>&1 | head -20; }
  command -v squeue >/dev/null 2>&1 && { echo; echo "-- queue right now --"; squeue -o '%.10i %.12P %.10u %.2t %.10M %R' 2>&1 | head -20; }
else
  echo "No scheduler found. GPUs are used directly, so a training run is a"
  echo "tmux or nohup command and the budget is wall-clock on a shared node."
fi

line "cpu and memory"
echo "cores:  $(nproc)"
free -h 2>/dev/null | head -2
echo "cgroup memory limit (what this container may actually use):"
cat /sys/fs/cgroup/memory.max 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
  || echo "  not reported"

line "storage"
df -h "$HOME" /tmp /scratch /data /raid 2>/dev/null | sort -u
echo
echo "quota, if one is enforced:"
quota -s 2>/dev/null || echo "  no quota command, or none set"
echo
echo "writable scratch candidates:"
for d in "$HOME" /scratch /raid /data /tmp; do
  [ -d "$d" ] && [ -w "$d" ] && echo "  $d  ($(df -h "$d" 2>/dev/null | awk 'NR==2{print $4}') free)"
done

line "python and frameworks"
echo "python:  $(python3 -V 2>&1)   ($(command -v python3))"
echo "pip:     $(python3 -m pip --version 2>&1 | cut -c1-60)"
python3 - <<'PY' 2>&1
for m in ("torch", "torchaudio", "transformers", "librosa", "soundfile", "numpy"):
    try:
        mod = __import__(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<14} MISSING ({type(e).__name__})")
try:
    import torch
    print(f"  cuda available: {torch.cuda.is_available()}   devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  device 0: {torch.cuda.get_device_name(0)}  "
              f"capability {torch.cuda.get_device_capability(0)}")
        print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
except Exception as e:
    print(f"  torch cuda check failed: {e}")
PY
echo
echo "can I install into my own user site? (needed if torch is missing)"
python3 -m pip install --dry-run --quiet --user packaging >/dev/null 2>&1 \
  && echo "  yes" || echo "  no, or pip is restricted"

line "network"
for h in github.com huggingface.co pypi.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$h" 2>/dev/null)
  echo "  https://$h  ->  ${code:-no answer}"
done
echo "proxy env: ${HTTPS_PROXY:-none}${https_proxy:+ (https_proxy set)}"

line "session persistence"
for c in tmux screen nohup; do
  command -v "$c" >/dev/null 2>&1 && echo "found: $c"
done
echo
echo "If tmux is present, training survives a closed laptop. If it is not,"
echo "nohup does the same job with no reattach. If neither works, the"
echo "notebook kernel dies with the browser tab and nothing long can run."

line "existing work"
ls -la "$HOME" 2>/dev/null | head -15
echo
echo "indic-tts-bench already here? $([ -d "$HOME/indic-tts-bench" ] && echo yes || echo no)"

line "done"
echo "Written to $OUT"
echo "Send that file back, or paste it."
