#!/usr/bin/env bash
# Is the download actually moving? Run in a SECOND macOS Terminal tab while
# the export is running. Prints the Hub cache size every 20 s.
# Growing = downloading, wait. Static for several minutes = stuck, Ctrl-C it.
CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo "watching $CACHE"
prev=""
while true; do
  cur=$(du -sk "$CACHE" 2>/dev/null | cut -f1)
  if [ -z "$cur" ]; then
    echo "$(date +%H:%M:%S)  cache does not exist yet"
  elif [ -z "$prev" ]; then
    printf '%s  %8.2f GB\n' "$(date +%H:%M:%S)" "$(echo "$cur/1048576" | bc -l)"
  else
    d=$((cur - prev))
    printf '%s  %8.2f GB   %+.1f MB since last check%s\n' \
      "$(date +%H:%M:%S)" "$(echo "$cur/1048576" | bc -l)" \
      "$(echo "$d/1024" | bc -l)" \
      "$([ "$d" -eq 0 ] && echo '   <-- not moving' || echo '')"
  fi
  prev="$cur"
  sleep 20
done
