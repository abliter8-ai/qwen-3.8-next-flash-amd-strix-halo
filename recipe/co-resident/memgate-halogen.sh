#!/usr/bin/env bash
# Only valid before Halogen starts: its pinned weights distort Linux MemAvailable.
set -euo pipefail
for ((attempt=0; attempt<12; attempt++)); do
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available_kib >= 112 * 1024 * 1024 )); then
    exit 0
  fi
  sleep 5
done
echo "Halogen startup needs 112 GiB MemAvailable. Stop Bonsai and other large workloads first." >&2
exit 1
