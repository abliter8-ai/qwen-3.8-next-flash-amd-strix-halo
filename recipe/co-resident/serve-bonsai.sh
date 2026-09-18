#!/usr/bin/env bash
set -euo pipefail
BONSAI_IMAGE="${BONSAI_IMAGE:-localhost/ternary-bonsai-strix:prism-1a07bfa-sync}"
BONSAI_MODEL_ROOT="${BONSAI_MODEL_ROOT:-${HOME}/models/bonsai}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8080}"
ALIAS="${ALIAS:-ternary-bonsai-2-27b}"

for file in Bonsai-2-27B-PQ2_0-CRACK.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf; do
  test -r "${BONSAI_MODEL_ROOT}/${file}" || { echo "Missing ${file}" >&2; exit 1; }
done
exec podman run --rm --name bonsai-strix \
  --ulimit core=0 --ulimit memlock=-1:-1 --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --security-opt label=disable --network host \
  -v "${BONSAI_MODEL_ROOT}:/models:ro" "${BONSAI_IMAGE}" \
  -m /models/Bonsai-2-27B-PQ2_0-CRACK.gguf \
  --mmproj /models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --no-repack -ngl 99 -c 262144 \
  --cache-type-k q4_0 --cache-type-v q4_0 -np 1 -t 8 -fa on \
  --jinja --reasoning off \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5 --repeat-penalty 1.0 \
  --fit off --cache-ram 0 \
  --host "${SERVER_HOST}" --port "${SERVER_PORT}" --alias "${ALIAS}"
