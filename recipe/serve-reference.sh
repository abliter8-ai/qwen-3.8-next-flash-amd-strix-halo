#!/usr/bin/env bash
# Qwen3.8-Flash-Next on Strix Halo — REFERENCE profile (no speculation).
#
# ~22 tok/s flat across content types, and byte-reproducible: identical input
# gives identical output at temperature 0 (measured 1 unique result across 5
# identical greedy runs, and 5/5 byte-equal pairs in a 30-minute soak).
#
# Same engine and same weights as serve-fast.sh — the only difference is the
# absence of the draft head and the --spec-* flags.
set -euo pipefail

: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the GGUF shards}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODEL="${MODEL:-${MODEL_ROOT}/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}"
MMPROJ="${MMPROJ:-}"                      # optional vision tower, if your quant ships one
ALIAS="${ALIAS:-qwen3.8-flash-next}"
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_PORT="${SERVER_PORT:-8080}"
CTX="${CTX:-262144}"
THREADS="${THREADS:-4}"
ENGRAM="${ENGRAM:-ssd}"

case "${ENGRAM}" in
  ssd) ENGRAM_ARGS=(-lm mmap --tensor-read-lazy on) ;;
  ram) ENGRAM_ARGS=(-lm none) ;;
  *)   echo "ENGRAM must be 'ssd' or 'ram'" >&2; exit 2 ;;
esac

VISION_ARGS=()
if [ -n "${MMPROJ}" ]; then
  test -r "${MMPROJ}" || { echo "mmproj not readable: ${MMPROJ}" >&2; exit 1; }
  # this projector wants a minimum image-token budget to be accurate
  VISION_ARGS=(--mmproj "${MMPROJ}" --image-min-tokens 1024)
fi

test -r "${MODEL}" || { echo "model not readable: ${MODEL}" >&2; exit 1; }

# See serve-fast.sh: GGML_HIP_ENABLE_UNIFIED_MEMORY is deliberately NOT set.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export LLAMA_QSA_GATHER="${LLAMA_QSA_GATHER:-1}"

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  "${VISION_ARGS[@]}" \
  --alias "${ALIAS}" \
  --host "${SERVER_HOST}" --port "${SERVER_PORT}" \
  -ngl 999 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  "${ENGRAM_ARGS[@]}" \
  -c "${CTX}" \
  -b 8192 -ub 2048 \
  -t "${THREADS}" \
  --parallel 1 \
  --jinja \
  --no-webui \
  "$@"
