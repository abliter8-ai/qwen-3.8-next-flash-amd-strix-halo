#!/usr/bin/env bash
# Qwen3.8-Flash-Next on Strix Halo — FAST profile (speculation on).
#
# ~66-82 tok/s on code, ~55 on structured output, ~22 on prose.
# Output is NOT byte-reproducible in this profile: verifying batches of drafted
# tokens changes floating-point accumulation order. Use serve-reference.sh when
# you need identical output from identical input.
#
# Every path comes from the environment. Defaults bind to loopback ONLY.
set -euo pipefail

: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the GGUF shards}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODEL="${MODEL:-${MODEL_ROOT}/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}"
DRAFT="${DRAFT:-${MODEL_ROOT}/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
ALIAS="${ALIAS:-qwen3.8-flash-next}"
SERVER_HOST="${SERVER_HOST:-localhost}"   # loopback by default; set deliberately to expose on a network
SERVER_PORT="${SERVER_PORT:-8080}"
CTX="${CTX:-262144}"
THREADS="${THREADS:-4}"                   # thread count barely matters in mmap mode

# Engram (n-gram lookup table) placement:
#   ssd  = table streams from disk, ~1.2 GB resident, full context      (default)
#   ram  = table pinned in memory, fastest, contexts up to ~48K only
ENGRAM="${ENGRAM:-ssd}"
case "${ENGRAM}" in
  ssd) ENGRAM_ARGS=(-lm mmap --tensor-read-lazy on) ;;
  ram) ENGRAM_ARGS=(-lm none)
       [ "${CTX}" -gt 49152 ] && echo "warning: RAM mode with ctx > ~48K deadlocks the first request" >&2 ;;
  *)   echo "ENGRAM must be 'ssd' or 'ram'" >&2; exit 2 ;;
esac

test -r "${MODEL}" || { echo "model not readable: ${MODEL}" >&2; exit 1; }
test -r "${DRAFT}" || { echo "draft not readable: ${DRAFT} (see export-mtp-sidecar.sh)" >&2; exit 1; }

# gfx1151 target. Deliberately NOT setting GGML_HIP_ENABLE_UNIFIED_MEMORY:
# on this hardware it moves the whole model into anonymous process memory
# (measured 76.9 GB anon / 0.2 GiB GPU) and defeats the file-backed mapping.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
export ROCBLAS_USE_HIPBLASLT="${ROCBLAS_USE_HIPBLASLT:-1}"
export LLAMA_QSA_GATHER="${LLAMA_QSA_GATHER:-1}"   # sparse-attention gather; 0 if --parallel > 1

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  -md "${DRAFT}" \
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
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 4 \
  --spec-draft-p-min 0.75 \
  "$@"

# Notes on the flags that are not obvious:
#   -ctk/-ctv q8_0   never bf16 on this attention shape (head-dim 256 re-converts
#                    the whole cache every call); quantized KV other than q8_0 can
#                    assert on some builds of this architecture
#   -ub 2048         needs the per-block QSA bias from the tuned branch
#   --parallel 1     multi-slot trips this architecture's cache handling; if you
#                    do run multi-slot, set LLAMA_QSA_GATHER=0
#   --spec-draft-p-min 0.75   the confidence gate that stops prose regressing
#   (never --no-mmap: it silently disables the lazy-read path this recipe needs)
