#!/usr/bin/env bash
# Export the model's own multi-token-prediction (MTP) draft head as a standalone
# GGUF sidecar, straight from a Hugging Face checkpoint.
#
# Public GGUF conversions strip the MTP block. This pulls ONLY the tensors the
# draft head needs (~8 GB of traffic) instead of downloading the full checkpoint
# (~336 GB for the BF16 release).
#
# A prebuilt sidecar for the stock checkpoint exists, if you would rather not:
#   https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF
#
# Usage: ./export-mtp-sidecar.sh <hf-repo-id> [outdir]
set -euo pipefail

REPO_ID="${1:?usage: export-mtp-sidecar.sh <hf-repo-id> [outdir]}"
OUT_DIR="${2:-$(pwd)/models}"
SRC_ROOT="${SRC_ROOT:-$(pwd)/engine}"       # the EngramHalo.cpp checkout
QUANT="${QUANT:-Q8_0}"                      # Q8_0 measured better than BF16: half the
                                            # draft reads, and quant-matched errors give
                                            # higher acceptance than a more precise draft
mkdir -p "${OUT_DIR}"
BF16_OUT="${OUT_DIR}/mtp-$(basename "${REPO_ID}")-BF16.gguf"
FINAL_OUT="${OUT_DIR}/mtp-$(basename "${REPO_ID}")-${QUANT}.gguf"

# --mtp    export only the draft block
# --no-ple REQUIRED alongside --mtp: the MTP-only tensor filter drops the n-gram
#          lookup block, but the metadata pass still tries to read its hash
#          constants and dies with
#          "PLE constant 'ple_embedding.layer_multipliers' missing from the checkpoint"
# --remote stream tensors from the Hub instead of downloading the checkpoint
#
# Gated repos: export HF_TOKEN first — the remote reader reads it from the
# environment, not from the CLI's cached login.
python3 "${SRC_ROOT}/convert_hf_to_gguf.py" \
  --remote "${REPO_ID}" \
  --mtp --no-ple \
  --outtype bf16 \
  --outfile "${BF16_OUT}"

"${SRC_ROOT}/build-hip/bin/llama-quantize" "${BF16_OUT}" "${FINAL_OUT}" "${QUANT}" "$(nproc 2>/dev/null || echo 8)"

echo
echo "draft sidecar: ${FINAL_OUT}"
ls -l "${FINAL_OUT}"
echo "(the BF16 intermediate can be deleted once you are happy with the quantised file)"

# Sanity check the result: a correct sidecar is ONE transformer block plus the
# shared embedding/output tensors — roughly 90 tensors, a couple of GB at Q8_0.
# A file anywhere near the size of the target model means the trunk came along
# by mistake.
