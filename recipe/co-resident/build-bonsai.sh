#!/usr/bin/env bash
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${SRC_ROOT:-$(pwd)/engine/prism-bonsai}"
BONSAI_IMAGE="${BONSAI_IMAGE:-localhost/ternary-bonsai-strix:prism-1a07bfa-sync}"
COMMIT=1a07bfa5f4144274c8f1c9963821dd9d9a51854b

if [ ! -e "${SRC_ROOT}" ]; then
  mkdir -p "$(dirname -- "${SRC_ROOT}")"
  git init -q "${SRC_ROOT}"
  git -C "${SRC_ROOT}" remote add origin https://github.com/PrismML-Eng/llama.cpp.git
  git -C "${SRC_ROOT}" fetch --depth 1 origin "${COMMIT}"
  git -C "${SRC_ROOT}" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "${SRC_ROOT}" rev-parse HEAD)" != "${COMMIT}" ]; then
  echo "Use a new SRC_ROOT: this directory is not the pinned Prism revision" >&2
  exit 1
fi
python3 "${RECIPE_DIR}/patch-strix-input-sync.py" "${SRC_ROOT}"
podman build -f "${RECIPE_DIR}/Containerfile.bonsai" -t "${BONSAI_IMAGE}" "${SRC_ROOT}"
