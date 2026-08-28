#!/usr/bin/env bash
# Build EngramHalo.cpp HIP-only for gfx1151.
#
# Why HIP-only: the draft head can select the wrong device when two GPU backends
# are compiled into one binary (upstream issue), and this patch set is tuned for
# HIP — on Vulkan it is reported, and was independently observed here, to be a
# net loss. Build Vulkan separately if you need it.
#
# Prerequisites: ROCm (7.x), cmake, a C++ compiler, git. Inside a ROCm container
# is the easy path; a ROCm host works too.
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$(pwd)/engine}"
REPO="${REPO:-https://github.com/Aristo94/EngramHalo.cpp.git}"
BRANCH="${BRANCH:-strix-halo-qwen4exp}"
# Pin. The branch moves; this is the revision every number in this repo came from.
COMMIT="${COMMIT:-4ff3affc2}"   # short SHA; the revision every number in results/ came from
GPU_TARGET="${GPU_TARGET:-gfx1151}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

if [ ! -d "${SRC_ROOT}/.git" ]; then
  git clone --branch "${BRANCH}" "${REPO}" "${SRC_ROOT}"
fi
cd "${SRC_ROOT}"
git fetch --all --quiet
git checkout --quiet "${COMMIT}"
echo "building $(git rev-parse --short HEAD) for ${GPU_TARGET}"

cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_VULKAN=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_HIP_FORCE_MMQ=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_ARCHITECTURES="${GPU_TARGET}" \
  -DGPU_TARGETS="${GPU_TARGET}" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DLLAMA_USE_PREBUILT_WEBUI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_BUILD_TESTS=OFF

cmake --build build-hip -j "${JOBS}" \
  --target llama-server llama-cli llama-bench llama-quantize

echo
echo "built: ${SRC_ROOT}/build-hip/bin/llama-server"
"${SRC_ROOT}/build-hip/bin/llama-server" --version 2>&1 | tail -2

# Sanity: the binary must know this architecture, and must see exactly one GPU.
echo
echo "architecture support: $(grep -ac qwen4exp "${SRC_ROOT}/build-hip/bin/llama-server" || echo 0) match(es) in the binary"
"${SRC_ROOT}/build-hip/bin/llama-server" --list-devices 2>&1 | tail -3
