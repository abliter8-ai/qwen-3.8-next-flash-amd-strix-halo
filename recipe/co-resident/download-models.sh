#!/usr/bin/env bash
# Pinned files only: do not download the optional Halogen speed overlay or GGUF draft head.
set -euo pipefail
RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QWEN_MODEL_ROOT="${QWEN_MODEL_ROOT:-${HOME}/models/halogen}"
BONSAI_MODEL_ROOT="${BONSAI_MODEL_ROOT:-${HOME}/models/bonsai}"
hf download peonist-ai/halogen-qwen3.8-flash-next \
  --revision d09cd7daaad3a06d14fb5f09322ed46c34e150f9 \
  --include qwen38-flash-next-w4b.hgn qwen38-flash-next-w4b.overlay.hgn \
    qwen38-flash-next-vision.hgn 'tokenizer/*' --local-dir "${QWEN_MODEL_ROOT}"
hf download dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF \
  Bonsai-2-27B-PQ2_0-CRACK.gguf --revision 3d36486a5fb2a3868116b8f6e768179e2391d28d \
  --local-dir "${BONSAI_MODEL_ROOT}"
hf download prism-ml/Ternary-Bonsai-2-27B-GGUF \
  Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  --local-dir "${BONSAI_MODEL_ROOT}"
(cd "${QWEN_MODEL_ROOT}" && sha256sum -c "${RECIPE_DIR}/halogen.sha256")
(cd "${BONSAI_MODEL_ROOT}" && sha256sum -c "${RECIPE_DIR}/bonsai.sha256")
