# Qwen3.8-Flash-Next + Ternary Bonsai on one Strix Halo

**Two resident models on the same 128 GB Ryzen AI MAX+ 395, each configured for
262,144 tokens.** Qwen runs on Halogen 0.9.1; Ternary Bonsai 2 27B **CRACK PQ2_0**
runs on Prism's llama.cpp fork, compiled for HIP `gfx1151` with the input
synchronization patch in this directory.

Measured on 2026-09-18: Bonsai decoded ordinary prose at **22.4 tokens/s**. Qwen's
cold 24k retrieval took **19.44s before** the memory change and **20.94s with Bonsai
resident but idle**, about 8% longer. Qwen's short decode stayed near **48 tokens/s**.
See [the measurements and limits](../../results/CO-RESIDENT.md).

This is a separate runtime recipe from the [August EngramHalo study](../../README.md#august-engramhalo-study).
Its settings and results must not be mixed with that study's GGUF/MTP profiles.

## Runtime and model pins

| Component | Tested pin |
|---|---|
| Hardware | Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`, 128 GB unified memory |
| Host | Linux 7.1.5; kernel parameters `amdgpu.gttsize=126976 ttm.pages_limit=32505856` |
| Qwen engine | [Halogen 0.9.1](https://github.com/peonist-ai/halogen-flash-server/tree/v0.9.1), container digest in [the Quadlet](halogen-flash.container) |
| Qwen weights | [peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/tree/d09cd7daaad3a06d14fb5f09322ed46c34e150f9), revision `d09cd7daaad3a06d14fb5f09322ed46c34e150f9` |
| Bonsai engine | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/1a07bfa5f4144274c8f1c9963821dd9d9a51854b), commit `1a07bfa5f4144274c8f1c9963821dd9d9a51854b` plus [the synchronization patch](patch-strix-input-sync.py) |
| Bonsai build | ROCm 7.2.4, Ubuntu 24.04; [base-image digest and build commands](Containerfile.bonsai) |
| Bonsai weights | [dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF](https://huggingface.co/dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF/tree/3d36486a5fb2a3868116b8f6e768179e2391d28d), `Bonsai-2-27B-PQ2_0-CRACK.gguf`, 7.21 GB |
| Vision projector | [Prism Q8_0 mmproj](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-GGUF/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b), `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`, 0.63 GB |

Qwen and Bonsai have separate userspace ROCm environments: the pinned Halogen
image carries ROCm 7.14; the Bonsai build uses 7.2.4. This recipe retains the
qualified versions. It does not claim to qualify newer Halogen releases.

Use Prism's fork for these weights. Its Hadamard/ternary implementation is required;
the general runtime reference is [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo).
The HIP patch below is the additional change tested on this APU.

<a name="memory"></a>
<img src="https://get.abliter8.ai/doc-assets/whyitworks.png" alt="Why it works" height="58" />

## Make room without shrinking context

Change **both** Qwen prefill settings:

```ini
Environment=HALOGEN_MAX_TOK=8192
Environment=HALOGEN_PREFILL_CHUNK=8192
```

These control the size of each prefill batch and its working allocation. They do
**not** set the context window or the response token limit. Prompts longer than
8,192 tokens are processed in pieces. The supported variables are documented in
[Halogen 0.9.1 FLAGS.md](https://github.com/peonist-ai/halogen-flash-server/blob/v0.9.1/docs/FLAGS.md).

| Qwen allocation/settings | Standalone baseline | Co-resident recipe |
|---|---:|---:|
| Prefill allocation / chunk | 32,768 / 32,768 | 8,192 / 8,192 |
| Reported working memory | 21.3 GiB | 8.1 GiB |
| Per-request context | 262,144 | 262,144 |
| Generation slots | 4 | 4 |
| Shared KV positions | 524,288 | 524,288 |
| Weight pinning, quality overlay, vision, MTP, prompt lookup, prompt cache | Enabled | Enabled |

This frees **13.2 GiB**. After the change, Halogen reported 68.0 GiB of locked
weights, 14.4 GiB of KV and 8.1 GiB of working memory. The 47.7 GiB n-gram lookup
table is read through the file cache; it is not another fully resident allocation.
Four Qwen slots share the KV pool: it holds two full-window conversations, or more
shorter conversations, rather than four full windows at once.

**Linux `MemAvailable` is misleading after Halogen loads.** On the measured host
it counted about 68 GiB of GPU-locked file-backed weights as available. Use the
engine's startup accounting and GPU allocations together; do not treat that
Linux figure as room for another model. Bonsai added about 14.2 GB of GPU/GTT
allocation and 1.2 GB of host RSS. Corrected remaining room was roughly 10 GiB,
an estimate, not a guaranteed allocation budget. GB and GiB are kept distinct here.

## Fix stale HIP inputs before building

Unpatched Prism answered two 1,630-token document questions with its answer to
the previous sky question. Turning off graph capture did not fix it. The patched
build answered both exact lookups and the next arithmetic question correctly.

On this integrated GPU, HIP can read pinned host inputs directly. The patch calls
`ggml_backend_sched_synchronize(sched.get())` immediately before
`res->set_inputs(&ubatch)`, so the next input write waits for earlier GPU work.
Graph capture and reuse remain enabled. The patch script checks the expected
source block, preserves a backup and can be run again without adding another call.

<a name="quickstart"></a>
<img src="https://get.abliter8.ai/doc-assets/quickstart.png" alt="How to start" height="58" />

## Download and build

Prerequisites: the Linux/AMD GPU setup above, rootless Podman with Quadlet and GPU
device access, Git, Python 3, `sha256sum`, and the
[Hugging Face `hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
The scripts use Bash. Allow **136 GB (127 GiB) for the selected model files**, plus
space for the container images, source and build. Reuse existing verified weights
instead of making another copy.

```bash
git clone https://github.com/abliter8-ai/qwen-3.8-next-flash-amd-strix-halo.git "$HOME/qwen-strix"
cd "$HOME/qwen-strix"
./recipe/co-resident/download-models.sh
./recipe/co-resident/build-bonsai.sh
```

The downloader pins all three model repositories and verifies the five large
files against [Halogen](halogen.sha256) and [Bonsai](bonsai.sha256) SHA-256 manifests.
It keeps the quality overlay and both models' vision components. Defaults are
`$HOME/models/halogen` and `$HOME/models/bonsai`; set `QWEN_MODEL_ROOT` and
`BONSAI_MODEL_ROOT` to download elsewhere, then adjust the unit paths to match.

The build script fetches the exact Prism commit into `engine/prism-bonsai`, applies
the patch and builds `localhost/ternary-bonsai-strix:prism-1a07bfa-sync`. `SRC_ROOT`
and `BONSAI_IMAGE` override those locations. A different existing source revision
is rejected. The AMD base image needs `hipblas-dev`, `rocblas-dev` and
`hipblaslt-dev` added before CMake can configure HIP.

**Packaging boundary:** the measurements used this pinned source, patch, compiler
and CMake configuration in an existing ROCm 7.2.4 build container. The supplied
Containerfile packages that build for reuse; a clean-machine image rebuild is not
part of the recorded qualification. The public Halogen template uses a loopback
port mapping; the measured deployment used host networking. Model settings are
the same.

## Start Qwen first, then Bonsai

For a new setup, install the included user units. They assume the clone path above:

```bash
mkdir -p "$HOME/.config/containers/systemd" "$HOME/.config/systemd/user"
cp recipe/co-resident/halogen-flash.container "$HOME/.config/containers/systemd/"
cp recipe/co-resident/bonsai-strix.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable bonsai-strix.service
systemctl --user start halogen-flash.service
```

For an existing Halogen service, keep its model paths and port settings. Stop
Bonsai, apply the two 8,192-token prefill settings to the existing Quadlet, retain
the context/KV/feature settings in the table above, and install the Bonsai unit.
Reload systemd and restart `halogen-flash.service` to apply the change.

Qwen's pre-start gate requires at least 112 GiB `MemAvailable` **before Qwen
loads**. Bonsai then waits for Qwen's engine to respond. `After`, `BindsTo`,
`PartOf` and `WantedBy` keep the pair ordered: restarting Qwen stops Bonsai first
and brings it back after Qwen. This ordering passed in the measured deployment.
The Quadlet's `[Install]` enables Qwen at user-manager startup; enable user
lingering with `loginctl enable-linger "$USER"` if it must run before login.

```bash
systemctl --user status halogen-flash.service bonsai-strix.service --no-pager
journalctl --user -u halogen-flash.service -u bonsai-strix.service -n 80 --no-pager
curl --fail http://127.0.0.1:8731/health
curl --fail http://127.0.0.1:8080/props
```

Check Qwen's actual context, slots, KV pool and memory report after startup;
Halogen can reduce a requested KV pool if the available device budget is smaller.
Check Bonsai's `/props` for `n_ctx=262144`, one slot and its sampling defaults.
These checks establish allocation and readiness; run the generation checks below.

## Bonsai serving defaults

[`serve-bonsai.sh`](serve-bonsai.sh) contains the full launch command:

| Setting | Value |
|---|---|
| Context / slots | 262,144 / 1 |
| GPU / attention / KV | Full GPU offload (`-ngl 99`), flash attention, Q4_0 keys and values |
| CPU threads / repacking | 8 / `--no-repack` |
| Extra RAM prompt cache / auto-fit | Disabled (`--cache-ram 0`, `--fit off`) |
| Reasoning | Off by default |
| Temperature / top-p / top-k / min-p | 0.7 / 0.8 / 20 / 0 |
| Presence / repetition / frequency penalty | 1.5 / 1.0 / 0 (frequency is the runtime default) |
| Vision | Prism Q8_0 projector attached |
| Endpoint / model ID | `http://127.0.0.1:8080/v1`, `ternary-bonsai-2-27b` |

Sampling follows the [Prism model card's non-thinking profile](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-GGUF/blob/6ed5e12bf84b7a63069882c91dd9e9218647d17b/README.md#best-practices).
Slot prefix reuse remains available even with the extra RAM cache disabled.
`SERVER_HOST`, `SERVER_PORT`, `ALIAS`, `BONSAI_MODEL_ROOT` and `BONSAI_IMAGE` can be
set through a systemd service override. Both public templates default to loopback.

For a gateway, expose each server as a separate OpenAI-compatible model. Preserve
the sampling defaults and advertise Bonsai's native image input and 262,144
combined prompt/output context. For OpenWebUI, enable vision/file input and use
direct chat by default; leave automatic built-in tools off. Native image input
passed through LiteLLM and two OpenWebUI instances in the measured deployment.
Explicit tool selection has not been qualified by this recipe.

## Verify responses

```bash
python3 harness/co-resident-smoke.py --output evidence/co-resident-smoke.json
```

This runs an unrelated Bonsai prompt, two exact document lookups, arithmetic on
both models, a native red-image request and an SSE response with `[DONE]` and
usage. It checks exact answers rather than only a healthy endpoint or non-empty
text. Temperature 0 is used for these checks; it does not change the serving defaults.
Use `--qwen-url`, `--bonsai-url`, `--qwen-model` or `--bonsai-model` for other local
endpoints. This is a short functional check, not a full-window quality benchmark.

The [recorded results](../../results/CO-RESIDENT.md) separate cold requests,
cached requests and simultaneous prefill. Both models can stay loaded; heavy
prefill on both at once shares GPU bandwidth and is substantially slower.
Full 262,144-token recall remains unqualified.

## Return to standalone Qwen

```bash
systemctl --user disable --now bonsai-strix.service
```

Qwen can remain at the smaller prefill allocation. To restore the measured
standalone baseline, change both `HALOGEN_MAX_TOK` and `HALOGEN_PREFILL_CHUNK`
back to `32768` in its Quadlet, then run:

```bash
systemctl --user daemon-reload
systemctl --user restart halogen-flash.service
```

Do not start Bonsai beside the larger allocation without rechecking memory.
The weights and built image can remain on disk for reuse.
