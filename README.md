# Qwen3.8-Flash-Next on one AMD Strix Halo

## September update: Qwen + Ternary Bonsai resident together

The [co-resident recipe](recipe/co-resident/README.md) runs **Qwen3.8-Flash-Next on
Halogen 0.9.1** alongside **Ternary Bonsai 2 27B CRACK on patched Prism HIP**, on
the same 128 GB Strix Halo. Both retain **262,144-token context**.

Reducing Qwen's prefill batches from 32,768 to 8,192 releases 13.2 GiB of working
memory. This changes batching, not context. Qwen keeps four slots, its 524,288-position
shared KV pool, pinned weights, quality overlay, vision and drafting.

Recorded on 2026-09-18: Bonsai prose decode **22.4 tok/s**; Qwen cold 24k retrieval
**19.44s before / 20.94s with Bonsai resident but idle**, with short decode still
near **48 tok/s**. Heavy simultaneous prefill slows both models. These are single
checks; full-window recall is not qualified.

- [Setup, pinned downloads, HIP fix, launch units and rollback](recipe/co-resident/README.md)
- [Measurements, memory accounting and qualification limits](results/CO-RESIDENT.md)
- [Direct generation and stale-input regression checks](harness/co-resident-smoke.py)

### Repeated prose benchmark — 19 September

**48 llama-benchy trials**, using Sherlock Holmes prose excerpts, compare each
model with the other resident but idle against both running together. At 512
input / 256 output tokens, concurrent decode averages **20.7 tok/s for Qwen and
16.2 tok/s for Bonsai**, versus **43.1 and 22.3 tok/s** with the peer idle:
usable interactive prose streams from both models on one APU.

Decode speed is only part of the experience. At 32k input, concurrent first-token
delays reach **132.68s for Qwen and 215.48s for Bonsai**. These are prose throughput
results, not coding or reasoning benchmarks.

- [Full benchmark, latency tables, overlap and findings](results/CO-RESIDENT-BENCHMARK.md)
- [All 48 trial rows](results/co-resident-benchy-2026-09-19.csv) · [metadata and stock reports](results/co-resident-benchy-2026-09-19.json)

## August EngramHalo study

The original single-model recipe and results follow. They use a different engine
and quantization from the September setup.

Serving a **180B-parameter / ~6B-active mixture-of-experts model on a single 128 GB
unified-memory APU** (Ryzen AI MAX+ 395, Radeon 8060S, `gfx1151`) — the launch recipe, the
engine builds that were needed to get there, the traps that cost real time, and the
measurement harness that produced every number.

**Full write-up with charts:**
https://artifacts.abliter8.ai/recipes/qwen3-8-flash-next-strix-halo-180b-on-one-apu.html

Measured 2026-08-28. One machine, one thermal environment, runs serialised with nothing else
on the GPU. Every number here came from a run on that box; see [Honest gaps](#honest-gaps).

---

## Headline

| | |
|---|---|
| Code decode, fastest configuration | **82 tok/s** (3.7× the same box without speculation) |
| Structured / JSON decode | **55 tok/s** at 0.93 draft acceptance |
| Prose decode | ~22 tok/s — speculation neither helps nor hurts, *with* a confidence gate |
| Context served | **262,144** — a 201,647-token prompt answered coherently, single slot |
| Weights resident | 98.5 GiB at 4.78 bpw, with the 47.7 GiB lookup table streamed from SSD |
| Executable code probe | **8 / 8** single-shot programs passed held-out assertions |
| 30-minute mixed soak | 92/92 requests, one process, zero kernel faults |

---

## The model, briefly

48 layers; 512 experts (10 routed + 1 shared); a hybrid attention stack of 36 gated-delta-net
layers interleaved with 12 sparse-attention (QSA) layers. **Only those 12 grow a KV cache**, so
a quarter-million-token context costs ~6 GiB rather than ~100.

The awkward part is a **51B-parameter n-gram lookup table** (per-layer token embedding, the
"engram" table) — pure random access, not matrix maths. On a 128 GB unified-memory APU, where
that table lives and *in what shape* is the decision that separates a lane that serves from one
that gets OOM-killed thirty seconds into loading.

## Engine

The architecture landed upstream in llama.cpp days before this work; the quantisation formats
that make it fit on this hardware live in a **separate community fork**; the multi-token
prediction (MTP) draft head lives in a **third**. No single published build had all three.

The fastest configuration here uses **[EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp)**
(branch `strix-halo-qwen4exp`) — Strix-Halo kernel patches, a true sparse-attention gather, the
MTP draft head, and SSD-backed engram streaming. Build it HIP-only with
[`recipe/build-engine.sh`](recipe/build-engine.sh).

> **Backend note, confirmed independently here:** this patch set is HIP/ROCm-tuned. The Vulkan
> path is a different story (see [Trap 2](#trap-2--one-tensor-too-big-for-the-backend)).

## Quick start

```bash
# 1. Build the engine (inside a ROCm container or on a ROCm host)
MODEL_ROOT=/path/to/models ./recipe/build-engine.sh

# 2. Fetch or build the MTP draft sidecar (~8 GB of traffic, not 336 GB)
./recipe/export-mtp-sidecar.sh Qwen/Qwen3.8-Flash-Next

# 3. Serve — fast profile (speculation on)
MODEL_ROOT=/path/to/models ./recipe/serve-fast.sh

#    …or the reproducible profile (no speculation, bit-identical output)
MODEL_ROOT=/path/to/models ./recipe/serve-reference.sh
```

Both scripts take everything from environment variables and default to **loopback only**. Set
`SERVER_HOST` deliberately if you want it on a network.

---

## The two traps

### Trap 1 — the unified-memory flag that inverts itself

`GGML_HIP_ENABLE_UNIFIED_MEMORY=1` sounds mandatory on a unified-memory APU. It is the opposite
of what you want. Measured on this hardware, same launch line otherwise:

| | RssAnon (process) | GPU (GTT) used |
|---|---|---|
| With the variable set | **76.9 GB** | 0.2 GiB |
| Without it | **0.7–1.4 GB** | ~73 GiB |

With it set, the entire model lands in anonymous process memory: unreclaimable, invisible to GPU
telemetry, one allocation from the OOM killer. Without it, weights sit in GPU memory and the
weight file stays a reclaimable page-cache mapping. The variable belongs to discrete-GPU recipes
where oversubscription is the point.

### Trap 2 — one tensor too big for the backend

On Vulkan the server was killed ~30 s into loading, every time. Cause: the lookup table ships as
**one 47.7 GiB tensor**, and Vulkan's maximum single-buffer binding is 4 GiB. Too large to bind,
it fell back to a host-side copy that exhausted the machine.

EngramHalo.cpp ships the fix — `gguf_split_ple_heads.py` splits the table into 16 per-head
tensors and **copies the quantised bytes through untouched** (no dequantise, no requantise, no
quality change). After splitting: 133 MB anonymous, 101.9 GiB resident on the GPU, full load.
See [`recipe/split-ple-heads.md`](recipe/split-ple-heads.md).

### Trap 3 — reasoning is on by default, and a short cap returns *nothing*

The chat template enables thinking by default at high effort. A request capped at 256 output
tokens spends the whole budget inside the think block and returns **empty visible content** with
`finish_reason: length`. The measured floor for a simple vision answer on this stack is **544
completion tokens**.

**Caller contract:** thinking-on callers send `max_tokens >= 1024`; short-cap and vision callers
send `chat_template_kwargs: {"enable_thinking": false}`. This bit an early soak run here — ten of
ten vision requests returned empty, and the smoke suite had masked it by counting reasoning text
as output.

---

## Measured results

Full tables in [`results/RESULTS.md`](results/RESULTS.md); raw sweep output in
[`results/`](results/).

### Decode throughput — same box, same prompts, medians of 5 at temperature 0

| Configuration | prose | code | structured |
|---|---|---|---|
| Reference HIP build, no speculation | 22.1 | 21.9 | 22.1 |
| Vulkan + MTP | 22.3 | 30.0 | 33.9 |
| **Tuned HIP + MTP** (q8_0 KV) | 21.9 | **66.0** | **55.3** |
| **Tuned HIP + MTP** (f16 KV) | 23.1 | **82.0** | 41.7 |

Draft acceptance tracks how predictable the text is: **0.93 structured, 0.88 code, 0.59 prose**.
`--spec-draft-p-min 0.75` is what keeps prose from going backwards — without the confidence gate,
rejected drafts cost more than accepted ones save.

### Depth

Prefill on the tuned build holds nearly flat across the first 48K (290 → 249 tok/s). The
untuned reference build falls from 244 tok/s at 8K to 87 at 200K — EngramHalo's own docs name the
causes (sparse attention that ran dense with a mask; a top-k selection that fell back to CPU past
~1K context) and its patches address both.

**Speculation is a tax at depth on unpredictable text**: on generic continuations the standard
sweep showed 16.0 vs 19.4 tok/s at 8K and 12.9 vs 14.1 at 48K, draft on vs off. The large wins are
workload-shaped, not universal.

### Determinism

| Configuration | Unique outputs / 5 identical greedy runs |
|---|---|
| HIP, no speculation | **1** (prose, code, structured) |
| Tuned HIP, no speculation | **1** |
| Tuned HIP + MTP | 5 prose · 1 code · 3 structured |
| Vulkan, no speculation | 5 |

Verifying a batch of drafted tokens changes floating-point accumulation order, which flips
genuinely close token choices. Every output stayed correct and coherent — the code probe scored
8/8 in exactly this configuration — but byte-equality is not available with speculation on.
**Both profiles run on the same engine and weights, one flag apart.**

### Does the fast path still write correct code?

Eight programming tasks generated single-shot at temperature 0 on a ~3-bit-class quantisation,
then **extracted and executed** against assertions the model never saw: balanced brackets,
run-length encoding, interval merging, an LRU cache, CSV column sum, IPv4 extraction (rejecting a
`999` octet), iterative Fibonacci plus an exact arithmetic check, and topological sort with cycle
detection. **8/8 passed.** Run it yourself: [`harness/code-qual.py`](harness/code-qual.py).

---

## Harness

Standard-library Python, no dependencies, all pointed at an OpenAI-compatible endpoint:

| Script | What it does |
|---|---|
| [`smoke.py`](harness/smoke.py) | 7 functional checks: text, multi-turn recall, reasoning on/off, forced tool call, vision, corruption guard |
| [`server-bench.py`](harness/server-bench.py) | fixed prose/code/structured prompts, warm-up discarded, median/min/max/MAD, output hashes |
| [`determinism.py`](harness/determinism.py) | 20 fixed prompts at temp 0, capture and compare two arms |
| [`ladder.py`](harness/ladder.py) | depth ladder using the server's own `/tokenize` to hit exact token targets |
| [`code-qual.py`](harness/code-qual.py) | generates code, extracts it, executes it against held-out asserts |
| [`soak.py`](harness/soak.py) | 30-minute mixed-capability soak, health + residency sampled every 30 s |

---

## Honest gaps

- **The fastest numbers and the code probe are from a ~3-bit-class quantisation** — a lossier
  build than the ~4.8-bit one used for the correctness gates. Speed and quality figures are not
  from the same file.
- **Prefill curves are not superimposable**: the tuned build was swept 0–48K at a fixed
  2,048-token prefill; the reference build was measured at 8K/111K/200K with built prompts. Depth
  *shape* is comparable, absolute values across methods are not.
- **The code probe is eight single-shot tasks.** It detects gross quantisation damage. It is not
  a coding benchmark.
- **Single machine, single sample.** No cross-machine replication. The EngramHalo author likewise
  reports n=1 hardware.
- **Long context is a batch shape, not an interactive one**: the 200K prompt took 17.3 minutes to
  first token.
- **Vision, tool calling and the reasoning toggle were verified functionally**, not scored for
  quality.

## Credits

- Architecture support: upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) maintainers
  and the model's quantisation community (PR #27742, merged 2026-08-27).
- Strix-Halo kernel tuning, sparse-attention gather, MTP draft head, table-splitting utility and
  SSD streaming: **[Aristo94/EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp)**.
- Pre-built MTP sidecar:
  **[EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF)**.
- Container patterns and host tuning groundwork:
  [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).

## Licence

Scripts and documentation in this repository: MIT (see [LICENSE](LICENSE)).
The **model** is under the Qwen Community License 1.0 — read it before serving it to anyone else;
it carries Model-as-a-Service restrictions. Engine code belongs to its respective projects.
