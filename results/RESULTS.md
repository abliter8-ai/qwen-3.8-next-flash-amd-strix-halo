# Measured results

Single 128 GB unified-memory AMD APU (Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`), Fedora,
rootless containers. 2026-08-28. Runs serialised with nothing else on the GPU.

Two measurement methods appear here and they are **not interchangeable**:

- **Fixed-prompt server bench** — three held prompts (prose, code, structured), one warm-up
  discarded, five retained runs, temperature 0, `max_tokens` 1000. Reports the median of the
  server's own `predicted_per_second`, plus min/max/MAD and a SHA-256 of each output.
  ([`server-bench.py`](../harness/server-bench.py))
- **llama-benchy sweep** — generic continuations at a fixed 2,048-token prefill and 256-token
  generation, across depths, two runs per cell. Raw output in this directory.

---

## 1. Decode throughput, fixed-prompt bench (tok/s, median of 5, temp 0)

| Configuration | prose | code | structured |
|---|---:|---:|---:|
| Reference HIP build (quant fork + arch port), no speculation | 22.09 | 21.89 | 22.14 |
| Vulkan (MTP-capable branch) + MTP n=2 | 22.34 | 29.98 | 33.90 |
| **EngramHalo HIP + MTP, q8_0 KV** | 21.94 | **66.03** | **55.26** |
| **EngramHalo HIP + MTP, f16 KV** | 23.13 | **82.04** | 41.70 |

Spread within an arm is small: the reference lane's code median 21.89 spans 21.6–22.1 with a MAD
of 0.19. The f16-KV arm is noisier on structured output (48.2–65.1).

### Draft acceptance (fraction of drafted tokens the target accepted)

| Content | acceptance |
|---|---:|
| structured / JSON | 0.93 |
| code | 0.88 |
| prose | 0.59 |

This is the whole story of speculative decoding on this model: the jointly-trained draft head
agrees with the trunk where output is constrained, and disagrees where it is open-ended.
`--spec-draft-p-min 0.75` gates on draft confidence and is what keeps prose from regressing.

---

## 2. Depth sweep — llama-benchy, pp 2048 / tg 256, concurrency 1, 2 runs

EngramHalo HIP build, engram table streamed from SSD, 262,144 allocation.

| Depth | no speculation — tg | with MTP — tg | no spec — prefill | MTP — prefill |
|---:|---:|---:|---:|---:|
| 0 | 21.24 | 21.74 | 289.8 | 319.3 |
| 8,192 | 19.42 | 15.98 | 286.8 | 272.8 |
| 24,576 | 17.17 | 15.45 | 274.2 | 254.2 |
| 49,152 | 14.11 | 12.89 | 248.9 | 231.4 |

**Speculation is a tax at depth on this workload.** These are generic continuations, which draft
like prose. It costs 8–18% from 8K onward. The large wins in section 1 are real and
workload-shaped — code and structured output — not universal. Choose the profile per lane.

Prefill holds up well: **−14% across the first 48K**.

### Reference build, same box, at its qualification rungs

Measured with built prompts sized via the server's `/tokenize`, full 262,144 allocation:

| Prompt tokens | prefill tok/s | decode tok/s | wall | finish |
|---:|---:|---:|---:|---|
| 7,832 | 243.6 | 18.05 | 32.7 s | stop |
| 111,791 | 160.0 | 15.50 | 653.7 s | stop |
| 201,647 | 87.0 | 12.48 | 1,039.5 s | stop |

The 201,647-token answer was coherent and recalled the prompt body correctly. Time to first token
was **17.3 minutes** — this is a batch shape, not an interactive one.

Do not overlay these on the sweep above: different build, different prompt construction, different
allocation. The *shape* is the comparable part — the untuned build loses about two thirds of its
prefill rate by 200K where the tuned build loses ~14% across 48K.

---

## 3. Determinism at temperature 0

Unique outputs across five identical greedy requests:

| Configuration | prose | code | structured |
|---|---:|---:|---:|
| Reference HIP, no speculation | 1 | 1 | 1 |
| EngramHalo HIP, no speculation | 1 | — | — |
| EngramHalo HIP + MTP | 5 | 1 | 3 |
| Vulkan, no speculation | 5 | — | — |

Batch verification of drafted tokens changes floating-point accumulation order, which flips
genuinely close token choices. Outputs stay correct — the code probe below scored full marks in
the speculative configuration — but byte-equality is unavailable with speculation on, and
unavailable on Vulkan at all.

---

## 4. Executable code probe

Eight tasks, generated single-shot at temperature 0 on the ~3-bit-class quant through the
speculative profile, extracted from the reply and executed against assertions the model never saw.
([`code-qual.py`](../harness/code-qual.py))

| Task | Hidden tests | Result |
|---|---|---|
| Balanced brackets | nesting, empty string, unclosed run, mixed text | pass |
| Run-length encoding | empty input, single char, repeated non-adjacent runs | pass |
| Merge intervals | overlap, touching endpoints, empty list | pass |
| LRU cache class | eviction order, get-counts-as-use, miss sentinel | pass |
| CSV column sum | header handling, float accumulation | pass |
| IPv4 extraction | rejects a 999 octet, preserves order | pass |
| Fibonacci + arithmetic | f(10)=55 and f(37)×3−1000 exactly | pass |
| Topological sort | valid ordering, cycle returns empty | pass |

**8/8.** The arithmetic task was deliberate: a sibling dense model on the same box at similar
quantisation reliably produces valid-looking JSON with wrong sums. This one did not.

Eight single-shot tasks is a screen for gross quantisation damage, not a coding benchmark.

---

## 5. Stability — 30-minute mixed soak, single process

10 multi-turn conversations, 10 reasoning-on, 10 reasoning-off, 10 forced tool calls, 10 image
requests, 5 deterministic repeat-pairs, one 48K-token prompt, then continuous generation to fill
the window. Health and residency sampled every 30 s. ([`soak.py`](../harness/soak.py))

| | |
|---|---|
| Requests completed | 92 / 92 |
| Deterministic pairs byte-equal | 5 / 5 |
| Health samples OK | 61 / 61, one constant process id |
| Kernel OOM / GPU reset / device loss | 0 |
| Process anonymous memory | bounded (0.7 GB → 8.5 GB across the window) |

An earlier run of this soak failed honestly and is worth repeating as a warning: **ten of ten
vision requests returned empty answers**, because the harness sent a 256-token cap with reasoning
left on. Not a vision fault — the model spent the whole budget inside its think block. See Trap 3
in the top-level README.

---

## 6. Residency

| Configuration | RssAnon (process) | GPU resident |
|---|---:|---:|
| HIP with `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` | 76.9 GB | 0.2 GiB |
| HIP without it | 0.7–1.4 GB | ~73 GiB |
| Vulkan, joined 47.7 GiB table | ~30 GB, climbing | OOM-killed |
| Vulkan, table split per head | 133 MB | 101.9 GiB |
| EngramHalo, SSD engram mode, after 24K prompt | 8.3 GB | — |

The weight file stays a reclaimable page-cache mapping in every working configuration. Never
`--no-mmap`.
