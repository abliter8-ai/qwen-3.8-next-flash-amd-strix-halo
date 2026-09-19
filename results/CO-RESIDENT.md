# Halogen Qwen + Ternary Bonsai: recorded results

**19 September update:** [48 repeated llama-benchy prose trials](CO-RESIDENT-BENCHMARK.md)
now compare each model with the peer idle against simultaneous work, including
decode speed, first-token delay and time to complete both jobs. The 18 September
single-check measurements below remain separate.

Measured 2026-09-18 on one Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
128 GB unified memory. [Recipe and runtime pins](../recipe/co-resident/README.md).
[Machine-readable measurements](co-resident-2026-09-18.json) contain selected
fields from the deployment receipts, with private network and operational details
removed. These are **single checks, not medians or confidence intervals**.

The Qwen runtime is Halogen 0.9.1 with the quality-overlay HGN checkpoint. Bonsai
is the CRACK PQ2_0 GGUF on Prism HIP plus the input synchronization patch. Neither
set of figures is interchangeable with the August EngramHalo measurements.

## Qwen with Bonsai idle

| Qwen configuration | Cold retrieval input | Wall time | Prefill tok/s | Short prose decode tok/s |
|---|---:|---:|---:|---:|
| Original, prefill 32,768, Qwen alone | 23,944 | 19.44s | 1,242.527 | 47.850 |
| Prefill 8,192, Qwen alone | 23,944 | 20.99s | 1,150.048 | 47.759 |
| Prefill 8,192, Bonsai resident but idle | 23,950 | 20.94s | 1,152.784 | 47.723 |

All three retrieval checks returned the exact planted code `873219`. The last
prompt adds a six-token prefix to avoid a cache hit. Cold elapsed time rose by
about **8%** after reducing the prefill allocation. No additional slowdown from
idle Bonsai was visible in this check. This does not establish a statistical
performance bound.

The short prose answers each generated 70 tokens. Compare their decode phase,
not total wall time: the original short request reused 16 prompt tokens, while
the later requests were cold. A repeat of the 23,950-token request reused 23,943
tokens and answered in **0.23s**; that is a cached result, not cold-prefill speed.
Bonsai's slot monitor sampled 30 times during this arm, with zero active samples
and no new task ID.

## Bonsai correctness and responsiveness

| Check | Result |
|---|---|
| Ordinary prose | 96 output tokens, **22.390 tok/s** decode, 4.87s total |
| Exact document lookup after an unrelated sky prompt | 1,630 input tokens; `101370`, correct, 7.61s |
| Second exact lookup on that document | 1,630 input tokens, 1,114 cached; `105480`, correct, 2.78s |
| Subsequent arithmetic | `391`, correct, 0.77s |
| Native image input | 224 × 224 solid red image; `Red`, 1.04s |
| SSE | `72`, correct; 0.584s first token, 0.70s total; `[DONE]` and usage present |

Before synchronization was added, both document questions returned the prior sky
answer. Disabling graph capture reproduced the failure. Synchronizing before the
next input write fixed both lookups and the unrelated arithmetic check. Graphs
remain enabled. The public [smoke harness](../harness/co-resident-smoke.py)
repeats this type of exact-answer test; it does not merely check for non-empty text.

Publication check: the bundled harness passed **7/7 checks** against the running
pair on 2026-09-18. Its document prompts contain 1,171 tokens, so this is a separate
check from the original 1,630-token probes above. Bonsai returned both exact codes,
both models returned `391`, native vision returned `Red`, and SSE returned `72`
with `[DONE]` and usage. [Public-harness receipt](co-resident-smoke-2026-09-18.json).

## Both models prefilling at once

| Model | Cold input tokens | Exact answer | Wall time |
|---|---:|---|---:|
| Qwen | 23,948 | `873219` | **69.58s** |
| Bonsai | 10,678 | `100685, 127400, 154115` | **72.95s** |

These requests ran together. Both answered correctly, but competing for GPU
bandwidth made this much slower than leaving the other model idle. Co-residence
does not imply that two large cold prompts keep their standalone latency.

## Memory, recovery and remaining limits

Qwen's working allocation fell from **21.3 to 8.1 GiB**, freeing **13.2 GiB**.
The smaller setting retains 262,144 context, four slots and 524,288 shared KV
positions. Weight pinning, the quality overlay, vision and drafting stay enabled.
Bonsai retains 262,144 context, one slot, Q4 KV and its Q8 vision projector.

After an ordered service restart, both direct/gateway arithmetic requests passed.
Bonsai's native red-image request also passed through two OpenWebUI instances
with automatic built-in tools off. Service health and advertised metadata alone
were not treated as response qualification.

- **Full-window recall at 262,144 tokens is unqualified.** That window is configured
  and allocated; the exact retrieval checks above cover shorter inputs.
- Explicit tools and broad vision quality are unqualified. The image test is a
  native-input plumbing check.
- The public scripts and units adapt the deployed recipe to portable paths and
  loopback endpoints. A clean-host build of the new Containerfile was not part
  of these measurements.
- The recorded comparison is one host and one sample per request. It is evidence
  for this deployment, not a general hardware ceiling.
