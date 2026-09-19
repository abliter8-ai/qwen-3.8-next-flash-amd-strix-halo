# Coding throughput with Qwen MTP + Bonsai

**Qwen was faster on this code prompt; Bonsai was essentially unchanged.** Both remained resident throughout. Qwen explicitly used its MTP speculative drafter; Bonsai used its existing non-speculative Prism HIP profile. MTP was already enabled in the earlier prose run, so this compares workloads, not switching speculation on versus off.

Measured 19 September 2026 on the same [co-resident recipe](../recipe/co-resident/README.md). [All 24 response records, engine draft counters, generated outputs and executable checks](co-resident-code-2026-09-19.json) · [CSV](co-resident-code-2026-09-19.csv).

## Matched 512-input / 256-output comparison

Three repeats per model and condition, temperature 0, thinking off. Both workloads use the same nominal 512-token input and exactly 256 output tokens. Values are mean decode tokens/s; code spread is population standard deviation. The [prose reference](CO-RESIDENT-BENCHMARK.md) uses Sherlock Holmes excerpts.

| Model | Peer state | Prose tok/s | Code tok/s | Code change |
|---|---|---:|---:|---:|
| Qwen3.8-Flash-Next | Idle | 43.1 | 51.0 ± 1.2 | +18.2% |
| Qwen3.8-Flash-Next | Active | 20.7 | 26.6 ± 0.8 | +28.3% |
| Bonsai 2 27B | Idle | 22.3 | 22.2 ± 0.0 | -0.1% |
| Bonsai 2 27B | Active | 16.2 | 16.0 ± 0.5 | -1.1% |

The code prompt requests a token-bucket rate-limiter module with types, validation, per-key LRU storage, thread safety, decorators and context managers, with no comments or docstrings. The 256-token samples are **code-generation prefixes**, not complete modules and not a correctness result. Complete, executable tasks are checked separately below.

| Model | Peer state | Code first token, s | Code response, s |
|---|---|---:|---:|
| Qwen3.8-Flash-Next | Idle | 1.46 | 6.46 |
| Qwen3.8-Flash-Next | Active | 5.81 | 15.41 |
| Bonsai 2 27B | Idle | 2.44 | 13.90 |
| Bonsai 2 27B | Active | 2.89 | 18.87 |

Qwen generates the fixed code response faster, but its paired first-token delay is 5.81s versus 4.78s in the prose test. Total paired Qwen response time improves from 17.18s to 15.41s. Decode rate and time to first output are separate measurements.

## Speculation was exercised

Qwen received `drafter="mtp"` on every request. Every Qwen response reported nonzero proposed and accepted draft counts. The counters cover the MTP head and prompt-lookup proposals; they do not isolate the head alone.

| Peer state | Prose accepted / proposed | Code accepted / proposed |
|---|---:|---:|
| Idle | 324 / 520 (62.3%) | 375 / 469 (80.0%) |
| Active | 305 / 543 (56.2%) | 380 / 465 (81.7%) |

The higher acceptance accompanies the higher code throughput. This is not a measured serial-to-MTP speedup: no serial control was run. Prose acceptance is derived from before/after runtime counters for its three matching trials; code acceptance comes from per-response engine timings.

## Complete coding tasks

Three tasks from the existing [code-quality harness](../harness/code-qual.py): balanced brackets, an LRU cache, and topological sorting including cycle detection. Each model completed each task with the peer idle and then with both active. **All 12 programs passed the existing executable assertions, without repairing or retrying the generated code.** These are narrow functional checks, not a general coding-quality score.

| Task | Model | Idle / paired decode, tok/s | Idle / paired response, s | Output tokens, idle / paired | Checks |
|---|---|---:|---:|---:|---|
| balanced | Qwen3.8-Flash-Next | 44.8 / 24.7 | 3.10 / 5.40 | 94 / 94 | 2/2 pass |
| balanced | Bonsai 2 27B | 22.9 / 16.7 | 4.59 / 6.56 | 92 / 92 | 2/2 pass |
| lru | Qwen3.8-Flash-Next | 48.8 / 26.6 | 5.60 / 7.50 | 218 / 150 | 2/2 pass |
| lru | Bonsai 2 27B | 22.5 / 17.6 | 8.14 / 10.87 | 169 / 169 | 2/2 pass |
| topo | Qwen3.8-Flash-Next | 47.5 / 24.9 | 5.64 / 10.63 | 216 / 219 | 2/2 pass |
| topo | Bonsai 2 27B | 22.6 / 15.4 | 7.61 / 11.38 | 158 / 156 | 2/2 pass |

These shorter task prompts use their natural output lengths, with a 1,200-token cap; all completed normally. Their rates should not be treated as a fixed-token speedup against the prose corpus. Inputs, outputs and held assertions are in the JSON.

## Method and limits

- Same Halogen 0.9.1 and patched Prism HIP deployment as the prose comparison. Both contexts remained 262,144; Qwen retained four slots and Bonsai one. No service configuration changed.
- The coding collector uses llama-benchy 0.4.0 token accounting and metric calculations, and additionally captures native SSE `timings` for speculative counters. The prose run used the stock CLI. Client-side latency includes network overhead; no latency estimate is subtracted.
- Fixed code input is padded to 512 local tokens; both servers reported 524 including their chat template. A unique leading request identifier prevents prefix reuse. Paired requests share the same task prompt and are launched together. Both complete before the next group starts.
- Temperature 0, thinking off, presence/frequency penalties 0; streaming usage supplies authoritative token totals. Both services remained resident. No peer activity was observed during separate tests, and responses reported no cached prompt tokens.
- The initial module prompt requested docstrings and produced a documentation-heavy prefix. Those calibration samples were retained locally but excluded from the code-only result. The reported fixed-code set contains all 12 subsequent measurements, each exactly 256 output tokens.
- Code uses one fixed programming instruction plus a cache-avoidance identifier; prose uses random excerpts. Three samples are descriptive, not a statistical or universal claim that coding is faster. Prompt wording, output style, speculative acceptance and context depth can change the result.
- Functional tasks were streamed to completion, then their generated Python was parsed and executed with the existing assertions in separate bounded processes. No measured code was rewritten to pass.
- Both services remained active with the same process IDs and no restarts after testing. Longer-context coding performance, serial-versus-speculative speedup and broader coding quality were not tested.
