# Prose throughput: Qwen + Bonsai, idle peer versus simultaneous work

**Coding follow-up:** [matched code/prose rates, verified MTP draft acceptance and 12 executable program checks](CO-RESIDENT-CODING.md).

Measured **19 September 2026**, using **llama-benchy 0.4.0** on the
[co-resident recipe](../recipe/co-resident/README.md). These are **prose throughput
tests using random excerpts from the default Sherlock Holmes corpus**. They do
not measure coding, reasoning, or answer quality.

**Both models retain usable streaming speeds for interactive prose with short
prompts.** At 512 input / 256 output tokens, concurrent decode averaged
**20.7 tok/s for Qwen and 16.2 tok/s for Bonsai**, with first tokens at **4.78s and
3.20s** respectively. With the peer idle, decode averaged **43.1 and 22.3 tok/s**.

Long cold inputs are a different latency experience. At 32k input, concurrent
streams averaged **7.0 tok/s for Qwen and 12.6 tok/s for Bonsai once generation
began**, but first-token delays were **132.68s and 215.48s**. Qwen's complete
response rose from **36.36s to 169.21s**. Finishing both jobs together took
**235.74s versus 239.58s back-to-back**, only **1.6% less elapsed time**.

Both models stayed loaded in every condition. The baseline is **the other model
resident but idle**, not an unloaded-peer baseline. The earlier
[18 September memory/retrieval checks](CO-RESIDENT.md) answer a different question;
do not treat these trials as a repeat of their pre-change versus post-change test.

[Per-trial CSV](co-resident-benchy-2026-09-19.csv) ·
[Metadata, aggregates and all 48 stock llama-benchy reports](co-resident-benchy-2026-09-19.json)

## Decode speed

Mean ± population standard deviation, three accepted trials per cell. Each trial generated exactly 256 tokens.

| Input target | Model | Other model idle, tok/s | Both active, tok/s | Decode change |
|---:|---|---:|---:|---:|
| 512 | Qwen3.8-Flash-Next | 43.1 ± 6.8 | 20.7 ± 1.7 | -52.0% |
| 512 | Bonsai 2 27B | 22.3 ± 0.0 | 16.2 ± 0.3 | -27.5% |
| 2,048 | Qwen3.8-Flash-Next | 40.5 ± 4.1 | 19.4 ± 1.0 | -52.1% |
| 2,048 | Bonsai 2 27B | 21.1 ± 0.0 | 15.3 ± 0.9 | -27.8% |
| 8,192 | Qwen3.8-Flash-Next | 35.7 ± 0.7 | 17.1 ± 0.1 | -52.1% |
| 8,192 | Bonsai 2 27B | 19.0 ± 0.0 | 19.0 ± 0.0 | +0.0% |
| 32,768 | Qwen3.8-Flash-Next | 37.7 ± 2.7 | 7.0 ± 0.3 | -81.5% |
| 32,768 | Bonsai 2 27B | 12.6 ± 0.0 | 12.6 ± 0.0 | -0.0% |

## First-token delay and whole-request time

| Input target | Model | First token: idle / paired, s | Complete response: idle / paired, s |
|---:|---|---:|---:|
| 512 | Qwen3.8-Flash-Next | 1.42 / 4.78 | 7.47 / 17.18 |
| 512 | Bonsai 2 27B | 2.42 / 3.20 | 13.87 / 19.00 |
| 2,048 | Qwen3.8-Flash-Next | 2.82 / 12.48 | 9.18 / 25.67 |
| 2,048 | Bonsai 2 27B | 9.45 / 11.79 | 21.51 / 28.55 |
| 8,192 | Qwen3.8-Flash-Next | 7.39 / 23.91 | 14.53 / 38.80 |
| 8,192 | Bonsai 2 27B | 39.35 / 51.30 | 52.77 / 64.71 |
| 32,768 | Qwen3.8-Flash-Next | 29.57 / 132.68 | 36.36 / 169.21 |
| 32,768 | Bonsai 2 27B | 182.97 / 215.48 | 203.22 / 235.74 |

## Effective prompt processing

These are llama-benchy client-side rates from actual prompt tokens and time to first response. They include request overhead; they are not isolated engine kernel timings.

| Input target | Model | Other idle, tok/s | Paired, tok/s |
|---:|---|---:|---:|
| 512 | Qwen3.8-Flash-Next | 370.1 ± 4.2 | 109.6 ± 1.3 |
| 512 | Bonsai 2 27B | 216.2 ± 1.6 | 163.8 ± 7.4 |
| 2,048 | Qwen3.8-Flash-Next | 731.3 ± 9.1 | 167.8 ± 22.6 |
| 2,048 | Bonsai 2 27B | 218.0 ± 0.4 | 175.4 ± 10.6 |
| 8,192 | Qwen3.8-Flash-Next | 1110.8 ± 1.5 | 343.2 ± 1.6 |
| 8,192 | Bonsai 2 27B | 208.5 ± 0.2 | 159.9 ± 0.3 |
| 32,768 | Qwen3.8-Flash-Next | 1108.9 ± 13.9 | 247.1 ± 0.5 |
| 32,768 | Bonsai 2 27B | 179.2 ± 0.1 | 152.1 ± 0.2 |

## Time to finish both jobs

Sequential time is the sum of the two idle-peer mean response times. Paired time is measured from the first request start to the last completion. The corpus excerpts vary between trials.

| Input target per model | Sequential, s | Paired, s | Paired elapsed change | Pair output rate, tok/s |
|---:|---:|---:|---:|---:|
| 512 | 21.34 | 19.00 | -11.0% | 26.95 |
| 2,048 | 30.69 | 28.56 | -7.0% | 17.93 |
| 8,192 | 67.29 | 64.71 | -3.8% | 7.91 |
| 32,768 | 239.58 | 235.74 | -1.6% | 2.17 |

## Overlap between the two requests

Both requests begin together. They need not reach generation together. Decode overlap is the intersection of their first-token-to-completion intervals.

| Input target | Both requests in flight, s | Both in generation phase, s |
|---:|---:|---:|
| 512 | 17.18 | 12.39 |
| 2,048 | 25.67 | 12.34 |
| 8,192 | 38.80 | 0.00 |
| 32,768 | 169.21 | 0.00 |

If generation overlap is near zero, a model can show little decode loss even though its first token arrives much later. This must not be read as proof that sustained simultaneous decoding has no cost.

## Method and limits

- One Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), 128 GB unified memory.
  Runtime/model pins are in the [recipe](../recipe/co-resident/README.md).
  Qwen uses Halogen 0.9.1, W4B HGN plus the quality overlay and MTP. Bonsai uses
  CRACK PQ2_0 on patched Prism HIP with full GPU offload and Q4 KV.
- Context stayed at 262,144 for both models. Qwen kept four slots, a 524,288-position
  shared KV pool, and 8,192-token prefill/MAX_TOK settings; Bonsai kept one slot.
  The benchmark used one request per model, two total when paired.
- Four prompt targets, three accepted repeats per model/condition, 48 accepted
  trials, exactly 256 generated tokens each. Server input counts were 11–14 tokens
  above the nominal target because of chat-template overhead and retokenization.
- Thinking off; temperature 0; presence/frequency penalties 0. These request-level
  settings do not change service defaults. Models were warmed by validation and
  calibration requests; per-trial warmup and prompt-size adaptation were disabled.
- Each cell used an unmodified llama-benchy CLI invocation. A coordinator launched
  the two paired processes together and waited for both to finish before proceeding.
  Measured request-start skew was 0.000–0.008 seconds. Repeats 1 and 3 used Qwen,
  Bonsai, paired order; repeat 2 reversed that order.
- `--no-cache` and fresh random corpus excerpts were used. Halogen recorded zero
  cache hits and zero cached prompt tokens across the measured groups. Bonsai
  received `cache_prompt=false`. One-second runtime observations found no peer
  inference during accepted separate-model trials, with no unknown samples.
  Routes remained open: this was observational isolation, not fenced traffic.
- The tokenizer came from the running Qwen service. Both servers agreed on a
  400-content-token difference between validation prompts. Its SHA-256 is in the
  JSON. No GPT-2 tokenizer fallback occurred. Output totals came from streaming
  server usage. Without token IDs, llama-benchy interpolates timestamps; its
  peak-rate estimates should not be read as exact hardware measurements.
- One calibration Qwen response stopped at 183 tokens despite `--exact-tg`.
  The calibration set was excluded. The measured run required full-length output
  and would repeat the whole pair if either response ended early. No additional
  early-ending attempts occurred; all 48 accepted trials produced 256 tokens
  without request errors. This measures fixed-length generation, not typical
  natural-answer length.
- Measurements came from a remote client calling the runtime APIs directly, with
  no gateway or fallback. Latency mode was `none`, so no network-latency estimate
  was subtracted. First-token and total times include client/network overhead.
- Three repeats show variability, not a statistical performance bound. Corpus
  seeds were not fixed, and excerpts vary between conditions. Qwen's speculative
  acceptance can vary with content. Commands reproduce the workload definition,
  not the exact excerpt. Standard deviations use the population convention used
  by llama-benchy; displayed values are rounded.
- The longest input measured here was approximately 32k tokens. This does not
  qualify full-window 262k recall/performance, coding, reasoning, tool use, vision,
  or sustained simultaneous decoding at long context. Qwen finished before Bonsai
  started decoding at 8k and 32k: the unchanged Bonsai decode rate is not evidence
  that continuously running both decoders is free of contention.
- Post-run checks confirmed unchanged context/slot settings and process IDs, no
  service restarts, and correct `391` responses from both models for `17 × 23`.
  Bonsai's API build label was `b0-unknown`; the executable hash is in the JSON,
  while its source/patch identity comes from the pinned deployment recipe.

## Reproduce the workload

With llama-benchy 0.4.0 available, the per-trial invocation is below. Set the
endpoint, served model identifier and matching tokenizer for the model being
measured. The host paths and model aliases in the recorded environment have been
removed from the public evidence; the timing and token-count values are unchanged.

```bash
llama-benchy \
  --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --pp "$PP" --tg 256 --exact-tg --depth 0 --runs 1 --concurrency 1 \
  --no-cache --no-warmup --no-adapt-prompt --skip-coherence \
  --latency-mode none \
  --extra-body 'temperature=0,enable_thinking=false,reasoning_effort=none,presence_penalty=0,frequency_penalty=0' \
  --format json --save-result "$RESULT" --emit-progress "$PROGRESS" \
  --exit-on-first-fail
```

Run each model with the other idle, then launch one invocation per model together
and await both. Repeat this for `PP=512`, `2048`, `8192` and `32768`, with three
accepted repetitions per condition. Warm and validate the models first; the
[functional smoke harness](../harness/co-resident-smoke.py) is separate from these
throughput trials. Check the actual output count in progress events rather than
assuming `--exact-tg` was honoured, and exclude/repeat a paired attempt if either
response is short.

## Finding

Two resident models provide useful interactive prose streams on one APU. The
cost of concurrent work is most visible in individual response latency, especially
Qwen during Bonsai prefill. Paired execution reduced time to finish both fixed
jobs by about 11% at 512 tokens, falling to 1.6% at 32k, while increasing each
request's latency. Use the first-token and whole-response measurements alongside
decode speed when deciding how to schedule large inputs.
