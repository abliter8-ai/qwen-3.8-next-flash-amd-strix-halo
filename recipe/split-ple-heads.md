# Splitting the engram table (only needed on Vulkan)

## The failure

On Vulkan the server dies ~30 s into loading, killed by the kernel, with no useful error from
llama.cpp itself. The kernel log shows the OOM killer taking `llama-server` with tens of GB of
anonymous RSS.

## The cause

Some GGUF builds of this architecture ship the 51B-parameter n-gram lookup table
(`per_layer_token_embd.weight`) as **one tensor**. At Q5_1 that is ~47.7 GiB.

Vulkan's `maxStorageBufferRange` is **4 GiB** on essentially every device. A single tensor larger
than that cannot be bound to the GPU at all, so the runtime falls back to keeping it host-side —
which on a unified-memory machine means an anonymous copy that competes with the very pool the
model is trying to load into.

HIP has no such limit, which is why the same file loads fine on the ROCm path.

## The fix

EngramHalo.cpp ships `gguf-py/gguf/scripts/gguf_split_ple_heads.py`. Each n-gram head is a
contiguous row range of the table, ~1.3 GiB each, and rows are a whole number of quantisation
blocks — so the split happens on block boundaries and the **quantised bytes are copied through
untouched**:

```bash
python3 engine/gguf-py/gguf/scripts/gguf_split_ple_heads.py \
  /path/to/model-00001-of-000NN.gguf \
  /path/to/model-PLESPLIT.gguf
```

No dequantise, no requantise, no quality change — only the packaging differs. Head bounds are
read from the file's own metadata.

## Measured, before and after

| | RssAnon | GPU resident | Result |
|---|---|---|---|
| Joined table, Vulkan | ~30 GB and climbing | — | OOM-killed ~30 s into load |
| Split table, Vulkan | **133 MB** | **101.9 GiB** | full load, serves |

## Also worth knowing

Even with the table split, a **cold boot at the full 262,144 context OOM'd on Vulkan** here in
the upload transient. Booting at 65,536 worked. Stepping back up was not probed.

On HIP none of this applies — the joined file loads, and the table streams from page cache with
`-lm mmap --tensor-read-lazy on`.
