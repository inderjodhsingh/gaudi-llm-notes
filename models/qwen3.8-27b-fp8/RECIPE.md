# Recipe — ~70–73 tok/s (greedy, thinking off, TP=1)

Checkpoint: `orcarouter/Qwen3.8-27B-Uncensored-FP8`  
Hardware: 1× Intel Gaudi2 96 GB.

## Stock vllm-gaudi (no extra patches)

This is the portable floor, typically **~48 tok/s T=1** or **~60–64 tok/s** once MTP k=1 + compiled drafter are healthy.

```bash
export PT_HPU_LAZY_MODE=0
export VLLM_GDN_COMPUTE_FP32=1
export VLLM_HPU_FORCE_CHANNEL_FP8=true
export VLLM_HPU_SPEC_KEEP_BT=0
export VLLM_HPU_SPEC_COMMIT_ALL=1
export VLLM_SKIP_WARMUP=true          # optional after a recipe cache exists
export PT_HPU_RECIPE_CACHE_CONFIG=/path/to/cache/,true,16384

vllm serve /path/to/Qwen3.8-27B-Uncensored-FP8 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --block-size 128 \
  --mamba-block-size 128 \
  --gpu-memory-utilization 0.70 \
  --limit-mm-per-prompt '{"image":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Also required in vllm-gaudi (already true for this architecture):

- `flatten_input` for Qwen3.5 / Qwen3.8 hybrids
- pin spec **padded batch = number of decode requests** when `num_tokens > 1` (do **not** use `bs × (1+k)` virtual rows)
- `torch.compile` the MTP drafter the same way as the target (regional compilation)

## Out-of-tree patch that hit 70–73 tok/s

Stock MTP k=1 still spends several milliseconds on:

1. a second `Sampler` over the 248k-row bonus logits, and
2. `.cpu()` of those logits **before** the drafter runs.

For **greedy** (temperature 0) the accept rule is argmax. Do this instead:

1. `target_ids = logits[target_indices].argmax(-1)`
2. `bonus_ids  = logits[bonus_indices].argmax(-1)`
3. Pack `[target_ids | bonus_ids]` as int32 **on HPU** (`[B, 2]` for k=1).
4. **Run `drafter.propose` now** (uses the pack / last-token index).
5. Only then `.cpu()` the `[B, 2]` pack and parse placeholders.
6. Skip `_prepare_sampling` and `RejectionSampler` / `Sampler` on this path.

Do **not** `.cpu()` the four rejection-sampler tensors onto host to “go faster” *before* the draft — that serializes the 24 ms target graph into the sample phase. Conversely, running the tiny reject-compare on HPU was **slower** than vllm-gaudi’s CPU PyTorch reject; leave that CPU path for the non-greedy case.

Probe: temperature 0, thinking off, “count from 1, one integer per line.” Expect **1…30** and ~70 tok/s e2e on a 128-token completion after graphs are warm.

## Do not

| Flag / idea | Why |
|---|---|
| `PT_HPU_LAZY_MODE=1` + wrap whole 27B in HPU graphs | OOM |
| `VLLM_HPU_SPEC_KEEP_BT=1` | Garbage / early EOS on this hybrid |
| MTP k=2 or k=3 | Collapse (~3 tok/s) or garbage |
| `tpc-clang` without `-march=gaudi2` | Gaudi1 ISA on Gaudi2 → device reset |
| `GC_KERNEL_PATH` as a colon-separated list | 1.24 treats the string as one `isfile()` |
| Custom fused GDN TPC instead of MME 128×128 | Correct math, worse tok/s |
| Concurrent batch 2 in one compiled decode graph | Dynamo crash |
| Async scheduling + spec decode | Disabled in vllm-gaudi |

## Versions this recipe was measured on

Habana **1.24**, vLLM **0.28.1rc1** + matching vllm-gaudi. Re-time after a driver bump.
