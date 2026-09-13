# Qwen3.8-Flash-Next-Uncensored-FP8 on Intel Gaudi2

Decode measurements for the **`qwen4_exp` MoE / GDN hybrid** checkpoint
(`orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`, ~186 GB FP8 incl. the 51 GB n-gram table) on
**4× Gaudi2 96 GB, TP=4 / EP=4**, vLLM 0.29.0 + vllm-gaudi `releases/v0.29.0`, Habana SynapseAI **1.24.1**,
thinking **off**.

This architecture has **no upstream HPU support** (the vLLM implementation is CUDA/ROCm only); everything below
runs on an out-of-tree port inside vllm-gaudi (see [RECIPE.md](RECIPE.md) → "Out-of-tree").

Best single-stream decode measured: **38.7 tok/s median / 38.7 p90** (512-token completions, 5 runs, box idle);
**40.2 / 41.0** on 128-token completions. Batch-4 aggregate ≈ **75 tok/s**. Context verified to the model's
**262,144** max (needle test passes at every rung up to 255K tokens).

Copy-paste flags: [RECIPE.md](RECIPE.md).

## Model

| | |
|---|---|
| Weights | FP8 block-quantised, 131 shards, ~186 GB (≈122 GB dense+experts, ≈51 GB PLE n-gram table) |
| Layout | 48 layers: **36 Gated DeltaNet + 12 full attention** (every 4th) |
| Hidden | 2560, 4 hyper-connection streams (hc_count 4, low-rank 320) |
| MoE | 512 experts, top-10 + 1 shared expert, FP8 |
| GDN | 16 key heads, 48 value heads, K=V=128, conv kernel 4 |
| Extra | PLE n-gram embedding layer at layer 1 (ngram 3, 16 heads, 320 M rows), QSA sparse-attention indexer (budget 2048), MTP-1 head, vision tower |
| KV | 2 KV heads × 256 on the 12 attention layers → 24.6 KB/token bf16 (24.6 GB per 1 M tokens) |

TP=4 is the practical layout: the weights need ≥2 cards, TP=2 leaves ~8 GB for KV, TP=8 is slower (8-way
collectives cost more than the per-rank work they save).

## Where the step goes

The decode step is **launch-bound**, not bandwidth-bound: per step the device executes only **~6.7 ms of
kernels** while the step takes ~25 ms. The rest is the gap structure of ~100–190 small recipes (two all-reduce graph
splits per layer × 48 layers) plus per-region dynamo guard evaluation (~0.2 ms per compiled region entry) and
~3.5 ms of host input preparation before the device can start. Sampling is 0.5 ms.

Consequences: bf16 storage for the fp32 elementwise work can save ≤1–2 ms; a fused 6-layer region with the
collectives kept inside the graph (`PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0`) cut the recipe count to ~12 per group and
gave **2×** on an 8-layer test model, but **breaks batched decode on the real model** (poisoned server, `!!!!`), so it
is not in the recipe. Single-token decode on this stack tops out near 40 tok/s at TP=4; the multiplier would have to
come from speculative decode (MTP head present, not yet ported to HPU: the plugin GDN kernel has no accepted-token
rollback).

## What reached the top rate

1. **Compile mode** (`PT_HPU_LAZY_MODE=0`, regional `torch.compile`). Lazy mode + HPU graphs fail in the GDN path
   under graph capture.
2. **Gathered-expert FP8 MoE** (`VLLM_HPU_MOE_GATHER=1`): 20.7 → 28.7 tok/s at TP=8.
3. **TP=4 / EP=4** instead of TP=8: 28.7 → 35.2.
4. **Sampling-metadata cache** (vllm-gaudi `hpu_input_batch.py`): indexing six per-request device tensors with a
   Python list cost a blocking H2D copy each (5.2 ms/step) → cached for an unchanged batch: 35.2 → 39.9.
5. **Fast top-k/top-p sampler** (256-candidate top-k instead of a full-vocab sort over 248 320 logits).
6. **Keep the per-layer compiled frames lean.** Every extra Python call level around a layer call is one more dynamo
   frame (guard evaluation) per layer per step: a helper method in the group loop cost 15 % (31.6 vs 37 tok/s).
7. Correctness fixes that also removed collapses: prefill state gather computed **eagerly** (inside the compiled
   region the gather silently used the state-slot index instead of the query length), and the FP8 dequant LUT for
   the n-gram table built **on the CPU** (on the HPU, `float8_e4m3fn` codes 120–127 / 248–255 decode to ±inf/NaN).

## Failed or not worth it

| Attempt | Outcome |
|---|---|
| `PT_HPU_LAZY_MODE=1` + HPU graphs | GDN path fails under graph capture; int64 buffer > 2^31 on device |
| Static shapes (`VLLM_T_COMPILE_DYNAMIC_SHAPES=false`) on the real model | 20 tok/s (recompiles per bucket/shape); +5 % only on the 8-layer test model |
| TP=8 / EP=8 | 35.8 vs 39.9 at TP=4 |
| Dense-replicated attention/GDN (no attention all-reduce) | no gain; the MoE collective dominates |
| Packed pre-transposed expert weights | ~17 % faster MoE kernel, +6 GB/rank, no step gain |
| `--async-scheduling` | first request hangs (`RPC call to sample_tokens timed out`) |
| Fused 6-layer regions (unrolled, one graph) with default all-reduce splits | **slower** (38.8 → 30 tok/s): fewer launches but each partition carries far more tensors |
| Fused regions + `PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0` | 40.2 tok/s single-stream but batch-4 decode collapses and poisons the server |
| `PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0` on the per-layer-frame path (test model, TP=2) | 103 → 90 tok/s |
| `VLLM_GDN_COMPUTE_FP32=0` | 99 → 91 tok/s (test model); keep fp32 |
| In-graph `index_put`/`index_copy_` state writes (read + write of one state tensor in one graph) | silently mis-ordered/dropped by the HPU backend → write eagerly, or write in a *later* region |
| Reading per-request metadata from the global forward context inside a compiled region | values baked in from the first traced request → resolve eagerly at model level and pass as inputs |
| Vectorised mamba state-index prep (1.6 ms Python loop → tensor ops) | correct, no step gain (host overlaps the device tail) |
| 12+ layers per compiled region | no gain |
| Editing a Python source file while a server that may still compile is running | HPU backend reads source lazily → `'NoneType' object has no attribute 'file'` crash |

## Shapes that matter

- Decode buckets are `(batch, query=1, blocks)`; batch buckets 2/3/4 compile on first use (~60 s each with
  `VLLM_SKIP_WARMUP=true`).
- Max context: keep the **prompt bucket at the chunked-prefill chunk size** (`VLLM_PROMPT_SEQ_BUCKET_MAX=8192`) — the
  largest prompt bucket's workspace, not KV, is what starves the KV budget. `--num-gpu-blocks-override` must cover
  max-model-len (≥ len/128) but a 7-attention-layer KV group tensor above ~3.5 GB fails to allocate (8192 blocks fail,
  3840 work for 256K).

| max-model-len | needle 25/50/75 % | TTFT after bucket compile | decode at depth |
|---|---|---|---|
| 32K | PASS | 3.3 s | ~26 tok/s |
| 64K | PASS | 7.2 s | ~25 tok/s |
| 128K | PASS | 17.8 s | ~25 tok/s |
| 256K | PASS | 49 s | 29–33 tok/s |

## Quality

- 12 arithmetic word problems, greedy, thinking off, 400 tokens: **7/12**; thinking on (low), 3000 tokens: **10/12**.
- Greedy 512-token code / prose / repeat prompts and three back-to-back 2k-token essays: coherent, no `!` runs, no NaN.
- Batch-4 greedy equals single-stream greedy **up to the first exact logit tie** (top-1 − top-2 = 0.000 in bf16);
  the padded argmax breaks ties differently. Deterministic per batch size.

## Versions

- Habana / driver **1.24.1** (`vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0`)
- vLLM **v0.29.0** (`98dff2a8`), vllm-gaudi **releases/v0.29.0** (`2dd55f97`) + the out-of-tree port
- transformers 5.16.1, torch 2.11.0a0
