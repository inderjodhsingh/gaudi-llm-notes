# Recipe — ~39 tok/s single-stream (greedy, thinking off, TP=4), 32K–256K context

Checkpoint: `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`  
Hardware: 4× Intel Gaudi2 96 GB (one HLS-2 half). Two such replicas fit one 8-card box.

## Serve (production setting used on bot2: 32K context, 4 sequences)

```bash
export PT_HPU_LAZY_MODE=0
export VLLM_SKIP_WARMUP=true
export VLLM_HPU_FORCE_CHANNEL_FP8=true
export VLLM_HPU_MOE_GATHER=1              # gathered-expert FP8 MoE (biggest single win)
export ENABLE_EXPERIMENTAL_FLAGS=1
export VLLM_PROMPT_SEQ_BUCKET_MAX=8192    # = the chunked-prefill chunk; never the full max-model-len
export VLLM_DECODE_BLOCK_BUCKET_MAX=256   # max-model-len / 128
export PT_HPU_RECIPE_CACHE_CONFIG=/path/to/recipe_cache,false,16384

python3 -m vllm.entrypoints.openai.api_server \
  --model orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8 \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --num-gpu-blocks-override 4000 --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching --language-model-only --trust-remote-code
```

Longer context: raise `--max-model-len` and `VLLM_DECODE_BLOCK_BUCKET_MAX` (= len/128); keep the prompt bucket at
8192; `--num-gpu-blocks-override` between len/128 × 1.5 and ~4000 (256K: 3840). Needle retrieval passes at 32K, 64K,
128K and 256K with these settings.

Benchmark used: streamed chat completion, `temperature 0.7` (model defaults top-k 20 / top-p 0.95), thinking off,
512 new tokens, 5 runs, nothing else running on the box → **38.7 tok/s median, 38.7 p90**.

## Out-of-tree (required — the architecture is not supported by vllm-gaudi)

`qwen4_exp` is CUDA/ROCm-only upstream. The port lives in vllm-gaudi (`vllm_gaudi/models/qwen4_exp.py` and
friends) and registers `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration`:

1. **Hyper-connections**: upstream pure-torch `GatedResidual` (4 streams, low-rank mixers).
2. **Gated DeltaNet**: the plugin's `HPUGatedDeltaNetAttention` (non-interleaved GQA layout, fp32 recurrent update).
3. **Full attention**: `Qwen3NextAttention` dense; the **QSA sparse indexer is not ported** (its weights are loaded but
   unused) — dense attention is exact for ≤2048 keys and still passes the needle test at 255K.
4. **MoE**: FP8 per-channel gathered-expert path (`VLLM_HPU_MOE_GATHER=1`), 128 local experts per rank at EP=4.
5. **PLE n-gram layer** as a `MambaBase` state layer (`mamba_type SHORT_CONV`): compact per-request state
   (`conv_state [9, 10240]` bf16 + last-2-token ids), n-gram hash reproduced **bit-exactly in int32 8-bit limbs**
   (HPU int64 arithmetic truncates to 32 bit), 320 M-row FP8 table stored as **uint8 + a 256-entry LUT built on the
   CPU** (Gaudi2 decodes e4m3 codes 120–127 to inf), TP-row-sharded with mask + all-reduce.
6. **State writes**: prefill state gather done eagerly (dynamo-disabled); decode writes eager `index_copy_`. Never read
   and write the same state tensor inside one compiled graph.
7. **Per-request metadata** (state slots, query lengths) resolved eagerly at model level and passed into the
   compiled regions as inputs.
8. Runner/platform glue: `SHORT_CONV` in the GDN mamba types, `.ple` as a mamba-like layer with its cache-group index,
   the two architectures in the hybrid-cache allowlists, `qwen4_exp` in the Qwen-MoE model types, and
   `check_runner_kv_caches_multi_layer` disabled (1-layer cache groups).
9. **Sampling-metadata cache** in `hpu_input_batch.py` (skip re-indexing per-request device tensors for an unchanged
   batch) and a **256-candidate top-k/top-p** sampler (`ops/hpu_topk_topp.py`).

Layers are compiled as individual regions (the runner's regional compilation) inside a plain Python loop; fusing them
into larger regions is slower unless the collectives stay in-graph, which breaks batching (see the note).

## Do not

| Flag / idea | Why |
|---|---|
| `PT_HPU_LAZY_MODE=1` (HPU graphs) | GDN path fails under graph capture |
| `VLLM_T_COMPILE_DYNAMIC_SHAPES=false` | 20 tok/s: recompiles per shape on the real model |
| `--async-scheduling` | first request hangs (`sample_tokens` RPC timeout) |
| `PT_HPU_ENABLE_ALLREDUCE_GRAPH_SPLIT=0` | batched decode collapses to `!!!!` and poisons the server |
| TP=8 | slower than TP=4 (35.8 vs 39.9) |
| `VLLM_PROMPT_SEQ_BUCKET_MAX` = max-model-len | the prompt-bucket workspace starves the KV budget at ≥128K |
| `--num-gpu-blocks-override 8192` | a 7-layer KV group tensor > 3.5 GB fails (`PT_DEVMEM Allocation failed`) |
| `VLLM_GDN_COMPUTE_FP32=0` | slower and riskier; keep fp32 |
| Speculative decode (`ngram`, `mtp`) | not supported by the HPU GDN/PLE state path yet (no accepted-token rollback) |
| Editing model source while the server runs | the HPU backend reads source lazily at compile → worker crash |

## Versions this recipe was measured on

Habana **1.24.1**, vLLM **v0.29.0** (`98dff2a8`), vllm-gaudi **releases/v0.29.0** (`2dd55f97`) + the port above,
transformers 5.16.1, torch 2.11.0a0. Measured 2026-09-12/13 on bot2.
