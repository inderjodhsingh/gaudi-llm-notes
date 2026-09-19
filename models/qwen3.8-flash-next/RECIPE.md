# Recipe — ~39 tok/s single-stream (greedy, thinking off, TP=4), 32K–256K context

Checkpoint: `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`  
Hardware: 4× Intel Gaudi2 96 GB (one HLS-2 half). Two such replicas fit one 8-card box.

## Serve (32K context, 4 sequences, TP=4)

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_SKIP_WARMUP=true
export VLLM_HPU_FORCE_CHANNEL_FP8=true
export VLLM_HPU_MOE_GATHER=1              # gathered-expert FP8 MoE (biggest single win)
export ENABLE_EXPERIMENTAL_FLAGS=1
export ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true
export VLLM_GRAPH_RESERVED_MEM=0.1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PROMPT_SEQ_BUCKET_MAX=8192    # = the chunked-prefill chunk; never the full max-model-len
export VLLM_DECODE_BLOCK_BUCKET_MAX=256   # max-model-len / 128
export PT_HPU_RECIPE_CACHE_CONFIG=/path/to/recipe_cache,false,16384

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.8-Flash-Next-FP8 \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --num-gpu-blocks-override 4000 --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```

Longer context: raise `--max-model-len` and `VLLM_DECODE_BLOCK_BUCKET_MAX` (= len/128); keep the prompt bucket at
8192; `--num-gpu-blocks-override` between len/128 × 1.5 and ~4000 (256K: 3840). Needle retrieval passes at 32K, 64K,
128K and 256K with these settings.

Benchmark used: streamed chat completion, `temperature 0.7` (model defaults top-k 20 / top-p 0.95), thinking off,
512 new tokens, 5 runs, nothing else running on the box → **38.7 tok/s median, 38.7 p90**.

## Weights

The port loads **stock FP8 block-quant** (`e4m3`, `weight_block_size [128,128]`, dynamic activations, n-gram 3 /
`ngram_vocab_size_base=20e6` / PLE at layer 2). No custom n-gram table rewrite.

- `Qwen/Qwen3.8-Flash-Next-FP8` — loaded and served. Same layout as the OrcaRouter uncensored FP8 used for the
  numbers below. `--language-model-only` drops `model.visual.*`. If the runner asks for 3-D M-RoPE positions on a
  text-only serve, strip `mrope_section` / `mrope_interleaved` from `text_config.rope_parameters` (1-D RoPE is
  equivalent for text). Do not rewrite `modules_to_not_convert` on 0.29; that ignored-layers edit was a 0.28 leftover.
- `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` — same quant + n-gram format, different post-train. Drop-in for this
  port.

Vision / MTP / QSA-indexer weights are present in the checkpoint; the mapper loads indexer tensors but does not use
them (dense attention fallback). MTP is not wired.

## Tool calling and thinking

`--tool-call-parser qwen3_xml --reasoning-parser qwen3 --enable-auto-tool-choice`. Thinking off is
`chat_template_kwargs.enable_thinking: false` (or a 400-token cap). Thinking on (`low`) was used for the 10/12
arithmetic run at 3000 tokens. Both modes generate; tool-call markup is the Qwen3 XML parser. No later quality
movement past **7/12** greedy / **10/12** thinking-on.

## Out-of-tree (required — the architecture is not supported by vllm-gaudi)

The patch is [`patches/0001-qwen4-exp-hpu-port.patch`](patches/0001-qwen4-exp-hpu-port.patch) against vllm-gaudi
`2dd55f97`. vLLM `98dff2a8` is stock. Container: [`docker/Dockerfile`](docker/Dockerfile).

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

- Base image: `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` digest `sha256:b257eaeffdc6ba5e1deaa4ca3aad8ec9ed0d777d00794a0635050e0160f09fd8`
- Pip pins that differ from the base: **transformers 5.16.1**, vLLM `0.29.0+g98dff2a81` (empty/HPU target), matching vllm-gaudi; torch stays the image's `2.11.0a0`. Full freeze: `docker/constraints.txt` (same pin set as the GLM image).
- Host: Ubuntu 24.04.4, kernel **6.8.0-138-generic**, `habanalabs-dkms` **1.24.1-482**, driver **1.24.1-b336d5e**, HL-SMI `hl-1.24.0-fw-62.6.2.0`, SPI preboot **hl-gaudi2-1.24.0-fw-62.6.2-sec-11**, CPLD `0x10` (2023-10-30).
- No custom ops / TPC / shared libraries.

Measured 2026-09-12/13 on 4× Gaudi2 96 GB.
