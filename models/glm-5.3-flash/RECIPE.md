# Recipe — 742 output tok/s at 128 concurrent, 17.7 tok/s single stream, 256K context

Checkpoint: `orcarouter/GLM-5.3-Flash-Uncensored-FP8` (fallback `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`; official
`zai-org/GLM-5.3-Flash` is the same 62-shard ~306 GB FP8 layout — drop-in).
Hardware: 8× Intel Gaudi2 96 GB, TP=8 / EP=8. The weights are ~306 GB FP8, so all eight cards are needed.

Download to a **local directory** and always pass that path, never a repo id: the `glm5next` processor joins paths
against the model directory and needs `processor_config.json` next to the weights (stock Hub file; a copy is
[`processor_config.json`](processor_config.json) in this folder). The port does not depend on a modified quant block
layout — OrcaRouter is an uncensored copy of the same FP8, not a different scheme.

```bash
hf download zai-org/GLM-5.3-Flash --local-dir /path/to/glm-5.3-flash
# or: hf download orcarouter/GLM-5.3-Flash-Uncensored-FP8 --local-dir /path/to/glm-5.3-flash
```

## Common environment

```bash
export PT_HPU_LAZY_MODE=0                       # compile mode; lazy + HPU graphs fails in the KDA path
export VLLM_T_COMPILE_DYNAMIC_SHAPES=false      # static shapes; dynamic recipes do not reload across restarts
export VLLM_HPU_FORCE_CHANNEL_FP8=true
export PT_HPU_RECIPE_CACHE_CONFIG=/path/to/recipe_cache,false,16384
export VLLM_SKIP_WARMUP=false                   # warm up; first-request compiles are ~30 s each otherwise
```

## Throughput setting — 4K context, 128 sequences (742 output tok/s at C=128)

```bash
export VLLM_DECODE_BLOCK_BUCKET_MAX=4096        # >= max_seqs * ceil(max_model_len/128)

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/glm-5.3-flash \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 128 \
  --max-num-batched-tokens 4096 --no-enable-chunked-prefill \
  --num-gpu-blocks-override 1536 --gpu-memory-utilization 0.75 \
  --no-enable-prefix-caching --language-model-only \
  --generation-config vllm --trust-remote-code
```

`--generation-config vllm` matters: the checkpoint's `generation_config.json` sets `repetition_penalty 1.1`, which
silently changes greedy output if you are comparing against a reference.

## Long-context setting — 128K, 8 sequences (needle 10/10 at 96K and 128K)

```bash
export VLLM_DECODE_BLOCK_BUCKET_MAX=8192        # 8 seqs * 131072/128

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/glm-5.3-flash \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --max-model-len 131072 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --num-gpu-blocks-override 2048 --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching --language-model-only \
  --generation-config vllm --trust-remote-code
```

Warmup ~26 min. Keep **chunked prefill on** above ~64K: it bounds the KDA prefill working set, which is what the
context ceiling is actually made of. Unchunked, a single 64K prefill reaches 92 GiB per card and a 192K warmup dies
on a 1,536 MiB allocation inside `_kda_chunk_prefill`.

## Maximum verified context — 256K, 2 sequences (needle 10/10 at 256K)

```bash
export VLLM_DECODE_BLOCK_BUCKET_MAX=4096        # 2 seqs * 262144/128

python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/glm-5.3-flash \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --max-model-len 262144 --max-num-seqs 2 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --num-gpu-blocks-override 1500 --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching --language-model-only \
  --generation-config vllm --trust-remote-code
```

Warmup 28 min, serving headroom 14.0 GiB per card, needle 10/10 at 256,000 tokens at both depths. Budget the
time-to-first-token: a 256K prompt prefills in **74.9 s**. Decode is unaffected by context — 55.5 ms/token at one
stream, the same as at 4K. For 192K instead, use `--max-model-len 196608 --num-gpu-blocks-override 1100` and
`VLLM_DECODE_BLOCK_BUCKET_MAX=3072`: 50.6 s prefill and 22.7 GiB of headroom.

Warm the prompt buckets **ascending** (smallest first) so that a run that runs out of memory still leaves a usable
partial curve. Expect warmup to touch the allocator pool limit (96.9 GiB) and survive; that happens on every
configuration here, including 32K, and is not by itself a failure.

## Tool calling and reasoning

```bash
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 --enable-prompt-tokens-details
```

Multi-turn tool chains and JSON structured output both work. Pass
[`chat_templates/chat_template.enable-thinking-switch.jinja`](chat_templates/chat_template.enable-thinking-switch.jinja)
as `--chat-template`. It honours `chat_template_kwargs.enable_thinking`: `false` emits `<think></think>` so
generation starts in content mode (no leaked reasoning, no stray `</think>`); `true` (default) opens `<think>`
like the stock template. Tool calls parse in both modes. `reasoning_effort` (`low`, `high`, or the default `max`)
still works; `high` versus the default made no measurable difference on a 12-turn agent task.

## Sparse DSA (optional, slower)

```bash
export VLLM_HPU_DSA_SPARSE=1
```

The k-pool indexer is correct — perplexity within 2 % of dense on held-out 32K documents, bit-identical over the
first 2051 positions, needle 10/10 at 32K — but prefill is 1.9× dense at 32K and 2.2× at 64K, and decode is
unchanged. Dense is the default for serving. Only enable this if you are working on the indexer.

## Out-of-tree (required — upstream vLLM ships this architecture CUDA-only)

`vllm/models/glm5next/` is an NVIDIA-only subpackage upstream. The port is a numbered `git am` series in
[`patches/`](patches/) against vLLM `98dff2a8` and vllm-gaudi `2dd55f97`. **01–48** are the port, **45 is the
sampling-metadata cache-key fix** (included), 49–50 add warmup ordering and per-rank memory logging. That 01–50 set
is what the measured image (`glm53-gaudi2:20260915`) contains. 51–65 live under [`patches/optional/`](patches/optional/)
and are **not** in the shared image (MTP experiments, expert-drop; the latter slows decode).

Build: [`docker/Dockerfile.glm53`](docker/Dockerfile.glm53) + [`docker/apply_patches.sh`](docker/apply_patches.sh).
No custom TPC / `.so`.

What the port covers, in rough order of importance:

1. **Vendor the `glm5next` package** and register the config and architectures for HPU.
2. **MoE routing under expert parallelism.** The Habana fused MoE kernel ignores `experts_min`, so ranks 1–7 route to
   the wrong experts. Localize routing ids per rank. Without this the model is silently wrong.
3. **KDA (Kimi Delta Attention).** A gated-delta implementation, real **chunked prefill** (`chunk_size=64`; the naive
   token-by-token loop takes 30 s for 1024 tokens), padded decode indices, eagerly merged conv weights, NaN-safe
   initial conv state.
4. **NoPE MLA.** `qk_rope_head_dim=0` means empty rope splits, which break several upstream paths; plus latent cache
   rank handling and skipping the separate V cache.
5. **mHC hyper-connections.** 4-stream expand/contract flattened for HPU, native fused RMSNorm.
6. **Compile hygiene.** Static shapes, no unbacked `block_list` guard, dynamo cache limits sized for
   layers × buckets, an explicit MoE overload to avoid a graph break, non-contiguous MLA decode query reshape.
7. **Sampling metadata cache key** (slot ids + request ids + `max_num_logprobs` + `all_greedy` + `all_random`).
   Keying on slot indices alone returns HTTP 500 for logprobs and makes sampled requests silently greedy.
8. **Sparse DSA k-pool indexer** behind `VLLM_HPU_DSA_SPARSE`, with a top-k KV gather instead of an L×L bias mask and
   the chunk loop kept outside the compiled layer graph.

## Do not

| Flag / idea | Why |
|---|---|
| `--enable-prefix-caching` | Cache hits corrupt the KDA recurrent state under compact GDN. Silent wrong output, not a slowdown. See the note |
| `PT_HPU_LAZY_MODE=1` | KDA path fails under graph capture |
| `VLLM_T_COMPILE_DYNAMIC_SHAPES=true` | recipes do not reload across restarts on this model |
| `VLLM_DECODE_BLOCK_BUCKET_MAX` below `max_seqs × ceil(len/128)` | a new graph compiles every step; host OOM mid-run |
| `--speculative-config '{"method":"mtp",...}'` | rejected at config time, and three more blockers behind that |
| `--enable-prefix-caching` with `--no-enable-chunked-prefill` | assertion: mamba `align` cache mode requires chunked prefill |
| Unchunked prefill above ~64K | the KDA fp32 whole-prompt working set exhausts the allocator (192K warmup dies; 160K breaches the floor at a 64K prefill) |
| Trusting vLLM dataclass config defaults | they disagree with the checkpoint (`num_experts_per_token`, `first_k_dense_replace`); load the local HF config |
| Passing a repo id instead of a local directory | the processor joins paths against the model dir |
| Comparing greedy output across concurrent requests | temperature 0 is only reproducible at concurrency 1 |

## Versions this recipe was measured on

- Base image: `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` digest `sha256:b257eaeffdc6ba5e1deaa4ca3aad8ec9ed0d777d00794a0635050e0160f09fd8`
- Pip pins that differ from the base: **transformers 5.16.1**, vLLM `0.29.1.dev0+g98dff2a81`, matching vllm-gaudi, `setuptools==84.0.0`, `torchaudio==2.11.0+cpu` (no-deps). Full freeze: [`docker/constraints-glm53.txt`](docker/constraints-glm53.txt).
- Host: Ubuntu 24.04.3, kernel **6.8.0-137-generic**, `habanalabs-dkms` **1.24.1-482**, driver **1.24.1-b336d5e**, HL-SMI `hl-1.24.0-fw-62.6.2.0`. SPI preboot on this box was **hl-gaudi2-1.21.2-fw-61.0.0-sec-11** (older than the 1.24.0-fw-62.6.2-sec-11 Flash-Next box); CPLD `0x10` (2023-10-30). A failure that only reproduces on one SPI level is a host problem, not the port.
- No custom ops / TPC / shared libraries.

Measured 2026-09-13 to 2026-09-16 on 8× Gaudi2 96 GB.
