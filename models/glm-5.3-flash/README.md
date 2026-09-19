# GLM-5.3-Flash-Uncensored-FP8 on Intel Gaudi2

Decode measurements for the **`glm5_next` MoE hybrid** checkpoint
(`orcarouter/GLM-5.3-Flash-Uncensored-FP8`, 62 shards, ~306 GB FP8; `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` is an
equivalent drop-in) on **8× Gaudi2 96 GB, TP=8 / EP=8**, vLLM 0.29.1.dev + vllm-gaudi `releases/v0.29.0`,
Habana SynapseAI **1.24.1**, vision tower skipped (`--language-model-only`).

Upstream vLLM ships this architecture as a **CUDA-only subpackage** (`vllm/models/glm5next/nvidia/`). Everything below
runs on an out-of-tree port inside vllm-gaudi (see [RECIPE.md](RECIPE.md) → "Out-of-tree").

Best single-stream greedy decode: **17.7 tok/s** (57 ms/token), and that figure is flat in context — 57.2 ms/token at 4K and 55.5 ms/token at 255K. Throughput scales far better than single stream: **742 output tok/s at 128 concurrent**
(4K context). Context verified to **262,144** tokens with needle retrieval 10/10 at 256K, both depths.

Copy-paste flags: [RECIPE.md](RECIPE.md).

## Model

| | |
|---|---|
| Weights | FP8 block-quantised e4m3, dynamic activation scheme, 62 shards, ~306 GB |
| Layout | 45 layers: **34 Kimi Delta Attention (KDA, linear) + 11 DeepSeek sparse attention (DSA/MLA)** every 4th |
| Hidden | 4096, vocab 154,880, dense MLP on layers 0–2 |
| MoE | 288 routed experts, top-8, 1 shared, sigmoid router with `noaux_tc`, `n_group=1` |
| KDA | 64 heads × 128, conv kernel 4, bounded-sigmoid gate (`gate_lower_bound=-5.0`), not softplus |
| DSA / MLA | q-LoRA 1536, kv-LoRA 512, `qk_nope_head_dim=256`, **`qk_rope_head_dim=0` (NoPE)**, `v_head_dim=256` |
| Indexer | `index_topk=2048`, `index_kpool=4`, 32 heads × 128 — selects 2048/4 = 512 pools plus the tail |
| mHC | 4-stream hyper-connection residual, Sinkhorn 20, fp32 `hc_*` |
| Not quantised | KDA, mHC, indexer, norms, router and correction bias, `lm_head`, `embed_tokens`, conv1d |
| Extra (unused here) | MTP head at layer 45, 24-layer vision tower |

The checkpoint advertises a 1M context. 128K is what this box warms up and serves; see "Context" below for where the
ceiling actually comes from.

## What reached the top rate

1. **Fix expert-parallel MoE first.** The Habana fused MoE kernel ignores `experts_min` under EP, so ranks 1–7 were
   silently routing to the wrong experts. Output looked fluent and was wrong. Everything downstream is meaningless
   until this is patched.
2. **Real chunked KDA prefill.** The first port walked the KDA recurrence token by token: a 1024-token prefill took
   30 s. Chunked (`chunk_size=64`) brought it to **0.87 s**, a 180× step change.
3. **Compile mode** (`PT_HPU_LAZY_MODE=0`, regional `torch.compile` per decoder layer) with **static shapes**
   (`VLLM_T_COMPILE_DYNAMIC_SHAPES=false`) and an on-disk recipe cache. Dynamic shapes failed to reload recipes across
   restarts on this model.
4. **Skip the MLA V-cache.** The NoPE MLA path stores only the latent; dropping the separate V tensor freed the KV
   budget that 64- and 128-sequence configs need.
5. **Sampling-metadata cache keyed correctly.** An inherited fast path keyed only on batch slot indices, so a new
   request in a reused slot inherited the previous request's `max_num_logprobs` (HTTP 500 on any logprobs request) and
   its `all_greedy` flag — meaning **sampled requests were silently greedy**. Key on slot ids, request ids,
   `max_num_logprobs`, `all_greedy` and `all_random`.
6. **Size the decode block buckets from KV blocks, not sequences.** Decode block buckets count **128-token** KV blocks:
   `VLLM_DECODE_BLOCK_BUCKET_MAX >= max_seqs × ceil(max_len/128)`. Getting this wrong lazily compiles a new graph every
   step and the host runs out of RAM mid-benchmark.
7. **Chunked prefill for long context.** It bounds the KDA prefill working set, which is the real context ceiling
   (see below). With it, a 128K serve sits at 68 GiB per card; without it, a single 64K prefill reaches 92 GiB.

## Throughput

4K context, 1024 in / 1024 out, `vllm bench serve`, 128-sequence config:

| concurrency | output tok/s | total tok/s | median TPOT | mean TTFT |
|---|---|---|---|---|
| 1 | 17.7 | 35 | 57 ms | 0.6 s |
| 16 | 239 | 479 | 63 ms | 4.4 s |
| 32 | 436 | 872 | 68 ms | 6.2 s |
| 64 | 641 | 1282 | 94 ms | 9.0 s |
| 128 | **742** | 1483 | 166 ms | 14.3 s |

Single-stream decode is flat in context length: 57 ms/token at 4K, 62 ms at 64K, ~80 ms at 90K with 8 streams.
A 16-concurrent soak (1024 in / 256 out, 512 requests) completed 512/512 in 1037 s with 0 recompiles.

TPOT 57 → 68 ms from 1 to 32 streams is **~0.35 ms per extra stream**. That is the v0.29 compile-mode Gaudi2
shape (a dense 27B on the same stack dropped from ~2.8 ms to ~0.7 ms per extra stream moving 0.26 → 0.29). The
sampling-metadata cache (patch 45) removes a ~5 ms H2D that would sit on **every** step regardless of batch; static
shapes stop per-step recompiles that would destroy the curve. The step is still host/launch bound (~56 ms vs a
~17 ms weight-bandwidth floor; a decode profile saw 241 recipe enqueues and an eager MoE op every layer from patch
37). We did not add a new batched GEMM.

## Context

Dense MLA (the serving default), needle retrieval = a 6-digit code inserted at a depth in filler prose, greedy answer:

| max-model-len | seqs | warmup | prefill (single request) | needle @ depth 0.5 / 0.95 | 8-way decode |
|---|---|---|---|---|---|
| 32,768 | 8 | 19 min | 32K in 5.9 s | 5/5 · 5/5 at 8K, 16K, 32K | 8K ctx, 1K out: 115 tok/s, TPOT 61 ms |
| 65,536 | 8 | 21 min | 64K in 11.8 s | 5/5 · 5/5 at 48K and 64K | 8 × 60K, 512 out: 129 tok/s steady state |
| 131,072 | 8 | 26 min | 128K in 25.5 s | **10/10 · 10/10** at 96K and 128K | 8 × 90K, 256 out: 99–103 tok/s |
| 196,608 | 2 | 28 min | 192K in 50.6 s | **10/10 · 10/10** at 192K | 190K in / 256 out: 57.2 ms TPOT at 1 stream, 153 ms at 2 |
| 262,144 | 2 | 28 min | 256K in 74.9 s | **10/10 · 10/10** at 256K | 255K in / 256 out: 55.5 ms TPOT at 1 stream, 200 ms at 2 |

The 192K and 256K rows need **chunked prefill**; the others do not. Serving headroom is 22.7 GiB per card at 192K and 14.0 GiB at 256K.

> **The two prefill figures for 128K are both correct and are not comparable.** The table above says 25.5 s; the
> scaling table below says 30.8 s. The first is unchunked at 8 sequences, the second is a point on a single chunked
> 2-sequence curve measured end to end for the fit. Chunking costs about 20 % at 128K and is what makes 192K and 256K
> possible at all. Compare within a table, never across them.

### Prefill scaling

Six points, one request at a time, three timed runs each (repeat spread ≤ 0.054 s), fitting `t(L) = aL + bL²` with no
intercept:

| L | prefill | s per 1k tokens | quadratic share |
|---|---|---|---|
| 16,384 | 3.10 s | 0.189 | 3.4 % |
| 32,768 | 6.35 s | 0.194 | 6.5 % |
| 65,536 | 13.56 s | 0.207 | 12.2 % |
| 131,072 | 30.42 s | 0.232 | 21.7 % |
| 163,840 | 40.13 s | 0.245 | 25.7 % |
| 196,608 | 50.58 s | 0.257 | 29.4 % |
| 262,144 | 74.87 s | 0.286 | 35.2 % |

`a = 0.185 ms/token` (95 % CI 0.1843–0.1860), `b = 0.383 ns/token²` (0.3792–0.3869), residual RMS 0.019 s.
**`b` is clearly non-zero** (t = 315), so prefill is measurably superlinear here — but the crossover where `bL²`
overtakes `aL` is at **483,292 tokens**, about 1.8× the largest context this hardware warms up. Two independent
fits, one ending at 192K and one at 256K, agree on `b` to 0.4 % and on the crossover to 2 %. Inside the served
range the linear term dominates everywhere. Part of `b` is the chunking scheme rather than the attention kernel:
each 8,192-token chunk attends to all preceding KV. An unchunked partial curve over 16K–64K gives `a = 0.168`,
`b = 0.130` and a crossover near 1.3 M tokens.

Decode does not scale with context at all: 57.2 ms/token at 4K, 57.2 ms at 190K, 55.5 ms at 255K. What grows is
time-to-first-token.

**What actually caps context: the KDA prefill working set.** `_kda_chunk_prefill` upcasts q/k/v/g/beta for the whole
prompt to fp32 and holds roughly nine tensors of shape `[S, heads, chunks, C, D]` live at once, plus a
`torch.cat`-ed intra-chunk attention across all chunks — all linear in prompt length. Unchunked, a 192K warmup dies on
a single **1,536 MiB** device allocation inside it (`PT_DEVMEM Allocation failed for size::1610612736`,
`hpu_kda_pytorch.py:257`), and a 160K serve reaches **91.9 GiB** in use on a mere 64K prefill. Chunked prefill bounds
it to one chunk, which is the whole reason 192K serves at 22.7 GiB of headroom and 128K sits at 68 GiB. The fix
direction is to keep the per-group loop without concatenating the intra-chunk attention, and to drop the whole-prompt
fp32 copies.

Warmup memory is a separate phenomenon and it is not cosmetic: on **every** configuration measured, including the
known-good 32K one, some ranks reach the allocator pool limit (96.9 GiB) during warmup while serving afterwards peaks
20–30 GB lower. The 192K warmup that succeeded peaked at 96,894 MiB of 96,895. Cause unknown. Judge warmup by whether
it completes; enforce a headroom floor only during serving.

## Sparse DSA: implemented, correct, and not the default

The k-pool indexer is ported behind an env flag, with a top-k KV gather replacing the original L×L mask.

| 32K, one stream | sparse (gather) | dense |
|---|---|---|
| prefill, 32,000 tokens | 10.9 s | **5.7 s** |
| decode TPOT median | 55.7 ms | 55.9 ms |
| warmup device workspace | 2.6 GiB | — |

At 64K it is 25.1 s against 11.6 s. Quality is fine — perplexity on two held-out 32K documents is within 2 % of dense
(−0.32 % and −1.42 %), bit-identical over the first 2051 positions where the dense path is exact, and needle is 10/10
at 32K — but prefill is 1.9–2.2× slower than dense because the per-query gathers are bandwidth-bound and the dense
call is still needed for the exact early rows. **Dense stays the serving default.** The earlier mask-based path also
needed 37.7 GiB of warmup workspace; the gather removed ~35 GiB of that.

The prefill curve above closes the question of whether sparse could ever pay off here. At 192K — the largest context
this hardware serves, in the configuration that serves it — the quadratic part of prefill is under 30 % of the time.
A sparse path that made that term free would still have to beat dense on the other 70 %, and it currently loses there
by roughly 2×. Sparse DSA is not worth further engineering at any context this box can serve.

## Failed or not worth it

| Attempt | Outcome |
|---|---|
| Stock vLLM on HPU | `glm5next` is a CUDA-only subpackage upstream; nothing loads |
| Habana fused MoE under EP, unpatched | ranks 1–7 route to the wrong experts, silently; fluent wrong output |
| Token-by-token KDA prefill | 30 s for 1024 tokens; chunked is 180× faster |
| `VLLM_T_COMPILE_DYNAMIC_SHAPES=true` | recipes do not reload across restarts on this model |
| Sparse DSA as the serving default | 1.9× (32K) to 2.2× (64K) slower prefill than dense, decode unchanged |
| Sparse DSA via an L×L bias mask | works, but 37.7 GiB of warmup workspace; replaced by a top-k gather |
| Prefix caching | **corrupts output** — see below. Keep it off |
| Speculative decode (`mtp`) | rejected at config time; four independent blockers, see below |
| `reasoning_effort: high` vs the default | 1,225 tokens / 240 s vs 1,210 / 224 s over a 12-turn agent task; no gain |
| 256K and 192K without chunked prefill | warmup allocation failure in the KDA prefill path |
| Decode block bucket max below `max_seqs × len/128` | lazy recompiles every step, host OOM mid-run |

## Prefix caching corrupts generation — keep it off

This is the one finding to carry to any other GDN/linear-attention hybrid on this stack, because it is **not specific
to this model and not caused by the out-of-tree port**. The implicated code is byte-identical in a pristine
`releases/v0.29.0` checkout.

The plugin auto-enables a "compact GDN" state layout for any model with `linear_attention` or `gdn_attention` layers.
Under it the worker derives each recurrent-state index from the request's **own recycled slot** and never reads the
scheduler's block table; slots are handed out from a free list on admission, returned only when a request finishes,
and never zeroed; and `has_initial_states` is set purely from `num_computed_tokens > 0`. A request admitted on a
prefix-cache hit therefore skips prefill on all 45 layers and then lets the 34 KDA layers start from whatever
recurrent state the previous occupant of that slot left behind. No error, no crash, plausible output.

It has stayed hidden because hits are rare: the hybrid cache lookup takes the minimum hit length across groups, and
the Mamba group in `align` mode commits almost nothing, so hits are usually clamped to zero. That is a masking
effect, not a guard. Four concurrent agents sharing a long prefix produce stable hits.

Any benchmark row reporting `cached_tokens > 0` on this stack is void, not merely optimistic.

## Temperature-0 output is not reproducible under concurrency

Separate defect, found as the control for the one above. Same conversations at `temperature: 0`, prefix caching off:
one request in flight matches **69/69** identical-prompt pairs, including a 1,536-token generation; four concurrent
matches **6/21**.

Sending N byte-identical prompts together returns N *different* answers (N=2 and N=4) — but the same multiset of
answers every round, with only lane assignment shuffling. So it is deterministic per batch slot, not random.
`max_tokens` of 1 and 4 are bit-identical; divergence starts around output token 26–33.

The mechanism is a batch-wide reduction: flat paged attention reduces per-block partials with
`block2batch(t, m) = matmul(m.T, t)` where `m` is a one-hot `[blocks_in_batch, batch]` matrix, so every sequence's sum
runs over every block in the batch and its offsets depend on batch position; the softmax renormalisation goes through
the same round trip. Not isolated to that single site — the FP8 MoE under expert parallelism also changes per-expert
tile shapes with batch composition.

**Practical rule: run every correctness gate at concurrency 1.**

## MTP: present in the checkpoint, not runnable

`num_nextn_predict_layers: 1`, the layer-45 weights are there, the module exists upstream, and the MTP block is built
as a DSA layer rather than KDA. It still does not run:

1. `method: "mtp"` is not honoured when passed explicitly; it is inferred from the draft model's `model_type` against
   an allowlist that has no `glm5_next` entry. Fails at config time in 45 s.
2. The MTP layer calls a **Triton-only** fused norm with no eager fallback; Triton is disabled on HPU.
3. The HPU Eagle proposer calls the draft model with a signature the MTP module does not have, and knows nothing
   about hybrid or mamba groups.
4. **No speculative KDA state.** Verifying a draft advances 34 recurrent states and a rejection must roll them back;
   compact GDN keeps one in-place state per live request with no checkpoint. Items 1–3 are plumbing; this one needs a
   design.

## Quality

- **Correctness against a CPU F32 reference is measured, not passed by a strict bar, and not claimed.** Teacher-forced
  greedy agreement is 58–64 of 64 tokens across five prompts (one prompt fully identical); prompt top-1 agreement
  91–96 %; mean KL 0.01–0.03. Divergences are single near-tie positions after which the sequences re-agree.
- The residual is consistent with **FP8 W8A8** activation quantisation in every MoE/MLP layer: per-layer MoE error
  3–8 % accumulating over 45 layers, against 2–5 % for a BF16 CPU run of the same weights. Whether that precision is
  acceptable, or the MoE activations need per-group scales or BF16, is a judgement call, not a bug.
- Needle retrieval 10/10 at 96K and 128K, both depths. Tool calling works end to end with the `glm47` parser and the
  `glm45` reasoning parser, including multi-turn chains and JSON structured output.
- Thinking switch: use [`chat_templates/chat_template.enable-thinking-switch.jinja`](chat_templates/chat_template.enable-thinking-switch.jinja)
  (`--chat-template`). Stock template ignores `enable_thinking: false` and leaks reasoning plus a stray `</think>`
  into `content`. The switch template emits `<think></think>` when thinking is off.

## Versions

- Habana / driver **1.24.1** (`vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0`)
- vLLM **0.29.1.dev** (`98dff2a8`), vllm-gaudi **releases/v0.29.0** (`2dd55f97`) + [`patches/`](patches/) 01–50
- torch 2.11.0a0, `PT_HPU_LAZY_MODE=0`. Host/firmware: see [RECIPE.md](RECIPE.md).
