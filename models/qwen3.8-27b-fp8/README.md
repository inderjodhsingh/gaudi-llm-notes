# Qwen3.8-27B-Uncensored-FP8 on Intel Gaudi2

Decode measurements for a **dense GDN hybrid** 27B-class checkpoint
(`orcarouter/Qwen3.8-27B-Uncensored-FP8`) on **1× Gaudi2 96 GB**, **TP=1**,
vLLM + vllm-gaudi, Habana SynapseAI **1.24**, thinking **off**.

Best single-stream greedy rate measured: **~70–73 tok/s** (MTP k=1).
A CUDA H100 baseline on the same checkpoint with MTP-3 sits around **128–156 tok/s**.
That gap is mostly **HBM**, not a missing flag.

Copy-paste flags: [RECIPE.md](RECIPE.md).

## Model

| | |
|---|---|
| Weights | FP8, ~27 GB |
| Layout | 64 layers: **48 Gated DeltaNet + 16 full attention** |
| Hidden | 5120 |
| GDN | HV=48, V=K=128 |
| Extra | vision tower, MTP-1 head |

This is **not** MoE. MoE gather patches do not apply.

## Roofline

| Measurement | Result |
|---|---|
| 48 sequential `fp8_gemm_v2` in **one** compiled `hpu_backend` graph | **0.79 ms, 1.60 TB/s** |
| Same 48 GEMMs eager | 16.8 ms, 75 GB/s |
| Isolated M=1 GEMM | ~0.16 ms (launch-dominated) |
| 48 fused `matmul` 5120×5120 (bf16 microbench) | **~2.00 TB/s** |
| Gaudi2 HBM peak (datasheet) | ~2.45 TB/s |

Weight-only T=1 at 1.60 TB/s: 27 GB / 1.60 TB/s ≈ **17 ms ≈ 59 tok/s**.
Live T=1 is ~21 ms ≈ **48–50 tok/s**.

**128 tok/s at T=1 would need ~3.5 TB/s.** That is above Gaudi2 peak.
H100’s 128–156 is consistent with HBM3 plus MTP-3 (more than one token per weight pass).

**Implication:** beating ~59 tok/s on one Gaudi2 requires **speculative decode** whose verify pass is close to **1× T=1**, not two full T=1 forwards.

## What reached the top rate

Stack, in order of impact:

1. **Compile mode** (`PT_HPU_LAZY_MODE=0`). Lazy mode + full HPU graphs **OOM**’d this 27B.
2. **Channel FP8 GEMM** (`VLLM_HPU_FORCE_CHANNEL_FP8=true`) → `torch.ops.hpu.fp8_gemm_v2`.
3. **Stock TPC** already in Habana 1.24 `libtpc_kernels.so`:
   - `gdn_read_decayed_state` (bind via PT2 custom op if `torch.ops.hpu.*` is missing)
   - `causal_conv1d_update` (vllm-gaudi native path)
4. **MTP k=1** with `COMMIT_ALL` (write last GDN/conv state in-graph; no stash clones).
5. **Do not inflate spec batch** to `bs × (1+k)`. Keep one row per request, flatten to virtual-batch T=1 for attention; GDN sees **B=1, T=2** and uses a compiled **unroll-2**.
6. **`torch.compile` the MTP drafter** (regional). Uncompiled draft was ~14.5 ms/step; compiled ~5.1–5.7 ms.
7. **Greedy on-device pack** (out-of-tree, see recipe):
   - `argmax` on target + bonus logits on HPU
   - run the drafter **before** any `.cpu()` of the 248k-row logits
   - skip a second `Sampler` pass and skip `_prepare_sampling` on the spec path
   - parse a `[B, 2]` int32 pack after the draft

Profiler (one warm decode, CPU table; HPU sort key not exposed):

- 67 recipe launches (48 GDN compiled regions ~193 µs CPU each, 16 attn ~226 µs)
- no single **device** op ≥ 3 ms besides the fused GEMM body
- RMSNorm is already a fused HPU op

So the remaining 73 → ~83 tok/s (k=1 arithmetic cap = 2 tokens / 24.2 ms target graph) is **launch/host tail**, not another kernel.

## Failed or not worth it

| Attempt | Outcome |
|---|---|
| `VLLM_GDN_COMPUTE_FP32=0` + HPU graphs | Garbage (`!!!!`) |
| ngram spec | `IndexError` / quality collapse |
| KEEP_BT: keep spec as `[B, T]` for `flat_pa` q_len>1 | Garbage text / early EOS |
| `seq_len>1` decode `flat_pa` without KEEP_BT buffers | Dynamo `matmul [2,2] @ [1, s]` |
| Custom fused `gdn_one_step` TPC | Numerically OK (~2e-4 vs eager) but **slower than MME** 128×128 in the 27B graph (~46 tok/s) |
| `tpc-clang` **without** `-march=gaudi2` | Defaults to **Gaudi1 ISA** and can **reset a Gaudi2** |
| `GC_KERNEL_PATH=a:b` | Habana 1.24 `isfile()`s the whole string — **one file only**; glue must `dlopen` stock `libtpc_kernels.so` |
| Two `TORCH_LIBRARY(custom_op)` .so files | Clash; merge ops |
| MTP k=2 / k=3 | ~3.6 / ~2.7 tok/s, garbage or low accept |
| Concurrent B=2 | Dynamo engine death |
| `PT_HPU_LAZY_MODE=1` | `PT_DEVMEM` OOM |
| `regional_compilation=false` (full module compile) | `HpuModelAdapter` has no `config` |
| Defer `parse_output` until after draft (CPU lists / mixed devices) | 500 (`index` on CPU tensor) or no gain |
| Keep rejection sampler on HPU (drop the four `.cpu()`s in vllm-gaudi’s PyTorch rejection path) | **Slower** (~41 tok/s). CPU compare is faster for this tiny op. |
| Skip “commit last state on reject” (force shorter commit) | Quality fail (early EOS) |
| Extra MTP lookahead token (d2) while k=1 verify | Quality fail (`1 2 33 4455`) |
| Env sweep: compact GDN, util 0.80, GDN bf16, channel-FP8 already on | Noise around 56–59 compact e2e; not a new ceiling |
| `get_forward_context()` elision / shared sidx buffers | Dynamo shape bugs or **garbage** (hybrid `cache_group_idx`) |
| `VLLM_CONFIG_HIDDEN_LAYERS` | **Lazy-only**; no-op in compile mode |

## Shapes that matter

MTP k=1 verify (KEEP_BT off):

- `token_ids` **(2, 1)** — virtual batch, attn `seq_len=1`
- GDN **B=1, T=2**, compiled unroll-2 (not padded to T=4)

Decode bucket lookup in vllm-gaudi is `(batch, query=1, blocks)`. Spec buckets inflate **batch** to `bs×(1+k)` unless you pin `padded_batch = num_decodes` for T>1.

## Quality

Greedy “count from 1, one integer per line” is the probe. The 70–73 recipe usually prints **1…30**. MTP sometimes **skips 6** (accept length still ~1.97–2.00). Treat skip-6 as a known MTP wart, not as “it works.”

## Versions

- Habana / driver **1.24.x**
- vLLM **0.28.1rc1** (or current vllm-gaudi pin)
- `vllm-gaudi` with HPU rejection sampler (PyTorch, not Triton)

Re-measure after a Habana bump: `torch.ops.hpu.gdn_read_decayed_state` may exist natively later.
