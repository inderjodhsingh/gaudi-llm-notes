# Optional GLM patches (51–65)

Not part of the serving image. Apply on top of 01–50 only if you are working on that item.

| Patch | What |
|---|---|
| 51 | `VLLM_HPU_ALLREDUCE_MODE=gather_sum` (compile-safe rewrite). Not a determinism fix. |
| 52 | Proof logs for the MoE allreduce branch. |
| 53–57 | MTP plumbing (speculative config, eager `fused_eh_norm`, loader skip, proposer `image_token_index`, spec warmup batches). Still blocked on the HPU Eagle proposer assuming a flat full-attention draft, and on speculative KDA state rollback. |
| 58 | Honour `min_p` on the selective sampler (stock path silently ignored it). |
| 59–60 | KDA decode batched matmul + state row view. Small decode win, not required. |
| 61–65 | Expert histogram + physical expert drop (`GLM53_EXPERT_DROP`). Measured: no unused experts, drop ≤0.2 % costs PPL and **slows** decode. Default stays dense 288. |

Do not enable 61–65 on a replica of the published recipe.
