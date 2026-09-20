# Qwen4Exp HPU port (Qwen3.8-Flash-Next)

Applies to:

| Tree | Ref |
|---|---|
| [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) `releases/v0.29.0` | `2dd55f97eec0bcfcb1f2c52c2d9aee2c4f43f38c` |
| [vLLM](https://github.com/vllm-project/vllm) v0.29.0 | `98dff2a81d747d1dba01a47f939f48c3526d4206` (**stock — no patch**) |

vLLM 0.29 already vendors `vllm/models/qwen4_exp/` (CUDA/ROCm). The HPU port imports the pure-torch hyper-connection helper from that package and registers its own `vllm_gaudi.models.qwen4_exp` implementation. There is no vLLM-tree diff.

```bash
git clone https://github.com/vllm-project/vllm-gaudi.git && cd vllm-gaudi
git checkout 2dd55f97eec0bcfcb1f2c52c2d9aee2c4f43f38c
git am /path/to/0001-qwen4-exp-hpu-port.patch
git am /path/to/0002-ple-prefill-state-length.patch
```

`0002` is required. Without it, thinking-off identifier copy and short-list prompts fail below the 64-token prompt bucket: PLE decode n-grams hash against pad ids. Rebuild the image or `git am` 0002 on a tree that already has 0001.

Or build [`../docker/Dockerfile`](../docker/Dockerfile).

## What the patch touches

New: `qwen4_exp.py` (and a `QWEN4_LEGACY_MODEL=1` fallback), PLE n-gram hasher, gathered-expert FP8 MoE, 256-candidate top-k/top-p.

Edited: model registry, Qwen3.5 GDN helper, `hpu_fp8.py`, platform allowlists, hybrid runner (`mamba_like_arch`, SHORT_CONV / `.ple` cache groups), sampling-metadata cache.

## Compiled pieces

None. No custom TPC kernels, no extra `.so`. Device work uses stock Habana 1.24 `libtpc_kernels.so` (`fp8_gemm_v2`, GDN, conv1d, fused MoE).
