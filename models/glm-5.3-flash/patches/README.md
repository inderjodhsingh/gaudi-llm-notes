# GLM-5.3-Flash HPU port

Numbered `git am` series against:

| Tree | Ref |
|---|---|
| [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) `releases/v0.29.0` | `2dd55f97eec0bcfcb1f2c52c2d9aee2c4f43f38c` |
| [vLLM](https://github.com/vllm-project/vllm) 0.29.1.dev | `98dff2a81d747d1dba01a47f939f48c3526d4206` |

`../docker/apply_patches.sh` sends a patch to the **vLLM** tree when its first `diff --git` path is under `vllm/`, otherwise to **vllm-gaudi**.

```bash
# from models/glm-5.3-flash/
docker build -f docker/Dockerfile.glm53 -t glm53-gaudi2:YYYYMMDD .
```

## Serving series (01–50)

This is the series baked into the measured image (`glm53-gaudi2:20260915`, id `43a3d4c625fd`). Patches 01–48 are the port; 49–50 add ascending warmup and per-rank memory logs. **Patch 45 is the sampling-metadata cache-key fix** (slot ids + request ids + `max_num_logprobs` + `all_greedy` + `all_random`). Without it, a reused batch slot inherits the previous request's greedy/logprobs flags.

vLLM-tree patches in this series include `01-vendor-glm5next-933876c` (vendors the CUDA `glm5next` package so it can be registered on HPU), plus 03–06, 08–10, 12, 43b.

Debug/trace hooks (23, 24, 26, 32, 46, 50) are env-gated no-ops unless you set `GLM53_TRACE_DIR` / `GLM53_DEBUG_SYMBOLIC` / `GLM53_NLL_DIR`. Do not set those on a performance run — they graph-break.

There is no patch 42.

## Optional (51–65)

See [`optional/README.md`](optional/README.md). Not in the measured image. MTP, expert-drop, and allreduce experiments.

## Compiled pieces

None. No custom TPC kernels, no extra `.so`. Stock Habana 1.24 fused MoE / GEMM / RMSNorm only.
