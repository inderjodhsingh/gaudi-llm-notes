# Model name on Intel Gaudi2

One paragraph: checkpoint id, architecture (dense / MoE / GDN hybrid), TP, thinking on/off.

Best single-stream greedy decode: **X tok/s**. Recipe: [RECIPE.md](RECIPE.md).

## Model

| | |
|---|---|
| Weights | |
| Layers | |
| Hidden | |

## Roofline / bandwidth

Record in-graph GEMM TB/s if you have it. Weight-only T=1 cap = bytes / bandwidth.

## What reached the best rate

Numbered list, highest impact first.

## Failed or not worth it

Table: attempt → outcome (one line).

## Quality probe

What you asked the model and what “pass” means.

## Versions

Habana, vLLM, vllm-gaudi pins.
