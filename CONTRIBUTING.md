# Contributing

## What belongs here

- Hardware class (Gaudi2, HBM size), software versions (Habana / vLLM / vllm-gaudi).
- Model id, architecture notes (dense vs MoE, GDN/hybrid, MTP).
- Measured single-stream decode tok/s, quality check used, and the exact flags.
- Failed experiments with a one-line cause (so the next person does not repeat them).

## What does not belong here

- Hostnames, IP addresses, inventory, usernames, tokens, API keys.
- Internal dashboards, ports, or process ids.
- Personal or customer data.

## New model note

```text
templates/model-note/   →   models/<huggingface-or-short-slug>/
  README.md               paper: setup, roofline, tried/failed, result
  RECIPE.md               copy-paste env + `vllm serve` line
```

Use the same quality probe across notes when possible (e.g. greedy “count 1…30”).
Mark any **out-of-tree patch** clearly so a stock vLLM checkout is not assumed.

## Commit style

- Imperative subject, ≤72 characters: `Add Qwen3.8-27B-FP8 Gaudi2 decode note`.
- One model per commit when you can.
