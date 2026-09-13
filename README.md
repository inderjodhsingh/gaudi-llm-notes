# Gaudi LLM notes

Public decode notes for large language models on **Intel Gaudi2** (HPU) with **vLLM** and **vllm-gaudi**.

Each model gets a folder under [`models/`](models/):

| Model | Note | Best measured decode (thinking off, TP=1, 1× Gaudi2 96 GB) |
|---|---|---|
| [Qwen3.8-27B-Uncensored-FP8](models/qwen3.8-27b-fp8/) | [paper](models/qwen3.8-27b-fp8/README.md) · [recipe](models/qwen3.8-27b-fp8/RECIPE.md) | **~70–73 tok/s** (MTP k=1) |
| [Qwen3.8-Flash-Next-Uncensored-FP8](models/qwen3.8-flash-next/) | [paper](models/qwen3.8-flash-next/README.md) · [recipe](models/qwen3.8-flash-next/RECIPE.md) | **~39 tok/s** (TP=4, 4× Gaudi2, MoE/GDN hybrid, out-of-tree port; 256K context verified) |
| [Lightricks/LTX-2.5](models/ltx-2-5/) | [paper](models/ltx-2-5/README.md) · [recipe](models/ltx-2-5/RECIPE.md) | **~1.37 s/step** (distilled 8-step clip, not tok/s) |

This is engineering documentation, not a vendor benchmark. Numbers are single-stream greedy (temperature 0) unless stated otherwise.

## Add another model

1. Copy [`templates/model-note/`](templates/model-note/) to `models/<slug>/`.
2. Fill **README.md** (what you tried) and **RECIPE.md** (flags that hit the best rate).
3. Add a row to the table above.
4. Open a pull request, or push to `main`.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Cite with [CITATION.cff](CITATION.cff).
