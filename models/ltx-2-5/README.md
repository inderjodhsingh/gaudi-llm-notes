# LTX-2.5 on Intel Gaudi2

Bring-up of the **Lightricks/LTX-2.5** audio-video DiT pack (distilled 22B transformer + Gemma4Unified 12B text encoder + conv VAE) on **1× Gaudi2 96 GB**, **TP=1**. This is a **clip**, not a CausalLM. `vllm serve` of the HuggingFace id does not apply.

Shipped clip: **all-HPU**, real text encoder, distilled 8-step CFG=1, 512×320×9, conv VAE. Finite, non-uniform (`pix_std ≈ 0.41`). Mean step **1366 ms**. Recipe: [RECIPE.md](RECIPE.md).

## Model

| | |
|---|---|
| Weights | Distilled DiT BF16 39.1 GiB + Gemma4 TE BF16 24.5 GiB + conv VAE 1.35 GiB |
| DiT | `LTX2VideoTransformer3DModel`, 48 layers, hidden 4096, 32 heads × 128, audio twin 32×64, split RoPE, `rope_double_precision` |
| TE | `Gemma4UnifiedForConditionalGeneration` LTX bundle (`gemma4-12b-ltx-v1`), hidden 3840, 48 layers, sliding 1024 / full every 6th, `head_dim` 256 / `global_head_dim` 512 |
| VAE | Conv video decoder, scale (8, 32, 32) |
| Extra | Audio VAE + vocoder, spatial ×2 latent upscaler |

No vllm-gaudi cousin for the DiT. The TE is **not** stock `google/gemma-4-12B-it` and is **not** the PLE Gemma4 family already in vllm-gaudi.

## Quality probe

Distilled 8-step, CFG=1, short clip (512×320, 9 frames). Pass = finite pixels, `pix_std > 0.1`, non-uniform imageio mp4, `used_real_te=true`. Count-1…30 is the wrong gate.

| Clip | TE | Shape | mean step | pix_std | bytes |
|---|---|---|---|---|---|
| dummy-TE (rejected) | noise context | 512×320×9 | 1024 ms | 0.41 | 434 kB |
| CPU-fp32 TE | `emb.golden` | 512×320×9 | 1498 ms | 0.42 | 444 kB |
| **HPU eager TE (shipped)** | `emb.native` | 512×320×9 | **1366 ms** | 0.41 | 421 kB |
| HPU eager TE | `emb.native` | 768×512×9 | 1294 ms | 0.41 | 1.0 MB |
| HPU eager TE | `emb.native` | 512×320×17 | 1050 ms | 0.41 | 773 kB |

## What actually broke (TE NaN)

First HPU encode of the LTX Gemma4 bundle produced **all-NaN hidden states** from **layer 0 sliding attention**, with a finite embedding (`max_abs ≈ 43`). The 512-vs-240 RoPE crash was a separate, earlier bug: LTX `_populate_rotary_v5` passed `global_head_dim` into a per-layer config that only has `head_dim`, so full-attention RoPE was 240-wide against Q of 512. Rebuilding rotary from `per_layer_config.head_dim` (or skipping the LTX overwrite and keeping transformers-native rotary) **fixes the crash, not the NaN**.

Bisection:

| Attempt | Result |
|---|---|
| Native rotary, skip LTX overwrite | inv_freq 128/256 correct; still NaN layer 0 sliding |
| CPU fp32, same prompt, pad-1024 | **finite**, connector video+audio finite |
| CPU bf16, pad-1024 | **finite** — not a bf16 issue |
| fp32 on HPU (~72 GiB) | still NaN layer 0 sliding |
| Unpadded seq=17 | **finite** (`max_abs = 3040`) |
| Default attn pad-32 / 64 / 128 | **finite** |
| Default attn pad-256 / 512 / 1024 | **NaN layer 0 sliding** |
| `attn_implementation="eager"`, pad-1024 | **finite**; hooks clean through 48 layers + final norm |
| True `sdpa` on HPU, pad-1024 | **NaN layer 0 sliding** |
| Float-to-bool SDPA mask | still NaN |
| Connector on random finite `[1,1024,3840]×49` | **finite** — connector is not the source |
| Unpadded seq=17 into connector | assert `seq_len % num_learnable_registers == 0` |

**Cause:** Habana SDPA + a long padded sequence (threshold between 128 and 256) NaNs sliding-window attention. **Fix:** `attn_implementation="eager"` at the pack's native pad-1024, *or* pad ≤ 128 with default attn. Connector still wants `seq_len` a multiple of its register count (32 works).

## Throughput (eager, no HPUGraph)

`PT_HPU_LAZY_MODE=0`. Distilled DiT 8-step, conv VAE.

| Setup | mean step | HBM | notes |
|---|---|---|---|
| Dummy-TE 512×320×9 (320 tok) | 1676 ms | 40 GiB | first DiT smoke |
| Dummy-TE 768×512×9 (768 tok) | 1100 ms | 43 GiB | |
| Dummy-TE 512×320×17 (480 tok) | 1171 ms | 41 GiB | |
| Real TE, 512×320×9 | **1366 ms** | 40 GiB | shipped |
| Lazy, no HPUGraph, 512×320×9 | ~650 ms after compile | 40 GiB | first two steps 276 s + 245 s; **not the default** |

VAE decode ~80–125 ms. Spatial ×2 latent upscaler 48 ms (ran clean on a card that reported 8 uncorrected events). Audio decoder 28 ms; vocoder must be **fp32** (215 ms) — bf16 `conv1d` failed.

Full pack on one 96 GiB card is ~66 GiB of weights. Practical split: TE on one module, DiT+VAE on another.

## Failed or not worth it

| Attempt | Outcome |
|---|---|
| `vllm serve Lightricks/LTX-2.5` | Not a CausalLM |
| Serving stock `google/gemma-4-12B-it` as the campaign product | Wrong weights; LTX ships its own TE bundle |
| Count 1…30 as the video gate | Wrong probe |
| NVFP4 / Comfy int8-convrot | Not on this stack |
| HPU SDPA TE at pad ≥ 256 | NaN from layer 0 sliding |
| fp32 HPU TE as a NaN fix | Still NaN, 72 GiB |
| Bool-cast SDPA mask | Still NaN |
| Lazy mode as the shipped default | Faster after a 9-minute compile; first-step cost is not a clip recipe |
| HPUGraph capture | Not used; lazy graph wrap is a later class |

## Versions

Habana **1.24.1**, PyTorch 2.11, ltx-core from the public LTX-2 tree. DiT path is eager HPU PyTorch, not vLLM.
