# Recipe — distilled LTX-2.5 clip on 1× Gaudi2 (~1.37 s/step)

Checkpoint: `Lightricks/LTX-2.5` Comfy pack (not the Diffusers repo).  
Hardware: 1× Intel Gaudi2 96 GB, TP=1.  
Shipped clip: distilled 22B DiT, 8 steps, CFG=1, 512×320×9, conv VAE, **HPU eager text encoder**.

This is **not** `vllm serve`. Load with `ltx-core` `SingleGPUModelBuilder` on `hpu`.

## Environment

```bash
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_H2D_PIPELINE=0
export HABANA_VISIBLE_DEVICES=all
export HABANA_VISIBLE_MODULES=<module_id>   # module_id, not hl-smi index
```

No HPUGraph wrap. No warmup. `safetensors.safe_open` has no `hpu` device — open on CPU, then `.to("hpu")`.

## Weights (BF16, from the Comfy pack)

| Piece | File |
|---|---|
| Distilled DiT | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` |
| Gemma4 TE + LTX projections | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` |
| Conv VAE | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` |
| Audio VAE + vocoder | `vae/ltx-2.5-audio-vae-bf16.safetensors` |
| Spatial ×2 upscaler (optional) | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` |

Do **not** load Comfy int8-convrot or NVFP4 on HPU.

## Text encoder (the NaN fix)

1. Rebuild Gemma4 rotary from `per_layer_config.head_dim` (256 sliding / 512 full). LTX `_populate_rotary_v5` otherwise builds a 240-wide full-attention RoPE and crashes against Q of 512. Skipping the overwrite and keeping transformers-native rotary is equivalent.
2. **`attn_implementation="eager"`** on HPU. Pad-1024 is then finite (~516–846 ms, ~24 GiB). Default SDPA is finite only for **pad ≤ 128**; pad ≥ 256 NaNs at layer 0 sliding.
3. Connector requires `seq_len % num_learnable_registers == 0` (32 works; raw unpadded 17 hits the assert). Video context is `[1, S, 4096]`, audio `[1, S, 2048]`.
4. Fallback if you cannot set eager: CPU fp32 encode (~13.5 s) is finite at pad-1024 and produces a usable golden embedding.

Measured HPU eager TE + distilled DiT + conv VAE: **1366 ms/step**, 40 GiB, `pix_std ≈ 0.41`, finite 512×320×9 mp4.

## DiT + VAE

- Distilled transformer, 8 steps, CFG=1, math SDPA (`AttentionFunction.SDPA_MATH`).
- Conv VAE decode ~80 ms at 512×320×9.
- Place TE on one module and DiT+VAE on another if you want headroom. 66 GiB of weights on one card leaves no graph room.

## Audio

- Audio VAE decoder: bf16 HPU, ~28 ms, finite mel.
- Vocoder: **fp32 on HPU** (~215 ms). bf16 `conv1d` is not finite.

## Do not

| Idea | Why |
|---|---|
| `vllm serve Lightricks/LTX-2.5` | Not a CausalLM |
| Stock `google/gemma-4-12B-it` as this TE | LTX version-checks its own bundle |
| HPU SDPA TE at pad ≥ 256 | NaN from layer 0 sliding (bf16, fp32, bool mask) |
| fp32 HPU TE as a NaN fix | Still NaN, ~72 GiB |
| Count 1…30 | Wrong probe for a DiT |
| NVFP4 / Comfy int8-convrot | Not this stack |
| Quote lazy ~650 ms/step as the default | First two steps 276 s + 245 s; no HPUGraph; eager is the shipped path |
| Dummy / NaN / `pix_std ≤ 0.1` clip as a win | Rejected |

## Versions this recipe was measured on

Habana **1.24.1**, PyTorch 2.11, public `ltx-core`. Re-time after a driver bump.
