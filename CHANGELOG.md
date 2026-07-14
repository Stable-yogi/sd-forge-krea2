# Changelog

## v1.1.2 — 2026-07-03 (download fixes + quantized DiT support)
- **Fixed: the downloader saved checkpoints to `models/Stablediffusion` (no dash)** — a folder Forge never scans. It now downloads into Forge's real **`models/Stable-diffusion`** (or your `--ckpt-dirs`), so the model actually shows up.
- **Fixed: two download-corruption edge cases** — (a) if a server ignores the HTTP Range header the resume no longer appends a full body onto a partial file; (b) the `.part` is verified complete against Content-Length before being finalized, so a dropped connection can't leave a truncated file that looks done (re-run to resume).
- **New: quantized DiT support** — **INT8-ConvRot** and **GGUF** Krea 2 checkpoints now load with the matching ops (they were loading with plain ops → black/NaN images or load errors).
- Added the **Detail Boost Apache-2.0 attribution** to the README credits.

## v1.1.1 — 2026-07-03
- **UI: the Detail Boost "Enable" now lives in the accordion header** (Forge `InputAccordion`),
  matching LoRA Block Weight — flip it and read its on/off status from the top of the panel
  without expanding. Falls back to a plain accordion on older Forge builds. No behaviour change.

## v1.1.0 — 2026-06-29
- **New: Detail Boost** (opt-in accordion in txt2img, off by default). Rebalances Krea 2's
  12-layer Qwen3-VL conditioning toward the deep fine-detail taps (identity/texture) with
  RMS-safe renormalisation, so the overall conditioning magnitude stays constant — sharper
  results without oversaturation. Presets: balanced / detail / subtle + strength slider.
  Technique credit: huwhitememes/comfyui-krea2-conditioning (Apache-2.0), fork of
  nova452/ComfyUI-ConditioningKrea2Rebalance.
- The full **Enhancement Suite** (Prompt-Adherence engine + custom per-layer control) is
  available free at stableyogi.com.

## v1.0.1 — 2026-06-29 (community-feedback fixes)
- **Fixed: `krea2` UI preset missing from the dropdown.** The preset now registers **resiliently and independently** of the architecture registration — the dropdown entry is added first, with the fewest possible dependencies, so it appears even on Forge Neo versions where a later step hit a version-specific snag.
- **Fixed: renamed VAE / text-encoder / checkpoint files not detected** ("extension asks for download only with proper name"). Files are now identified by their **safetensors keys (content)**, not just the filename — rename them however you like and the extension still finds and auto-loads them.
- **Fixed: fp8 text encoder log spam.** Stripped the unused Qwen3-VL vision tower from the **fp8** TE (removes the `Unexpected: model.visual.*` dump and saves a little VRAM).

**To upgrade:** `git pull` in `extensions/sd-forge-krea2` (or re-download the ZIP), then **restart Forge**.

## v1.0.0 — 2026-06-29
- Initial release: native Krea 2 in Forge Neo — full + piecewise loading, fp8 support, one-click model downloader tab, and the `krea2` UI preset.
