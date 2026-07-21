# Changelog

## v1.2.3 — 2026-07-22
- **Fixed: the `krea2` preset now FORCES its Turbo defaults (8 steps / CFG 1 / Euler / Simple) on every selection.** Earlier versions seeded these with `setdefault`, which never overwrites a value already saved in `config.json`. Anyone who installed before v1.2.1 had the old RAW-style **28 steps / CFG 4.5** written into their config, and updating the extension could not dislodge it — so selecting the preset kept silently applying 28 / 4.5. On the 8-step distilled Turbo/Muse checkpoints that produces exactly the reported symptoms: hatching/hachures in dark areas, over-sharpening, and apparent "hallucinations above ~20 steps" (all classic too-high-CFG artifacts). The preset now sets the correct sampling values every time it is selected; your checkpoint / VAE / TE choices stay per-user (still `setdefault`, never overwritten).

## v1.2.2 — 2026-07-16
- **Preset default resolution is now 896×1152** (portrait, ~1 MP) for the `krea2` preset, replacing 1024×1536. Sampler/steps/CFG unchanged (Euler / Simple / 8 / 1).

## v1.2.1 — 2026-07-15
- **Preset fix: the `krea2` UI preset now defaults to 8 steps / CFG 1** (Turbo + Muse) instead of RAW's 28 / 4.5. Since Turbo and Muse are the recommended 8-step models, the preset matches them out of the box. RAW users bump steps→28 and CFG→4.5.
- **Wan 2.1 VAE support** — the auto-loader now recognizes and attaches the **Wan 2.1 VAE** (which Muse 1.5+ pairs with), alongside the Qwen-Image VAE.
- **Default resolution is now 1024×1536** (portrait) for the krea2 preset.
- **Preset-loading hardened** — the preset's sampler/steps/CFG and module options are ensured right before Forge reads them, with a diagnostic log line on selection.

## v1.2.0 — 2026-07-15 (Muse by Stable Yogi + download hardening)
- **New: one-click "Muse by Stable Yogi" download** — the Krea 2 tab now features **Muse** (Krea 2 v1.5 Turbo, photoreal), fetched from Civitai as **Q8_0 GGUF (recommended)**, **Q4_0 GGUF (low-VRAM)**, or **fp8**. Muse is a bare DiT, so the button also pulls the base Krea 2 TE + VAE it needs. No Civitai token required.
- **GGUF bare DiTs now auto-attach their TE + VAE** — the "seamless pieces" loader recognizes `.gguf` (not just `.safetensors`), so a GGUF Muse checkpoint loads without manually picking modules.
- **Download hardening** — a magic-byte check rejects an HTML error page saved under a model name; a complete-but-unpromoted `.part` is now recovered instead of dead-ending on HTTP 416; and per-variant DiT detection is filename-precise, so installing a Muse build no longer mislabels or silently blocks the vanilla Turbo/RAW rows.

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
