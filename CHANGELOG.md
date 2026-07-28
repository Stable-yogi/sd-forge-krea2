# Changelog

## v1.3.2 — 2026-07-29
- **New: one-click Wan 2.1 VAE download** — the Krea 2 tab has a new **"⬇️ Wan 2.1 VAE (alt)"** button (from Comfy-Org/Wan_2.1_ComfyUI_repackaged). The Wan 2.1 VAE **also works with Krea 2 and in some cases gives better results** — download it and pick it in the VAE dropdown instead of the Qwen VAE to try. Shown as an optional row in the status table; setup readiness still only requires the Qwen VAE.

## v1.3.1 — 2026-07-29 (LoRA × quantized-checkpoint fixes — GGUF crash, INT8 "no effect", fp8-scaled noise)
- **Fixed: GGUF checkpoints crashed when used with a LoRA** — `RuntimeError: Creating a Parameter from an instance of type ParameterGGUF requires that detach() returns an instance of the same type...`. Two causes, both fixed:
  - Checkpoints loaded via the **API or a preset** missed Forge's "GGUF requires fp16 LoRA" override (it only fires in the UI path), so the LoRA was **baked into the GGUF weights**. The extension now forces **on-the-fly LoRA** at the true decision point (`add_patches` on a model that actually has GGUF weights) — immune to load-path and options-state differences.
  - The LoRA **unpatch/restore** path re-wrapped a backed-up quantized weight with `torch.nn.Parameter(...)`, tripping torch ≥ 2.9's strict subclass check during sampling's memory juggling — the extension now restores already-Parameter values as-is (subclass preserved).
- **Fixed: LoRAs silently had NO EFFECT on INT8-ConvRot checkpoints** ("LoRA loads in the terminal but the character never appears"). Baking merges tiny fp deltas into int8 storage where they round to zero. The extension now (a) flags int8 builds so Forge wraps them in its **INT8ModelPatcher** (the ConvRot-aware dynamic-LoRA engine) and (b) forces those LoRAs **on-the-fly**, applying them post-matmul at full strength. Verified: fixed-seed INT8+LoRA render now matches the bf16 ground truth's character near pixel-perfect.
- **Fixed: fp8-SCALED checkpoints (e.g. Comfy-Org `krea2_*_fp8_scaled`) broke** — first misrouted to the int8 ops (`mat1 and mat2 must have the same dtype, but got BFloat16 and Float`), and on plain fp8 ops they render pure noise because their `.weight_scale` tensors are real scales. They now route through **Forge's mixed-precision ops** (the same path as NVFP4), which applies the scales correctly. INT8 detection additionally requires genuinely int8-stored weights, so the two formats can never cross-route again.
- All fixes apply to **both Forge generations** and are feature-detected (no core edits). Verified end-to-end on the old-generation Forge (the RunPod/desktop bundle) with a character LoRA across GGUF / INT8-ConvRot / fp8-scaled / plain fp8 / bf16, including LoRA strength changes mid-session.

## v1.3.0 — 2026-07-27 (new-Forge compatibility — fixes fresh-install failures)
- **Fixed: the extension broke Krea 2 on NEW Forge Neo builds** (the ones that ship Krea 2 natively — `backend/nn/krea.py` + a native `krea` preset). On those builds the extension's registration crashed mid-way (`AttributeError: 'tuple' object has no attribute 'append'` — `possible_models` is immutable now), leaving a **half-patched loader**: our detection/build patches hijacked the native Krea 2 pipeline while the TE/VAE auto-attach never registered. Net result on a fresh install: **`AssertionError: You do not have Qwen3 state dict!`** when loading the Turbo or Muse DiT (exactly as reported on Civitai and Discord), plus a broken `krea2` entry alongside the native `krea` preset.
- **The extension is now native-aware.** On new Forge it steps aside — Forge's built-in Krea 2 arch and its native **`krea`** preset do the loading — while the extension keeps providing what Forge doesn't have: the **one-click model downloader tab** (base + Muse), **auto TE + VAE attach for bare DiTs** (this alone fixes the Qwen3 assert: pick the checkpoint and go, no manual module selection), and **Detail Boost**, now ported to run inside the native engine too.
- **Old Forge Neo builds are unchanged** — the extension still registers the `krea2` arch + preset exactly as before (that path is also hardened against the immutable-tuple case).
- The Setup tab's instructions now name the right preset for your Forge (`krea` on new builds, `krea2` on old ones).

## v1.2.4 — 2026-07-23
- **Fixed: NVFP4 (and other `comfy_quant` mixed-precision) checkpoints crashed on load-and-generate** with `RuntimeError: self and mat2 must have the same dtype, but got Float and BFloat16` (raised from `krea2/dit.py` at `img = self.first(img)`). The extension picked its quantization ops by testing for `.weight_scale` keys — but NVFP4 files carry those *as well as* their own `.weight_scale_2` per-tensor scales, so every NVFP4 checkpoint was misrouted to the INT8-ConvRot ops (the log gave it away: `[krea2] INT8-ConvRot -> int8 ops` on an NVFP4 file). The layers NVFP4 deliberately keeps at high precision (`first` / `last` / `txtfusion` / projectors, stored bf16+fp32) then reached a raw `F.linear` with mismatched dtypes. NVFP4 is now detected first via its `.weight_scale_2` marker and handed to Forge's own mixed-precision ops (`detect_quantization` → `mixed_precision_ops`) — the same path Forge's native loader uses — built on-device without a blanket dtype cast (which would corrupt the packed 4-bit weights) and with manual casting enabled for the high-precision leftovers.
- INT8-ConvRot, fp8-scaled and GGUF routing is unchanged (they carry no `.weight_scale_2`); verified by re-running an INT8-ConvRot checkpoint after the change.
- Requires a GPU with NVFP4 compute (Blackwell / compute capability 10.0+) for the fast path; on older GPUs the unsupported formats are dequantized instead and the reason is logged.

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
