# Changelog

## v1.0.1 — 2026-06-29 (community-feedback fixes)
- **Fixed: `krea2` UI preset missing from the dropdown.** The preset now registers **resiliently and independently** of the architecture registration — the dropdown entry is added first, with the fewest possible dependencies, so it appears even on Forge Neo versions where a later step hit a version-specific snag.
- **Fixed: renamed VAE / text-encoder / checkpoint files not detected** ("extension asks for download only with proper name"). Files are now identified by their **safetensors keys (content)**, not just the filename — rename them however you like and the extension still finds and auto-loads them.
- **Fixed: fp8 text encoder log spam.** Stripped the unused Qwen3-VL vision tower from the **fp8** TE (removes the `Unexpected: model.visual.*` dump and saves a little VRAM).

**To upgrade:** `git pull` in `extensions/sd-forge-krea2` (or re-download the ZIP), then **restart Forge**.

## v1.0.0 — 2026-06-29
- Initial release: native Krea 2 in Forge Neo — full + piecewise loading, fp8 support, one-click model downloader tab, and the `krea2` UI preset.
