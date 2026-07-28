# sd-forge-krea2 — Krea 2 for Forge

Run **Krea 2** (Krea AI's 12B single-stream DiT) in **Stable Diffusion WebUI Forge (Neo)**.
The first open-source Krea 2 integration for Forge. Built by **[stableyogi.com](https://stableyogi.com)**.

Krea 2 = a 12B DiT image model (Qwen3-VL text encoder + Qwen-Image VAE, flow-matching).
Two variants: **RAW** (base, best quality) and **Turbo** (8-step distilled, fast).

<p align="center">
  <img src="assets/lake.png" width="66%" alt="Krea 2 — mountain lake at sunrise"/>
</p>
<p align="center">
  <img src="assets/apple.png" width="24%" alt="apple"/>
  <img src="assets/puppy.png" width="24%" alt="puppy"/>
  <img src="assets/coffee.png" width="24%" alt="latte art"/>
  <img src="assets/rose.png" width="24%" alt="rose"/>
</p>
<p align="center"><sub>All generated in Forge with Krea 2 RAW · the <code>krea2</code> preset · Euler / Simple. No cherry-picking.</sub></p>

---

## ✨ Features
- **⭐ One-click [Muse by Stable Yogi](https://civitai.com/models/2741166)** (Krea 2 v1.5 Turbo, photoreal) — GGUF or fp8, straight from the "Krea 2" tab.
- Native Krea 2 architecture in Forge — no ComfyUI needed.
- **Both loading streams supported:**
  - **Full model** — one combined checkpoint with everything baked in.
  - **Pieces** — a bare DiT checkpoint; the TE + VAE are **auto-loaded** (or pick them yourself).
- **fp8 supported** for the DiT and the text encoder (half the size/VRAM).
- **One-click model downloader** — the **"Krea 2"** tab fetches every file into the right folder.
- **`krea2` UI preset** — auto-sets sampler/steps/CFG **and** auto-selects the TE + VAE.
- **Detail Boost** (opt-in) — rebalances the 12-layer text conditioning toward the deep,
  fine-detail taps with RMS-safe renormalisation: sharper identity/texture, no oversaturation.
  *Want more? The full **Enhancement Suite** (advanced Prompt-Adherence engine + custom
  per-layer control) is **free** →
  [Stable-yogi/sd-forge-krea2-enhancements](https://github.com/Stable-yogi/sd-forge-krea2-enhancements)
  (by [stableyogi.com](https://stableyogi.com)).*

## ✅ Requirements
- **Forge Neo** (Haoming02/sd-webui-forge-classic, `neo` branch) — **both generations supported:**
  - **Newer builds with native Krea 2** (they have a built-in `krea` preset): the extension auto-detects this and uses Forge's native engine — it adds the downloader tab, auto TE/VAE attach, and Detail Boost on top. Use UI Preset **`krea`**.
  - **Older builds (e.g. `neo-2.23`) without native Krea 2**: the extension registers the whole architecture itself. Use UI Preset **`krea2`**.
- An NVIDIA GPU with enough VRAM (fp8 set ≈ 16–20 GB; bf16 set ≈ 24 GB+).
- No extra Python packages — uses Forge's existing dependencies.

## 📦 Installation
1. Copy the `sd-forge-krea2` folder into your Forge `extensions/` directory.
2. Restart Forge.
3. Open the new **"Krea 2"** tab.

## ⬇️ Getting the models (easy way)
In the **Krea 2** tab:
1. **⭐ Download Muse** (recommended) — [Muse by Stable Yogi](https://civitai.com/models/2741166), a photoreal Krea 2 v1.5 Turbo checkpoint. Pick a build (**Q8 GGUF** / **Q4 GGUF** low-VRAM / **fp8**); the button also grabs the base **TE + VAE** it needs. No Civitai token required.
2. Or grab the **vanilla base Krea 2** (Turbo/RAW DiT + TE + VAE) with the buttons below.
3. Files land in Forge's standard folders automatically — **no command-line flags needed.**

### Manual download (alternative)
All files are in the public HF repo **[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)**:

| File | Put it in |
|---|---|
| `diffusion_models/krea2_turbo_fp8_scaled.safetensors` (or `_bf16`) | `models/Stable-diffusion/` |
| `diffusion_models/krea2_raw_fp8_scaled.safetensors` (or `_bf16`) | `models/Stable-diffusion/` |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` (or `_bf16`) | `models/text_encoder/` |
| `vae/qwen_image_vae.safetensors` | `models/VAE/` |
| *(optional)* `wan_2.1_vae.safetensors` from [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) — also works with Krea 2, in some cases gives better results | `models/VAE/` |

## 🚀 Usage
1. In **txt2img**, set **UI Preset → `krea2`** (applies Euler / Simple / 8 steps / CFG 1 and auto-selects the TE + VAE).
2. Pick a **Krea 2 checkpoint**:
   - **Turbo** → 8 steps, CFG 1.0
   - **RAW** → 28 steps, CFG 4.5
3. Sampler **Euler**, scheduler **Simple**, Clip skip **1**.
4. Use **natural-language prompts** (Qwen3-VL works poorly with raw JSON).
5. Generate.

### Two ways to load a model
- **Full model:** select a combined checkpoint (e.g. one you baked with everything inside) — just works.
- **Pieces:** select a bare DiT checkpoint — the extension **auto-loads** a Qwen3-VL TE (bf16 preferred) + Qwen-Image VAE from your module folders. To use fp8 or a specific TE/VAE, pick them in the **VAE / Text Encoder** dropdown (that choice wins).

## 🛠 Troubleshooting
- **"You do not have Qwen3 state dict!" / fails to load a bare DiT** → the TE/VAE weren't found. Make sure `qwen3vl_4b_*.safetensors` is in `models/text_encoder/` and `qwen_image_vae.safetensors` is in `models/VAE/` (the Krea 2 tab does this for you).
- **Washed-out / doubled / garbled images** → wrong settings. Use **Euler + Simple**, Clip skip **1**, discard-penultimate-sigma **off**, and a prose prompt. The `krea2` preset sets these for you.
- **TE/VAE dropdown empty** → put the files in `models/text_encoder` + `models/VAE`, then hit the 🔄 refresh next to the dropdown.

## 💬 Help & Support
Questions, bugs, or want to show off your results? **Bring your issues to the Stable Yogi community → [stableyogi.com](https://stableyogi.com)** — that's where we help, share presets, and post guides.

## 📜 Credits & License
- DiT implementation ported from **ComfyUI** (`comfy/ldm/krea2`) — therefore this extension is **GPL-3.0**.
- Model weights: **Krea 2 Community License** (download from Comfy-Org/Krea-2; not redistributed here).
- **Detail Boost** technique adapted from [huwhitememes/comfyui-krea2-conditioning](https://github.com/huwhitememes/comfyui-krea2-conditioning) (**Apache-2.0**), a fork of nova452/ComfyUI-ConditioningKrea2Rebalance.
- Integration & packaging by **[stableyogi.com](https://stableyogi.com)**.

---

### More free tools by Stable Yogi

Small, free, open tools for local AI art — Forge / Forge Neo, AUTOMATIC1111, and ComfyUI.
Browse them all at **[github.com/Stable-yogi](https://github.com/Stable-yogi)** · more at **[stableyogi.com](https://stableyogi.com)**.
