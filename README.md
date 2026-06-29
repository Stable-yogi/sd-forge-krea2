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
- Native Krea 2 architecture in Forge — no ComfyUI needed.
- **Both loading streams supported:**
  - **Full model** — one combined checkpoint with everything baked in.
  - **Pieces** — a bare DiT checkpoint; the TE + VAE are **auto-loaded** (or pick them yourself).
- **fp8 supported** for the DiT and the text encoder (half the size/VRAM).
- **One-click model downloader** — the **"Krea 2"** tab fetches every file into the right folder.
- **`krea2` UI preset** — auto-sets sampler/steps/CFG **and** auto-selects the TE + VAE.

## ✅ Requirements
- **Forge Neo** (Haoming02/sd-webui-forge-classic, `neo` branch). Tested on `neo-2.23`.
- An NVIDIA GPU with enough VRAM (fp8 set ≈ 16–20 GB; bf16 set ≈ 24 GB+).
- No extra Python packages — uses Forge's existing dependencies.

## 📦 Installation
1. Copy the `sd-forge-krea2` folder into your Forge `extensions/` directory.
2. Restart Forge.
3. Open the new **"Krea 2"** tab.

## ⬇️ Getting the models (easy way)
In the **Krea 2** tab:
1. Pick a precision — **fp8** (smaller/faster) or **bf16** (max quality).
2. Click **Download Recommended Set** (Turbo DiT + Text Encoder + VAE).
3. Files land in Forge's standard folders automatically — **no command-line flags needed.**

### Manual download (alternative)
All files are in the public HF repo **[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)**:

| File | Put it in |
|---|---|
| `diffusion_models/krea2_turbo_fp8_scaled.safetensors` (or `_bf16`) | `models/Stablediffusion/` |
| `diffusion_models/krea2_raw_fp8_scaled.safetensors` (or `_bf16`) | `models/Stablediffusion/` |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` (or `_bf16`) | `models/text_encoder/` |
| `vae/qwen_image_vae.safetensors` | `models/VAE/` |

## 🚀 Usage
1. In **txt2img**, set **UI Preset → `krea2`** (applies Euler / Simple / 28 steps / CFG 4.5 and auto-selects the TE + VAE).
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
- Integration & packaging by **[stableyogi.com](https://stableyogi.com)**.
