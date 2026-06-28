"""
Krea 2 Setup tab — makes setup one-click:
  * detects which model files are already present
  * shows download links + the exact folder each goes in
  * one-click downloads each model into Forge's STANDARD folders
    (models/text_encoder, models/VAE, the checkpoint dir) so no launch flags are needed.

All files come from the public Comfy-Org/Krea-2 HF repo (no login/gate).
"""
import os
import traceback

import gradio as gr

try:
    from modules import script_callbacks, shared, paths
except Exception:
    script_callbacks = None

REPO = "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/"
REPO_PAGE = "https://huggingface.co/Comfy-Org/Krea-2/tree/main"
SY = "https://stableyogi.com"  # credit / traffic link

# component -> {label, bf16 path, fp8 path, dest kind, detect keywords}
COMPONENTS = {
    "dit_turbo": dict(label="Krea 2 Turbo DiT (8-step, fast — recommended)",
                      bf16="diffusion_models/krea2_turbo_bf16.safetensors",
                      fp8="diffusion_models/krea2_turbo_fp8_scaled.safetensors",
                      dest="ckpt", kw=("krea2_turbo", "turbo")),
    "dit_raw": dict(label="Krea 2 RAW DiT (base, best quality)",
                    bf16="diffusion_models/krea2_raw_bf16.safetensors",
                    fp8="diffusion_models/krea2_raw_fp8_scaled.safetensors",
                    dest="ckpt", kw=("krea2_raw", "raw")),
    "te": dict(label="Qwen3-VL Text Encoder (REQUIRED)",
               bf16="text_encoders/qwen3vl_4b_bf16.safetensors",
               fp8="text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
               dest="te", kw=("qwen3vl", "qwen3_vl")),
    "vae": dict(label="Qwen-Image VAE (REQUIRED)",
                bf16="vae/qwen_image_vae.safetensors",
                fp8="vae/qwen_image_vae.safetensors",
                dest="vae", kw=("qwen_image_vae",)),
}
_EXTS = (".safetensors", ".sft", ".gguf")


def _models_root():
    try:
        return paths.models_path
    except Exception:
        return os.path.join(os.getcwd(), "models")


def _dest_dirs(kind):
    """All dirs to SCAN for a kind, plus the preferred TARGET dir (first) to download into."""
    root = _models_root()
    co = getattr(shared, "cmd_opts", None)
    if kind == "ckpt":
        target = os.path.join(root, "Stablediffusion")
        scan = [target] + list(getattr(co, "ckpt_dirs", []) or [])
    elif kind == "te":
        target = os.path.join(root, "text_encoder")
        scan = [target] + list(getattr(co, "text_encoder_dirs", []) or [])
    else:  # vae
        target = os.path.join(root, "VAE")
        scan = [target] + list(getattr(co, "vae_dirs", []) or [])
    return target, [d for d in scan if d]


def _present(kw, kind):
    _, scan = _dest_dirs(kind)
    for d in scan:
        try:
            for n in os.listdir(d):
                low = n.lower()
                if low.endswith(_EXTS) and any(k in low for k in kw):
                    return os.path.join(d, n)
        except Exception:
            continue
    return None


def _status_md():
    rows = ["| Component | Folder | Status |", "|---|---|---|"]
    ready_req = True
    for key, c in COMPONENTS.items():
        target, _ = _dest_dirs(c["dest"])
        hit = _present(c["kw"], c["dest"])
        if hit:
            mark = f"✅ `{os.path.basename(hit)}`"
        else:
            mark = "❌ missing"
            if "REQUIRED" in c["label"]:
                ready_req = False
        rows.append(f"| {c['label']} | `{target}` | {mark} |")
    dit = _present(COMPONENTS["dit_turbo"]["kw"], "ckpt") or _present(COMPONENTS["dit_raw"]["kw"], "ckpt")
    ready = ready_req and bool(dit)
    banner = ("### ✅ Krea 2 is ready — pick UI Preset **krea2**, choose a Krea 2 checkpoint, generate."
              if ready else
              "### ⚠️ Setup incomplete — download the ❌ items below (you need at least one DiT + the TE + the VAE).")
    return banner + "\n\n" + "\n".join(rows)


def _download(url, dest_path, progress):
    import requests
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    resume = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    name = os.path.basename(dest_path)
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if r.status_code in (200, 206):
            pass
        else:
            r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume
        done = resume
        with open(tmp, "ab" if resume else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 21):  # 2MB
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    progress(min(done / total, 1.0), desc=f"{name}  {done/1e9:.1f}/{total/1e9:.1f} GB")
    os.replace(tmp, dest_path)
    return dest_path


def _do(keys, precision, progress=gr.Progress()):
    log = []
    for key in keys:
        c = COMPONENTS[key]
        rel = c["fp8"] if precision == "fp8 (smaller / faster)" else c["bf16"]
        target_dir, _ = _dest_dirs(c["dest"])
        dest = os.path.join(target_dir, os.path.basename(rel))
        if _present(c["kw"], c["dest"]):
            log.append(f"⏭️  {c['label']} — already present, skipped")
            continue
        try:
            progress(0.0, desc=f"Starting {os.path.basename(rel)} …")
            _download(REPO + rel, dest, progress)
            log.append(f"✅ {c['label']} → {dest}")
        except Exception as e:
            log.append(f"❌ {c['label']} FAILED: {e}")
    log.append("\nDone. Click 'Refresh status'. New models also need a checkpoint/module refresh in Forge (🔄 in the dropdowns).")
    return _status_md(), "\n".join(log)


def _build_tab():
    with gr.Blocks() as ui:
        gr.Markdown(
            "## 🎨 Krea 2 for Forge — Setup\n"
            "First open-source **Krea 2** on Forge. Install once, click download, generate. "
            f"All weights are pulled from the public [Comfy-Org/Krea-2]({REPO_PAGE}) repo. "
            f"Built by [sy.com]({SY})."
        )
        status = gr.Markdown(_status_md())
        with gr.Row():
            precision = gr.Radio(
                ["fp8 (smaller / faster)", "bf16 (full quality)"],
                value="fp8 (smaller / faster)", label="Precision",
                info="fp8 ≈ half the size/VRAM, near-identical quality. bf16 = maximum fidelity.",
            )
            refresh = gr.Button("🔄 Refresh status")
        gr.Markdown("**Quick start** — grab the essential set (Turbo DiT + Text Encoder + VAE):")
        get_all = gr.Button("⬇️  Download Recommended Set", variant="primary")
        with gr.Row():
            get_turbo = gr.Button("⬇️ Turbo DiT")
            get_raw = gr.Button("⬇️ RAW DiT")
            get_te = gr.Button("⬇️ Text Encoder")
            get_vae = gr.Button("⬇️ VAE")
        logbox = gr.Textbox(label="Download log", lines=8, interactive=False)
        gr.Markdown(
            "### How to use\n"
            "1. Download the **Recommended Set** above (or the bf16 set for max quality).\n"
            "2. In txt2img, set **UI Preset → krea2** (auto-applies Euler / Simple / 28 steps / CFG 4.5 **and auto-selects the TE + VAE**).\n"
            "3. Pick a **Krea 2 checkpoint** (Turbo → 8 steps, CFG 1; RAW → 28 steps, CFG 4.5).\n"
            "4. Use **natural-language prompts** (Qwen3-VL dislikes JSON). Generate.\n\n"
            "Files download into Forge's standard folders, so no command-line flags are needed. "
            f"Weights: [Comfy-Org/Krea-2]({REPO_PAGE}) · Krea 2 Community License."
        )

        def dl_all(precision, progress=gr.Progress()):
            return _do(["dit_turbo", "te", "vae"], precision, progress)

        def dl_turbo(precision, progress=gr.Progress()):
            return _do(["dit_turbo"], precision, progress)

        def dl_raw(precision, progress=gr.Progress()):
            return _do(["dit_raw"], precision, progress)

        def dl_te(precision, progress=gr.Progress()):
            return _do(["te"], precision, progress)

        def dl_vae(precision, progress=gr.Progress()):
            return _do(["vae"], precision, progress)

        refresh.click(lambda: _status_md(), outputs=[status])
        get_all.click(dl_all, inputs=[precision], outputs=[status, logbox])
        get_turbo.click(dl_turbo, inputs=[precision], outputs=[status, logbox])
        get_raw.click(dl_raw, inputs=[precision], outputs=[status, logbox])
        get_te.click(dl_te, inputs=[precision], outputs=[status, logbox])
        get_vae.click(dl_vae, inputs=[precision], outputs=[status, logbox])
    return [(ui, "Krea 2", "krea2_setup_tab")]


if script_callbacks is not None:
    try:
        script_callbacks.on_ui_tabs(_build_tab)
    except Exception:
        print("[krea2] setup tab failed to register:\n" + traceback.format_exc())
