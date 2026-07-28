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
    # Optional ALTERNATIVE VAE: the Wan 2.1 VAE also works with Krea 2 and in some cases gives
    # better results. Lives in a different HF repo, hence the absolute `url` (overrides REPO+rel).
    "vae_wan": dict(label="Wan 2.1 VAE (optional — alt to the Qwen VAE)",
                    bf16="wan_2.1_vae.safetensors",
                    fp8="wan_2.1_vae.safetensors",
                    url="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
                    dest="vae", kw=("wan_2.1_vae", "wan21_vae", "wan_vae")),
}
_EXTS = (".safetensors", ".sft", ".gguf")


def _native_krea2():
    """Newer Forge Neo ships Krea 2 natively; its UI preset is named 'krea' (ours is 'krea2')."""
    try:
        import backend.loader as _l
        return any(getattr(M, "__name__", "") == "Krea2" for M in getattr(_l, "possible_models", ()))
    except Exception:
        return False


PRESET_NAME = "krea" if _native_krea2() else "krea2"

# ── Featured checkpoint: "Muse by Stable Yogi" — Krea 2 v1.5 Turbo (photoreal) ──
# A bare Krea 2 DiT (turbo, 8-step); it plugs into the SAME base Krea 2 TE + VAE the
# downloader already fetches. Civitai serves these anonymously (307 -> signed R2 CDN,
# Range/resume supported), so no API token is needed. The two GGUF builds share one
# filename on Civitai, so we save them under disambiguated names (Q8_0 / Q4_0).
CIVITAI_MUSE_PAGE = "https://civitai.com/models/2741166"
# Filenames deliberately avoid the substrings "turbo"/"raw" (they'd collide with the base
# dit_turbo/dit_raw detection keywords) and use a specific "krea2muse" token instead of bare
# "muse" (which false-matches amuse/museum/...). All three lowercase to contain "krea2muse".
MUSE_KW = ("krea2muse",)
MUSE_VARIANTS = {
    "Q8_0 GGUF — recommended (~14 GB)": dict(
        url="https://civitai.com/api/download/models/3092719?type=Model&format=GGUF&size=pruned&quantType=Q8_0",
        fname="krea2Muse_v15_Q8_0.gguf"),
    "Q4_0 GGUF — low VRAM (~8 GB)": dict(
        url="https://civitai.com/api/download/models/3092719?type=Model&format=GGUF&size=pruned&quantType=Q4_0",
        fname="krea2Muse_v15_Q4_0.gguf"),
    "fp8 safetensors — compat (~12.5 GB)": dict(
        url="https://civitai.com/api/download/models/3092719",
        fname="krea2Muse_v15_fp8.safetensors"),
}


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
        # Forge's real checkpoint folder is "Stable-diffusion" (WITH the dash — see Forge core
        # modules/sd_models.py). Download where Forge actually scans: the first --ckpt-dirs if set,
        # else models/Stable-diffusion. (The old "Stablediffusion" put files where Forge never looks.)
        ckpt_dirs = list(getattr(co, "ckpt_dirs", []) or [])
        default = os.path.join(root, "Stable-diffusion")
        target = ckpt_dirs[0] if ckpt_dirs else default
        scan = [default] + ckpt_dirs
    elif kind == "te":
        target = os.path.join(root, "text_encoder")
        scan = [target] + list(getattr(co, "text_encoder_dirs", []) or [])
    else:  # vae
        target = os.path.join(root, "VAE")
        scan = [target] + list(getattr(co, "vae_dirs", []) or [])
    return target, [d for d in scan if d]


def _st_keys(path):
    """safetensors tensor names (header only). Empty set on failure."""
    try:
        if not str(path).lower().endswith((".safetensors", ".sft")):
            return set()
        import json
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        return set(h.keys())
    except Exception:
        return set()


def _classify(kind, keys):
    """Identify a Krea2 component by its keys (so renamed files still count as present)."""
    if kind == "te":
        return (any(("self_attn.q_norm" in k and "layers." in k and "visual" not in k) for k in keys)
                and any("embed_tokens" in k for k in keys))
    if kind == "vae":
        return (any(k.startswith("decoder.") for k in keys)
                and any(("downsamples" in k or "upsamples" in k) for k in keys)
                and not any("visual" in k for k in keys))
    if kind == "ckpt":  # krea2 DiT fingerprint
        return any(("blocks.0.mod.lin" in k or "txtfusion.projector" in k) for k in keys)
    return False


def _present(kw, kind, content=True):
    _, scan = _dest_dirs(kind)
    # 1. fast filename-keyword match
    for d in scan:
        try:
            for n in os.listdir(d):
                low = n.lower()
                if low.endswith(_EXTS) and any(k in low for k in kw):
                    return os.path.join(d, n)
        except Exception:
            continue
    # 2. content fallback — detect RENAMED files by their keys (cap per dir to stay fast).
    #    Skipped when content=False: a krea2 DiT fingerprint can't tell Muse/turbo/raw apart, so
    #    per-variant DiT rows pass content=False (filename-only) to avoid cross-attributing one
    #    file to another variant. TE/VAE keep content detection (their fingerprints are unique).
    if content:
        for d in scan:
            try:
                files = [n for n in os.listdir(d) if n.lower().endswith((".safetensors", ".sft"))]
                if len(files) > 80:           # don't header-scan a huge checkpoint dir
                    continue
                for n in files:
                    p = os.path.join(d, n)
                    if _classify(kind, _st_keys(p)):
                        return p
            except Exception:
                continue
    return None


def _any_dit():
    """True if any loadable Krea 2 DiT is present — Muse / turbo / raw by name, or a renamed
    krea2 DiT .safetensors by content. (GGUF is name-only; _st_keys can't read gguf headers.)"""
    if (_present(MUSE_KW, "ckpt", content=False)
            or _present(COMPONENTS["dit_turbo"]["kw"], "ckpt", content=False)
            or _present(COMPONENTS["dit_raw"]["kw"], "ckpt", content=False)):
        return True
    return bool(_present(("__krea2dit__",), "ckpt", content=True))  # bogus kw -> falls to content scan


def _status_md():
    rows = ["| Component | Folder | Status |", "|---|---|---|"]
    tgt_ckpt, _ = _dest_dirs("ckpt")
    muse_hit = _present(MUSE_KW, "ckpt", content=False)  # filename-only: never claim another DiT is "Muse"
    rows.append(f"| ⭐ Muse (Krea 2 v1.5 Turbo) | `{tgt_ckpt}` | "
                + (f"✅ `{os.path.basename(muse_hit)}`" if muse_hit else "— not downloaded") + " |")
    ready_req = True
    for key, c in COMPONENTS.items():
        target, _ = _dest_dirs(c["dest"])
        # DiT rows are filename-only (fingerprint can't tell turbo/raw/muse apart); the Wan-VAE row
        # too (the generic VAE fingerprint would claim the Qwen VAE as "wan"). TE/qwen-VAE keep content.
        hit = _present(c["kw"], c["dest"], content=(c["dest"] != "ckpt" and key != "vae_wan"))
        if hit:
            mark = f"✅ `{os.path.basename(hit)}`"
        elif "optional" in c["label"]:
            mark = "— optional, not downloaded"
        else:
            mark = "❌ missing"
            if "REQUIRED" in c["label"]:
                ready_req = False
        rows.append(f"| {c['label']} | `{target}` | {mark} |")
    ready = ready_req and _any_dit()
    banner = (f"### ✅ Krea 2 is ready — pick UI Preset **{PRESET_NAME}**, choose a Krea 2 checkpoint, generate."
              if ready else
              "### ⚠️ Setup incomplete — download the ❌ items below (you need at least one DiT + the TE + the VAE).")
    return banner + "\n\n" + "\n".join(rows)


def _verify_magic(path):
    """Reject an obvious non-model — e.g. a Cloudflare/login/error HTML page that a server returned
    with 200 and we saved under a .gguf/.safetensors name. Cheap header check, no full parse."""
    low = path.lower()
    with open(path, "rb") as f:
        head = f.read(8)
    if low.endswith(".gguf"):
        if head[:4] != b"GGUF":
            raise IOError("downloaded file is not a GGUF (looks like an error page) — delete the .part and retry")
    elif low.endswith((".safetensors", ".sft")):
        import struct as _s
        if len(head) < 8:
            raise IOError("downloaded file too small to be a safetensors")
        n = _s.unpack("<Q", head)[0]              # safetensors header = leading uint64 length
        if not (0 < n < os.path.getsize(path)):
            raise IOError("downloaded file is not a valid safetensors (looks like an error page) — delete the .part and retry")


def _download(url, dest_path, progress):
    import requests
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    resume = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": "Mozilla/5.0 (sd-forge-krea2)"}  # some CDNs reject a bare UA
    if resume:
        headers["Range"] = f"bytes={resume}-"
    name = os.path.basename(dest_path)
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if r.status_code == 416 and resume:
            # Range past EOF: the .part is already complete but was never promoted last time (a kill
            # between write and os.replace, or a locked dest). Promote it instead of failing forever.
            try:
                _verify_magic(tmp)
            except Exception:
                os.remove(tmp)
                raise
            os.replace(tmp, dest_path)
            return dest_path
        if r.status_code not in (200, 206):
            r.raise_for_status()
        resumed = (r.status_code == 206)            # server actually honored the Range request
        if not resumed:                             # full body (Range ignored, or fresh) -> start over,
            resume = 0                               # else appending the full body corrupts the .part
        total = int(r.headers.get("content-length", 0)) + resume
        done = resume
        with open(tmp, "ab" if resumed else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 21):  # 2MB
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    progress(min(done / total, 1.0), desc=f"{name}  {done/1e9:.1f}/{total/1e9:.1f} GB")
    # verify completeness before promoting .part -> final, so a silently-dropped connection can't
    # leave a truncated file that looks done (re-running resumes from the .part).
    if total and os.path.getsize(tmp) < total:
        raise IOError(f"{name}: incomplete download ({os.path.getsize(tmp)}/{total} bytes) — re-run to resume")
    # reject an error page / non-model saved under a model name (delete so a retry starts clean).
    try:
        _verify_magic(tmp)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise
    os.replace(tmp, dest_path)
    return dest_path


def _do(keys, precision, progress=gr.Progress()):
    log = []
    for key in keys:
        c = COMPONENTS[key]
        rel = c["fp8"] if precision == "fp8 (smaller / faster)" else c["bf16"]
        target_dir, _ = _dest_dirs(c["dest"])
        dest = os.path.join(target_dir, os.path.basename(rel))
        if _present(c["kw"], c["dest"], content=(c["dest"] != "ckpt" and key != "vae_wan")):
            log.append(f"⏭️  {c['label']} — already present, skipped")
            continue
        try:
            progress(0.0, desc=f"Starting {os.path.basename(rel)} …")
            _download(c.get("url") or (REPO + rel), dest, progress)  # `url` = component from another repo
            log.append(f"✅ {c['label']} → {dest}")
        except Exception as e:
            log.append(f"❌ {c['label']} FAILED: {e}")
    log.append("\nDone. Click 'Refresh status'. New models also need a checkpoint/module refresh in Forge (🔄 in the dropdowns).")
    return _status_md(), "\n".join(log)


def _do_muse(variant, precision, progress=gr.Progress()):
    """Download the featured Muse checkpoint from Civitai, then ensure the base TE + VAE.
    Muse is a bare DiT, so it needs the same Krea 2 text encoder + VAE as the base model."""
    log = []
    v = MUSE_VARIANTS.get(variant) or next(iter(MUSE_VARIANTS.values()))
    target_dir, scan = _dest_dirs("ckpt")
    # skip if THIS exact build already exists in any scanned ckpt dir (not just the download target),
    # so a custom --ckpt-dirs doesn't trigger a needless multi-GB re-download of a file Forge already sees.
    existing = next((os.path.join(d, v["fname"]) for d in scan
                     if os.path.exists(os.path.join(d, v["fname"]))), None)
    if existing:
        log.append(f"⏭️  Muse ({variant}) — already present, skipped")
    else:
        dest = os.path.join(target_dir, v["fname"])
        try:
            progress(0.0, desc=f"Starting {v['fname']} …")
            _download(v["url"], dest, progress)
            log.append(f"✅ Muse ({variant}) → {dest}")
        except Exception as e:
            log.append(f"❌ Muse ({variant}) FAILED: {e}")
    # Muse is a bare DiT — pull the base Krea 2 TE + VAE too (_do adds its own footer).
    _, tvlog = _do(["te", "vae"], precision, progress)
    return _status_md(), "\n".join(log) + "\n" + tvlog


def _build_tab():
    with gr.Blocks() as ui:
        gr.Markdown(
            "## 🎨 Krea 2 for Forge — Setup\n"
            "First open-source **Krea 2** on Forge. Install once, click download, generate. "
            f"All weights are pulled from the public [Comfy-Org/Krea-2]({REPO_PAGE}) repo. "
            f"Built by [stableyogi.com]({SY})."
        )
        status = gr.Markdown(_status_md())
        with gr.Row():
            precision = gr.Radio(
                ["fp8 (smaller / faster)", "bf16 (full quality)"],
                value="fp8 (smaller / faster)", label="Precision",
                info="fp8 ≈ half the size/VRAM, near-identical quality. bf16 = maximum fidelity.",
            )
            refresh = gr.Button("🔄 Refresh status")

        # ── Featured: Muse by Stable Yogi (Krea 2 v1.5 Turbo) ─────────────────
        gr.Markdown(
            "### ⭐ Featured — **Muse by Stable Yogi** · Krea 2 v1.5 Turbo (photoreal)\n"
            "Fast **8-step** photoreal checkpoint tuned by Stable Yogi "
            "(**8 steps · CFG 1 · Euler · Simple**). It's a bare DiT, so this also grabs the "
            f"base **TE + VAE** it needs. [Model page]({CIVITAI_MUSE_PAGE})."
        )
        with gr.Row():
            muse_variant = gr.Dropdown(
                choices=list(MUSE_VARIANTS.keys()), value=list(MUSE_VARIANTS.keys())[0],
                label="Muse build", scale=3,
                info="Q8 = best quality · Q4 = lowest VRAM · fp8 = widest compat. TE/VAE follow the Precision above.",
            )
            get_muse = gr.Button("⭐ Download Muse (+ TE + VAE)", variant="primary", scale=1)

        gr.Markdown("---\n**Vanilla base Krea 2** (optional) — stock Comfy-Org checkpoints; the TE + VAE are shared with Muse:")
        get_all = gr.Button("⬇️  Download base Set (Turbo DiT + Text Encoder + VAE)")
        with gr.Row():
            get_turbo = gr.Button("⬇️ Turbo DiT")
            get_raw = gr.Button("⬇️ RAW DiT")
            get_te = gr.Button("⬇️ Text Encoder")
            get_vae = gr.Button("⬇️ VAE")
            get_wan_vae = gr.Button("⬇️ Wan 2.1 VAE (alt)")
        gr.Markdown(
            "💡 **Tip:** the **Wan 2.1 VAE** also works with Krea 2 — and in some cases gives "
            "**better results**, so you can try it. Download it, then pick it in the VAE dropdown "
            "instead of the Qwen VAE."
        )
        logbox = gr.Textbox(label="Download log", lines=8, interactive=False)
        gr.Markdown(
            "### How to use\n"
            "1. Click **⭐ Download Muse** (recommended) — or the base set. Precision **fp8** = less VRAM, **bf16** = max fidelity (applies to the TE + VAE).\n"
            f"2. In txt2img, set **UI Preset → {PRESET_NAME}** (auto-applies the right sampler/steps/CFG; the TE + VAE are auto-attached).\n"
            "3. Pick a **Krea 2 checkpoint** (Muse / Turbo → 8 steps, CFG 1; RAW → 28 steps, CFG 4.5).\n"
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

        def dl_wan_vae(precision, progress=gr.Progress()):
            return _do(["vae_wan"], precision, progress)

        refresh.click(lambda: _status_md(), outputs=[status])
        get_muse.click(_do_muse, inputs=[muse_variant, precision], outputs=[status, logbox])
        get_all.click(dl_all, inputs=[precision], outputs=[status, logbox])
        get_turbo.click(dl_turbo, inputs=[precision], outputs=[status, logbox])
        get_raw.click(dl_raw, inputs=[precision], outputs=[status, logbox])
        get_te.click(dl_te, inputs=[precision], outputs=[status, logbox])
        get_vae.click(dl_vae, inputs=[precision], outputs=[status, logbox])
        get_wan_vae.click(dl_wan_vae, inputs=[precision], outputs=[status, logbox])
    return [(ui, "Krea 2", "krea2_setup_tab")]


if script_callbacks is not None:
    try:
        script_callbacks.on_ui_tabs(_build_tab)
    except Exception:
        print("[krea2] setup tab failed to register:\n" + traceback.format_exc())
