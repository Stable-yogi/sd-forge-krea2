"""
Krea 2 registration — runs on Forge load. Registers the Krea2 arch into Forge's
model system at runtime (no core edits): detection fingerprint, model_list BASE,
the loader's transformer builder (SingleStreamDiT), and the diffusion engine.

Everything is wrapped so a failure never breaks Forge boot.
"""
import os
import sys
import traceback

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # extensions/sd-forge-krea2
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)
CFG_DIR = os.path.join(EXT_ROOT, "hf_config", "Krea2")

# Recommended Krea2 RAW sampling defaults (RAW is cfg-based; Turbo users drop steps→8, cfg→1).
# These are READ-ONLY via getattr in on_preset_change, so a plain data inject suffices.
KREA2_PRESET_DEFAULTS = {
    "krea2_t2i_sampler": "Euler", "krea2_i2i_sampler": "Euler",
    "krea2_t2i_scheduler": "Simple", "krea2_i2i_scheduler": "Simple",
    "krea2_t2i_step": 28, "krea2_t2i_hr_step": 28, "krea2_i2i_step": 28,
    "krea2_t2i_cfg": 4.5, "krea2_t2i_hr_cfg": 4.5, "krea2_i2i_cfg": 4.5,
    "krea2_t2i_width": 1024, "krea2_t2i_height": 1024,
    "krea2_i2i_width": 1024, "krea2_i2i_height": 1024,
    "krea2_t2i_batch_size": 1, "krea2_i2i_batch_size": 1,
}
# These hidden per-preset opts are WRITTEN via opts.set() in checkpoint/modules/dtype_change,
# and opts.set does `self.data_labels[key]` (KeyError if unregistered) — so they MUST be in
# data_labels, not just data.
KREA2_HIDDEN_OPTS = {
    "forge_checkpoint_krea2": None,
    "forge_additional_modules_krea2": [],
    "forge_unet_storage_dtype_krea2": "Automatic",
}


def _find_krea2_modules():
    """Auto-locate the Qwen3-VL TE (bf16 preferred) + Qwen-Image VAE from Forge's module dirs:
    models/text_encoder + models/VAE plus any --text-encoder-dirs/--vae-dirs. Dir-scanning so it
    works even before the UI populates its module list. Returns [te_path, vae_path] (found ones)."""
    from modules import shared
    try:
        from modules import paths
        root = paths.models_path
    except Exception:
        root = os.path.join(os.getcwd(), "models")
    co = getattr(shared, "cmd_opts", None)
    te_dirs = [os.path.join(root, "text_encoder")] + list(getattr(co, "text_encoder_dirs", []) or [])
    vae_dirs = [os.path.join(root, "VAE")] + list(getattr(co, "vae_dirs", []) or [])

    def scan(dirs, all_kw=(), any_kw=(), prefer=(), avoid=()):
        cands = []
        for d in dirs:
            try:
                for n in os.listdir(d):
                    low = n.lower()
                    if (low.endswith((".safetensors", ".sft", ".gguf"))
                            and all(k in low for k in all_kw)
                            and (not any_kw or any(k in low for k in any_kw))
                            and not any(a in low for a in avoid)):
                        cands.append(os.path.join(d, n))
            except Exception:
                continue
        if not cands:
            return None
        for pk in prefer:
            for c in cands:
                if pk in os.path.basename(c).lower():
                    return c
        return cands[0]

    te = scan(te_dirs, any_kw=("qwen3vl", "qwen3_vl", "qwen3-vl"), prefer=("bf16",))   # bf16 default
    vae = scan(vae_dirs, all_kw=("vae",), any_kw=("qwen",), prefer=("qwen_image_vae",), avoid=("clear",))
    return [m for m in (te, vae) if m]


def _patch_preset_auto_modules():
    """Make selecting the 'krea2' preset auto-fill its VAE+TE the first time (when the user
    hasn't chosen modules for it yet). Respects any later manual choice (saved per-preset)."""
    import modules_forge.main_entry as ME
    from modules import shared

    if getattr(ME.on_preset_change, "_krea2", False):
        return
    _orig = ME.on_preset_change

    def _wrapped(preset):
        if preset == "krea2" and not (shared.opts.data.get("forge_additional_modules_krea2") or []):
            auto = _find_krea2_modules()
            if auto:
                shared.opts.data["forge_additional_modules_krea2"] = auto
        return _orig(preset)

    _wrapped._krea2 = True
    ME.on_preset_change = _wrapped


def _register_preset():
    """Add a 'krea2' entry to the UI Preset dropdown + its default sampler/steps/cfg + auto VAE/TE.
    Pure runtime patch (no core edit): on_preset_change reads getattr(opts, f'{preset}_*')
    dynamically and PresetArch.choices() feeds the dropdown, so this is upgrade-safe."""
    from modules import shared
    from modules.options import OptionInfo
    import modules_forge.presets as P

    if getattr(P.PresetArch.choices, "_krea2", False) is False:
        _orig_choices = P.PresetArch.choices

        def _choices():
            c = list(_orig_choices())
            if "krea2" not in c:
                c.append("krea2")
            return c

        _choices._krea2 = True
        P.PresetArch.choices = staticmethod(_choices)

    # hidden opts -> register in data_labels so opts.set() doesn't KeyError
    for k, default in KREA2_HIDDEN_OPTS.items():
        if k not in shared.opts.data_labels:
            oi = OptionInfo(default)
            oi.section = (None, "Forge Hidden Options")
            shared.opts.data_labels[k] = oi
        shared.opts.data.setdefault(k, default)

    # read-only sampling defaults -> data inject is enough
    for k, v in KREA2_PRESET_DEFAULTS.items():
        shared.opts.data.setdefault(k, v)

    # selecting the krea2 preset auto-picks the Qwen3-VL TE + Qwen-Image VAE
    _patch_preset_auto_modules()


def _register():
    import torch
    import backend.loader as loader
    from backend.operations import ForgeOperations
    # IMPORTANT: import the SAME package object the loader uses (top-level
    # `huggingface_guess`, since modules_forge/packages is on sys.path). Importing it
    # as modules_forge.packages.huggingface_guess yields a 2nd copy and the patches
    # would land on a module the loader never sees.
    from huggingface_guess import detection, latent, model_list

    # --- 1. Krea2 BASE config (inherits Z-Image: Qwen3 TE + flow-match) ---
    class Krea2Base(model_list.ZImage):
        huggingface_repo = CFG_DIR
        unet_config = {"image_model": "krea2"}
        sampling_settings = {"multiplier": 1.0, "shift": 1.15}
        unet_target = "transformer"
        latent_format = latent.Wan21          # Qwen-Image VAE: 16-ch Wan21 per-channel stats
        supported_inference_dtypes = [torch.bfloat16, torch.float32]

        def clip_target(self, state_dict):
            # baked combined ckpt carries text_encoders.qwen3.* ; the piecewise path
            # (replace_state_dict patch below) yields native text_encoders.qwen3_4b.transformer.*
            pref = self.text_encoder_key_prefix[0]
            if "{}qwen3_4b.transformer.model.embed_tokens.weight".format(pref) in state_dict:
                return {"qwen3_4b.transformer": "text_encoder"}
            return {"qwen3": "text_encoder"}

    if not any(getattr(m, "__name__", "") == "Krea2Base" for m in model_list.models):
        model_list.models.insert(0, Krea2Base)

    # --- 2. detection fingerprint (txtfusion / mod.lin are unique to Krea2) ---
    _orig_detect = detection.detect_unet_config

    def _patched_detect(state_dict, key_prefix):
        if ("{}txtfusion.projector.weight".format(key_prefix) in state_dict
                or "{}blocks.0.mod.lin".format(key_prefix) in state_dict):
            return {"image_model": "krea2"}
        return _orig_detect(state_dict, key_prefix)

    if getattr(detection.detect_unet_config, "_krea2", False) is False:
        _patched_detect._krea2 = True
        detection.detect_unet_config = _patched_detect

    # --- 3. loader: build SingleStreamDiT for cls_name "SingleStreamDiT" ---
    def _build_krea2_dit(guess, state_dict):
        from krea2.dit import SingleStreamDiT
        L = loader
        unet_config = {k: v for k, v in guess.unet_config.items() if k not in ("image_model", "audio_model")}
        sdtype = L.utils.weight_dtype(state_dict)
        storage_dtype = sdtype if sdtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16
        load_device = L.memory_management.get_torch_device()
        comp_dtype = L.memory_management.inference_cast(weight_dtype=storage_dtype, inference_device=load_device, supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
        params = L.utils.calculate_parameters(state_dict)
        init_device = L.memory_management.unet_initial_load_device(parameters=params, dtype=storage_dtype)
        need_cast = storage_dtype != comp_dtype
        to_args = dict(device=init_device, dtype=storage_dtype)
        with L.no_init_weights():
            with L.using_forge_operations(**to_args, manual_cast_enabled=need_cast):
                model = SingleStreamDiT(**unet_config, operations=ForgeOperations).to(**to_args)
        L.load_state_dict(model, state_dict)
        model.config = unet_config
        model.storage_dtype = storage_dtype
        model.computation_dtype = comp_dtype
        model.load_device = load_device
        model.initial_device = init_device
        model.offload_device = L.memory_management.unet_offload_device()
        return model

    _orig_lhc = loader.load_huggingface_component

    def _patched_lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if cls_name == "SingleStreamDiT":
            return _build_krea2_dit(guess, state_dict)
        return _orig_lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    if getattr(loader.load_huggingface_component, "_krea2", False) is False:
        _patched_lhc._krea2 = True
        loader.load_huggingface_component = _patched_lhc

    # --- 4. diffusion engine ---
    from krea2.engine import Krea2 as Krea2Engine
    Krea2Engine.matched_guesses = [Krea2Base]
    if Krea2Engine not in loader.possible_models:
        loader.possible_models.append(Krea2Engine)

    # --- 5. PIECEWISE + fp8: load the raw DiT as checkpoint and pick the Qwen3-VL TE
    #        (bf16 or fp8) + Qwen VAE from the UI module dropdowns (no 34GB bake).
    #        Forge's replace_state_dict qwen3 branch expects model.layers.* but a
    #        Qwen3-VL file nests under model.language_model.* and carries model.visual.* —
    #        flatten + drop-visual before the merge so the native branch fires (which then
    #        handles fp8/comfy_quant for free). ---
    _orig_replace = loader.replace_state_dict

    def _patched_replace(sd, asd, guess, path):
        # Qwen3-VL TE files ship an unused vision tower (model.visual.*) that otherwise rides into
        # the Qwen3 text-encoder load as "Unexpected" keys (wasted VRAM + a noisy log). Strip it.
        # The bf16 variant nests the LM under model.language_model.* (flatten that); the fp8 variant
        # is already flat (model.layers.*) — so strip visual INDEPENDENTLY of the rename.
        if any(k.startswith(("model.visual.", "visual.", "model.language_model.")) for k in asd):
            asd = {(k.replace("model.language_model.", "model.", 1) if k.startswith("model.language_model.") else k): v
                   for k, v in asd.items()
                   if not (k.startswith("model.visual.") or k.startswith("visual."))}
        return _orig_replace(sd, asd, guess, path)

    if getattr(loader.replace_state_dict, "_krea2", False) is False:
        _patched_replace._krea2 = True
        loader.replace_state_dict = _patched_replace

    # --- 6. UI preset 'krea2' (Euler / Simple / 28 steps / CFG 4.5) ---
    try:
        _register_preset()
    except Exception:
        print("[krea2] preset registration skipped:\n" + traceback.format_exc())

    # --- 7. SEAMLESS PIECES: a bare krea2 DiT auto-loads its TE+VAE, so loading "pieces"
    #        works exactly like the full bake with no manual module-picking. Both streams
    #        supported: full combined checkpoint OR bare DiT + auto TE/VAE.
    #        (sd_models binds forge_loader by name at import, so patch THAT reference.) ---
    import json as _json
    import struct as _struct
    import modules.sd_models as _sdm

    def _is_bare_krea2_dit(path):
        try:
            p = str(path)
        except Exception:
            return False
        if not p.lower().endswith((".safetensors", ".sft")):
            return False
        try:
            with open(p, "rb") as f:
                n = _struct.unpack("<Q", f.read(8))[0]
                keys = list(_json.loads(f.read(n)).keys())
        except Exception:
            return False
        has_krea2 = any(("blocks.0.mod.lin" in k) or ("txtfusion.projector" in k) for k in keys)
        has_te = any(k.startswith("text_encoders.") or (".language_model." in k) for k in keys)
        return has_krea2 and not has_te

    _orig_fl = _sdm.forge_loader

    def _auto_pieces_forge_loader(sd, additional_state_dicts=None):
        try:
            if _is_bare_krea2_dit(sd):
                asd = list(additional_state_dicts or [])
                if not any("qwen3vl" in os.path.basename(str(p)).lower() for p in asd):
                    auto = _find_krea2_modules()
                    if auto:
                        additional_state_dicts = auto + asd
                        print("[krea2] bare DiT -> auto-loaded modules: "
                              + ", ".join(os.path.basename(m) for m in auto))
        except Exception:
            pass
        return _orig_fl(sd, additional_state_dicts)

    if getattr(_sdm.forge_loader, "_krea2", False) is False:
        _auto_pieces_forge_loader._krea2 = True
        _sdm.forge_loader = _auto_pieces_forge_loader
        loader.forge_loader = _auto_pieces_forge_loader

    print("[krea2] registered: arch 'krea2' (SingleStreamDiT + Qwen3-VL + Qwen VAE) + "
          "piecewise/fp8 + auto TE/VAE for bare DiTs + 'krea2' UI preset.")


try:
    _register()
except Exception:
    print("[krea2] registration FAILED (Forge boot unaffected):\n" + traceback.format_exc())
