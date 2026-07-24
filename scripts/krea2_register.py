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

# Recommended Krea2 Turbo / Muse sampling defaults (8-step distilled, cfg 1 = guidance off).
# RAW users bump steps->28 and cfg->4.5. READ-ONLY via getattr in on_preset_change (plain inject).
KREA2_PRESET_DEFAULTS = {
    "krea2_t2i_sampler": "Euler", "krea2_i2i_sampler": "Euler",
    "krea2_t2i_scheduler": "Simple", "krea2_i2i_scheduler": "Simple",
    "krea2_t2i_step": 8, "krea2_t2i_hr_step": 8, "krea2_i2i_step": 8,
    "krea2_t2i_cfg": 1.0, "krea2_t2i_hr_cfg": 1.0, "krea2_i2i_cfg": 1.0,
    "krea2_t2i_width": 896, "krea2_t2i_height": 1152,
    "krea2_i2i_width": 896, "krea2_i2i_height": 1152,
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


def _st_header_keys(path):
    """Read a safetensors file's tensor names (header only — cheap). Empty set on failure."""
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


def _is_qwen3_te(keys):
    return (any(("self_attn.q_norm" in k and "layers." in k and "visual" not in k) for k in keys)
            and any("embed_tokens" in k for k in keys))


def _is_qwen_vae(keys):
    return (any(k.startswith("decoder.") for k in keys) and any(k.startswith("encoder.") for k in keys)
            and any(("downsamples" in k or "upsamples" in k) for k in keys)
            and not any("visual" in k for k in keys))


def _is_wan_vae(keys):
    """Wan 2.1 VAE (Muse 1.5+ pairs with this instead of the Qwen-Image VAE): encoder/decoder
    present, no vision tower. Structurally distinct from the Qwen VAE but both are Krea2-compatible."""
    return (any(k.startswith("decoder.") for k in keys) and any(k.startswith("encoder.") for k in keys)
            and not any("visual" in k for k in keys))


def _content_scan(dirs, classifier, prefer=()):
    """Fallback when filenames don't match (user renamed files): identify the right module by its
    safetensors keys instead of its name. Only runs when the keyword scan found nothing."""
    cands = []
    for d in dirs:
        try:
            for n in os.listdir(d):
                if n.lower().endswith((".safetensors", ".sft")):
                    p = os.path.join(d, n)
                    if classifier(_st_header_keys(p)):
                        cands.append(p)
        except Exception:
            continue
    if not cands:
        return None
    for pk in prefer:
        for c in cands:
            if pk in os.path.basename(c).lower():
                return c
    return cands[0]


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
    if te is None:                                  # renamed file? fall back to key-based detection
        te = _content_scan(te_dirs, _is_qwen3_te, prefer=("bf16",))
    # VAE: Krea2 checkpoints pair with EITHER the Qwen-Image VAE (V1) or the Wan 2.1 VAE (Muse 1.5+).
    vae = scan(vae_dirs, all_kw=("vae",), any_kw=("qwen", "wan"), prefer=("wan_2.1_vae", "qwen_image_vae"), avoid=("clear",))
    if vae is None:
        vae = _content_scan(vae_dirs, lambda ks: _is_qwen_vae(ks) or _is_wan_vae(ks))
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
        if preset == "krea2":
            try:
                # FORCE the krea2 preset's sampler/scheduler/steps/cfg/size to the intended turbo
                # values every time it's selected. (Was setdefault, which left stale saved values —
                # e.g. an old 28 steps / CFG 4.5 — stuck; that's why cfg/steps "didn't apply".)
                for k, v in KREA2_PRESET_DEFAULTS.items():
                    shared.opts.data[k] = v
                # Checkpoint / modules stay per-user (setdefault, never overwrite their choice).
                for k, v in KREA2_HIDDEN_OPTS.items():
                    shared.opts.data.setdefault(k, v)
                if not (shared.opts.data.get("forge_additional_modules_krea2") or []):
                    auto = _find_krea2_modules()
                    if auto:
                        shared.opts.data["forge_additional_modules_krea2"] = auto
            except Exception:
                print("[krea2] preset pre-fill error:\n" + traceback.format_exc())
        try:
            result = _orig(preset)
            if preset == "krea2":                       # diagnostic: shows what actually gets applied
                mods = [os.path.basename(m) for m in (shared.opts.data.get("forge_additional_modules_krea2") or [])]
                print("[krea2] preset selected -> step=%s cfg=%s sampler=%s scheduler=%s modules=%s"
                      % (shared.opts.data.get("krea2_t2i_step"), shared.opts.data.get("krea2_t2i_cfg"),
                         shared.opts.data.get("krea2_t2i_sampler"), shared.opts.data.get("krea2_t2i_scheduler"), mods))
            return result
        except Exception:
            print("[krea2] on_preset_change('%s') FAILED:\n%s" % (preset, traceback.format_exc()))
            raise

    _wrapped._krea2 = True
    ME.on_preset_change = _wrapped


def _register_preset():
    """Add a 'krea2' entry to the UI Preset dropdown + its default sampler/steps/cfg + auto VAE/TE.
    Resilient by design: the dropdown entry is added FIRST with the fewest possible dependencies,
    so a later (Forge-version-specific) failure can never hide the preset. This is the fix for the
    'krea2 preset missing from the UI preset' reports."""
    # 1. Add 'krea2' to the dropdown — minimal deps. MUST succeed even if step 2/3 fail.
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

    # 2. Per-preset options (sampler/steps/cfg + hidden checkpoint/modules opts). Wrapped so a
    #    version difference (e.g. the OptionInfo import path) can't prevent the dropdown entry.
    try:
        from modules import shared
        try:
            from modules.options import OptionInfo
        except Exception:
            from modules.shared import OptionInfo  # older Forge layout
        for k, default in KREA2_HIDDEN_OPTS.items():
            if k not in shared.opts.data_labels:
                oi = OptionInfo(default)
                oi.section = (None, "Forge Hidden Options")
                shared.opts.data_labels[k] = oi
            shared.opts.data.setdefault(k, default)
        for k, v in KREA2_PRESET_DEFAULTS.items():
            shared.opts.data.setdefault(k, v)
    except Exception:
        print("[krea2] preset OPTS step skipped (the 'krea2' dropdown entry was still added):\n"
              + traceback.format_exc())

    # 3. Auto-pick TE+VAE when the preset is selected (non-critical).
    try:
        _patch_preset_auto_modules()
    except Exception:
        pass


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

        # Krea2's DiT is built with an explicit ComfyUI-style `operations=` class, which bypasses the
        # per-format ops selection Forge's normal loader does. So we must pick the matching quant ops
        # ourselves — otherwise every quantized checkpoint (INT8-ConvRot, GGUF, ...) loads with plain
        # ops, the quant metadata (weight_scale/comfy_quant, or the gguf blocks) is dropped, and you
        # get a black/NaN image or a load error. This mirrors backend/loader.py's format handling.
        import backend.operations as _ops
        load_device = L.memory_management.get_torch_device()
        params = L.utils.calculate_parameters(state_dict)
        sdtype = L.utils.weight_dtype(state_dict)          # a torch dtype, or the string "gguf"/"nf4"/"fp4"

        # comfy_quant MIXED-PRECISION checkpoints (NVFP4 / mxfp8 / per-layer mixed). These carry a
        # per-tensor `.weight_scale_2` that INT8-ConvRot does NOT — but they ALSO carry `.weight_scale`,
        # so this MUST be tested BEFORE the int8 branch, or an NVFP4 model is misrouted to int8 ops and
        # dies at the first un-quantized layer with
        #   "RuntimeError: self and mat2 must have the same dtype, but got Float and BFloat16".
        # Forge already knows how to build the right ops from the comfy_quant descriptors, so instead of
        # hand-picking a class we ask it for the same MixedPrecisionOps its native loader would use.
        _quant_cfg = None
        if any(k.endswith(".weight_scale_2") for k in state_dict):
            try:
                from backend.state_dict import detect_quantization
                _quant_cfg = detect_quantization(state_dict, is_unet=True)
            except Exception:
                _quant_cfg = None

        is_mixed = _quant_cfg is not None
        is_int8_convrot = (not is_mixed) and any(k.endswith(".weight_scale") for k in state_dict)
        is_gguf = (sdtype == "gguf")

        if is_mixed:
            from backend.operations_mixed_precision import mixed_precision_ops
            _mm = L.memory_management
            _cdt = torch.bfloat16 if _mm.should_use_bf16(load_device) else torch.float32
            _disabled = set()
            for _flag, _names in ((("supports_nvfp4_compute"), ("nvfp4",)),
                                  (("supports_mxfp8_compute"), ("mxfp8",)),
                                  (("supports_fp8_compute"), ("float8_e4m3fn", "float8_e5m2"))):
                _fn = getattr(_mm, _flag, None)
                if _fn is not None and not _fn(load_device):
                    _disabled.update(_names)
            _cfg = dict(_quant_cfg)
            _full = _cfg.pop("TE", False)
            _dit_ops = mixed_precision_ops(quant_config=_cfg, compute_dtype=_cdt,
                                           full_precision_mm=_full, disabled=_disabled)
            storage_dtype = torch.bfloat16
            _tag = ("comfy_quant (NVFP4/mixed) -> mixed-precision ops"
                    + (f"; unsupported on this GPU, dequantized: {sorted(_disabled)}" if _disabled else ""))
        elif is_int8_convrot:
            # int8 weights + per-row weight_scale + comfy_quant descriptor; ForgeOperationsInt8 applies
            # the dequant + ConvRot. dynamic_quantize=False keeps the deliberately-fp8 leftover layers
            # (txtfusion / first / last / projections) at their stored precision instead of re-int8-ing.
            _dit_ops = _ops.ForgeOperationsInt8
            _dit_ops.excluded_names = []
            _dit_ops.dynamic_quantize = False
            storage_dtype = torch.bfloat16
            _tag = "INT8-ConvRot -> int8 ops (per-row scale + ConvRot)"
        elif is_gguf:
            # gguf weights are quantized blocks dequantized on the fly by the gguf Linear.
            _dit_ops = _ops.ForgeOperationsGGUF
            storage_dtype = "gguf"
            _tag = "GGUF -> gguf ops (dequant on the fly)"
        else:
            _dit_ops = ForgeOperations
            storage_dtype = sdtype if sdtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16
            _tag = None

        _wd = storage_dtype if isinstance(storage_dtype, torch.dtype) else torch.bfloat16
        comp_dtype = L.memory_management.inference_cast(weight_dtype=_wd, inference_device=load_device, supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
        init_device = L.memory_management.unet_initial_load_device(parameters=params, dtype=_wd)
        if _tag:
            print(f"[krea2] {_tag}")

        with L.no_init_weights():
            if is_gguf:
                # move to device but NEVER cast dtype — a .to(dtype) would dequantize the gguf blocks.
                with L.using_forge_operations(device=init_device, dtype=comp_dtype, manual_cast_enabled=False, bnb_dtype="gguf"):
                    model = SingleStreamDiT(**unet_config, operations=_dit_ops).to(device=init_device)
            elif is_mixed:
                # Like gguf: move to device, never blanket-cast dtype (the packed 4-bit weights and
                # their fp8/fp32 scales must keep their stored dtypes). manual_cast handles the
                # un-quantized leftovers (first / last / txtfusion / projectors) at forward time.
                with L.using_forge_operations(device=init_device, dtype=comp_dtype,
                                              manual_cast_enabled=True, bnb_dtype=dict(_quant_cfg)):
                    model = SingleStreamDiT(**unet_config, operations=_dit_ops).to(device=init_device)
            else:
                # INT8-ConvRot keeps some layers in fp8/bf16 (txtfusion / first / last / projectors).
                # Those non-int8 weights must be cast to the activation dtype at forward — the DiT
                # feeds fp32 latents into `first`, so an fp8 weight x fp32 input crashes F.linear.
                # Forcing manual_cast makes them cast, exactly like the working fp8 checkpoint does
                # (its storage fp8 != compute, so it already casts). Plain fp8/bf16 models: cast only
                # when storage != compute, as before.
                need_cast = is_int8_convrot or (storage_dtype != comp_dtype)
                to_args = dict(device=init_device, dtype=storage_dtype)
                with L.using_forge_operations(**to_args, manual_cast_enabled=need_cast):
                    model = SingleStreamDiT(**unet_config, operations=_dit_ops).to(**to_args)
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

    # --- 6. UI preset 'krea2' is registered INDEPENDENTLY at the module bottom, so it appears in
    #        the dropdown even if an earlier arch step hits a Forge-version snag. ---

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
            low = p.lower()
        except Exception:
            return False
        try:
            if low.endswith((".safetensors", ".sft")):
                with open(p, "rb") as f:
                    n = _struct.unpack("<Q", f.read(8))[0]
                    keys = list(_json.loads(f.read(n)).keys())
                has_krea2 = any(("blocks.0.mod.lin" in k) or ("txtfusion.projector" in k) for k in keys)
                has_te = any(k.startswith("text_encoders.") or (".language_model." in k) for k in keys)
                return has_krea2 and not has_te
            if low.endswith(".gguf"):
                # GGUF stores tensor names as plain strings near the file start; sniff the krea2 DiT
                # fingerprint (+ absence of a bundled TE) without a full GGUF parse. This lets a bare
                # GGUF checkpoint (e.g. Muse Q8/Q4) auto-attach its TE+VAE just like the safetensors one.
                with open(p, "rb") as f:
                    head = f.read(16 << 20)
                has_krea2 = (b"txtfusion.projector" in head) or (b"blocks.0.mod.lin" in head)
                has_te = (b"language_model" in head) or (b"text_encoders." in head)
                return has_krea2 and not has_te
        except Exception:
            return False
        return False

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
    print("[krea2] arch registration FAILED (Forge boot unaffected):\n" + traceback.format_exc())

# Register the UI preset INDEPENDENTLY of the arch registration above — it's a pure UI convenience
# and should show in the dropdown even if the arch step hit a version-specific snag. Idempotent.
try:
    _register_preset()
except Exception:
    print("[krea2] preset registration skipped:\n" + traceback.format_exc())
