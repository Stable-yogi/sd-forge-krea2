"""
Krea 2 Detail Boost — opt-in per-layer conditioning rebalance (UI).

Off by default; enabling it emphasises the deep Qwen3-VL taps (fine detail / identity /
texture) while RMS-renormalising so the overall conditioning magnitude stays constant.
Technique credit: huwhitememes/comfyui-krea2-conditioning (Apache-2.0), fork of
nova452/ComfyUI-ConditioningKrea2Rebalance. See krea2/enhance.py.
"""
import os
import sys

import gradio as gr

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXT_ROOT not in sys.path:                      # scripts load alphabetically; don't rely on
    sys.path.insert(0, EXT_ROOT)                  # krea2_register having run first

from krea2 import enhance
from modules import scripts


class Krea2DetailBoost(scripts.Script):
    def title(self):
        return "Krea 2 Detail Boost"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Krea 2 Detail Boost", open=False):
            gr.Markdown(
                "Rebalances Krea 2's 12-layer text conditioning toward the deep, fine-detail "
                "taps — sharper identity/texture without oversaturating (RMS-safe). "
                "Only affects Krea 2 models. *Full Enhancement Suite (advanced prompt adherence "
                "+ more) free at [stableyogi.com](https://stableyogi.com).*"
            )
            enabled = gr.Checkbox(label="Enable", value=False)
            preset = gr.Dropdown(label="Preset", choices=["balanced", "detail", "subtle"], value="balanced")
            strength = gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
            renormalize = gr.Checkbox(
                label="RMS renormalize (recommended)", value=True,
                info="Hold overall conditioning magnitude constant — quality-preserving mode.",
            )
        return [enabled, preset, strength, renormalize]

    def process(self, p, enabled=False, preset="balanced", strength=1.0, renormalize=True):
        if not enabled or strength == 0.0:
            enhance.CONFIG["detail_boost"] = None
            return
        weights = enhance.PRESET_WEIGHTS.get(preset, enhance.PRESET_WEIGHTS["balanced"])
        enhance.CONFIG["detail_boost"] = {
            "weights": list(weights),
            "strength": float(strength),
            "renormalize": bool(renormalize),
        }
        p.extra_generation_params["Krea2 Detail Boost"] = f"{preset} (s={strength:g}, rms={'on' if renormalize else 'off'})"

    def postprocess(self, p, processed, *args):
        enhance.CONFIG["detail_boost"] = None
