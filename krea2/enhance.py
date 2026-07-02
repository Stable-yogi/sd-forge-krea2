"""
Krea 2 Detail Boost — per-layer conditioning rebalance, applied inside the DiT forward.

Krea 2 conditions on a 12-layer Qwen3-VL tap. The deep taps carry fine detail / identity /
texture that alignment training under-weights; boosting them sharpens results. Raw boosting
inflates the conditioning magnitude and wrecks likeness/colour, so by default we RMS-renormalise:
shift the *ratios* between taps while holding the overall magnitude constant.

Technique adapted from huwhitememes/comfyui-krea2-conditioning (Apache-2.0), itself a fork of
nova452/ComfyUI-ConditioningKrea2Rebalance (Apache-2.0). Credit to both authors.

Applied in-model (on the unpacked (B, seq, 12, 2560) context) rather than on the cached cond,
so toggling it always takes effect on the next generation — no stale-cond-cache issues.
"""
from __future__ import annotations

from typing import List, Optional

import torch

# Preset per-layer gains, shallow -> deep, matching the tap order [2,5,8,...,35].
PRESET_WEIGHTS: dict = {
    "balanced": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0],
    "detail":   [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.2, 3.0, 6.0, 1.5, 5.0, 1.2],
    "subtle":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.5, 1.0],
}
PRESET_NAMES = ["balanced", "detail", "subtle", "custom"]

# Runtime config, set per-generation by the UI script and cleared afterwards.
# {"weights": [12 floats], "strength": float, "renormalize": bool} or None (off).
CONFIG: dict = {"detail_boost": None}


def parse_weights(text: str) -> Optional[List[float]]:
    try:
        parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
        vals = [float(p) for p in parts]
        return vals if len(vals) >= 2 else None
    except Exception:
        return None


def maybe_detail_boost(context: torch.Tensor) -> torch.Tensor:
    """Apply the configured per-layer rebalance to the unpacked (B, seq, N, D) context.
    No-op (and zero overhead) when disabled or on shape mismatch."""
    cfg = CONFIG.get("detail_boost")
    if not cfg:
        return context
    try:
        weights = cfg.get("weights")
        if not weights or context.dim() != 4 or context.shape[2] != len(weights):
            return context
        strength = float(cfg.get("strength", 1.0))
        if strength == 0.0:
            return context
        renormalize = bool(cfg.get("renormalize", True))

        orig_dtype = context.dtype
        x = context.float()
        # interpolate gains toward 1.0 by strength: eff = 1 + s*(w-1)
        gains = torch.tensor(weights, dtype=x.dtype, device=x.device)
        gains = 1.0 + strength * (gains - 1.0)

        if renormalize:
            ref_rms = x.pow(2).mean(dim=(1, 2, 3)).sqrt()          # (B,)

        x = x * gains.view(1, 1, -1, 1)

        if renormalize:
            new_rms = x.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
            x = x * (ref_rms / new_rms).view(-1, 1, 1, 1)

        return x.to(orig_dtype)
    except Exception:
        return context
