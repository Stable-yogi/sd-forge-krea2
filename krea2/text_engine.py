"""
Krea 2 text engine — Qwen3-VL-4B with the 12-layer tap.

Subclasses Forge's Qwen3TextProcessingEngine (Z-Image path): sets the 12-layer
list tap (Forge's Llama2_.forward returns cat -> (B,12,seq,2560)), applies Krea's
system+user template, strips the template prefix, and flattens the 12-layer axis
into the feature dim -> (B, seq, 12*2560). The DiT's `txtfusion` adapter unpacks it.

Logic ported from ComfyUI comfy/text_encoders/krea2.py.
"""
import torch
from backend.text_processing.qwen3_engine import Qwen3TextProcessingEngine

KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
KREA2_TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
                  "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                  "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")

_IM_START = 151644
_USER = 872
_NL = 198


class Krea2TextProcessingEngine(Qwen3TextProcessingEngine):
    def __init__(self, text_encoder, tokenizer):
        super().__init__(text_encoder, tokenizer)
        self.llama_template = KREA2_TEMPLATE
        self.intermediate_output = list(KREA2_TAP_LAYERS)   # list tap -> (B,12,seq,2560)

    def process_tokens(self, batch_tokens, batch_multipliers):
        z = super().process_tokens(batch_tokens, batch_multipliers)   # (B, 12, seq, 2560)
        if z.dim() != 4:
            return z   # safety: unexpected shape, pass through

        # Strip the system + "user\n" prefix (find the 2nd <|im_start|>, then skip user + newline).
        toks = [int(t) for t in batch_tokens[0]]
        template_end = -1
        count = 0
        for i, t in enumerate(toks):
            if t == _IM_START and count < 2:
                template_end = i
                count += 1
        if template_end >= 0 and z.shape[2] > template_end + 3:
            if len(toks) > template_end + 2 and toks[template_end + 1] == _USER and toks[template_end + 2] == _NL:
                template_end += 3
        if template_end > 0:
            z = z[:, :, template_end:, :]

        # Flatten the 12-layer axis into the feature dim: (B, seq, 12*2560).
        b, n, seq, h = z.shape
        return z.permute(0, 2, 1, 3).reshape(b, seq, n * h)
