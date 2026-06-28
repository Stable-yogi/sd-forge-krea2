"""
Krea 2 diffusion engine — adapted from backend/diffusion_engine/zimage.py.
Qwen3-VL text encoder (12-layer tap) + Qwen-Image VAE + flow-matching (shift 1.15).
"""
import torch

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from .text_engine import Krea2TextProcessingEngine


class Krea2(ForgeDiffusionEngine):
    matched_guesses = []   # filled in by the register hook with the Krea2 model_list class

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={"qwen3": huggingface_components["text_encoder"]},
            tokenizer_dict={"qwen3": huggingface_components["tokenizer"]},
        )
        vae = VAE(model=huggingface_components["vae"], is_wan=True)
        k_predictor = PredictionDiscreteFlow(estimated_config)
        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config,
        )

        self.text_processing_engine = Krea2TextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3,
            tokenizer=clip.tokenizer.qwen3,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()
        self.is_wan = True
        # Krea2 RAW uses a fixed flow shift of 1.15 (sampling_settings). use_shift=True would
        # let the UI Distilled-CFG slider override it, so keep it False to lock 1.15.
        self.use_shift = False

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine.tokenize([prompt])[0])
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 2) * 2.0 - 1.0
        return sample.to(x)
