"""
Image Generator – text-to-image using Stable Diffusion (local, free).

Models (auto-downloaded from HuggingFace on first use):
  - runwayml/stable-diffusion-v1-5   (~4 GB, default, best compatibility)
  - stabilityai/sdxl-turbo            (~6 GB, faster inference, higher quality)
  - stabilityai/stable-diffusion-2-1  (~5 GB, alternative)
"""
import os
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils.logger import setup_logger
from utils.gpu_utils import get_device, clear_gpu_cache

logger = setup_logger(__name__)

ROOT = Path(__file__).parent.parent

STYLE_PRESETS = {
    "photorealistic": "RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
    "cinematic":      "cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    "portrait":       "professional portrait photography, studio lighting, sharp focus, 85mm lens, f/1.8",
    "anime":          "anime artwork, anime style, key visual, vibrant, studio anime, highly detailed",
    "artistic":       "concept art, digital painting, artstation, illustration, sharp, highly detailed",
    "3d_render":      "3d render, octane render, highly detailed, volumetric lighting, 8k",
    "cartoon":        "3d cartoon character, pixar style, cute, expressive, vibrant colors, smooth shading, soft lighting, highly detailed",
    "none":           "",
}

NEGATIVE_BASE = (
    "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, "
    "extra limbs, poorly drawn face, poorly drawn hands, missing fingers, "
    "watermark, signature, text, logo"
)

# Anime checkpoints respond to booru-style quality tags rather than photo terms.
NEGATIVE_ANIME = (
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, worst quality, "
    "low quality, jpeg artifacts, signature, watermark, text, blurry, nsfw"
)

SD_VAE_MSE = "stabilityai/sd-vae-ft-mse"          # sharper faces/eyes for SD 1.5 photo models
LCM_LORA_SD15 = "latent-consistency/lcm-lora-sdv1-5"  # Fast mode: ~6 steps instead of 30

# Per-model loading tweaks; models not listed load with defaults.
#   variant: download the fp16 weights (half the size) when running in fp16
#   vae:     replace the bundled VAE
#   anime:   anime negatives, anime upscaler, no GFPGAN
MODEL_PROFILES = {
    "Lykon/dreamshaper-8":                  {"variant": "fp16"},
    "SG161222/Realistic_Vision_V5.1_noVAE": {"vae": SD_VAE_MSE},
    "gsdf/Counterfeit-V2.5":                {"anime": True},
}

# Styles where the subject is drawn/rendered, not photographed. GFPGAN would
# paste a photographic face onto these, so face restoration is skipped.
DRAWN_STYLES = {"anime", "artistic", "cartoon", "3d_render"}


def _is_sd15_family(model_id: str) -> bool:
    m = model_id.lower()
    return not ("xl" in m or "turbo" in m or "stable-diffusion-2" in m)


class ImageGenerator:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg.get("image_generation", {})
        self.device = device
        self._pipe = None
        self._current_model = None
        self._default_scheduler = None
        self._fast = False      # whether LCM-LoRA is currently loaded into _pipe
        self._restorer = None   # GFPGAN, loaded lazily and reused
        self._upsamplers = {}   # Real-ESRGAN by weights name, loaded lazily and reused
        self._pooled = None
        self._negative_pooled = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        style: str = "photorealistic",
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        model_id: str = "Lykon/dreamshaper-8",
        output_path: Optional[str] = None,
        fast: bool = False,
    ) -> str:
        """Generate an image and return its file path."""
        profile = MODEL_PROFILES.get(model_id, {})
        is_anime = profile.get("anime", False)
        is_drawn = is_anime or style in DRAWN_STYLES

        # Build full prompt with style prefix
        style_prefix = STYLE_PRESETS.get(style, "")
        full_prompt = f"{style_prefix}, {prompt}".strip(", ") if style_prefix else prompt
        neg_base = NEGATIVE_ANIME if is_anime else NEGATIVE_BASE
        full_negative = f"{neg_base}, {negative_prompt}".strip(", ") if negative_prompt else neg_base

        pipe = self._load_pipeline(model_id)

        # LCM-LoRA only exists for SD 1.5-family models
        fast = fast and _is_sd15_family(model_id)
        self._set_fast_mode(pipe, fast)
        if fast:
            steps = 6 if steps > 8 else max(4, steps)
            guidance_scale = min(guidance_scale, 1.5)  # LCM needs low CFG

        logger.info(
            f"Generating image: '{prompt[:60]}…' | {width}×{height} | {steps} steps"
            f"{' | fast' if fast else ''} | {model_id}"
        )

        import torch
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        is_turbo = "turbo" in model_id.lower()
        is_xl = "xl" in model_id.lower() or "sdxl" in model_id.lower()

        # Encode prompts via compel so prompts longer than 77 tokens work correctly
        prompt_embeds, negative_embeds = self._encode_prompt(
            pipe, full_prompt, full_negative if not is_turbo else "", is_xl
        )

        kwargs = dict(width=width, height=height, generator=generator)
        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = prompt_embeds
            if not is_turbo and negative_embeds is not None:
                kwargs["negative_prompt_embeds"] = negative_embeds
            if is_xl:
                kwargs["pooled_prompt_embeds"] = self._pooled
                if not is_turbo and self._negative_pooled is not None:
                    kwargs["negative_pooled_prompt_embeds"] = self._negative_pooled
        else:
            # compel unavailable — fall back to raw string prompts
            kwargs["prompt"] = full_prompt
            if not is_turbo:
                kwargs["negative_prompt"] = full_negative

        if is_turbo:
            kwargs.update(num_inference_steps=max(1, steps // 10), guidance_scale=0.0)
        else:
            kwargs.update(num_inference_steps=steps, guidance_scale=guidance_scale)

        result = pipe(**kwargs)
        image = result.images[0]

        # GFPGAN face restoration (photos only), then Real-ESRGAN upscaling
        if not is_drawn:
            image = self._restore_faces(image)
        image = self._upscale(image, anime=is_drawn)

        # Save
        if output_path is None:
            out_dir = ROOT / "outputs" / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(out_dir / f"img_{ts}.png")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        # Return cached MPS/CUDA allocations to the OS; on an 8 GB Mac the
        # allocator otherwise holds GBs between jobs and pushes the system into swap.
        clear_gpu_cache()
        gc.collect()
        logger.info(f"Image saved → {output_path}")
        return output_path

    def _restore_faces(self, image):
        """Run GFPGAN face restoration on a PIL image. Returns original if unavailable."""
        try:
            import numpy as np
            import cv2
            from PIL import Image
            from gfpgan import GFPGANer

            model_path = ROOT / "models" / "gfpgan" / "GFPGANv1.4.pth"
            if not model_path.exists():
                return image

            restorer = GFPGANer(
                model_path=str(model_path),
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device if self.device != "mps" else "cpu",
            )
            # upscale=1: only restore faces here. Upscaling is Real-ESRGAN's job;
            # doubling here too would hand it 4× the pixels to process.
            if self._restorer is None:
                self._restorer = GFPGANer(
                    model_path=str(model_path),
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device if self.device != "mps" else "cpu",
                )
            restorer = self._restorer

            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            _, _, restored = restorer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
            logger.info("GFPGAN face restoration applied")
            return Image.fromarray(restored_rgb)
        except Exception as e:
            logger.warning(f"Face restoration skipped: {e}")
            return image

    def _upscale(self, image, anime: bool = False):
        """Run Real-ESRGAN 2× upscale on a PIL image. Returns original if unavailable.

        anime=True uses the anime_6B weights, which keep flat colours and clean
        line art instead of adding photographic texture.
        """
        try:
            import numpy as np
            import cv2
            from PIL import Image
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            weights = "RealESRGAN_x4plus_anime_6B" if anime else "RealESRGAN_x4plus"
            model_path = ROOT / "models" / "realesrgan" / f"{weights}.pth"
            if not model_path.exists():
                if not anime:
                    return image
                weights = "RealESRGAN_x4plus"
                model_path = ROOT / "models" / "realesrgan" / f"{weights}.pth"
                if not model_path.exists():
                    return image

            # MPS works for RRDBNet (~11s for 512→1024 on M2 vs ~6 min on CPU).
            # fp16 only on CUDA; on MPS use fp32 for reliable output.
            if weights not in self._upsamplers:
                import torch
                num_block = 6 if weights.endswith("anime_6B") else 23
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=4)
                self._upsamplers[weights] = RealESRGANer(
                    scale=4,
                    model_path=str(model_path),
                    model=model,
                    tile=512,
                    tile_pad=10,
                    pre_pad=0,
                    half=("cuda" in self.device),
                    device=torch.device(self.device),
                )
            upsampler = self._upsamplers[weights]

            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            upscaled_bgr, _ = upsampler.enhance(img_bgr, outscale=2)
            upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
            result_img = Image.fromarray(upscaled_rgb)
            logger.info(f"Real-ESRGAN upscaled {image.size} → {result_img.size}")
            return result_img
        except Exception as e:
            logger.warning(f"Real-ESRGAN upscaling skipped: {e}")
            return image

    # ── Prompt encoder (compel — handles >77 tokens) ──────────────────────────

    def _encode_prompt(self, pipe, prompt: str, negative_prompt: str, is_xl: bool):
        self._pooled = None
        self._negative_pooled = None
        try:
            from compel import Compel, ReturnedEmbeddingsType
            if is_xl:
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                )
                pos_embeds, self._pooled = compel([prompt])
                neg_embeds, self._negative_pooled = compel([negative_prompt])
            else:
                compel = Compel(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                )
                pos_embeds = compel([prompt])
                neg_embeds = compel([negative_prompt])

            [pos_embeds, neg_embeds] = compel.pad_conditioning_tensors_to_same_length(
                [pos_embeds, neg_embeds]
            )
            return pos_embeds, neg_embeds
        except Exception as e:
            logger.warning(f"compel encoding failed, falling back to raw prompt: {e}")
            return None, None

    # ── Pipeline loader ────────────────────────────────────────────────────────

    def _load_pipeline(self, model_id: str):
        if self._pipe is not None and self._current_model == model_id:
            return self._pipe

        # Unload previous model
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            clear_gpu_cache()
            gc.collect()

        import torch
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            AutoPipelineForText2Image,
            DPMSolverMultistepScheduler,
        )

        logger.info(f"Loading model: {model_id} (first run downloads from HuggingFace)…")

        is_xl = "xl" in model_id.lower() or "sdxl" in model_id.lower()
        is_turbo = "turbo" in model_id.lower()

        # float16 for CUDA and MPS (M2 supports fp16 well); float32 only for plain CPU
        dtype = torch.float32
        if "cuda" in self.device or self.device == "mps":
            dtype = torch.float16

        if is_turbo:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
            )
        elif is_xl:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        else:
            profile = MODEL_PROFILES.get(model_id, {})
            extra = {}
            if profile.get("variant") and dtype == torch.float16:
                extra["variant"] = profile["variant"]
            if profile.get("vae"):
                from diffusers import AutoencoderKL
                extra["vae"] = AutoencoderKL.from_pretrained(profile["vae"], torch_dtype=dtype)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                **extra,
            )
            # DPM++ 2M Karras: good quality at 20–30 steps. Set algorithm_type
            # explicitly — some checkpoints (DreamShaper) ship a DEIS config
            # whose settings would otherwise carry over and fail.
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                use_karras_sigmas=True,
            )

        # Memory optimisations
        if self.device == "mps":
            pipe = pipe.to("mps")
            pipe.enable_attention_slicing()
        elif "cuda" in self.device:
            pipe = pipe.to(self.device)
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
        else:
            pipe = pipe.to("cpu")
            pipe.enable_attention_slicing()

        self._pipe = pipe
        self._current_model = model_id
        self._default_scheduler = pipe.scheduler
        self._fast = False
        logger.info("Model loaded")
        return pipe

    def _set_fast_mode(self, pipe, fast: bool):
        """Load/unload LCM-LoRA and swap the scheduler to match."""
        if fast == self._fast:
            return
        if fast:
            from diffusers import LCMScheduler
            logger.info("Enabling fast mode (LCM-LoRA)…")
            try:
                # weight_name is required when HF is offline / served from cache
                pipe.load_lora_weights(
                    LCM_LORA_SD15, weight_name="pytorch_lora_weights.safetensors", adapter_name="lcm"
                )
            except Exception:
                pipe.unload_lora_weights()  # don't leave a half-loaded adapter behind
                raise
            pipe.scheduler = LCMScheduler.from_config(self._default_scheduler.config)
        else:
            pipe.unload_lora_weights()
            pipe.scheduler = self._default_scheduler
        self._fast = fast
