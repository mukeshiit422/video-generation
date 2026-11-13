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
    "none":           "",
}

NEGATIVE_BASE = (
    "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, "
    "extra limbs, poorly drawn face, poorly drawn hands, missing fingers, "
    "watermark, signature, text, logo"
)


class ImageGenerator:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg.get("image_generation", {})
        self.device = device
        self._pipe = None
        self._current_model = None
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
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate an image and return its file path."""

        # Build full prompt with style prefix
        style_prefix = STYLE_PRESETS.get(style, "")
        full_prompt = f"{style_prefix}, {prompt}".strip(", ") if style_prefix else prompt
        full_negative = f"{NEGATIVE_BASE}, {negative_prompt}".strip(", ") if negative_prompt else NEGATIVE_BASE

        logger.info(f"Generating image: '{prompt[:60]}…' | {width}×{height} | {steps} steps")

        pipe = self._load_pipeline(model_id)

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

        # Apply GFPGAN face restoration then Real-ESRGAN upscaling
        image = self._restore_faces(image)
        image = self._upscale(image)

        # Save
        if output_path is None:
            out_dir = ROOT / "outputs" / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(out_dir / f"img_{ts}.png")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
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
                model_rootpath=str(ROOT / "models" / "gfpgan"),
            )

            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            _, _, restored = restorer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
            logger.info("GFPGAN face restoration applied")
            return Image.fromarray(restored_rgb)
        except Exception as e:
            logger.warning(f"Face restoration skipped: {e}")
            return image

    def _upscale(self, image):
        """Run Real-ESRGAN 4× upscale on a PIL image. Returns original if unavailable."""
        try:
            import numpy as np
            import cv2
            from PIL import Image
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_path = ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth"
            if not model_path.exists():
                return image

            dev = self.device if self.device != "mps" else "cpu"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=(dev != "cpu"),
                device=dev,
            )

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
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            # Faster scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

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
        logger.info("Model loaded")
        return pipe
