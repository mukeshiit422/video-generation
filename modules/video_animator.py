"""
SVD Animator – uses Stable Video Diffusion to generate actual motion video
from a still image. Works with any image type: human, cartoon, animal, scene.

Model: stabilityai/stable-video-diffusion-img2vid-xt (25 frames, fp16)
Falls back to Ken Burns if model unavailable or OOM.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from utils.logger import setup_logger
from utils.gpu_utils import clear_gpu_cache

logger = setup_logger(__name__)

ROOT = Path(__file__).parent.parent
SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"


class SVDAnimator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableVideoDiffusionPipeline

        logger.info("Loading Stable Video Diffusion…")
        self._pipe = StableVideoDiffusionPipeline.from_pretrained(
            SVD_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        # Memory optimisations — critical for M2 unified memory
        try:
            self._pipe.enable_vae_slicing()          # older diffusers
        except AttributeError:
            try:
                self._pipe.vae.enable_slicing()      # newer diffusers
            except Exception:
                pass
        try:
            self._pipe.unet.enable_forward_chunking()
        except Exception:
            pass
        try:
            self._pipe.enable_model_cpu_offload()    # offload between MPS/CPU as needed
        except Exception:
            target = self.device if self.device in ("cuda", "mps") else "cpu"
            self._pipe = self._pipe.to(target)
        logger.info(f"SVD loaded on {self.device}")

    def animate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        motion_bucket_id: int = 100,
        num_steps: int = 10,
    ) -> str:
        """
        Generate a looping animated video from a still image, then mux audio.
        The 25-frame clip is ping-pong looped to fill the full audio duration.
        """
        import torch

        self._load()

        # Get audio duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip() or "5")

        # Prepare image — SVD works at 1024×576 (landscape) or 576×1024 (portrait)
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w >= h:  # landscape / square
            target_size = (1024, 576)
        else:        # portrait
            target_size = (576, 1024)
        img = img.resize(target_size, Image.LANCZOS)

        logger.info(f"SVD: generating 25 frames @ {target_size} (motion={motion_bucket_id})…")
        with torch.inference_mode():
            result = self._pipe(
                img,
                num_frames=25,
                num_inference_steps=num_steps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=0.02,
                decode_chunk_size=4,
            )
        frames = result.frames[0]  # list of PIL Images, 25 frames
        clear_gpu_cache()

        # Ping-pong loop to fill audio duration
        fps = 7
        clip_len = len(frames) / fps  # ~3.5s
        looped = list(frames)
        while len(looped) / fps < duration + clip_len:
            looped.extend(reversed(frames))  # reverse
            looped.extend(frames)            # forward
        target_n = int(duration * fps) + 1
        looped = looped[:target_n]

        # Write silent clip
        tmp_silent = output_path + "_svd_silent.mp4"
        self._write_frames(looped, tmp_silent, fps)

        # Mux audio + upscale to 1920×1080
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_silent,
            "-i", audio_path,
            "-shortest",
            "-vf", "scale=1920:1080:flags=lanczos",
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ], check=True, capture_output=True)
        Path(tmp_silent).unlink(missing_ok=True)

        logger.info(f"SVD output → {output_path}")
        return output_path

    def _write_frames(self, frames: list, path: str, fps: int):
        """Write PIL frame list to MP4 via FFmpeg pipe."""
        import io, struct

        # Write to temp dir as individual JPEG then concat via ffmpeg
        with tempfile.TemporaryDirectory(prefix="svd_frames_") as td:
            td = Path(td)
            for i, f in enumerate(frames):
                f.save(str(td / f"f{i:06d}.jpg"), quality=92)

            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(td / "f%06d.jpg"),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-pix_fmt", "yuv420p",
                path,
            ], check=True, capture_output=True)
