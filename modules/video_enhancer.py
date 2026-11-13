"""
Video Enhancer – upscales and restores frames to Full HD (1920×1080).

Pipeline:
  1. Real-ESRGAN  → 4× super-resolution on every frame
  2. GFPGAN       → face restoration / detail recovery
  3. FFmpeg       → re-encode at target resolution
"""
import os
import cv2
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple

from utils.logger import setup_logger
from utils.gpu_utils import clear_gpu_cache

logger = setup_logger(__name__)

ROOT = Path(__file__).parent.parent


class VideoEnhancer:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg.get("video_enhancement", {})
        self.device = device
        self.target_w, self.target_h = self.cfg.get("target_resolution", [1920, 1080])
        self._realesrgan = None
        self._gfpgan = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def enhance(
        self,
        video_path: str,
        output_path: str = "outputs/temp_enhanced.mp4",
    ) -> str:
        """Enhance video to Full HD. Returns output path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        pipeline = self.cfg.get("pipeline", ["realesrgan", "gfpgan"])

        # Extract frames
        frames_dir = Path(tempfile.mkdtemp(prefix="enhance_frames_"))
        fps = self._extract_frames(video_path, frames_dir)

        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        logger.info(f"Enhancing {len(frame_paths)} frames with pipeline: {pipeline}")

        enhanced_dir = Path(tempfile.mkdtemp(prefix="enhanced_frames_"))

        for i, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            if "realesrgan" in pipeline:
                frame = self._apply_realesrgan(frame)

            if "gfpgan" in pipeline:
                frame = self._apply_gfpgan(frame)

            # Resize to exact target resolution
            frame = self._resize_to_target(frame)

            out_path = enhanced_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(out_path), frame)

            if (i + 1) % 25 == 0:
                logger.debug(f"  Enhanced {i+1}/{len(frame_paths)} frames")

        # Re-encode to video (no audio – audio is merged in VideoComposer)
        self._frames_to_video(enhanced_dir, output_path, fps)

        # Cleanup
        shutil.rmtree(str(frames_dir), ignore_errors=True)
        shutil.rmtree(str(enhanced_dir), ignore_errors=True)
        clear_gpu_cache()

        logger.info(f"Enhanced video → {output_path}")
        return output_path

    # ── Frame Extraction ──────────────────────────────────────────────────────

    def _extract_frames(self, video_path: str, out_dir: Path) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), frame)
            idx += 1
        cap.release()
        logger.debug(f"Extracted {idx} frames at {fps:.1f} fps")
        return fps

    # ── Real-ESRGAN ───────────────────────────────────────────────────────────

    def _load_realesrgan(self):
        if self._realesrgan is not None:
            return

        model_path = ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth"
        cfg = self.cfg.get("realesrgan", {})

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            dev = self.device

            self._realesrgan = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=cfg.get("tile", 512),
                tile_pad=cfg.get("tile_pad", 10),
                pre_pad=cfg.get("pre_pad", 0),
                half=(dev != "cpu"),
                device=dev,
            )
            logger.info("Real-ESRGAN loaded")
        except Exception as e:
            logger.warning(f"Real-ESRGAN load failed: {e}. Upscaling via bicubic.")
            self._realesrgan = None

    def _apply_realesrgan(self, frame: np.ndarray) -> np.ndarray:
        self._load_realesrgan()
        if self._realesrgan is None:
            # Fallback: bicubic 2×
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        try:
            out, _ = self._realesrgan.enhance(frame, outscale=4)
            return out
        except Exception as e:
            logger.warning(f"ESRGAN frame failed: {e}")
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # ── GFPGAN ───────────────────────────────────────────────────────────────

    def _load_gfpgan(self):
        if self._gfpgan is not None:
            return

        model_path = ROOT / "models" / "gfpgan" / "GFPGANv1.4.pth"
        cfg = self.cfg.get("gfpgan", {})

        try:
            from gfpgan import GFPGANer
            dev = self.device
            self._gfpgan = GFPGANer(
                model_path=str(model_path),
                upscale=1,  # already upscaled by ESRGAN
                arch=cfg.get("arch", "clean"),
                channel_multiplier=cfg.get("channel_multiplier", 2),
                bg_upsampler=None,
                device=dev,
                model_rootpath=str(model_path.parent),
            )
            logger.info("GFPGAN loaded")
        except Exception as e:
            logger.warning(f"GFPGAN load failed: {e}")
            self._gfpgan = None

    def _apply_gfpgan(self, frame: np.ndarray) -> np.ndarray:
        self._load_gfpgan()
        if self._gfpgan is None:
            return frame
        try:
            # GFPGAN expects BGR
            _, _, restored = self._gfpgan.enhance(
                frame, has_aligned=False, only_center_face=False, paste_back=True
            )
            return restored if restored is not None else frame
        except Exception as e:
            logger.warning(f"GFPGAN frame failed: {e}")
            return frame

    # ── Resize & Re-encode ────────────────────────────────────────────────────

    def _resize_to_target(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # Fit inside target, maintaining aspect ratio, then pad
        ratio = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        y_off = (self.target_h - new_h) // 2
        x_off = (self.target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def _frames_to_video(self, frames_dir: Path, output_path: str, fps: float):
        import subprocess
        frame_pattern = str(frames_dir / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-vf", f"scale={self.target_w}:{self.target_h}",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
