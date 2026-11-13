"""
Video Enhancer – upscales and restores frames to Full HD (1920×1080).

Pipeline:
  1. Real-ESRGAN  → 2× super-resolution on every frame
  2. GFPGAN       → face restoration (skipped when no face in source)
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


def _has_face(image_path: str) -> bool:
    """Quick check: does the source image contain a human face?"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
        return len(faces) > 0
    except Exception:
        return False


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
        source_image: str = None,
    ) -> str:
        """Enhance video to Full HD. Returns output path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Fast path: if input is already at or near target resolution, just re-encode
        import subprocess as _sp2
        _probe = _sp2.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True,
        )
        _dims = _probe.stdout.strip().split()
        in_w = int(_dims[0]) if len(_dims) > 0 else 0
        in_h = int(_dims[1]) if len(_dims) > 1 else 0
        if in_w >= self.target_w * 0.9 and in_h >= self.target_h * 0.9:
            logger.info(f"Input already {in_w}x{in_h} — skipping ESRGAN/GFPGAN, re-encoding only")
            import subprocess as _sp
            _sp.run([
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", output_path,
            ], check=True, capture_output=True)
            return output_path

        pipeline = self.cfg.get("pipeline", ["realesrgan", "gfpgan"])

        # Decide upfront whether GFPGAN is worth running (needs a face)
        run_gfpgan = "gfpgan" in pipeline
        if run_gfpgan and source_image:
            run_gfpgan = _has_face(source_image)
            if not run_gfpgan:
                logger.info("No face in source image — skipping GFPGAN for speed")

        # Extract frames as JPEG (much faster I/O than PNG)
        frames_dir = Path(tempfile.mkdtemp(prefix="enhance_frames_"))
        fps = self._extract_frames(video_path, frames_dir)

        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        logger.info(f"Enhancing {len(frame_paths)} frames with pipeline: {pipeline}")

        enhanced_dir = Path(tempfile.mkdtemp(prefix="enhanced_frames_"))

        for i, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            if "realesrgan" in pipeline:
                frame = self._apply_realesrgan(frame)

            if run_gfpgan:
                frame = self._apply_gfpgan(frame)

            # Resize to exact target resolution
            frame = self._resize_to_target(frame)

            # Write as JPEG for speed; FFmpeg reads them fine
            out_path = enhanced_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if (i + 1) % 50 == 0:
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
            # JPEG quality 95: ~10× smaller than PNG, negligible quality loss
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            idx += 1
        cap.release()
        logger.debug(f"Extracted {idx} frames at {fps:.1f} fps")
        return fps

    # ── Real-ESRGAN ───────────────────────────────────────────────────────────

    def _load_realesrgan(self):
        if self._realesrgan is not None:
            return

        cfg = self.cfg.get("realesrgan", {})

        # Prefer lighter x2plus model (faster); fall back to x4plus at outscale=2
        x2_path = ROOT / "models" / "realesrgan" / "RealESRGAN_x2plus.pth"
        x4_path = ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth"
        if x2_path.exists():
            model_path, scale = x2_path, 2
        elif x4_path.exists():
            model_path, scale = x4_path, 4
        else:
            logger.warning("No Real-ESRGAN weights found. Upscaling via bicubic.")
            self._realesrgan = None
            return

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            nb = 23 if scale == 4 else 23  # x2plus also uses 23 blocks
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=nb, num_grow_ch=32, scale=scale,
            )
            dev = self.device
            self._realesrgan = RealESRGANer(
                scale=scale,
                model_path=str(model_path),
                model=model,
                tile=cfg.get("tile", 256),   # smaller tile = less VRAM, faster on MPS
                tile_pad=cfg.get("tile_pad", 10),
                pre_pad=cfg.get("pre_pad", 0),
                half=(dev != "cpu"),
                device=dev,
            )
            logger.info(f"Real-ESRGAN {scale}x loaded from {model_path.name}")
        except Exception as e:
            logger.warning(f"Real-ESRGAN load failed: {e}. Upscaling via bicubic.")
            self._realesrgan = None

    def _apply_realesrgan(self, frame: np.ndarray) -> np.ndarray:
        self._load_realesrgan()
        if self._realesrgan is None:
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        try:
            # outscale matches model scale (2 or 4 depending on which weights exist)
            out, _ = self._realesrgan.enhance(frame, outscale=self._realesrgan.scale)
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
        src_ratio = w / h
        tgt_ratio = self.target_w / self.target_h

        if src_ratio >= tgt_ratio:
            # Wide or square → scale to target height, crop width
            scale = self.target_h / h
            new_w, new_h = int(w * scale), self.target_h
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            x_off = (new_w - self.target_w) // 2
            return resized[:, x_off:x_off + self.target_w]
        else:
            # Portrait → scale to target width, crop height
            scale = self.target_w / w
            new_w, new_h = self.target_w, int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            y_off = (new_h - self.target_h) // 2
            return resized[y_off:y_off + self.target_h, :]

    def _frames_to_video(self, frames_dir: Path, output_path: str, fps: float):
        import subprocess
        frame_pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
