"""
Face Animator – animates a still portrait image to match an audio track.

Primary:   SadTalker (OpenTalker/SadTalker) – best quality single-image talking head
Fallback:  Wav2Lip – alternative lip-sync engine
"""
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger
from utils.gpu_utils import clear_gpu_cache

logger = setup_logger(__name__)


def _preprocess_image(image_path: str, out_path: str) -> str:
    """
    Normalise an input image for SadTalker:
    - Convert to RGB (drop alpha, handle palette modes)
    - Resize so the longest edge is 512 px (SadTalker default output_size)
    - Save as PNG

    Returns out_path so callers can chain: image = _preprocess_image(src, dst)
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        max_dim = 512
        w, h = img.size
        scale = min(max_dim / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(out_path)
        logger.info(f"Pre-processed image saved to {out_path} ({img.size})")
        return out_path
    except Exception as exc:
        logger.warning(f"Image pre-processing failed ({exc}); using original")
        return image_path

ROOT = Path(__file__).parent.parent
SADTALKER_DIR = ROOT / "third_party" / "SadTalker"
WAV2LIP_DIR   = ROOT / "third_party" / "Wav2Lip"


class FaceAnimator:
    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg = cfg.get("face_animation", {})
        self.device = device
        self.backend = self.cfg.get("backend", "sadtalker")
        self._sadtalker_disabled = False  # set True after first failure; skip for rest of job

    def _assert_face_detected(self, image_path: str):
        """
        Raise ValueError if no high-confidence frontal face is found.
        Uses stricter thresholds than a basic Haar check so we avoid calling
        SadTalker on images its internal landmark detector will reject.
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Strict params: minNeighbors=5 reduces false positives (cartoons, etc.)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                raise ValueError(
                    "No high-confidence frontal face detected — routing to FOMM. "
                    "SadTalker requires a photo with a clearly visible, forward-facing human face."
                )

            # Additional check: face must cover at least 5% of the image area
            h, w = gray.shape
            img_area = h * w
            largest_face_area = max(fw * fh for (_, _, fw, fh) in faces)
            if largest_face_area < img_area * 0.05:
                raise ValueError(
                    f"Detected face is too small ({largest_face_area/img_area:.1%} of image) — routing to FOMM."
                )

            logger.info(f"Face pre-check passed — {len(faces)} face(s) detected")
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Face pre-check skipped: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def animate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str = "outputs/temp_raw.mp4",
        progress_cb=None,
    ) -> str:
        """Animate character image with audio. Auto-falls back to FOMM for non-human images."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try SadTalker first; fall back to FOMM on face-check failure or subprocess error.
        # Once SadTalker fails once, _sadtalker_disabled is set so subsequent images in the
        # same job skip the attempt entirely.
        if SADTALKER_DIR.exists() and not self._sadtalker_disabled:
            try:
                self._assert_face_detected(image_path)
                return self._animate_sadtalker(image_path, audio_path, output_path, progress_cb)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"SadTalker skipped ({e}). Falling back to FOMM.")
                self._sadtalker_disabled = True

        if self.backend == "wav2lip" and WAV2LIP_DIR.exists():
            return self._animate_wav2lip(image_path, audio_path, output_path)

        return self._animate_fomm(image_path, audio_path, output_path)

    # ── SadTalker ─────────────────────────────────────────────────────────────

    def _animate_sadtalker(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        progress_cb=None,
    ) -> str:
        """Call SadTalker inference script via subprocess."""
        st_cfg = self.cfg.get("sadtalker", {})
        checkpoints_dir = str(ROOT / "models" / "sadtalker" / "checkpoints")

        # Validate face is detectable before invoking SadTalker
        self._assert_face_detected(image_path)

        # Pre-process the image so SadTalker always receives a clean 512-px RGB PNG
        result_dir = str(ROOT / "outputs" / "sadtalker_tmp")
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        preprocessed_img = str(ROOT / "outputs" / "sadtalker_tmp" / "input_preprocessed.png")
        image_path = _preprocess_image(image_path, preprocessed_img)

        cmd = [
            sys.executable,
            str(SADTALKER_DIR / "inference.py"),
            "--source_image",     str(image_path),
            "--driven_audio",     str(audio_path),
            "--result_dir",       result_dir,
            "--checkpoint_dir",   checkpoints_dir,
            "--pose_style",       str(st_cfg.get("pose_style", 0)),
            "--expression_scale", str(st_cfg.get("expression_scale", 1.0)),
            "--preprocess",       st_cfg.get("preprocess", "resize"),
            "--size",             str(st_cfg.get("output_size", 512)),
        ]

        if st_cfg.get("still_mode", False):
            cmd.append("--still")

        # Device flag
        if "cuda" in self.device:
            pass  # default is CUDA if available
        elif self.device == "mps":
            # MPS fallback: let PyTorch use MPS where supported, CPU elsewhere
            pass  # no --cpu flag; PYTORCH_ENABLE_MPS_FALLBACK handles unsupported ops
        else:
            cmd.extend(["--cpu"])

        import re as _re

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SADTALKER_DIR) + ":" + env.get("PYTHONPATH", "")
        if self.device == "mps":
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTHONUNBUFFERED"] = "1"  # ensure subprocess stdout is not buffered

        logger.info(f"Running SadTalker: {' '.join(cmd[-6:])}")

        # Stream output so we can parse tqdm progress lines for the UI
        proc = subprocess.Popen(
            cmd,
            cwd=str(SADTALKER_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if progress_cb and "Face Renderer" in line:
                m = _re.search(r"(\d+)/(\d+)", line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    if total > 0:
                        progress_cb(done, total)

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                "SadTalker failed — most likely cause: no frontal face detected in the image. "
                "Use a photo where the face is clearly visible and roughly facing the camera. "
                f"(return code {proc.returncode})"
            )

        # Find generated video
        generated = list(Path(result_dir).glob("**/*.mp4"))
        if not generated:
            raise RuntimeError(f"SadTalker produced no .mp4 file in {result_dir}")

        # Move to final location
        shutil.move(str(generated[-1]), output_path)
        # Clean temp dir
        shutil.rmtree(result_dir, ignore_errors=True)

        clear_gpu_cache()
        logger.info(f"SadTalker output → {output_path}")
        return output_path

    # ── Ken Burns (slow zoom/pan) — best for scene illustrations, cartoons ──────

    # Four distinct motion styles — rotated per image so each segment looks different
    _KB_MOTIONS = [
        # slow zoom in from center
        "z='min(zoom+0.0003,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        # slow zoom out from center
        "z='if(eq(on,1),1.12,max(zoom-0.0003,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        # zoom in anchored to bottom-left (characters often bottom-centre)
        "z='min(zoom+0.0003,1.12)':x='iw*(1-1/zoom)*0.3':y='ih*(1-1/zoom)'",
        # zoom in anchored to top-right
        "z='min(zoom+0.0003,1.12)':x='iw*(1-1/zoom)':y='0'",
    ]

    def _animate_ken_burns(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        motion_idx: int = 0,
    ) -> str:
        """
        Ken Burns slow-zoom/pan effect for the full duration of audio_path.
        Far better than FOMM for scene illustrations — no warping artifacts.
        Runs entirely via FFmpeg (fast, no ML).
        """
        # Get audio duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip() or "5")
        fps = 25
        n_frames = max(1, int(duration * fps))

        motion = self._KB_MOTIONS[motion_idx % len(self._KB_MOTIONS)]

        # Scale source to 2160×2160 so zoompan has room to work at any aspect ratio.
        # The zoompan s=WxH crops out 1920×1080, covering the image crop-fill style.
        vf = (
            "scale=2160:2160:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop=2160:2160,"
            f"zoompan={motion}:d={n_frames}:fps={fps}:s=1920x1080"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", image_path,
            "-i", audio_path,
            "-filter_complex", f"[0:v]{vf}[v]",
            "-map", "[v]", "-map", "1:a",
            "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ken Burns FFmpeg failed:\n{result.stderr[-600:]}")
        logger.info(f"Ken Burns ({motion_idx % len(self._KB_MOTIONS)}) → {output_path}")
        return output_path

    def animate_bulk_ken_burns(self, image_audio_pairs: list) -> list:
        """
        Animate each image: tries SVD (actual motion) first, falls back to Ken Burns.
        image_audio_pairs: [(image_path, audio_path, output_path), ...]
        """
        # Lazy-init SVD animator once for the whole batch
        svd = self._get_svd_animator()

        output_paths = []
        for idx, (img, audio, out) in enumerate(image_audio_pairs):
            if svd is not None:
                try:
                    logger.info(f"SVD animating image {idx+1}/{len(image_audio_pairs)}…")
                    svd.animate(img, audio, out)
                    output_paths.append(out)
                    continue
                except Exception as e:
                    logger.warning(f"SVD failed for image {idx}: {e} — falling back to Ken Burns")

            # Ken Burns fallback
            self._animate_ken_burns(img, audio, out, motion_idx=idx)
            output_paths.append(out)
            logger.info(f"Ken Burns done [{idx+1}/{len(image_audio_pairs)}]")
        return output_paths

    def _get_svd_animator(self):
        """Return a loaded SVDAnimator, or None if SVD is unavailable."""
        if hasattr(self, "_svd"):
            return self._svd
        try:
            from modules.video_animator import SVDAnimator
            self._svd = SVDAnimator(device=self.device)
            self._svd._load()   # pre-load model once for the batch
            return self._svd
        except Exception as e:
            logger.warning(f"SVD unavailable ({e}) — will use Ken Burns")
            self._svd = None
            return None

    # Keep animate_bulk_fomm as an alias so old call sites don't break
    def animate_bulk_fomm(self, image_audio_pairs: list, output_dir: str = "") -> list:
        return self.animate_bulk_ken_burns(image_audio_pairs)

    def _animate_fomm(self, image_path: str, audio_path: str, output_path: str) -> str:
        """Single-image: try SVD, fall back to Ken Burns."""
        svd = self._get_svd_animator()
        if svd is not None:
            try:
                return svd.animate(image_path, audio_path, output_path)
            except Exception as e:
                logger.warning(f"SVD failed: {e} — Ken Burns fallback")
        return self._animate_ken_burns(image_path, audio_path, output_path)

    # ── Wav2Lip ───────────────────────────────────────────────────────────────

    def _animate_wav2lip(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Call Wav2Lip inference script via subprocess."""
        w2l_cfg = self.cfg.get("wav2lip", {})
        model_path = str(ROOT / "models" / "wav2lip" / "wav2lip_gan.pth")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Wav2Lip model not found at {model_path}. "
                "Download from: https://github.com/Rudrabha/Wav2Lip#getting-the-weights"
            )

        pads = w2l_cfg.get("pads", [0, 10, 0, 0])
        cmd = [
            sys.executable,
            str(WAV2LIP_DIR / "inference.py"),
            "--checkpoint_path", model_path,
            "--face",           str(image_path),
            "--audio",          str(audio_path),
            "--outfile",        str(output_path),
            "--pads",           *[str(p) for p in pads],
            "--resize_factor",  str(w2l_cfg.get("resize_factor", 1)),
        ]
        if w2l_cfg.get("nosmooth", False):
            cmd.append("--nosmooth")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(WAV2LIP_DIR) + ":" + env.get("PYTHONPATH", "")

        logger.info("Running Wav2Lip...")
        result = subprocess.run(
            cmd,
            cwd=str(WAV2LIP_DIR),
            env=env,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Wav2Lip failed with return code {result.returncode}")

        clear_gpu_cache()
        logger.info(f"Wav2Lip output → {output_path}")
        return output_path
