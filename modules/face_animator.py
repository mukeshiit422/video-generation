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

    # ── Public API ─────────────────────────────────────────────────────────────

    def animate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str = "outputs/temp_raw.mp4",
        progress_cb=None,
    ) -> str:
        """Animate character image with audio. progress_cb(done, total) called with frame counts."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "sadtalker" and SADTALKER_DIR.exists():
            return self._animate_sadtalker(image_path, audio_path, output_path, progress_cb)
        elif self.backend == "wav2lip" and WAV2LIP_DIR.exists():
            return self._animate_wav2lip(image_path, audio_path, output_path)
        else:
            if SADTALKER_DIR.exists():
                return self._animate_sadtalker(image_path, audio_path, output_path, progress_cb)
            elif WAV2LIP_DIR.exists():
                return self._animate_wav2lip(image_path, audio_path, output_path)
            else:
                raise RuntimeError(
                    "No face animation backend found. Run install.sh to clone SadTalker."
                )

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
            raise RuntimeError(f"SadTalker failed with return code {proc.returncode}")

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
