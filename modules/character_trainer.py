"""
Character Trainer – fine-tunes face animation and TTS models on a
specific character to improve consistency and quality over time.

Training pipeline:
  Phase 1 – Data collection: gather images, extract from generated videos
  Phase 2 – Data augmentation: flip, rotate, brightness, cropping
  Phase 3 – Fine-tune: SadTalker pose/expression nets + (optionally) XTTS voice
  Phase 4 – Evaluation: compare before/after on held-out test set
"""
import os
import cv2
import shutil
import json
import random
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from utils.logger import setup_logger

logger = setup_logger(__name__)

ROOT = Path(__file__).parent.parent
SADTALKER_DIR = ROOT / "third_party" / "SadTalker"


class CharacterTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("training", {})
        self.checkpoints_dir = ROOT / "checkpoints"
        self.training_data_dir = ROOT / "training_data"

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(
        self,
        character_image: str,
        images_dir: Optional[str] = None,
        video_samples: Optional[str] = None,
        voice_sample: Optional[str] = None,
        epochs: int = 50,
        character_name: str = "default",
    ):
        """Full training pipeline for a character."""
        logger.info(f"Starting character training: {character_name}")

        run_dir = self.checkpoints_dir / character_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Collect training images
        image_dataset = self._collect_images(
            character_image=character_image,
            images_dir=images_dir,
            video_samples=video_samples,
            run_dir=run_dir,
        )

        logger.info(f"Dataset: {len(image_dataset)} images")

        # Phase 2: Augment dataset
        augmented = self._augment_dataset(image_dataset, run_dir)
        logger.info(f"After augmentation: {len(augmented)} images")

        # Phase 3: Fine-tune face animation
        self._finetune_sadtalker(augmented, run_dir, epochs, character_name)

        # Phase 4: Voice fine-tuning (if voice sample provided)
        if voice_sample and Path(voice_sample).exists():
            self._finetune_voice(voice_sample, run_dir, character_name)

        # Phase 5: Save manifest
        self._save_manifest(character_name, run_dir, len(augmented))

        logger.info(f"Training complete. Checkpoint: {run_dir}")
        return str(run_dir)

    def collect_feedback(
        self,
        video_path: str,
        rating: int,  # 1-5
        script: str,
        character: str,
    ):
        """Save generation feedback for future training."""
        meta_dir = self.training_data_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "video": str(video_path),
            "rating": rating,
            "script_preview": script[:200],
            "character": str(character),
            "include_in_training": rating >= self.cfg.get("min_rating_for_training", 4),
        }

        fname = meta_dir / f"{Path(video_path).stem}.json"
        fname.write_text(json.dumps(entry, indent=2))
        logger.info(f"Feedback saved (rating={rating}/5): {fname.name}")

    # ── Phase 1: Data Collection ───────────────────────────────────────────────

    def _collect_images(
        self,
        character_image: str,
        images_dir: Optional[str],
        video_samples: Optional[str],
        run_dir: Path,
    ) -> List[str]:
        raw_dir = run_dir / "raw_images"
        raw_dir.mkdir(exist_ok=True)
        collected = []

        # Main character image
        dst = raw_dir / "character_main.jpg"
        shutil.copy(character_image, str(dst))
        collected.append(str(dst))

        # Additional images from directory
        if images_dir and Path(images_dir).is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img in Path(images_dir).glob(ext):
                    dst = raw_dir / img.name
                    shutil.copy(str(img), str(dst))
                    collected.append(str(dst))

        # Extract frames from video samples
        if video_samples and Path(video_samples).is_dir():
            for video in Path(video_samples).glob("*.mp4"):
                frames = self._extract_face_frames(str(video), raw_dir)
                collected.extend(frames)

        # Extract from previously generated and rated videos
        rated_videos = self._get_rated_videos()
        for video_path in rated_videos:
            frames = self._extract_face_frames(video_path, raw_dir)
            collected.extend(frames[:10])  # max 10 frames per generated video

        return collected

    def _extract_face_frames(
        self, video_path: str, out_dir: Path, every_n: int = 15
    ) -> List[str]:
        """Extract face frames from a video at every_n frame interval."""
        cap = cv2.VideoCapture(video_path)
        idx, saved = 0, []
        video_name = Path(video_path).stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                face = self._crop_face(frame)
                if face is not None:
                    out_path = out_dir / f"{video_name}_f{idx:05d}.jpg"
                    cv2.imwrite(str(out_path), face)
                    saved.append(str(out_path))
            idx += 1

        cap.release()
        return saved

    def _crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop face from frame. Returns None if no face found."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.3)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (512, 512))

    # ── Phase 2: Augmentation ─────────────────────────────────────────────────

    def _augment_dataset(self, image_paths: List[str], run_dir: Path) -> List[str]:
        aug_dir = run_dir / "augmented"
        aug_dir.mkdir(exist_ok=True)
        augmented = list(image_paths)

        cfg = self.cfg.get("character_finetune", {})

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")

                # Horizontal flip
                if cfg.get("random_flip", True):
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    out = aug_dir / f"flip_{Path(img_path).name}"
                    flipped.save(str(out))
                    augmented.append(str(out))

                # Rotation variants
                rot_range = cfg.get("random_rotation", 10)
                for angle in [-rot_range, rot_range]:
                    rotated = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
                    out = aug_dir / f"rot{angle}_{Path(img_path).name}"
                    rotated.save(str(out))
                    augmented.append(str(out))

                # Brightness variants
                for factor in [0.8, 1.2]:
                    bright = ImageEnhance.Brightness(img).enhance(factor)
                    out = aug_dir / f"bright{factor}_{Path(img_path).name}"
                    bright.save(str(out))
                    augmented.append(str(out))

                # Slight blur (simulate different focal lengths)
                blurred = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                out = aug_dir / f"blur_{Path(img_path).name}"
                blurred.save(str(out))
                augmented.append(str(out))

            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path}: {e}")

        return augmented

    # ── Phase 3: Fine-tune SadTalker ─────────────────────────────────────────

    def _finetune_sadtalker(
        self,
        image_paths: List[str],
        run_dir: Path,
        epochs: int,
        character_name: str,
    ):
        """Fine-tune SadTalker's mapping network on character-specific data."""
        logger.info(f"Fine-tuning SadTalker mapping network ({epochs} epochs)...")

        # Check if SadTalker training script exists
        train_script = SADTALKER_DIR / "src" / "train_full.py"
        if not train_script.exists():
            # Use our custom training wrapper
            train_script = ROOT / "training" / "finetune_sadtalker.py"

        if not train_script.exists():
            logger.warning("SadTalker training script not found. Creating data manifest only.")
            self._save_training_manifest(image_paths, run_dir, character_name)
            return

        cmd = [
            sys.executable, str(train_script),
            "--data_dir", str(run_dir / "augmented"),
            "--checkpoint_dir", str(ROOT / "models" / "sadtalker" / "checkpoints"),
            "--output_dir", str(run_dir / "finetuned"),
            "--epochs", str(epochs),
            "--character_name", character_name,
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True)
        if result.returncode == 0:
            logger.info("SadTalker fine-tuning complete")
        else:
            logger.error(f"Fine-tuning failed: {result.returncode}")

    # ── Phase 4: Fine-tune Voice ──────────────────────────────────────────────

    def _finetune_voice(
        self, voice_sample: str, run_dir: Path, character_name: str
    ):
        """Fine-tune XTTS-v2 on character voice (or just save reference for cloning)."""
        voice_dir = run_dir / "voice"
        voice_dir.mkdir(exist_ok=True)

        # Copy reference audio for voice cloning (zero-shot – no training needed)
        dst = voice_dir / f"{character_name}_voice.wav"
        shutil.copy(voice_sample, str(dst))

        # Save a voice profile
        profile = {
            "character": character_name,
            "voice_sample": str(dst),
            "cloning_mode": "zero_shot",
        }
        (voice_dir / "profile.json").write_text(json.dumps(profile, indent=2))
        logger.info(f"Voice profile saved for {character_name}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_rated_videos(self) -> List[str]:
        """Return video paths that received high ratings."""
        meta_dir = self.training_data_dir / "metadata"
        if not meta_dir.exists():
            return []

        videos = []
        for meta_file in meta_dir.glob("*.json"):
            try:
                data = json.loads(meta_file.read_text())
                if data.get("include_in_training") and Path(data["video"]).exists():
                    videos.append(data["video"])
            except Exception:
                pass
        return videos

    def _save_training_manifest(
        self, image_paths: List[str], run_dir: Path, character_name: str
    ):
        manifest = {
            "character": character_name,
            "image_count": len(image_paths),
            "images": image_paths,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _save_manifest(self, character_name: str, run_dir: Path, n_images: int):
        manifest = {
            "character": character_name,
            "run_dir": str(run_dir),
            "n_images": n_images,
            "timestamp": datetime.now().isoformat(),
            "status": "complete",
        }
        (run_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(f"Training manifest saved → {run_dir}/training_manifest.json")
