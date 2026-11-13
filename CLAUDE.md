# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always activate the venv before running anything:
```bash
source venv/bin/activate
```

Python version: **3.11** (3.12+ breaks Coqui TTS). The venv uses `/opt/homebrew/bin/python3.11`.

Hardware target: **Apple M2** — use MPS (Metal) for GPU, not CUDA. `utils/gpu_utils.py:get_device()` handles this automatically.

## Common Commands

```bash
# Start web UI + API server (primary way to use the system)
python agent.py serve

# Generate a video from CLI
python agent.py generate -s scripts/example_script.txt -c characters/portrait.jpg

# Generate with voice cloning
python agent.py generate -s scripts/my_script.txt -c characters/portrait.jpg -v voice_sample.wav

# Download all model weights (run once after install) — includes Image-mode SD models
python setup_models.py --all
# Image-mode models only (DreamShaper 8, Realistic Vision 5.1 + VAE, Counterfeit 2.5, LCM-LoRA, Qwen2.5-1.5B prompt enhancer; ~13 GB)
python setup_models.py --image

# Apply patches after any pip install/upgrade
python patch_dependencies.py

# Fine-tune on a character
python agent.py train -c characters/portrait.jpg --name mychar --epochs 100

# Rate a generated video (feeds training pipeline)
python agent.py feedback --video outputs/video_xyz.mp4 --rating 5
```

## Architecture

The system is a local, open-source text-to-video pipeline. No paid APIs — everything runs on-device.

### Generation pipeline (sequential)

```
Script text
  → modules/script_processor.py   (split into Segment objects, handle [PAUSE]/[BREAK] markers)
  → modules/tts_engine.py          (Segment list → WAV audio; backends: XTTS-v2 → Edge-TTS → pyttsx3)
  → modules/face_animator.py       (image + WAV → raw MP4 via SadTalker subprocess)
  → modules/video_enhancer.py      (raw MP4 → 1080p via Real-ESRGAN + GFPGAN, frame-by-frame)
  → modules/video_composer.py      (enhanced MP4 + WAV → final MP4 via FFmpeg)
```

### Web server

`api_server.py` creates a FastAPI app (returned from `create_app(cfg)`). Jobs run as background tasks; the UI polls `/jobs/{job_id}` every 1.5s. The web UI is a single-file SPA at `web/index.html` — no build step, served directly by FastAPI.

Two modes in the UI: **Video** (the pipeline above) and **Image** (Stable Diffusion via `modules/image_generator.py`).

### Key design decisions

- **SadTalker is called via subprocess**, not imported. It runs as `third_party/SadTalker/inference.py` with `sys.executable`. Its output MP4 is moved from a temp `result_dir` to the final path.
- **TTS falls back to silence** if all backends fail — `tts_engine.py:_generate_silence()` estimates duration from character count so SadTalker always gets a valid audio file.
- **Multiple character images** can be uploaded; the primary (index 0 after reordering) goes to SadTalker, the rest are stored for training data.
- All config lives in `config.yaml`. Modules receive the full parsed dict and pull their own subtree (e.g. `cfg.get("tts", {})`).

### Dependency compatibility issues (ongoing)

Third-party packages have numpy 2.x and torchvision 0.16+ incompatibilities. `patch_dependencies.py` fixes them all — run it after any `pip install` that touches numpy, scikit-image, basicsr, or torchvision. Key known patches:
- `basicsr/data/degradations.py`: `functional_tensor` → `functional`
- `skimage/util/dtype.py`: `np.bool8` → `np.bool_`
- SadTalker source: `np.VisibleDeprecationWarning` → `DeprecationWarning`

numpy must stay at **1.26.4** — 2.x breaks scikit-image, cv2, and SadTalker simultaneously.

### Model weights location

| Component    | Path                                        |
|-------------|---------------------------------------------|
| SadTalker   | `models/sadtalker/checkpoints/`             |
| GFPGAN      | `models/sadtalker/gfpgan/weights/` and `models/gfpgan/` |
| Real-ESRGAN | `models/realesrgan/`                        |
| TTS (XTTS)  | Cached by Coqui in `~/.local/share/tts/`   |
| SD models   | Cached by HuggingFace in `~/.cache/huggingface/` |

### Training pipeline

`modules/character_trainer.py` collects frames from rated videos (`feedback` command sets `include_in_training: true` for ratings ≥4), augments them (flip/rotate/brightness), and calls `training/finetune_sadtalker.py` as a subprocess. XTTS voice cloning is zero-shot (reference WAV only, no actual training).
