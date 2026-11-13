# VideoGen Agent

**Local · Open-Source · No Paid APIs**

Turn a character portrait + text script into a realistic Full HD talking-head video — entirely on your machine.

---

## Pipeline

```
Script ──► Script Processor ──► Segments
                                    │
Character Image ──► XTTS-v2 TTS ──► Audio
                                    │
Image + Audio ──► SadTalker ──► Raw video (512px)
                                    │
                   Real-ESRGAN ──► 4× upscale
                                    │
                   GFPGAN     ──► Face restoration
                                    │
                   FFmpeg     ──► Final 1080p MP4
```

---

## Quick Start

### 1. Install

```bash
# Give execute permission
chmod +x install.sh

# Run installer (installs ffmpeg, clones SadTalker, downloads models)
./install.sh
```

### 2. Activate environment

```bash
source venv/bin/activate
```

### 3. Generate a video

```bash
# Using example script + your character image
python agent.py generate \
  --script scripts/example_script.txt \
  --character characters/your_portrait.jpg

# With voice cloning (provide a 5-30s WAV sample)
python agent.py generate \
  --script scripts/example_script.txt \
  --character characters/your_portrait.jpg \
  --voice your_voice_sample.wav

# In Hindi, without HD enhancement (faster)
python agent.py generate \
  --script scripts/my_hindi_script.txt \
  --character characters/portrait.jpg \
  --language hi \
  --no-enhance
```

### 4. Web UI

```bash
python agent.py serve
# Open: http://localhost:8000
```

### 5. Train on your character

```bash
# Provide main portrait + optional folder of more images
python agent.py train \
  --character characters/portrait.jpg \
  --images-dir characters/more_poses/ \
  --name "my_character" \
  --epochs 100
```

### 6. Rate outputs (improves training data)

```bash
python agent.py feedback --video outputs/video_xyz.mp4 --rating 5
```

---

## Script Format

```
Hello, welcome to this demonstration.

[PAUSE]

This is the second paragraph with a half-second pause before it.

[BREAK]

[ENERGETIC]
This line will be delivered with more energy!

[PAUSE:1.5]

A custom 1.5-second pause before this line.
```

**Markers:** `[SCENE]` `[CUT]` `[PAUSE]` `[BREAK]` `[LONG_PAUSE]` `[PAUSE:N]`
**Emotions:** `[HAPPY]` `[SAD]` `[ANGRY]` `[WHISPER]` `[ENERGETIC]`

---

## Languages Supported

| Code | Language   |
|------|------------|
| en   | English    |
| hi   | Hindi      |
| es   | Spanish    |
| fr   | French     |
| de   | German     |
| ja   | Japanese   |
| zh   | Chinese    |
| ar   | Arabic     |
| pt   | Portuguese |
| ru   | Russian    |
| ko   | Korean     |
| it   | Italian    |

---

## Models Used (All Free / Open-Source)

| Component       | Model              | License     |
|----------------|--------------------|-------------|
| TTS (primary)  | Coqui XTTS-v2      | CPML        |
| TTS (fallback) | Edge-TTS           | Free        |
| Face animation | SadTalker          | CC BY-NC    |
| Upscaling      | Real-ESRGAN x4+    | BSD-3       |
| Face restore   | GFPGANv1.4         | Apache 2.0  |
| Video encoding | FFmpeg             | LGPL/GPL    |

---

## Directory Structure

```
automate_videoGen/
├── agent.py              ← Main CLI
├── api_server.py         ← FastAPI web server
├── config.yaml           ← All configuration
├── install.sh            ← Full installer
├── setup_models.py       ← Model downloader
├── modules/
│   ├── tts_engine.py     ← TTS (XTTS-v2, Bark, Edge-TTS)
│   ├── face_animator.py  ← SadTalker / Wav2Lip wrapper
│   ├── video_enhancer.py ← Real-ESRGAN + GFPGAN
│   ├── video_composer.py ← FFmpeg composition
│   ├── script_processor.py
│   └── character_trainer.py
├── training/
│   └── finetune_sadtalker.py
├── utils/
│   ├── model_downloader.py
│   ├── gpu_utils.py
│   └── logger.py
├── web/
│   └── index.html        ← Browser UI
├── characters/           ← Your portrait images
├── scripts/              ← Script text files
├── outputs/              ← Generated videos
├── models/               ← Downloaded model weights
├── third_party/          ← Cloned repos (SadTalker, etc.)
├── training_data/        ← Collected training data + ratings
└── checkpoints/          ← Fine-tuned model checkpoints
```

---

## Tips for Best Results

1. **Portrait quality** – Use a well-lit, front-facing portrait. 512×512 or larger works best.
2. **Voice sample** – 10–20 seconds of clear speech (no music, no noise) gives best cloning.
3. **Script length** – Keep individual sentences under 300 characters. Use `[BREAK]` between sections.
4. **Still mode** – Use `--still` for a news-anchor / studio look with minimal head movement.
5. **Train regularly** – Rate your best outputs 4–5 stars. Run `train` monthly to improve quality.

---

## Hardware Requirements

| Config           | Min                  | Recommended        |
|-----------------|----------------------|--------------------|
| RAM              | 8 GB                 | 16+ GB             |
| Storage          | 15 GB free           | 40+ GB free        |
| GPU              | None (CPU works)     | Apple M1/M2 or NVIDIA 8GB+ |
| Generation time  | 5–15 min (CPU)       | 1–3 min (GPU/MPS)  |
