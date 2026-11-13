"""
FastAPI server – exposes the video generation pipeline over HTTP
and serves the web UI.
"""
import os
import sys
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from typing import List
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def create_app(cfg: dict) -> FastAPI:
    app = FastAPI(
        title="VideoGen Agent API",
        description="Open-source text-to-video generation",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve web UI
    web_dir = ROOT / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

    # In-memory job tracker
    jobs: dict = {}

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = ROOT / "web" / "index.html"
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.post("/generate")
    async def generate_video(
        background_tasks: BackgroundTasks,
        script: str = Form(...),
        language: str = Form(default="en"),
        enhance: bool = Form(default=True),
        still_mode: bool = Form(default=False),
        subtitles: bool = Form(default=False),
        characters: List[UploadFile] = File(...),
        voice_sample: Optional[UploadFile] = File(default=None),
    ):
        """Submit a video generation job. Returns job_id immediately."""
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        job_dir = ROOT / "outputs" / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save all character images; first one is used for video generation
        char_paths = []
        for i, char_file in enumerate(characters):
            if char_file and char_file.filename:
                p = job_dir / f"character_{i}{Path(char_file.filename).suffix}"
                p.write_bytes(await char_file.read())
                char_paths.append(str(p))

        if not char_paths:
            raise HTTPException(400, "At least one character image is required")
        char_path = char_paths[0]  # primary image for SadTalker

        voice_path = None
        if voice_sample and voice_sample.filename:
            voice_path = job_dir / f"voice{Path(voice_sample.filename).suffix}"
            voice_path.write_bytes(await voice_sample.read())

        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "output": None,
            "error": None,
        }

        background_tasks.add_task(
            _run_generation,
            job_id=job_id,
            jobs=jobs,
            cfg=cfg,
            script=script,
            char_path=char_path,
            char_paths=char_paths,
            voice_path=str(voice_path) if voice_path else None,
            language=language,
            enhance=enhance,
            still_mode=still_mode,
            subtitles=subtitles,
            job_dir=job_dir,
        )

        return JSONResponse({"job_id": job_id, "status": "queued"})

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        return JSONResponse(jobs[job_id])

    @app.get("/jobs/{job_id}/download")
    async def download_video(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        job = jobs[job_id]
        if job["status"] != "done":
            raise HTTPException(400, f"Job not ready: {job['status']}")
        output_path = job["output"]
        if not output_path or not Path(output_path).exists():
            raise HTTPException(500, "Output file missing")
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"video_{job_id}.mp4",
        )

    @app.get("/outputs")
    async def list_outputs():
        out_dir = ROOT / "outputs"
        videos = sorted(out_dir.glob("video_*.mp4"), reverse=True)
        return JSONResponse([
            {
                "name": v.name,
                "path": str(v),
                "size_mb": round(v.stat().st_size / 1024 / 1024, 2),
                "created": datetime.fromtimestamp(v.stat().st_ctime).isoformat(),
            }
            for v in videos[:20]
        ])

    @app.post("/generate-image")
    async def generate_image(
        background_tasks: BackgroundTasks,
        prompt: str = Form(...),
        negative_prompt: str = Form(default=""),
        style: str = Form(default="photorealistic"),
        width: int = Form(default=512),
        height: int = Form(default=512),
        steps: int = Form(default=30),
        guidance_scale: float = Form(default=7.5),
        seed: int = Form(default=-1),
        model_id: str = Form(default="runwayml/stable-diffusion-v1-5"),
    ):
        """Submit a text-to-image job. Returns job_id immediately."""
        job_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        jobs[job_id] = {
            "id": job_id, "status": "queued", "progress": 0,
            "message": "Queued", "output": None, "error": None, "type": "image",
        }
        background_tasks.add_task(
            _run_image_generation,
            job_id=job_id, jobs=jobs, cfg=cfg,
            prompt=prompt, negative_prompt=negative_prompt,
            style=style, width=width, height=height,
            steps=steps, guidance_scale=guidance_scale,
            seed=None if seed == -1 else seed,
            model_id=model_id,
        )
        return JSONResponse({"job_id": job_id, "status": "queued"})

    @app.get("/jobs/{job_id}/image")
    async def download_image(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        job = jobs[job_id]
        if job["status"] != "done":
            raise HTTPException(400, f"Job not ready: {job['status']}")
        output_path = job["output"]
        if not output_path or not Path(output_path).exists():
            raise HTTPException(500, "Output file missing")
        return FileResponse(output_path, media_type="image/png", filename=f"image_{job_id}.png")

    @app.get("/outputs/images")
    async def list_images():
        img_dir = ROOT / "outputs" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(img_dir.glob("img_*.png"), reverse=True)
        return JSONResponse([
            {
                "name": i.name,
                "path": str(i),
                "size_mb": round(i.stat().st_size / 1024 / 1024, 2),
                "created": datetime.fromtimestamp(i.stat().st_ctime).isoformat(),
            }
            for i in images[:20]
        ])

    @app.post("/feedback")
    async def save_feedback(
        video_path: str = Form(...),
        rating: int = Form(...),
    ):
        from modules.character_trainer import CharacterTrainer
        trainer = CharacterTrainer(cfg)
        trainer.collect_feedback(video_path, rating, "", "")
        return JSONResponse({"ok": True, "rating": rating})

    @app.post("/train")
    async def start_training(
        background_tasks: BackgroundTasks,
        character: UploadFile = File(...),
        name: str = Form(default="default"),
        epochs: int = Form(default=50),
    ):
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job_dir = ROOT / "checkpoints" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        char_path = job_dir / f"character{Path(character.filename).suffix}"
        char_path.write_bytes(await character.read())

        jobs[job_id] = {"id": job_id, "status": "training", "progress": 0, "message": "Starting training…"}

        background_tasks.add_task(
            _run_training,
            job_id=job_id,
            jobs=jobs,
            cfg=cfg,
            char_path=str(char_path),
            name=name,
            epochs=epochs,
        )
        return JSONResponse({"job_id": job_id})

    return app


# ── Background Tasks ──────────────────────────────────────────────────────────

async def _run_generation(
    job_id, jobs, cfg, script, char_path, char_paths,
    voice_path, language, enhance, still_mode, subtitles, job_dir
):
    from utils.gpu_utils import get_device
    import time as _time

    job_started = _time.time()

    def update(step, msg, pct):
        jobs[job_id].update({
            "status": "running",
            "message": msg,
            "progress": pct,
            "step": step,
            "step_started_at": _time.time(),
            "job_started_at": job_started,
        })

    try:
        device = get_device()
        if still_mode:
            cfg.setdefault("face_animation", {}).setdefault("sadtalker", {})["still_mode"] = True

        update(1, "Processing script…", 10)
        from modules.script_processor import ScriptProcessor
        segments = ScriptProcessor(cfg).process(script)

        update(2, "Generating voice…", 25)
        from modules.tts_engine import TTSEngine
        audio_path = await asyncio.to_thread(
            TTSEngine(cfg, device=device).synthesize,
            segments, voice_path, language, "default",
            str(job_dir / "audio.wav"),
        )

        update(3, "Animating character…", 45)
        from modules.face_animator import FaceAnimator

        def _anim_progress(done, total):
            sub = done / total
            pct = 45 + int(sub * 19)  # 45 → 64%
            jobs[job_id].update({
                "status": "running",
                "message": f"Animating character… ({done}/{total} frames)",
                "progress": pct,
                "step": 3,
                "step_started_at": jobs[job_id].get("step_started_at", _time.time()),
                "job_started_at": job_started,
            })

        raw_video = await asyncio.to_thread(
            FaceAnimator(cfg, device=device).animate,
            char_path, audio_path,
            str(job_dir / "raw.mp4"),
            _anim_progress,
        )

        if enhance:
            update(4, "Enhancing to Full HD…", 65)
            from modules.video_enhancer import VideoEnhancer
            raw_video = await asyncio.to_thread(
                VideoEnhancer(cfg, device=device).enhance,
                raw_video,
                str(job_dir / "enhanced.mp4"),
            )

        update(5, "Compositing…", 85)
        output_path = str(job_dir / "final.mp4")
        from modules.video_composer import VideoComposer
        composer = VideoComposer(cfg)
        sub_path = None
        if subtitles:
            sub_path = composer.generate_subtitles(segments, output_path)
        final = await asyncio.to_thread(
            composer.compose, raw_video, audio_path, output_path, None, 0.1, sub_path
        )

        jobs[job_id] = {
            "id": job_id, "status": "done", "progress": 100,
            "message": "Complete!", "output": final, "error": None,
            "job_started_at": job_started,
            "total_seconds": round(_time.time() - job_started, 1),
        }

    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e), "message": f"Error: {e}"})
        import traceback; traceback.print_exc()


async def _run_image_generation(
    job_id, jobs, cfg, prompt, negative_prompt,
    style, width, height, steps, guidance_scale, seed, model_id
):
    try:
        jobs[job_id].update({"status": "running", "progress": 20, "message": "Loading model…"})
        from modules.image_generator import ImageGenerator
        from utils.gpu_utils import get_device
        gen = ImageGenerator(cfg, device=get_device())

        jobs[job_id].update({"progress": 50, "message": "Generating image…"})
        output_path = await asyncio.to_thread(
            gen.generate,
            prompt, negative_prompt, style, width, height,
            steps, guidance_scale, seed, model_id, None,
        )
        jobs[job_id].update({
            "status": "done", "progress": 100,
            "message": "Image ready!", "output": output_path,
        })
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e), "message": f"Error: {e}"})
        import traceback; traceback.print_exc()


async def _run_training(job_id, jobs, cfg, char_path, name, epochs):
    try:
        from modules.character_trainer import CharacterTrainer
        ckpt = await asyncio.to_thread(
            CharacterTrainer(cfg).train,
            char_path, None, None, None, epochs, name
        )
        jobs[job_id].update({"status": "done", "progress": 100, "message": f"Checkpoint: {ckpt}"})
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})
