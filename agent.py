#!/usr/bin/env python3
"""
Video Generator Agent – turn scripts into realistic Full HD talking-head videos.

Usage:
  python agent.py generate  -s <script> -c <character_image>
  python agent.py generate  -s <script> -c <character_image> -v <voice_sample.wav>
  python agent.py train     -c <character_image> [--images-dir dir] [--epochs 100]
  python agent.py serve     [--port 8000]
  python agent.py setup     (download all models)
  python agent.py feedback  --video outputs/out.mp4 --rating 5
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import click
import yaml
from dotenv import load_dotenv
load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.logger import setup_logger
from utils.gpu_utils import get_device

console = Console()
logger = setup_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    cfg_path = ROOT / config_path
    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── CLI Group ─────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("1.0.0", prog_name="VideoGen Agent")
def cli():
    """Open-source Text-to-Video agent. Local, private, no paid APIs."""
    pass


# ── generate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--script",         "-s", required=True,
              help="Script text or path to .txt file")
@click.option("--character",      "-c", required=True,
              help="Character portrait image (JPG/PNG)")
@click.option("--voice",          "-v", default=None,
              help="Voice sample .wav for cloning (3–30 s)")
@click.option("--output",         "-o", default=None,
              help="Output video path (default: outputs/<timestamp>.mp4)")
@click.option("--language",       "-l", default="en",
              help="Language code: en, hi, es, fr, de, ja, zh …")
@click.option("--config",               default="config.yaml")
@click.option("--no-enhance",           is_flag=True, default=False,
              help="Skip Real-ESRGAN / GFPGAN (much faster, lower quality)")
@click.option("--background-music",     default=None,
              help="Background music file (MP3/WAV, optional)")
@click.option("--subtitles",            is_flag=True, default=False,
              help="Burn subtitles into the video")
@click.option("--still",                is_flag=True, default=False,
              help="Minimal head movement (studio-cam look)")
def generate(
    script, character, voice, output, language, config,
    no_enhance, background_music, subtitles, still
):
    """Generate a talking-head video from a script + character image."""

    _banner()

    cfg = load_config(config)
    if still:
        cfg.setdefault("face_animation", {}).setdefault("sadtalker", {})["still_mode"] = True

    device = get_device()
    console.print(f"  Device  : [bold]{device}[/bold]")

    # Resolve script text
    script_text = Path(script).read_text() if Path(script).exists() else script
    console.print(f"  Script  : {len(script_text)} chars")
    console.print(f"  Character: [cyan]{character}[/cyan]")
    if voice:
        console.print(f"  Voice   : [cyan]{voice}[/cyan]")

    # Resolve output path
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(ROOT / "outputs" / f"video_{ts}.mp4")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    console.print()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        console=console,
        transient=False,
    ) as prog:

        # 1. Script processing
        tid = prog.add_task("[cyan]1/5  Processing script…", total=None)
        from modules.script_processor import ScriptProcessor
        processor = ScriptProcessor(cfg)
        segments = processor.process(script_text)
        prog.update(tid, description=f"[green]1/5  Script → {len(segments)} segments ✓")
        prog.stop_task(tid)

        # 2. TTS
        tid = prog.add_task("[cyan]2/5  Generating voice…", total=None)
        from modules.tts_engine import TTSEngine
        tts = TTSEngine(cfg, device=device)
        audio_path = tts.synthesize(
            segments=segments,
            voice_sample=voice,
            language=language,
            output_path=str(ROOT / "outputs" / "temp_audio.wav"),
        )
        prog.update(tid, description="[green]2/5  Voice generated ✓")
        prog.stop_task(tid)

        # 3. Face animation
        tid = prog.add_task("[cyan]3/5  Animating character…", total=None)
        from modules.face_animator import FaceAnimator
        animator = FaceAnimator(cfg, device=device)
        raw_video = animator.animate(
            image_path=character,
            audio_path=audio_path,
            output_path=str(ROOT / "outputs" / "temp_raw.mp4"),
        )
        prog.update(tid, description="[green]3/5  Character animated ✓")
        prog.stop_task(tid)

        # 4. Enhancement
        if not no_enhance:
            tid = prog.add_task("[cyan]4/5  Enhancing to Full HD…", total=None)
            from modules.video_enhancer import VideoEnhancer
            enhancer = VideoEnhancer(cfg, device=device)
            enhanced_video = enhancer.enhance(
                video_path=raw_video,
                output_path=str(ROOT / "outputs" / "temp_enhanced.mp4"),
            )
            prog.update(tid, description="[green]4/5  Full HD enhancement ✓")
            prog.stop_task(tid)
        else:
            enhanced_video = raw_video
            console.print("  [dim]Enhancement skipped (--no-enhance)[/dim]")

        # 5. Final composition
        tid = prog.add_task("[cyan]5/5  Compositing final video…", total=None)
        from modules.video_composer import VideoComposer
        composer = VideoComposer(cfg)

        sub_path = None
        if subtitles:
            sub_path = composer.generate_subtitles(segments, output)

        final = composer.compose(
            video_path=enhanced_video,
            audio_path=audio_path,
            output_path=output,
            background_music=background_music,
            subtitle_path=sub_path,
        )
        prog.update(tid, description="[green]5/5  Video ready ✓")
        prog.stop_task(tid)

    console.print()
    console.print(Panel.fit(
        f"[bold green]Video generated successfully![/bold green]\n\n"
        f"[dim]Output:[/dim]  [cyan]{final}[/cyan]",
        border_style="green",
    ))

    # Save generation metadata
    _save_generation_meta(script_text, character, voice, final)


# ── train ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--character",    "-c", required=True, help="Main character portrait")
@click.option("--images-dir",   "-i", default=None,  help="Folder with more character images")
@click.option("--video-samples","-v", default=None,  help="Folder with character video clips")
@click.option("--voice-sample",       default=None,  help="Voice sample for TTS cloning")
@click.option("--name",         "-n", default="default", help="Character name/ID")
@click.option("--epochs",             default=50, type=int)
@click.option("--config",             default="config.yaml")
def train(character, images_dir, video_samples, voice_sample, name, epochs, config):
    """Fine-tune the model on a specific character for better consistency."""
    _banner()
    cfg = load_config(config)

    console.print(Panel.fit(
        f"[bold yellow]Character Training[/bold yellow]\n"
        f"  Character : [cyan]{character}[/cyan]\n"
        f"  Name      : [yellow]{name}[/yellow]\n"
        f"  Epochs    : {epochs}",
        border_style="yellow",
    ))

    from modules.character_trainer import CharacterTrainer
    trainer = CharacterTrainer(cfg)
    ckpt = trainer.train(
        character_image=character,
        images_dir=images_dir,
        video_samples=video_samples,
        voice_sample=voice_sample,
        epochs=epochs,
        character_name=name,
    )
    console.print(f"\n[bold green]Training complete! Checkpoint: {ckpt}[/bold green]")


# ── feedback ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--video",    "-v", required=True, help="Path to the generated video")
@click.option("--rating",   "-r", required=True, type=click.IntRange(1, 5),
              help="Rating 1–5 (5 = excellent)")
@click.option("--config",         default="config.yaml")
def feedback(video, rating, config):
    """Rate a generated video to improve future training data quality."""
    cfg = load_config(config)
    from modules.character_trainer import CharacterTrainer
    trainer = CharacterTrainer(cfg)

    # Try to load metadata for context
    meta_path = Path(video).parent / (Path(video).stem + ".json")
    script = ""
    character = ""
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        script = data.get("script_preview", "")
        character = data.get("character", "")

    trainer.collect_feedback(video, rating, script, character)
    stars = "★" * rating + "☆" * (5 - rating)
    console.print(f"[bold green]Feedback saved:[/bold green] {stars} ({rating}/5) for {video}")
    if rating >= 4:
        console.print("[dim]  → This video will be included in the next training run[/dim]")


# ── serve ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host",   default="0.0.0.0")
@click.option("--port",   default=8000, type=int)
@click.option("--config", default="config.yaml")
def serve(host, port, config):
    """Launch the web UI + REST API server."""
    import uvicorn
    from api_server import create_app

    cfg = load_config(config)
    app = create_app(cfg)

    console.print(Panel.fit(
        f"[bold green]Web UI started[/bold green]\n\n"
        f"  Open: [cyan underline]http://localhost:{port}[/cyan underline]",
        border_style="green",
    ))
    uvicorn.run(app, host=host, port=port, reload=False)


# ── setup ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", default="config.yaml")
@click.option("--group", default="all", help="Model group: all | sadtalker | realesrgan | gfpgan")
def setup(config, group):
    """Download all required model weights."""
    cfg = load_config(config)
    from utils.model_downloader import ModelDownloader
    dl = ModelDownloader(cfg)
    if group == "all":
        dl.download_all()
    else:
        dl.download_group(group)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]VideoGen Agent[/bold cyan]  [dim]v1.0.0[/dim]\n"
        "[dim]Open-Source · Local · No Paid APIs[/dim]",
        border_style="cyan",
    ))
    console.print()


def _save_generation_meta(script: str, character: str, voice, output: str):
    meta = {
        "timestamp": datetime.now().isoformat(),
        "character": str(character),
        "voice_sample": str(voice) if voice else None,
        "script_preview": script[:300],
        "output": str(output),
        "rating": None,
    }
    meta_path = Path(output).parent / (Path(output).stem + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    cli()
