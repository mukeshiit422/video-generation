"""
Automatic model downloader – fetches all required model weights from
Hugging Face Hub and direct URLs using only open-source/free models.
"""
import os
import sys
import shutil
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
)

console = Console()

ROOT = Path(__file__).parent.parent

# ── Model Registry ────────────────────────────────────────────────────────────
MODELS = {
    "sadtalker": {
        "description": "SadTalker – talking head generation weights",
        "files": [
            {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
                "dest": "models/sadtalker/checkpoints/mapping_00229-model.pth.tar",
                "size_hint": "170M",
            },
            {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
                "dest": "models/sadtalker/checkpoints/mapping_00109-model.pth.tar",
                "size_hint": "170M",
            },
            {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
                "dest": "models/sadtalker/checkpoints/SadTalker_V0.0.2_256.safetensors",
                "size_hint": "726M",
            },
            {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
                "dest": "models/sadtalker/checkpoints/SadTalker_V0.0.2_512.safetensors",
                "size_hint": "726M",
            },
        ],
    },
    "sadtalker_gfpgan": {
        "description": "GFPGAN weights for SadTalker face enhancement",
        "files": [
            {
                "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                "dest": "models/sadtalker/gfpgan/weights/GFPGANv1.4.pth",
                "size_hint": "340M",
            },
        ],
    },
    "realesrgan": {
        "description": "Real-ESRGAN – video upscaling to Full HD",
        "files": [
            {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "dest": "models/realesrgan/RealESRGAN_x4plus.pth",
                "size_hint": "64M",
            },
            {
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                "dest": "models/realesrgan/RealESRGAN_x4plus_anime_6B.pth",
                "size_hint": "18M",
            },
        ],
    },
    "gfpgan": {
        "description": "Standalone GFPGAN for face restoration",
        "files": [
            {
                "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                "dest": "models/gfpgan/GFPGANv1.4.pth",
                "size_hint": "340M",
            },
            {
                "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
                "dest": "models/gfpgan/detection_Resnet50_Final.pth",
                "size_hint": "108M",
            },
            {
                "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
                "dest": "models/gfpgan/parsing_parsenet.pth",
                "size_hint": "85M",
            },
        ],
    },
    "wav2lip": {
        "description": "Wav2Lip – lip sync weights",
        "files": [
            {
                "url": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLoBxis-B38C1kc16scFmAinNg?e=TvAl9T",
                "dest": "models/wav2lip/wav2lip_gan.pth",
                "size_hint": "411M",
                "note": "Direct download may fail – see README for manual download",
            },
        ],
    },
}


class ModelDownloader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = ROOT

    def download_all(self):
        console.print("\n[bold cyan]Downloading all required models...[/bold cyan]\n")
        for key, spec in MODELS.items():
            if key == "wav2lip":
                console.print(f"[yellow]Skipping Wav2Lip (requires manual download – see models/wav2lip/README.md)[/yellow]")
                continue
            self._download_group(key, spec)
        console.print("\n[bold green]All models downloaded.[/bold green]")

    def download_group(self, group_name: str):
        if group_name not in MODELS:
            console.print(f"[red]Unknown model group: {group_name}[/red]")
            return
        self._download_group(group_name, MODELS[group_name])

    def _download_group(self, name: str, spec: dict):
        console.print(f"[cyan]─── {spec['description']} ───[/cyan]")
        for file_spec in spec["files"]:
            dest = self.root / file_spec["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                console.print(f"  [dim]✓ {dest.name} already present[/dim]")
                continue
            self._download_file(file_spec["url"], dest, file_spec.get("size_hint", "?"))

    def _download_file(self, url: str, dest: Path, size_hint: str):
        console.print(f"  Downloading [yellow]{dest.name}[/yellow] (~{size_hint})...")
        try:
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {dest.name}", total=None)

                def reporthook(block, block_size, total_size):
                    if total_size > 0 and progress.tasks[task].total is None:
                        progress.update(task, total=total_size)
                    progress.update(task, advance=block_size)

                urllib.request.urlretrieve(url, str(dest), reporthook)
            console.print(f"  [green]✓ {dest.name}[/green]")
        except Exception as e:
            console.print(f"  [red]✗ Failed: {e}[/red]")
            console.print(f"    Manual download: {url}")
            console.print(f"    Save to: {dest}")


def try_hf_download(repo_id: str, filename: str, local_dir: str) -> Optional[str]:
    """Download a file from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return path
    except Exception as e:
        console.print(f"[red]HF download failed for {repo_id}/{filename}: {e}[/red]")
        return None
