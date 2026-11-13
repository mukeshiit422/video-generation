#!/usr/bin/env python3
"""
Model setup script – downloads all required model weights.
Run this after install.sh completes.
"""
import sys
import click
import yaml
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def download_image_models():
    """Pre-download the Image-mode models into the HuggingFace cache.

    Fetches exactly what ImageGenerator loads (same variant / skipped
    components), so the first generation in the UI starts immediately.
    """
    from huggingface_hub import hf_hub_download
    from diffusers import AutoencoderKL, StableDiffusionPipeline
    from modules.image_generator import MODEL_PROFILES, LCM_LORA_SD15

    for model_id, profile in MODEL_PROFILES.items():
        click.echo(f"→ {model_id}")
        skip = {"safety_checker": None, "feature_extractor": None}
        if profile.get("vae"):
            skip["vae"] = None  # replaced at load time; don't fetch the bundled one
        StableDiffusionPipeline.download(model_id, variant=profile.get("variant"), **skip)

    vaes = {p["vae"] for p in MODEL_PROFILES.values() if p.get("vae")}
    for vae_id in sorted(vaes):
        click.echo(f"→ {vae_id}")
        AutoencoderKL.from_pretrained(vae_id)

    click.echo(f"→ {LCM_LORA_SD15} (fast mode)")
    hf_hub_download(LCM_LORA_SD15, "pytorch_lora_weights.safetensors")

    from huggingface_hub import snapshot_download
    from modules.prompt_enhancer import DEFAULT_MODEL as ENHANCER_MODEL
    click.echo(f"→ {ENHANCER_MODEL} (prompt enhancer)")
    snapshot_download(ENHANCER_MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
    click.echo("Image models ready.")


@click.command()
@click.option("--all",         "download_all", is_flag=True, default=False)
@click.option("--sadtalker",   is_flag=True, default=False)
@click.option("--realesrgan",  is_flag=True, default=False)
@click.option("--gfpgan",      is_flag=True, default=False)
@click.option("--image",       is_flag=True, default=False, help="Image-mode SD models + prompt enhancer (~13 GB)")
@click.option("--config",      default="config.yaml")
def main(download_all, sadtalker, realesrgan, gfpgan, image, config):
    """Download pre-trained model weights."""
    with open(ROOT / config) as f:
        cfg = yaml.safe_load(f)

    from utils.model_downloader import ModelDownloader
    dl = ModelDownloader(cfg)

    if download_all:
        dl.download_all()
        download_image_models()
        return

    if sadtalker:
        dl.download_group("sadtalker")
        dl.download_group("sadtalker_gfpgan")
    if realesrgan:
        dl.download_group("realesrgan")
    if gfpgan:
        dl.download_group("gfpgan")
    if image:
        download_image_models()

    if not any([download_all, sadtalker, realesrgan, gfpgan, image]):
        click.echo("Specify a model group: --all, --sadtalker, --realesrgan, --gfpgan, --image")


if __name__ == "__main__":
    main()
