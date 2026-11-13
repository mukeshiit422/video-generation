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


@click.command()
@click.option("--all",         "download_all", is_flag=True, default=False)
@click.option("--sadtalker",   is_flag=True, default=False)
@click.option("--realesrgan",  is_flag=True, default=False)
@click.option("--gfpgan",      is_flag=True, default=False)
@click.option("--config",      default="config.yaml")
def main(download_all, sadtalker, realesrgan, gfpgan, config):
    """Download pre-trained model weights."""
    with open(ROOT / config) as f:
        cfg = yaml.safe_load(f)

    from utils.model_downloader import ModelDownloader
    dl = ModelDownloader(cfg)

    if download_all:
        dl.download_all()
        return

    if sadtalker:
        dl.download_group("sadtalker")
        dl.download_group("sadtalker_gfpgan")
    if realesrgan:
        dl.download_group("realesrgan")
    if gfpgan:
        dl.download_group("gfpgan")

    if not any([download_all, sadtalker, realesrgan, gfpgan]):
        click.echo("Specify a model group: --all, --sadtalker, --realesrgan, --gfpgan")


if __name__ == "__main__":
    main()
