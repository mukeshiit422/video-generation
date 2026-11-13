import sys
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return the best available device string for PyTorch."""
    try:
        import torch

        if sys.platform == "darwin":
            # Apple Silicon – prefer MPS
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("Using Apple MPS (Metal Performance Shaders)")
                return "mps"
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"Using CUDA – {name} ({vram:.1f} GB VRAM)")
            return device
    except ImportError:
        pass

    logger.warning("No GPU found – falling back to CPU (generation will be slow)")
    return "cpu"


def get_vram_gb() -> float:
    """Return available VRAM in GB, or 0 for CPU/MPS."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def clear_gpu_cache():
    """Free cached GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
