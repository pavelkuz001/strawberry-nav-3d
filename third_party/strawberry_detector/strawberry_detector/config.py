"""
Configuration for strawberry_detector.

Goal: keep repo clean (git clone -> run), store large assets (weights, Depth-Anything-V2 repo)
in per-user cache directory (cross-platform: Ubuntu/macOS/Windows).
"""

from pathlib import Path
from platformdirs import user_cache_dir

# Package directory (kept for relative resources if needed)
PACKAGE_DIR = Path(__file__).resolve().parent

# Cross-platform user cache (e.g. ~/.cache on Linux, ~/Library/Caches on macOS)
CACHE_DIR = Path(user_cache_dir("strawberry_detector"))
CHECKPOINTS_DIR = CACHE_DIR / "checkpoints"

# Depth-Anything-V2 settings
DEPTH_ANYTHING_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_ANYTHING_DIR = CACHE_DIR / "Depth-Anything-V2"
DEPTH_WEIGHTS_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/"
    "depth_anything_v2_metric_hypersim_vitl.pth"
)
DEPTH_WEIGHTS_FILENAME = "depth_anything_v2_metric_hypersim_vitl.pth"
DEPTH_ENCODER = "vitl"  # Vision Transformer Large

# Segmentation classes
CLASS_NAMES = {0: "ripe", 1: "unripe"}


# YOLO settings
YOLO_WEIGHTS_GDRIVE_ID = "10cpgTPpNocwytHg77AqKypWX-yOdhsGY"
YOLO_WEIGHTS_FILENAME = "strawberry_yolo_best.pt"

# Default device (CLI restricts to cuda/cpu anyway)
DEFAULT_DEVICE = "cuda"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def get_depth_weights_path() -> Path:
    ensure_dirs()
    return CHECKPOINTS_DIR / DEPTH_WEIGHTS_FILENAME


def get_yolo_weights_path() -> Path:
    ensure_dirs()
    return CHECKPOINTS_DIR / YOLO_WEIGHTS_FILENAME
