"""Data layer: Shapes3D loader + deterministic shared splits."""

from .shapes3d import (
    FACTOR_NAMES,
    FACTORS,
    N_TOTAL,
    Factor,
    Shapes3D,
    download_shapes3d,
    ensure_image_memmap,
    load_arrays,
    memmap_path,
)
from .splits import make_splits

__all__ = [
    "FACTORS",
    "FACTOR_NAMES",
    "Factor",
    "N_TOTAL",
    "Shapes3D",
    "download_shapes3d",
    "ensure_image_memmap",
    "load_arrays",
    "memmap_path",
    "make_splits",
]
