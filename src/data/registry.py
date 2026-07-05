"""Dataset registry — select a dataset (+ its factor metadata) by name.

One config key ``data.dataset`` ("shapes3d" | "dsprites") drives the whole
pipeline (training, extraction, probing) without hard-coding either dataset.
Imports are lazy so importing this module needs neither h5py nor the data files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cls: type  # torch Dataset subclass (Shapes3D | DSprites)
    factors: tuple  # tuple[Factor, ...]
    n_total: int
    default_path: Path


def get_dataset(name: str = "shapes3d") -> DatasetSpec:
    if name == "shapes3d":
        from .shapes3d import DEFAULT_PATH, FACTORS, N_TOTAL, Shapes3D

        return DatasetSpec("shapes3d", Shapes3D, FACTORS, N_TOTAL, DEFAULT_PATH)
    if name == "dsprites":
        from .dsprites import DEFAULT_PATH, FACTORS, N_TOTAL, DSprites

        return DatasetSpec("dsprites", DSprites, FACTORS, N_TOTAL, DEFAULT_PATH)
    raise ValueError(f"unknown dataset {name!r}; choose 'shapes3d' or 'dsprites'")
