"""Shapes3D (3D Shapes) loader + factor metadata.

Shapes3D ships as a single HDF5 (``3dshapes.h5``, ~255 MB, no auth) with:
    images : (480000, 64, 64, 3) uint8
    labels : (480000, 6)        float64   # the 6 generative factors, in order

Factors (prereg §0): floor_hue, wall_hue, object_hue, scale, orientation are
CONTINUOUS (-> R²); shape is CATEGORICAL (-> normalized accuracy). Each unique
factor combination appears exactly once (10*10*10*8*4*15 == 480000), so every
image is a distinct instance — clean for instance-discrimination SSL.

Usage:
    python -m src.data.shapes3d --download        # fetch + verify the HDF5
    python -m src.data.shapes3d --info            # print schema / factor stats
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
DEFAULT_PATH = Path("data/raw/3dshapes.h5")
EXPECTED_BYTES = 267_573_662
N_TOTAL = 480_000
IMAGE_SIZE = 64


@dataclass(frozen=True)
class Factor:
    name: str
    index: int
    kind: str  # "continuous" | "categorical"
    n_values: int
    cyclic: bool = False

    @property
    def chance(self) -> float:
        """Chance accuracy for a categorical factor (else NaN)."""
        return 1.0 / self.n_values if self.kind == "categorical" else float("nan")


# Order matches the `labels` column order in the HDF5.
FACTORS: tuple[Factor, ...] = (
    Factor("floor_hue", 0, "continuous", 10),
    Factor("wall_hue", 1, "continuous", 10),
    Factor("object_hue", 2, "continuous", 10),
    Factor("scale", 3, "continuous", 8),
    Factor("shape", 4, "categorical", 4),
    # Shapes3D orientation spans [-30, 30] deg (a limited arc, no wrap), so plain
    # R² is well-defined here. The `cyclic` flag is carried for the Phase-2
    # probe-build decision and for dSprites' full-rotation orientation.
    Factor("orientation", 5, "continuous", 15, cyclic=True),
)
FACTOR_NAMES = tuple(f.name for f in FACTORS)


def download_shapes3d(path: str | Path = DEFAULT_PATH, force: bool = False) -> Path:
    """Download the Shapes3D HDF5 to ``path`` (idempotent; size-guarded)."""
    path = Path(path)
    if path.exists() and not force and path.stat().st_size == EXPECTED_BYTES:
        print(f"[shapes3d] already present: {path} ({path.stat().st_size} bytes)")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[shapes3d] downloading {URL} -> {path} (~255 MB) ...")
    urllib.request.urlretrieve(URL, path)
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise RuntimeError(
            f"download size {size} != expected {EXPECTED_BYTES}; file may be corrupt"
        )
    print(f"[shapes3d] done: {size} bytes")
    return path


def load_arrays(
    path: str | Path = DEFAULT_PATH, indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read images+labels for ``indices`` (sorted) into memory.

    Returns ``(images uint8 (N,64,64,3), labels float32 (N,6))``. With
    ``indices=None`` reads the whole 5.9 GB array — pass a split for subsets.
    """
    import h5py  # local import so module imports without h5py installed

    with h5py.File(path, "r") as f:
        if indices is None:
            images = f["images"][:]
            labels = f["labels"][:].astype(np.float32)
        else:
            idx = np.sort(np.asarray(indices))
            images = f["images"][idx]
            labels = f["labels"][idx].astype(np.float32)
    return images, labels


class Shapes3D(Dataset):
    """In-memory Shapes3D split.

    Args:
        indices: which rows of the HDF5 this split covers (see ``data.splits``).
        transform: callable on a PIL image. For SSL pass a ``TwoViewTransform``
            (returns two tensors); for eval pass ``eval_transform`` (one tensor).
        path: HDF5 location.
        return_label: include the 6-factor label vector (float32) per item.
    """

    def __init__(
        self,
        indices: np.ndarray,
        transform=None,
        path: str | Path = DEFAULT_PATH,
        return_label: bool = True,
    ):
        from PIL import Image  # local import

        self._Image = Image
        self.transform = transform
        self.return_label = return_label
        self.images, self.labels = load_arrays(path, indices)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = self._Image.fromarray(self.images[i])  # HWC uint8 -> PIL
        out = self.transform(img) if self.transform is not None else img
        if self.return_label:
            return out, self.labels[i]
        return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="Shapes3D data utility")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    args = ap.parse_args()

    if args.download:
        download_shapes3d(args.path)
    if args.info:
        import h5py

        with h5py.File(args.path, "r") as f:
            print("keys:", list(f.keys()))
            for k in f.keys():
                print(f"  {k}: shape={f[k].shape} dtype={f[k].dtype}")
            labels = f["labels"][:]
        print(f"\nN_TOTAL={len(labels)} (expected {N_TOTAL})")
        for fac in FACTORS:
            col = labels[:, fac.index]
            uniq = np.unique(col)
            print(
                f"  {fac.name:12s} {fac.kind:11s} nuniq={len(uniq):2d} "
                f"range=[{col.min():.3f},{col.max():.3f}] cyclic={fac.cyclic}"
            )


if __name__ == "__main__":
    _main()
