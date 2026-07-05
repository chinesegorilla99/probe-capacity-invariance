"""dSprites loader + factor metadata (the position arm).

dSprites ships as a single ~26 MB ``.npz`` (grayscale 64x64 binary sprites) with:
    imgs            : (737280, 64, 64) uint8  in {0, 1}
    latents_values  : (737280, 6)      float64
    latents_classes : (737280, 6)      int64   # integer factor indices

Latent columns (order): color, shape, scale, orientation, posX, posY. ``color``
is degenerate (a single value) and dropped. The retained factors (prereg §0):
shape = CATEGORICAL (normalized accuracy); scale, orientation, posX, posY =
CONTINUOUS (R^2). The headline position arm targets posX/posY. Orientation is a
FULL rotation here (cyclic) — sin/cos handling is a probe-build concern.

Labels are the integer ``latents_classes`` (uniformly spaced ordinal targets);
for R^2 an affine rescaling of the target is irrelevant, and the categorical
factor uses class indices directly. Each factor combination appears once
(1*3*6*40*32*32 == 737280), so every image is a distinct instance — clean for
instance-discrimination SSL, matching the Shapes3D setup.

Images are grayscale; ``__getitem__`` scales {0,1} -> {0,255} and converts to RGB
so the shared 3-channel backbone consumes them unchanged.

    python -m src.data.dsprites --download        # fetch + verify the npz
    python -m src.data.dsprites --build-cache      # decompress -> image memmap
    python -m src.data.dsprites --info             # schema / factor stats
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DEFAULT_PATH = Path("data/raw/dsprites.npz")
N_TOTAL = 737_280
IMAGE_SIZE = 64


@dataclass(frozen=True)
class Factor:
    name: str
    index: int  # column into the 6-wide latents array
    kind: str  # "continuous" | "categorical"
    n_values: int
    cyclic: bool = False

    @property
    def chance(self) -> float:
        return 1.0 / self.n_values if self.kind == "categorical" else float("nan")


# ``index`` matches the latents column order; color (col 0) is dropped.
FACTORS: tuple[Factor, ...] = (
    Factor("shape", 1, "categorical", 3),
    Factor("scale", 2, "continuous", 6),
    # dSprites orientation is the full [0, 2*pi) circle -> genuinely cyclic.
    Factor("orientation", 3, "continuous", 40, cyclic=True),
    Factor("pos_x", 4, "continuous", 32),
    Factor("pos_y", 5, "continuous", 32),
)
FACTOR_NAMES = tuple(f.name for f in FACTORS)


def download_dsprites(path: str | Path = DEFAULT_PATH, force: bool = False) -> Path:
    """Download the dSprites npz to ``path`` (idempotent; content-verified)."""
    path = Path(path)
    if path.exists() and not force:
        print(f"[dsprites] already present: {path} ({path.stat().st_size} bytes)")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[dsprites] downloading {URL} -> {path} (~26 MB) ...")
    urllib.request.urlretrieve(URL, path)
    # Verify by loading the archive header rather than an exact byte count.
    with np.load(path, allow_pickle=False) as f:
        shape = f["imgs"].shape
    if shape != (N_TOTAL, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"unexpected imgs shape {shape}; file may be corrupt")
    print(f"[dsprites] done: {path.stat().st_size} bytes, imgs {shape}")
    return path


def load_arrays(
    path: str | Path = DEFAULT_PATH, indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read images + integer factor labels for ``indices`` (sorted) into memory.

    Returns ``(images uint8 (N,64,64), labels float32 (N,6))``.
    """
    with np.load(path, allow_pickle=False) as f:
        imgs = f["imgs"]
        labels = f["latents_classes"]
        if indices is None:
            images = imgs[:]
            lab = labels[:].astype(np.float32)
        else:
            idx = np.sort(np.asarray(indices))
            images = imgs[idx]
            lab = labels[idx].astype(np.float32)
    return images, lab


def memmap_path(path: str | Path = DEFAULT_PATH) -> Path:
    """Location of the uncompressed image cache built next to the npz."""
    return Path(path).with_suffix(".images.npy")


def ensure_image_memmap(path: str | Path = DEFAULT_PATH) -> Path:
    """Decompress all images into an uncompressed ``.npy`` memmap once.

    The npz stores ``imgs`` as one compressed member, so a single access
    decompresses the whole (737280, 64, 64) array (~3 GB) — we stream it out to a
    flat on-disk uint8 memmap that datasets then mmap (rows page in on demand,
    shared across forked workers, no re-decompression). Idempotent — rebuilds
    only if the cache is missing or the wrong shape.
    """
    path = Path(path)
    mm_path = memmap_path(path)
    with np.load(path, allow_pickle=False) as f:
        shape = f["imgs"].shape
        if mm_path.exists():
            existing = np.load(mm_path, mmap_mode="r")
            ok = existing.shape == shape and existing.dtype == np.uint8
            del existing
            if ok:
                return mm_path
        imgs = f["imgs"]  # decompresses the full array into RAM
        mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8, shape=shape)
        block = 50_000
        for start in range(0, shape[0], block):
            stop = min(start + block, shape[0])
            mm[start:stop] = imgs[start:stop]
        mm.flush()
        del mm, imgs
    return mm_path


class DSprites(Dataset):
    """Memmap-backed dSprites split (mirrors :class:`data.shapes3d.Shapes3D`).

    Images are served from an uncompressed on-disk memmap of the full dataset
    (built once via :func:`ensure_image_memmap`); the memmap is opened lazily per
    process so spawn workers receive only the path, not the ~3 GB array. Grayscale
    {0,1} pixels are scaled to {0,255} and converted to RGB per item.

    Args mirror Shapes3D: ``indices``, ``transform`` (a ``TwoViewTransform`` for
    SSL or ``eval_transform`` for extraction), ``path``, ``return_label``,
    ``in_memory``.
    """

    def __init__(
        self,
        indices: np.ndarray,
        transform=None,
        path: str | Path = DEFAULT_PATH,
        return_label: bool = True,
        in_memory: bool = False,
    ):
        self.transform = transform
        self.return_label = return_label
        self.indices = np.sort(np.asarray(indices))
        self._mm_path = ensure_image_memmap(path)
        self._images = None
        self._ram = None
        if in_memory:
            full = np.asarray(np.load(self._mm_path, mmap_mode="r"))
            self._ram = np.ascontiguousarray(full[self.indices])
            del full
        self.labels = None
        if return_label:
            with np.load(path, allow_pickle=False) as f:
                self.labels = f["latents_classes"][self.indices].astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def _image_array(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self._mm_path, mmap_mode="r")
        return self._images

    def __getitem__(self, i: int):
        from PIL import Image

        arr = self._ram[i] if self._ram is not None else self._image_array()[self.indices[i]]
        # {0,1} uint8 -> {0,255} grayscale -> RGB for the 3-channel backbone.
        img = Image.fromarray(np.asarray(arr) * np.uint8(255)).convert("RGB")
        out = self.transform(img) if self.transform is not None else img
        if self.return_label:
            return out, self.labels[i]
        return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="dSprites data utility")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build-cache", action="store_true", help="build image memmap")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    args = ap.parse_args()

    if args.download:
        download_dsprites(args.path)
    if args.build_cache:
        mm = ensure_image_memmap(args.path)
        print(f"[dsprites] image cache ready: {mm} ({mm.stat().st_size} bytes)")
    if args.info:
        with np.load(args.path, allow_pickle=False) as f:
            print("keys:", list(f.keys()))
            for k in f.keys():
                print(f"  {k}: shape={f[k].shape} dtype={f[k].dtype}")
            labels = f["latents_classes"][:]
        print(f"\nN_TOTAL={len(labels)} (expected {N_TOTAL})")
        for fac in FACTORS:
            col = labels[:, fac.index]
            uniq = np.unique(col)
            print(
                f"  {fac.name:12s} {fac.kind:11s} nuniq={len(uniq):2d} "
                f"range=[{col.min():.0f},{col.max():.0f}] cyclic={fac.cyclic}"
            )


if __name__ == "__main__":
    _main()
