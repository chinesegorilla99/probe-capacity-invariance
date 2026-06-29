"""Augmentation conditions — the invariance "treatments".

``build_augmentation(condition, strength)`` returns a single-view transform
(PIL -> CHW float tensor in [0,1]). ``TwoViewTransform`` wraps it into the SimCLR
positive-pair generator.

Phase 1 trains only the ``standard`` SimCLR stack (a strong, canonical mix) for
the reference encoder, plus a ``control`` (minimal) stack. The per-factor
conditions (color / position / orientation / scale) are dispatched here so the
Phase-3 sweep reuses this function unchanged — they currently raise
NotImplementedError to keep Phase 1 scope honest.
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T

IMAGE_SIZE = 64

CONDITIONS = ("standard", "control", "color", "position", "orientation", "scale")
STRENGTHS = ("weak", "strong")


class TwoViewTransform:
    """Generate two independently-augmented views of one image (SimCLR pair)."""

    def __init__(self, base: Callable):
        self.base = base

    def __call__(self, x):
        return self.base(x), self.base(x)


def eval_transform() -> Callable:
    """Deterministic transform for probing/extraction: just PIL -> tensor."""
    return T.Compose([T.ToTensor()])


def build_augmentation(
    condition: str = "standard",
    strength: str = "strong",
    image_size: int = IMAGE_SIZE,
) -> Callable:
    """Return a single-view augmentation transform for a (condition, strength)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    if strength not in STRENGTHS:
        raise ValueError(f"unknown strength {strength!r}; choose from {STRENGTHS}")

    if condition == "standard":
        # Canonical SimCLR stack: crop + flip + color jitter + grayscale.
        jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.3, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.RandomApply([jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    if condition == "control":
        # Minimal augmentation: a light crop only — a non-trivial positive pair
        # that targets no specific factor (the prereg "Control-aug" baseline).
        crop_scale = (0.8, 1.0) if strength == "strong" else (0.9, 1.0)
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=crop_scale, antialias=True),
                T.ToTensor(),
            ]
        )

    # color / position / orientation / scale — wired for Phase 3, not Phase 1.
    raise NotImplementedError(
        f"condition {condition!r} is a Phase-3 per-factor treatment; "
        "Phase 1 trains only 'standard' (+ 'control')."
    )
