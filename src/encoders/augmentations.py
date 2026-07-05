"""Augmentation conditions — the invariance "treatments".

``build_augmentation(condition, strength)`` returns a single-view transform
(PIL -> CHW float tensor in [0,1]). ``TwoViewTransform`` wraps it into the SimCLR
positive-pair generator.

Phase 1 trains the ``standard`` SimCLR stack (a strong, canonical mix) for the
reference encoder, plus a ``control`` (minimal) stack. The per-factor conditions
(color / position / orientation / scale) are the Phase-3 invariance treatments:
each varies ONLY its targeted factor across the two views and leaves the image
otherwise intact, so the induced invariance is clean and interpretable (no
confounding geometry/colour). Each is offered at ``weak`` and ``strong`` strength.
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

    weak = strength == "weak"

    if condition == "color":
        # Target: object/floor/wall hue (Shapes3D). Colour-jitter only, no
        # geometry — the two views differ solely in colour, inducing hue
        # invariance. Grayscale added at strong strength (canonical SimCLR).
        if weak:
            jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.1)
            return T.Compose([T.RandomApply([jitter], p=0.8), T.ToTensor()])
        jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.5)
        return T.Compose(
            [
                T.RandomApply([jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )

    if condition == "position":
        # Target: x/y position (dSprites). Translate only — scale/rotation fixed
        # — so the views differ solely in object location. ``fill=0`` matches the
        # black dSprites background (also fine for the Shapes3D extension).
        frac = 0.10 if weak else 0.25
        return T.Compose(
            [
                T.RandomAffine(degrees=0, translate=(frac, frac), fill=0),
                T.ToTensor(),
            ]
        )

    if condition == "orientation":
        # Target: orientation. Rotate only. Strong spans the full circle to match
        # dSprites' full-rotation orientation factor; weak is a limited arc.
        degrees = 15 if weak else 180
        return T.Compose([T.RandomRotation(degrees, fill=0), T.ToTensor()])

    if condition == "scale":
        # Target: scale. Isotropic zoom only (no translation/rotation), so the
        # views differ solely in object size.
        scale_range = (0.9, 1.1) if weak else (0.6, 1.4)
        return T.Compose(
            [
                T.RandomAffine(degrees=0, scale=scale_range, fill=0),
                T.ToTensor(),
            ]
        )

    raise ValueError(f"unhandled condition {condition!r}")  # unreachable
