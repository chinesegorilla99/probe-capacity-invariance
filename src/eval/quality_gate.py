"""Encoder-quality gate (prereg §5) — the Phase-1 trustworthiness check.

Freeze each encoder, fit a SINGLE linear probe per factor, and report normalized
recoverability for SimCLR vs the random-encoder floor vs the supervised-reference
ceiling. PASS means the SimCLR encoder demonstrably learned real structure —
required before ANY probing/metric work (Phase 2).

These are Phase-1 trustworthiness thresholds, deliberately distinct from the
frozen Phase-3 epsilon_G metric layer (not built here).

    python -m src.eval.quality_gate --config configs/run/pilot_mps.yaml \
        --simclr results/encoders/standard_strong_seed0/backbone.pt \
        --supervised results/encoders/supervised_seed0/backbone.pt \
        --random-seed 0 --out results/quality_gate/pilot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.shapes3d import DEFAULT_PATH, FACTORS, N_TOTAL, Shapes3D
from ..data.splits import make_splits
from ..encoders.augmentations import eval_transform
from ..encoders.backbone import ResNet18CIFAR
from ..encoders.random_encoder import build_random_encoder
from ..utils.config import load_config
from ..utils.device import pick_device
from .extract import extract_features
from .metrics import (
    linear_recoverability_categorical,
    linear_recoverability_continuous,
)

# Gate thresholds (Phase-1 trustworthiness, NOT the frozen epsilon_G).
SHAPE_PASS = 0.90
SHAPE_MARGIN_OVER_RANDOM = 0.30
ABOVE_RANDOM_MARGIN = 0.05
MIN_FACTORS_ABOVE = 3

SHAPE_IDX = next(f.index for f in FACTORS if f.kind == "categorical")


def load_backbone(ckpt_path: str | Path, device: torch.device) -> ResNet18CIFAR:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet18CIFAR()
    model.load_state_dict(ckpt["backbone"])
    return model.to(device).eval()


def _features_for_splits(backbone, datasets, device, bs, nw):
    return {
        name: extract_features(backbone, ds, device, batch_size=bs, num_workers=nw)
        for name, ds in datasets.items()
    }


def evaluate_encoder(name: str, feats: dict) -> dict:
    """Per-factor normalized recoverability for one encoder's extracted features."""
    Htr, Ytr = feats["probe_train"]
    Hva, Yva = feats["probe_val"]
    Hte, Yte = feats["probe_test"]

    out: dict[str, dict] = {}
    for fac in FACTORS:
        if fac.kind == "categorical":
            ytr = np.rint(Ytr[:, fac.index]).astype(int)
            yva = np.rint(Yva[:, fac.index]).astype(int)
            yte = np.rint(Yte[:, fac.index]).astype(int)
            out[fac.name] = linear_recoverability_categorical(
                Htr, ytr, Hva, yva, Hte, yte, fac.chance
            )
        else:
            out[fac.name] = linear_recoverability_continuous(
                Htr, Ytr[:, fac.index], Hva, Yva[:, fac.index], Hte, Yte[:, fac.index]
            )
    return out


def evaluate_gate(simclr: dict, random: dict) -> dict:
    shape_recov = simclr["shape"]["recoverability"]
    rand_shape = random["shape"]["recoverability"]
    shape_ok = shape_recov >= SHAPE_PASS and shape_recov >= rand_shape + SHAPE_MARGIN_OVER_RANDOM

    above = [
        fac.name
        for fac in FACTORS
        if simclr[fac.name]["recoverability"]
        > random[fac.name]["recoverability"] + ABOVE_RANDOM_MARGIN
    ]
    above_ok = len(above) >= MIN_FACTORS_ABOVE

    return {
        "passed": bool(shape_ok and above_ok),
        "shape_ok": bool(shape_ok),
        "shape_recoverability": shape_recov,
        "shape_random": rand_shape,
        "factors_above_random": above,
        "n_above_random": len(above),
        "above_ok": bool(above_ok),
        "criteria": {
            "shape_pass": SHAPE_PASS,
            "shape_margin_over_random": SHAPE_MARGIN_OVER_RANDOM,
            "above_random_margin": ABOVE_RANDOM_MARGIN,
            "min_factors_above": MIN_FACTORS_ABOVE,
        },
    }


def _print_table(table: dict, encoders: list[str]) -> None:
    width = 13
    header = "factor".ljust(width) + "type".ljust(7) + "".join(
        e.ljust(width) for e in encoders
    )
    print("\n" + header)
    print("-" * len(header))
    for fac in FACTORS:
        row = fac.name.ljust(width) + ("cat" if fac.kind == "categorical" else "cont").ljust(7)
        for e in encoders:
            v = table[e][fac.name]["recoverability"]
            row += f"{v:+.3f}".ljust(width)
        print(row)


def run_gate(args) -> dict:
    cfg = load_config(args.config)
    device = pick_device(args.device)
    bs, nw = args.batch_size, args.num_workers

    n_total = cfg["data"].get("n_total", N_TOTAL)
    splits = make_splits(n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    path = cfg["data"].get("path", DEFAULT_PATH)
    print(f"[gate] device={device} | building probe splits "
          f"(train={len(splits['probe_train'])}, val={len(splits['probe_val'])}, "
          f"test={len(splits['probe_test'])})")
    datasets = {
        name: Shapes3D(splits[name], transform=eval_transform(), path=path, return_label=True)
        for name in ("probe_train", "probe_val", "probe_test")
    }

    table: dict[str, dict] = {}

    # Random-encoder floor (defines G later); average over given seeds if multiple.
    rand_seed = args.random_seed[0]
    rand_bb = build_random_encoder(rand_seed, device)
    table["random"] = evaluate_encoder(
        "random", _features_for_splits(rand_bb, datasets, device, bs, nw)
    )

    # SimCLR (primary = first ckpt; extras used for the reproducibility check).
    simclr_per_seed = {}
    for i, ck in enumerate(args.simclr):
        bb = load_backbone(ck, device)
        res = evaluate_encoder(f"simclr{i}", _features_for_splits(bb, datasets, device, bs, nw))
        simclr_per_seed[str(ck)] = res
        if i == 0:
            table["simclr"] = res

    encoders = ["random", "simclr"]
    if args.supervised:
        sup_bb = load_backbone(args.supervised, device)
        table["supervised"] = evaluate_encoder(
            "supervised", _features_for_splits(sup_bb, datasets, device, bs, nw)
        )
        encoders.append("supervised")

    _print_table(table, encoders)

    gate = evaluate_gate(table["simclr"], table["random"])

    # Reproducibility across SimCLR seeds: spread of shape recoverability.
    shape_by_seed = {
        k: v["shape"]["recoverability"] for k, v in simclr_per_seed.items()
    }
    spread = (max(shape_by_seed.values()) - min(shape_by_seed.values())) if len(shape_by_seed) > 1 else 0.0
    repro = {
        "shape_recoverability_by_ckpt": shape_by_seed,
        "shape_spread": spread,
        "n_seeds": len(shape_by_seed),
    }

    verdict = "PASS ✅" if gate["passed"] else "FAIL ❌"
    print(f"\n[gate] shape recoverability (SimCLR) = {gate['shape_recoverability']:+.3f} "
          f"(random {gate['shape_random']:+.3f}, need >= {SHAPE_PASS})")
    print(f"[gate] factors above random (+{ABOVE_RANDOM_MARGIN}): "
          f"{gate['n_above_random']} -> {gate['factors_above_random']}")
    if len(shape_by_seed) > 1:
        print(f"[gate] reproducibility: shape spread across {len(shape_by_seed)} seeds = {spread:.3f}")
    print(f"[gate] VERDICT: {verdict}")

    report = {
        "device": str(device),
        "config": args.config,
        "table": table,
        "gate": gate,
        "reproducibility": repro,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[gate] wrote {out}")
    return report


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--simclr", nargs="+", required=True, help="one or more backbone.pt")
    ap.add_argument("--supervised", default=None)
    ap.add_argument("--random-seed", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_gate(args)


if __name__ == "__main__":
    _main()
