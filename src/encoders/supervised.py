"""Supervised-reference encoder — recoverability CEILING (diagnostic-only).

A backbone trained end-to-end on the ground-truth factors (CE for shape, MSE for
standardized continuous factors). It enters NO decision rule (prereg §5) — it
just shows how recoverable each factor *can* be, as context for the gate and
(later) for encoder-gain G. Only the frozen backbone ``h`` is probed afterward,
exactly like the SimCLR encoder, so heads are discarded post-training.

    python -m src.encoders.supervised --config configs/run/supervised.yaml
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from seeds import seed_everything, worker_init_fn

from ..data.shapes3d import DEFAULT_PATH, FACTORS, N_TOTAL, Shapes3D
from ..data.splits import make_splits
from ..utils.config import apply_overrides, load_config
from ..utils.device import device_supports_amp, pick_device
from ..utils.logging import JsonlLogger
from .augmentations import eval_transform
from .backbone import FEATURE_DIM, ResNet18CIFAR

CONT_IDX = [f.index for f in FACTORS if f.kind == "continuous"]
SHAPE_IDX = next(f.index for f in FACTORS if f.kind == "categorical")
N_SHAPE = next(f.n_values for f in FACTORS if f.kind == "categorical")


class SupervisedModel(nn.Module):
    """Backbone + linear heads (shape classifier + continuous regressor)."""

    def __init__(self, n_cont: int):
        super().__init__()
        self.backbone = ResNet18CIFAR()
        self.head_shape = nn.Linear(FEATURE_DIM, N_SHAPE)
        self.head_cont = nn.Linear(FEATURE_DIM, n_cont)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head_shape(h), self.head_cont(h)


def train_supervised(cfg: dict) -> Path:
    run = cfg["run"]
    seed = int(run["seed"])
    seed_everything(seed, deterministic=run.get("deterministic", True))
    device = pick_device(run.get("device"))
    use_amp = device_supports_amp(device) and run.get("amp", True)

    n_total = cfg["data"].get("n_total", N_TOTAL)
    splits = make_splits(n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    train_idx = splits["encoder_train"]
    if cfg["data"].get("subset"):
        train_idx = train_idx[: int(cfg["data"]["subset"])]

    ds = Shapes3D(
        train_idx, transform=eval_transform(), path=cfg["data"].get("path", DEFAULT_PATH)
    )
    # Standardize continuous targets on this split (stored for reproducibility).
    cont = ds.labels[:, CONT_IDX]
    mean, std = cont.mean(0), cont.std(0) + 1e-8

    num_workers = run.get("num_workers", 4)
    loader = DataLoader(
        ds,
        batch_size=run["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    model = SupervisedModel(n_cont=len(CONT_IDX)).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=run["lr"], weight_decay=run["weight_decay"]
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_ctx = (
        (lambda: torch.autocast(device_type="cuda")) if use_amp else nullcontext
    )
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    out_dir = Path(cfg["output"]["dir"]) / f"supervised_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last_ckpt.pt"
    final_path = out_dir / "backbone.pt"

    # Resume from the last epoch checkpoint if present (see train_simclr.py).
    start_epoch = 0
    if run.get("resume", True) and ckpt_path.exists():
        resumed = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        opt.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        start_epoch = resumed["epoch"] + 1
        print(f"[supervised] resumed from {ckpt_path} at epoch {start_epoch}")

    if start_epoch >= run["epochs"] and final_path.exists():
        print("[supervised] already completed — skipping")
        return final_path

    logger = JsonlLogger(out_dir / "train_log.jsonl")
    print(
        f"[supervised] device={device} amp={use_amp} n={len(ds)} "
        f"epochs={run['epochs']} start_epoch={start_epoch}"
    )

    model.train()
    for epoch in range(start_epoch, run["epochs"]):
        running = 0.0
        for x, y in tqdm(loader, desc=f"sup epoch {epoch}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            shape_t = y[:, SHAPE_IDX].long()
            cont_t = (y[:, CONT_IDX] - mean_t) / std_t
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                pred_shape, pred_cont = model(x)
                loss = F.cross_entropy(pred_shape, shape_t) + F.mse_loss(
                    pred_cont, cont_t
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        avg = running / len(loader)
        logger.log(epoch=epoch, loss=avg)
        print(f"[supervised] epoch {epoch} loss={avg:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
            },
            ckpt_path,
        )

    logger.close()
    torch.save(
        {
            "kind": "supervised",
            "seed": seed,
            "backbone": model.backbone.state_dict(),
            "cont_idx": CONT_IDX,
            "cont_mean": mean.tolist(),
            "cont_std": std.tolist(),
            "config": cfg,
        },
        final_path,
    )
    print(f"[supervised] saved {final_path}")
    return final_path


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    train_supervised(cfg)


if __name__ == "__main__":
    _main()
