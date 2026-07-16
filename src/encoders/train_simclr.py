"""SimCLR encoder training — config-driven, device-agnostic, seed-addressable.

    python -m src.encoders.train_simclr --config configs/run/pilot_mps.yaml
    python -m src.encoders.train_simclr --config configs/run/reference_cuda.yaml \
        --set run.seed=1

Writes to ``results/encoders/<condition>_<strength>_seed<seed>/``:
    train_log.jsonl   per-epoch loss + collapse diagnostics
    metrics.json      final summary (loss, last diagnostics, NaN flag)
    backbone.pt       frozen-encoder checkpoint (backbone + projector state)
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from seeds import seed_everything, worker_init_fn

from ..data.registry import get_dataset
from ..data.splits import make_splits
from ..utils.config import apply_overrides, load_config
from ..utils.device import device_supports_amp, pick_device
from ..utils.logging import JsonlLogger
from .augmentations import TwoViewTransform, build_augmentation
from .backbone import ResNet18CIFAR
from .projector import MLPProjector
from .simclr import SimCLRModel, collapse_diagnostics, nt_xent_loss


def _build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup -> cosine decay over ``total_steps`` (per-step LambdaLR)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_id_from_cfg(cfg: dict) -> str:
    aug = cfg["augmentation"]
    return f"{aug['condition']}_{aug['strength']}_seed{cfg['run']['seed']}"


def _recover_final(log_path: Path) -> tuple[float, dict]:
    """Recover (final_loss, diagnostics) from the last train_log line."""
    last = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
    if not last or last.get("event") == "nan_abort":
        return float("nan"), {}
    keys = ("feat_std", "eff_rank", "alignment", "uniformity")
    diag = {k: last[k] for k in keys if k in last}
    return float(last.get("loss", float("nan"))), diag


def _finalize(out_dir, cfg, model, seed, final_loss, diag, epochs_run, nan_aborted=False):
    """Write metrics.json and the frozen backbone.pt checkpoint."""
    metrics = {
        "run_id": run_id_from_cfg(cfg),
        "seed": seed,
        "final_loss": final_loss,
        "nan_aborted": nan_aborted,
        "diagnostics": diag,
        "epochs_run": epochs_run,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(
        {
            "kind": "simclr",
            "seed": seed,
            "backbone": model.backbone.state_dict(),
            "projector": model.projector.state_dict(),
            "config": cfg,
            "final_loss": final_loss,
        },
        out_dir / "backbone.pt",
    )


def train_simclr(cfg: dict) -> Path:
    run = cfg["run"]
    seed = int(run["seed"])
    seed_everything(seed, deterministic=run.get("deterministic", True))
    device = pick_device(run.get("device"))
    use_amp = device_supports_amp(device) and run.get("amp", True)

    # --- data: encoder_train split, two-view SSL (no labels) ---
    spec = get_dataset(cfg["data"].get("dataset", "shapes3d"))
    n_total = cfg["data"].get("n_total", spec.n_total)
    splits = make_splits(n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    train_idx = splits["encoder_train"]
    if cfg["data"].get("subset"):
        train_idx = train_idx[: int(cfg["data"]["subset"])]

    aug = build_augmentation(
        cfg["augmentation"]["condition"], cfg["augmentation"]["strength"]
    )
    ds = spec.cls(
        train_idx,
        transform=TwoViewTransform(aug),
        path=cfg["data"].get("path", spec.default_path),
        return_label=False,
        in_memory=cfg["data"].get("in_memory", False),
    )
    num_workers = run.get("num_workers", 4)
    loader = DataLoader(
        ds,
        batch_size=run["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # --- model / optim / schedule ---
    model = SimCLRModel(
        ResNet18CIFAR(),
        MLPProjector(
            in_dim=512,
            hidden_dim=cfg["projector"]["hidden_dim"],
            out_dim=cfg["projector"]["out_dim"],
        ),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=run["lr"], weight_decay=run["weight_decay"]
    )
    steps_per_epoch = len(loader)
    total_steps = run["epochs"] * steps_per_epoch
    warmup_steps = run.get("warmup_epochs", 0) * steps_per_epoch
    sched = _build_scheduler(opt, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    temperature = float(cfg["simclr"]["temperature"])
    amp_ctx = (
        (lambda: torch.autocast(device_type="cuda")) if use_amp else nullcontext
    )

    # Compile the forward path for speed (kernel fusion only — same architecture,
    # loss, and hyperparameters, so the training regime is unchanged). Compile a
    # separate handle and keep ``model`` eager for diagnostics/checkpointing so
    # saved state_dicts stay prefix-clean. Default on for CUDA, off elsewhere.
    compile_cfg = run.get("compile")
    if compile_cfg is None:
        compile_cfg = device.type == "cuda"
    use_compile = bool(compile_cfg) and hasattr(torch, "compile")
    fwd = torch.compile(model) if use_compile else model

    out_dir = Path(cfg["output"]["dir"]) / run_id_from_cfg(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "last_ckpt.pt"
    final_path = out_dir / "backbone.pt"

    # Resume from the last epoch checkpoint if present (model/optim/sched/scaler;
    # RNG and dataloader state are not restored).
    start_epoch = 0
    if run.get("resume", True) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[simclr] resumed from {ckpt_path} at epoch {start_epoch}")

    if start_epoch >= run["epochs"]:
        if final_path.exists():
            print(f"[simclr] {run_id_from_cfg(cfg)} already completed — skipping")
            return final_path
        # Training reached the last epoch but finalization was interrupted before
        # backbone.pt was written. Finalize from the loaded checkpoint (recovering
        # loss/diagnostics from the last train_log line) instead of retraining.
        final_loss, last_diag = _recover_final(out_dir / "train_log.jsonl")
        _finalize(out_dir, cfg, model, seed, final_loss, last_diag, run["epochs"])
        print(f"[simclr] finalized {final_path} from checkpoint (no retrain)")
        return final_path

    logger = JsonlLogger(out_dir / "train_log.jsonl")
    print(
        f"[simclr] run={run_id_from_cfg(cfg)} device={device} amp={use_amp} "
        f"compile={use_compile} n={len(ds)} bs={run['batch_size']} "
        f"steps/ep={steps_per_epoch} epochs={run['epochs']} tau={temperature} "
        f"start_epoch={start_epoch}"
    )

    last_diag: dict = {}
    nan_aborted = False
    for epoch in range(start_epoch, run["epochs"]):
        model.train()
        running = 0.0
        v1 = v2 = None
        for v1, v2 in tqdm(loader, desc=f"ep {epoch}", leave=False):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                z1, z2 = fwd(v1), fwd(v2)
                loss = nt_xent_loss(z1, z2, temperature)
            if not torch.isfinite(loss):
                print(f"[simclr] NON-FINITE loss at epoch {epoch} — aborting")
                logger.log(epoch=epoch, loss=float("nan"), event="nan_abort")
                nan_aborted = True
                break
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        if nan_aborted:
            break

        # --- per-epoch collapse diagnostics on the last batch ---
        model.eval()
        with torch.no_grad():
            h1 = model.encode(v1)
            z1d = F.normalize(model.projector(h1), dim=1)
            z2d = model(v2)
            last_diag = collapse_diagnostics(h1.float(), z1d.float(), z2d.float())
        avg = running / steps_per_epoch
        logger.log(epoch=epoch, loss=avg, lr=sched.get_last_lr()[0], **last_diag)
        print(
            f"[simclr] epoch {epoch} loss={avg:.4f} feat_std={last_diag['feat_std']:.4f} "
            f"eff_rank={last_diag['eff_rank']:.1f} align={last_diag['alignment']:.3f} "
            f"unif={last_diag['uniformity']:.3f}"
        )
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
            },
            ckpt_path,
        )

    logger.close()
    final_loss = float("nan") if nan_aborted else avg
    _finalize(
        out_dir, cfg, model, seed, final_loss, last_diag,
        epoch + (0 if nan_aborted else 1), nan_aborted,
    )
    print(f"[simclr] saved {final_path} (final_loss={final_loss:.4f})")
    return final_path


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides a.b=c")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    train_simclr(cfg)


if __name__ == "__main__":
    _main()
