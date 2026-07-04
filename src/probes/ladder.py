"""Probe-capacity ladder (prereg §0) — the Phase-2 instrument's rungs.

Four rungs of monotone capacity, applied to the frozen 512-d encoder feature h:

    1. linear      Linear(512 -> out)
    2. mlp_small    512 -> 64  -> out
    3. mlp_large    512 -> 256 -> out
    4. mlp_deep     512 -> 256 -> 128 -> out

Capacity is reported as an explicit measure (trainable parameter count) per rung
(prereg §0 / FIX 3). Widths are NOT a frozen quantity (prereg §0) — locked here at
probe-build (decision D016). Probe-train size is held fixed across the ladder; the
only per-rung tuning is regularization (weight decay) selected on the val split,
never on test. Seed controls probe init (seeds.py convention).

Every rung is trained the same way for both factor kinds:
    categorical -> cross-entropy, reported as normalized accuracy (acc-chance)/(1-chance)
    continuous  -> MSE, reported as R^2 (unclipped — clipping biases encoder gain up)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, r2_score

from seeds import seed_everything

IN_DIM = 512  # backbone h

# Locked ladder (D016). hidden = () is the linear rung.
@dataclass(frozen=True)
class Rung:
    name: str
    hidden: tuple[int, ...]


LADDER: tuple[Rung, ...] = (
    Rung("linear", ()),
    Rung("mlp_small", (64,)),
    Rung("mlp_large", (256,)),
    Rung("mlp_deep", (256, 128)),
)

# Regularization tuned on val (prereg §0). Weight decay is the only per-rung knob.
WEIGHT_DECAY_GRID: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 4096
DEFAULT_LR = 1e-3


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def param_count(rung: Rung, out_dim: int) -> int:
    """Trainable parameters — the reported capacity measure for this rung."""
    n, d = 0, IN_DIM
    for h in rung.hidden:
        n += d * h + h
        d = h
    n += d * out_dim + out_dim
    return n


def _standardize(Xtr, *others):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return [(X - mu) / sd for X in (Xtr, *others)]


def _train_one(
    rung, Xtr, ytr, Xva, yva, Xte, yte, kind, out_dim, wd, seed, device, epochs, batch, lr
):
    """Train one probe at a fixed weight decay; return (val_metric, test_metric)."""
    seed_everything(seed)
    probe = MLPProbe(IN_DIM, rung.hidden, out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss() if kind == "categorical" else nn.MSELoss()

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(
        ytr, dtype=torch.long if kind == "categorical" else torch.float32, device=device
    )
    n = Xtr_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    probe.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = probe(Xtr_t[idx])
            target = ytr_t[idx]
            loss = loss_fn(out, target if kind == "categorical" else target.unsqueeze(1))
            loss.backward()
            opt.step()

    val_metric = _score(probe, Xva, yva, kind, out_dim, device)
    test_metric = _score(probe, Xte, yte, kind, out_dim, device)
    return val_metric, test_metric


@torch.no_grad()
def _score(probe, X, y, kind, out_dim, device) -> float:
    probe.eval()
    out = probe(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    if kind == "categorical":
        return float(accuracy_score(y, out.argmax(1)))
    return float(r2_score(y, out.ravel()))  # raw R^2 (val selection); unclipped


def fit_rung(
    rung: Rung,
    Xtr, ytr, Xva, yva, Xte, yte,
    kind: str,
    chance: float,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    wd_grid: tuple[float, ...] = WEIGHT_DECAY_GRID,
) -> dict:
    """Fit one ladder rung for one factor; select weight decay on val; report test.

    Returns normalized recoverability on test (norm-acc for categorical, R^2 for
    continuous), the raw score, the selected weight decay, and the rung's param count.
    """
    device = torch.device(device)
    out_dim = round(1.0 / chance) if kind == "categorical" else 1  # n_classes = 1/chance
    Xtr, Xva, Xte = _standardize(np.asarray(Xtr, np.float32),
                                 np.asarray(Xva, np.float32), np.asarray(Xte, np.float32))

    best = {"val": -np.inf}
    for wd in wd_grid:
        val_m, test_m = _train_one(
            rung, Xtr, ytr, Xva, yva, Xte, yte, kind, out_dim, wd, seed, device, epochs, batch, lr
        )
        if val_m > best["val"]:
            best = {"val": val_m, "test": test_m, "wd": wd}

    if kind == "categorical":
        recov = (best["test"] - chance) / (1.0 - chance)
        metric = "norm_acc"
    else:
        recov = best["test"]  # R^2, unclipped
        metric = "r2"
    return {
        "rung": rung.name,
        "recoverability": float(recov),
        "raw_test": float(best["test"]),
        "best_wd": float(best["wd"]),
        "params": param_count(rung, out_dim),
        "metric": metric,
    }


def fit_ladder(Xtr, ytr, Xva, yva, Xte, yte, kind, chance, **kw) -> list[dict]:
    """Fit all four rungs for one factor; returns one result dict per rung (in order)."""
    return [fit_rung(r, Xtr, ytr, Xva, yva, Xte, yte, kind, chance, **kw) for r in LADDER]
