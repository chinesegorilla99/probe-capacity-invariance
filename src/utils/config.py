"""Tiny YAML config loader with ``_base_`` inheritance and dotted overrides.

A run config inherits defaults via a top-level ``_base_`` (a path or list of
paths, relative to the config file) and overrides only what changes. This keeps
one config == one fully-specified experiment cell while avoiding duplication.

Example::

    _base_: ../base.yaml
    train: {epochs: 20, batch_size: 256}
    run: {device: mps}
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``_base_`` inheritance depth-first."""
    path = Path(path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}

    bases = cfg.pop("_base_", [])
    if isinstance(bases, (str, Path)):
        bases = [bases]

    merged: dict[str, Any] = {}
    for b in bases:
        merged = _deep_update(merged, load_config(path.parent / b))
    merged = _deep_update(merged, cfg)
    return merged


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` CLI overrides (values parsed as YAML scalars)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out
