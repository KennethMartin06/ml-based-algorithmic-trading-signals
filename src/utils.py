"""Shared utilities: configuration, paths, seeding, logging and JSON helpers."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML configuration and apply optional nested overrides."""
    cfg_path = resolve_path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return deep_update(cfg, overrides or {})


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # PyTorch is optional
        pass


def get_logger(name: str = "trading", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, indent=2)
    return p


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def class_names_for(scheme: str) -> list[str]:
    if scheme == "three_class":
        return ["Down", "Neutral", "Up"]
    if scheme == "binary":
        return ["Down", "Up"]
    raise ValueError(f"Unknown target scheme: {scheme!r}")


def fmt(value: Any, digits: int = 3, pct: bool = False) -> str:
    """Format a metric for Markdown tables; NaN/None -> 'N/A'."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.{max(digits - 1, 1)}f}%" if pct else f"{v:.{digits}f}"
