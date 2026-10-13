"""Classification targets aligned with the execution convention.

Notation: a *decision* is made after the close of day t using features from rows <= t.

* ``next_open`` (default): the order is filled at Open[t+1] and the position is held until the next
  decision is filled at Open[t+2]. The tradable return of that decision is

      r_t = Open[t+1+h] / Open[t+1] - 1

* ``close`` (market-on-close approximation, optimistic): r_t = Close[t+h] / Close[t] - 1.

Labels (three-class, neutral band ``theta``):  Down = 0 if r_t < -theta, Neutral = 1 if
|r_t| <= theta, Up = 2 if r_t > theta.  Binary: Up = 1 if r_t > 0 else Down = 0.

The label of row t depends on prices up to row ``t + label_end_offset``; rows whose label window
extends beyond the available data are NaN and are never used for training or evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import class_names_for

CONVENTIONS = ("next_open", "close")


def label_end_offset(horizon: int, convention: str) -> int:
    """Number of rows after t whose prices are needed to compute the label of row t."""
    if convention == "next_open":
        return horizon + 1
    if convention == "close":
        return horizon
    raise ValueError(f"Unknown execution convention {convention!r}; expected one of {CONVENTIONS}")


def forward_returns(df: pd.DataFrame, horizon: int = 1, convention: str = "next_open") -> pd.Series:
    """Tradable forward return for a decision taken after the close of each row."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if convention == "next_open":
        entry = df["open"].shift(-1)
        exit_ = df["open"].shift(-(horizon + 1))
    elif convention == "close":
        entry = df["close"]
        exit_ = df["close"].shift(-horizon)
    else:
        raise ValueError(f"Unknown execution convention {convention!r}")
    return (exit_ / entry - 1.0).rename(f"fwd_ret_{horizon}")


def period_returns(df: pd.DataFrame, convention: str = "next_open") -> pd.Series:
    """One-period return captured by a position decided at row t (used by the backtester)."""
    return forward_returns(df, horizon=1, convention=convention).rename("period_ret")


def make_labels(fwd_ret: pd.Series, scheme: str = "three_class", threshold: float = 0.001) -> pd.Series:
    """Map forward returns to integer class labels (float dtype so that NaN survives)."""
    if threshold < 0:
        raise ValueError("neutral threshold must be non-negative")
    labels = pd.Series(np.nan, index=fwd_ret.index, dtype=float, name="label")
    valid = fwd_ret.notna()
    if scheme == "three_class":
        labels[valid & (fwd_ret < -threshold)] = 0.0
        labels[valid & (fwd_ret.abs() <= threshold)] = 1.0
        labels[valid & (fwd_ret > threshold)] = 2.0
    elif scheme == "binary":
        labels[valid & (fwd_ret <= 0)] = 0.0
        labels[valid & (fwd_ret > 0)] = 1.0
    else:
        raise ValueError(f"Unknown target scheme {scheme!r}")
    return labels


def build_target_frame(df: pd.DataFrame, target_cfg: dict, convention: str) -> pd.DataFrame:
    """Forward return, label, the date the label becomes known, and the backtest period return."""
    h = int(target_cfg.get("horizon", 1))
    scheme = target_cfg.get("scheme", "three_class")
    theta = float(target_cfg.get("neutral_threshold", 0.001))
    fwd = forward_returns(df, h, convention)
    out = pd.DataFrame(index=df.index)
    out["fwd_ret"] = fwd
    out["label"] = make_labels(fwd, scheme, theta)
    offset = label_end_offset(h, convention)
    dates = pd.Series(df.index, index=df.index)
    out["label_end_date"] = dates.shift(-offset)
    out["period_ret"] = period_returns(df, convention)
    return out


def up_class_index(scheme: str) -> int:
    return len(class_names_for(scheme)) - 1


def down_class_index(scheme: str) -> int:
    return 0


def class_distribution(labels: pd.Series, scheme: str) -> pd.DataFrame:
    names = class_names_for(scheme)
    counts = labels.dropna().astype(int).value_counts().reindex(range(len(names)), fill_value=0)
    total = counts.sum()
    return pd.DataFrame(
        {"class": names, "count": counts.values, "frequency": counts.values / total if total else np.nan}
    )
