"""Rule-based trading baselines and naive classification benchmarks.

All baseline positions are indexed by decision date t and use information up to the close of t,
exactly like the ML signals, so every strategy is executed by the same backtester under the same
timing, cost and sizing rules (synopsis section 5.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import sma


def buy_and_hold_positions(index: pd.Index) -> pd.Series:
    """Fully invested from the first decision date onward (one entry trade, never exits)."""
    return pd.Series(1.0, index=index, name="position")


def ma_crossover_positions(close: pd.Series, short: int = 10, long: int = 50, allow_short: bool = False) -> pd.Series:
    """Long when SMA_short(t) > SMA_long(t), otherwise flat (or short if ``allow_short``).

    Both averages include the close of day t, which is known when the decision is made after the
    close; the trade is executed at the next fill time defined by the execution convention.
    Before both averages exist (warm-up) the position is flat.
    """
    if short >= long:
        raise ValueError("short window must be shorter than long window")
    s, l = sma(close, short), sma(close, long)
    pos = pd.Series(np.where(s > l, 1.0, -1.0 if allow_short else 0.0), index=close.index, name="position")
    return pos.where(s.notna() & l.notna(), 0.0)


def persistence_positions(close: pd.Series, allow_short: bool = False) -> pd.Series:
    """Naive momentum: long if today's close-to-close return was positive."""
    ret = close.pct_change()
    pos = pd.Series(np.where(ret > 0, 1.0, -1.0 if allow_short else 0.0), index=close.index, name="position")
    return pos.where(ret.notna(), 0.0)


# ---------------------------------------------------------------------------------------------
# classification benchmarks
# ---------------------------------------------------------------------------------------------


class MajorityClassBaseline:
    """Predicts the most frequent training class; probabilities = training class frequencies."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def fit(self, y: np.ndarray) -> "MajorityClassBaseline":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=self.n_classes).astype(float)
        self.prior_ = counts / counts.sum()
        self.majority_ = int(np.argmax(counts))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.majority_, dtype=int)

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.prior_, (n, 1))


def persistence_labels(close: pd.Series, scheme: str, threshold: float) -> pd.Series:
    """Predict tomorrow's class = class of today's close-to-close return (known at decision time)."""
    from .targets import make_labels

    return make_labels(close.pct_change(), scheme, threshold)
