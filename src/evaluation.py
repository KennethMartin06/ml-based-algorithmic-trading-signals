"""Classification metrics, probability calibration, block-bootstrap confidence intervals,
feature importance and regime analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from .backtester import sharpe_ratio

# ---------------------------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------------------------


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - 1[y = k])^2  (range 0..2)."""
    y = np.asarray(y_true, dtype=int)
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1)))


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], proba: np.ndarray | None = None
) -> dict:
    """Accuracy, balanced accuracy, macro/weighted precision-recall-F1, per-class scores,
    confusion matrix and (if probabilities are given) log loss and Brier score.

    Macro averaging weights every class equally (sensitive to the rare Neutral class); weighted
    averaging weights classes by their support.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "f1_weighted": float(fw),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "predicted_class_share": {
            name: float(np.mean(y_pred == i)) for i, name in enumerate(class_names)
        },
    }
    if proba is not None:
        proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
        out["brier"] = multiclass_brier(y_true, proba, len(class_names))
    else:
        out["log_loss"] = float("nan")
        out["brier"] = float("nan")
    return out


def reliability_curve(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed frequency of the event vs mean predicted probability in equal-width bins."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = ids == b
        if m.any():
            rows.append({"bin": b, "mean_predicted": prob[m].mean(), "observed_frequency": outcome[m].mean(), "count": int(m.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> float:
    curve = reliability_curve(prob, outcome, n_bins)
    if curve.empty:
        return float("nan")
    w = curve["count"] / curve["count"].sum()
    return float(np.sum(w * np.abs(curve["mean_predicted"] - curve["observed_frequency"])))


# ---------------------------------------------------------------------------------------------
# block bootstrap (time-series aware)
# ---------------------------------------------------------------------------------------------


def circular_block_bootstrap_indices(n: int, block_len: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Index matrix (n_boot x n) of circular block-bootstrap resamples.

    Blocks of consecutive observations preserve short-range autocorrelation and volatility
    clustering, which an i.i.d. bootstrap would destroy.
    """
    rng = np.random.default_rng(seed)
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def bootstrap_ci(values: np.ndarray, stat_fn, block_len: int = 20, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile CI of ``stat_fn`` applied to block-bootstrap resamples of ``values`` (rows)."""
    values = np.asarray(values)
    n = len(values)
    if n < 2 * block_len:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    idx = circular_block_bootstrap_indices(n, block_len, n_boot, seed)
    stats = np.array([stat_fn(values[i]) for i in idx], dtype=float)
    stats = stats[np.isfinite(stats)]
    alpha = (1 - ci) / 2
    return {
        "estimate": float(stat_fn(values)),
        "low": float(np.quantile(stats, alpha)) if len(stats) else float("nan"),
        "high": float(np.quantile(stats, 1 - alpha)) if len(stats) else float("nan"),
        "share_positive": float(np.mean(stats > 0)) if len(stats) else float("nan"),
    }


def sharpe_stat(annualization: int = 252, risk_free_rate: float = 0.0):
    return lambda r: sharpe_ratio(pd.Series(r), risk_free_rate, annualization)


def sharpe_difference_stat(annualization: int = 252, risk_free_rate: float = 0.0):
    """Statistic on a 2-column array [strategy, benchmark]: Sharpe(strategy) - Sharpe(benchmark).
    Both columns are resampled with the SAME block indices (paired bootstrap)."""
    return lambda a: (sharpe_ratio(pd.Series(a[:, 0]), risk_free_rate, annualization)
                      - sharpe_ratio(pd.Series(a[:, 1]), risk_free_rate, annualization))


def accuracy_difference_stat():
    """Statistic on [correct_model, correct_baseline] indicator columns."""
    return lambda a: float(a[:, 0].mean() - a[:, 1].mean())


# ---------------------------------------------------------------------------------------------
# feature analysis
# ---------------------------------------------------------------------------------------------


def permutation_importance_frame(model, X: np.ndarray, y: np.ndarray, feature_names: list[str], n_repeats: int = 10, seed: int = 0,
                                 n_classes: int | None = None) -> pd.DataFrame:
    """Permutation importance (increase in log loss when a feature is shuffled) on held-out data.

    Computed on the VALIDATION period so that the test set is not used for analysis that could
    influence modelling choices. Importance measures predictive reliance, not causality.
    """
    from .models import predict_proba_full, safe_log_loss

    k = n_classes or int(np.max(y)) + 1

    def neg_log_loss(est, X_, y_):
        return -safe_log_loss(y_, predict_proba_full(est, X_, k), k)

    res = permutation_importance(model, X, np.asarray(y, dtype=int), scoring=neg_log_loss, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (
        pd.DataFrame({"feature": feature_names, "importance_mean": res.importances_mean, "importance_std": res.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def mutual_information_frame(X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int = 0) -> pd.DataFrame:
    """Mutual information between each feature and the class label on TRAINING data."""
    mi = mutual_info_classif(X, np.asarray(y, dtype=int), random_state=seed)
    return pd.DataFrame({"feature": feature_names, "mutual_information": mi}).sort_values("mutual_information", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------------------------


def regime_labels(features: pd.DataFrame, train_index: pd.Index, vol_col: str = "vol_20", trend_col: str = "trend_regime") -> pd.DataFrame:
    """Volatility regime (above/below the TRAINING-period median vol) and trend regime (above/below
    the 200-day SMA), both known at decision time."""
    vol_median = features.loc[train_index, vol_col].median()
    out = pd.DataFrame(index=features.index)
    out["volatility_regime"] = np.where(features[vol_col] > vol_median, "high_vol", "low_vol")
    out["trend_regime"] = np.where(features[trend_col] > 0, "uptrend", "downtrend")
    return out


def f1_macro(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)), zero_division=0)
