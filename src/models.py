"""Tabular ML models (Logistic Regression, Random Forest, XGBoost) and purged time-series tuning.

* Logistic Regression is wrapped in a ``Pipeline(StandardScaler, LogisticRegression)`` so the scaler
  is re-fitted on the training part of every CV fold - the validation fold never influences it.
* Hyperparameters are chosen by an exhaustive search over a small grid, scored by mean multiclass
  log loss across ``TimeSeriesSplit`` folds with a ``gap`` equal to the label horizon (purging).
  Only training-period data is used; the validation and test periods are never touched.
* ``class_weight`` is treated as a hyperparameter (None vs 'balanced') for every model. XGBoost
  has no ``class_weight`` argument, so 'balanced' is implemented with per-sample weights.
* If ``xgboost`` cannot be imported, ``HistGradientBoostingClassifier`` is used instead and is
  always reported under the name "HistGradientBoosting (XGBoost fallback)" - never as XGBoost.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = xgb.__version__
except Exception:  # pragma: no cover - depends on the environment
    xgb = None
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = None

DISPLAY_NAMES = {
    "majority_class": "Majority class",
    "persistence": "Naive persistence",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGradientBoosting (XGBoost fallback)",
    "lstm": "LSTM",
    "ensemble": "Ensemble (soft vote)",
}

TABULAR_MODELS = ("logistic_regression", "random_forest", "xgboost")


def resolve_model_name(name: str) -> str:
    """Map 'xgboost' to the documented fallback when the library is unavailable."""
    if name == "xgboost" and not XGBOOST_AVAILABLE:
        return "hist_gradient_boosting"
    return name


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def balanced_sample_weight(y: np.ndarray, n_classes: int) -> np.ndarray:
    """w_i = n / (K * n_{y_i}) - identical to sklearn's class_weight='balanced'."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = np.nan
    w = len(y) / (n_classes * counts)
    return np.nan_to_num(w[y], nan=0.0)


def build_estimator(name: str, params: dict, n_classes: int, seed: int = 42):
    """Instantiate an unfitted estimator. ``params`` may contain 'class_weight'."""
    p = dict(params)
    cw = p.pop("class_weight", None)
    if name == "logistic_regression":
        clf = LogisticRegression(C=p.pop("C", 1.0), class_weight=cw, max_iter=5000, random_state=seed, **p)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "random_forest":
        return RandomForestClassifier(class_weight=cw, random_state=seed, n_jobs=-1, **p)
    if name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed")
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        return xgb.XGBClassifier(
            objective=objective,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **p,
        )
    if name == "hist_gradient_boosting":
        mapping = {"n_estimators": "max_iter", "reg_lambda": "l2_regularization"}
        hp = {mapping.get(k, k): v for k, v in p.items() if k not in XGB_ONLY_PARAMS}
        return HistGradientBoostingClassifier(class_weight=cw, random_state=seed, **hp)
    raise ValueError(f"Unknown model {name!r}")


# XGBoost options without a HistGradientBoosting equivalent (dropped - and not reported - for the fallback)
XGB_ONLY_PARAMS = ("subsample", "colsample_bytree", "min_child_weight")


def applicable_params(name: str, params: dict | None) -> dict:
    """Parameters that are actually applied by ``name`` (so reports never list ignored settings)."""
    params = dict(params or {})
    if name == "hist_gradient_boosting":
        return {k: v for k, v in params.items() if k not in XGB_ONLY_PARAMS}
    return params


def fit_estimator(name: str, params: dict, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 42):
    y = np.asarray(y, dtype=int)
    if name == "xgboost":
        # XGBoost needs contiguous labels 0..m-1: encode the classes present in this training set
        # and remember them so that predict_proba_full can map columns back to all K classes.
        present = np.unique(y)
        if len(present) < 2:
            raise ValueError(f"XGBoost needs at least two classes in the training data, got {present.tolist()}")
        encoded = np.searchsorted(present, y)
        model = build_estimator(name, params, len(present), seed)
        weights = balanced_sample_weight(encoded, len(present)) if params.get("class_weight") == "balanced" else None
        model.fit(X, encoded, sample_weight=weights)
        model.original_classes_ = present
        return model
    model = build_estimator(name, params, n_classes, seed)
    model.fit(X, y)
    return model


def predict_proba_full(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities with one column per class 0..K-1, even if a class was absent in training."""
    proba = model.predict_proba(X)
    classes = getattr(model, "original_classes_", None)
    if classes is None:
        classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["clf"].classes_
    out = np.zeros((len(X), n_classes))
    for j, c in enumerate(np.asarray(classes, dtype=int)):
        out[:, c] = proba[:, j]
    return out


def time_series_grid_search(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    grid: dict,
    fixed: dict | None,
    n_classes: int,
    n_splits: int = 4,
    gap: int = 2,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Exhaustive grid search with purged ``TimeSeriesSplit`` on training data.

    Returns the best parameter set (lowest mean validation-fold log loss) and the full CV table.
    """
    fixed = applicable_params(name, fixed)
    y = np.asarray(y, dtype=int)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for params in expand_grid(grid):
        full = {**fixed, **params}
        losses, f1s = [], []
        t0 = time.perf_counter()
        for tr, va in splitter.split(X):
            model = fit_estimator(name, full, X[tr], y[tr], n_classes, seed)
            proba = predict_proba_full(model, X[va], n_classes)
            losses.append(safe_log_loss(y[va], proba, n_classes))
            f1s.append(f1_score(y[va], proba.argmax(axis=1), average="macro", labels=list(range(n_classes)), zero_division=0))
        rows.append(
            {
                **{k: ("None" if v is None else v) for k, v in params.items()},
                "cv_log_loss_mean": float(np.mean(losses)),
                "cv_log_loss_std": float(np.std(losses)),
                "cv_macro_f1_mean": float(np.mean(f1s)),
                "fit_seconds": time.perf_counter() - t0,
            }
        )
    table = pd.DataFrame(rows).sort_values("cv_log_loss_mean").reset_index(drop=True)
    best_idx = int(np.argmin([r["cv_log_loss_mean"] for r in rows]))
    best = {**fixed, **expand_grid(grid)[best_idx]}
    return best, table


def safe_log_loss(y_true, proba, n_classes: int) -> float:
    """Multiclass log loss after clipping and re-normalising rows (float32 outputs may not sum to 1)."""
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(log_loss(np.asarray(y_true, dtype=int), p, labels=list(range(n_classes))))


def tree_importance(model, feature_names: list[str]) -> pd.Series | None:
    """Impurity (RF) or gain-based (XGBoost) importance, normalised to sum to 1."""
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=feature_names, dtype=float)
        total = imp.sum()
        return imp / total if total > 0 else imp
    if hasattr(est, "coef_"):
        # mean absolute standardised coefficient across classes
        imp = pd.Series(np.abs(est.coef_).mean(axis=0), index=feature_names, dtype=float)
        return imp / imp.sum()
    return None
