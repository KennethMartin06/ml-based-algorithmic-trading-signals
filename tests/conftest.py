"""Shared deterministic fixtures. Synthetic data is used ONLY to test code behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import adjust_ohlc, generate_synthetic_ohlcv, validate_ohlcv
from src.utils import load_config


@pytest.fixture(scope="session")
def raw_ohlcv() -> pd.DataFrame:
    """~3 years of deterministic synthetic raw (yfinance-style) OHLCV."""
    return generate_synthetic_ohlcv("TEST", "2015-01-01", "2017-12-31", seed=7)


@pytest.fixture(scope="session")
def prices(raw_ohlcv) -> pd.DataFrame:
    """Validated, adjusted lower-case OHLCV frame as used by the pipeline."""
    clean, _ = validate_ohlcv(raw_ohlcv, "TEST")
    return adjust_ohlc(clean)


@pytest.fixture()
def tiny_prices() -> pd.DataFrame:
    """Hand-checkable 6-day price frame."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 103.0, 102.0, 104.0, 106.0],
            "high": [101.0, 103.0, 104.0, 104.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0, 101.0, 103.0, 105.0],
            "close": [100.5, 102.0, 102.5, 103.5, 105.0, 106.5],
            "volume": [1e6, 1.1e6, 0.9e6, 1.2e6, 1.0e6, 1.05e6],
        },
        index=idx,
    )


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(cfg) -> dict:
    """Config with tiny grids so model-level tests run in seconds."""
    from src.utils import deep_update

    out = deep_update(cfg, {
        "split": {"train_end": "2016-06-30", "val_end": "2016-12-31"},
        "models": {"tuning": {"cv_splits": 2},
                   "random_forest": {"fixed": {"n_estimators": 20, "max_features": "sqrt"}},
                   "lstm": {"max_epochs": 3, "patience": 2, "tune": False, "seq_len": 5, "hidden_size": 8}},
        "robustness": {"permutation_repeats": 1, "bootstrap": {"n_boot": 50, "block_len": 5, "ci": 0.95}},
    })
    out["models"]["logistic_regression"]["grid"] = {"C": [0.1], "class_weight": [None]}
    out["models"]["random_forest"]["grid"] = {"max_depth": [3], "min_samples_leaf": [20], "class_weight": [None]}
    out["models"]["xgboost"]["grid"] = {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [20], "class_weight": [None]}
    out["features"]["regime_window"] = 60
    out["features"]["trend_window"] = 60
    return out


@pytest.fixture()
def rng():
    return np.random.default_rng(123)
