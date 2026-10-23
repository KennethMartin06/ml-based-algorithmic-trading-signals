"""Regression tests for issues found in the adversarial code review (see docs/submission_checklist.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import experiment as ex
from src.models import XGBOOST_AVAILABLE, applicable_params, fit_estimator, predict_proba_full, time_series_grid_search
from src.lstm_model import TORCH_AVAILABLE
from src.reporting import key_findings
from src.utils import deep_update

# ---------------------------------------------------------------------------------------------
# XGBoost with a class missing from the training data
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
@pytest.mark.parametrize("present", [(0, 2), (0, 1), (1, 2)])
def test_xgboost_handles_absent_class(rng, present):
    """XGBoost requires contiguous labels; fit_estimator encodes the present classes and
    predict_proba_full maps the columns back, leaving a zero column for the absent class."""
    X = rng.standard_normal((120, 4))
    y = np.where(X[:, 0] > 0, present[1], present[0])
    model = fit_estimator("xgboost", {"n_estimators": 10, "max_depth": 2, "class_weight": "balanced"}, X, y, 3)
    p = predict_proba_full(model, X, 3)
    absent = ({0, 1, 2} - set(present)).pop()
    assert p.shape == (120, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(p[:, absent] == 0.0)
    # the model learned the signal: high P(present[1]) when X0 > 0
    assert p[X[:, 0] > 0, present[1]].mean() > 0.8


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
def test_xgboost_single_class_raises_clear_error(rng):
    with pytest.raises(ValueError, match="at least two classes"):
        fit_estimator("xgboost", {"n_estimators": 5}, rng.standard_normal((20, 2)), np.zeros(20, dtype=int), 3)


def test_hist_gradient_boosting_reports_only_applied_params(rng):
    """The fallback must not report XGBoost-only settings it silently ignores."""
    fixed = {"subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0}
    assert applicable_params("hist_gradient_boosting", fixed) == {"reg_lambda": 1.0}
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] > 0).astype(int)
    best, _ = time_series_grid_search("hist_gradient_boosting", X, y, {"n_estimators": [20]}, fixed, 2, n_splits=2, gap=1)
    assert set(best) == {"reg_lambda", "n_estimators"}


# ---------------------------------------------------------------------------------------------
# LSTM early-stopping slice inside the training period
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n, frac, gap", [(1000, 0.2, 2), (501, 0.25, 1), (50, 0.2, 3)])
def test_lstm_internal_split_is_purged_and_inside_train(n, frac, gap):
    """Fitting rows precede the early-stopping slice by at least ``gap`` rows, so no fitting label
    window overlaps the slice; both parts come from the training rows only."""
    train_pos = np.arange(300, 300 + n)
    fit, es = ex.lstm_internal_split(train_pos, frac, gap)
    assert len(es) == max(1, int(round(n * frac)))
    assert es[-1] == train_pos[-1]
    assert fit.max() + gap < es.min()
    assert set(fit) | set(es) <= set(train_pos)
    assert len(fit) + gap + len(es) == n


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_holdout_lstm_never_uses_validation_rows(small_cfg, prices, monkeypatch):
    """The LSTM's fit/early-stopping positions all lie inside the training period."""
    from src import lstm_model

    seen = {}
    real_fit = lstm_model.LSTMSequenceClassifier.fit

    def spy(self, X, y, train_pos, val_pos, scaler_rows=None):
        seen["fit"], seen["es"], seen["scaler"] = np.asarray(train_pos), np.asarray(val_pos), np.asarray(scaler_rows)
        return real_fit(self, X, y, train_pos, val_pos, scaler_rows)

    monkeypatch.setattr(lstm_model.LSTMSequenceClassifier, "fit", spy)
    cfg = deep_update(small_cfg, {"models": {"lstm": {"enabled": True}}})
    res = ex.train_holdout_ticker("TEST", prices, cfg, None, with_lstm=True, with_analysis=False)
    data, _, _ = ex.prepare_ticker_dataset(prices, cfg)
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    for key in ("fit", "es", "scaler"):
        assert data.index[seen[key].max()] <= train_end, key
    assert "lstm__p0" in res["predictions"].columns


# ---------------------------------------------------------------------------------------------
# run isolation
# ---------------------------------------------------------------------------------------------


def test_build_config_isolates_partial_runs(cfg):
    base = cfg["experiment"]["results_dir"]
    full = ex.build_config(deep_update(cfg, {}))
    assert full["experiment"]["results_dir"] == base and full["_is_default_full_run"]
    one = ex.build_config(deep_update(cfg, {}), tickers=["AAPL"])
    assert one["experiment"]["results_dir"] == f"{base}_AAPL" and not one["_is_default_full_run"]
    nolstm = ex.build_config(deep_update(cfg, {}), skip_lstm=True, skip_walk_forward=True)
    assert nolstm["experiment"]["results_dir"] == f"{base}_nolstm_nowf"
    assert not nolstm["models"]["lstm"]["enabled"] and not nolstm["walk_forward"]["enabled"]
    quick = ex.build_config(deep_update(cfg, {}), quick=True)
    assert quick["experiment"]["results_dir"] == "results_quick" and not quick["_is_default_full_run"]
    same = ex.build_config(deep_update(cfg, {}), tickers=list(cfg["data"]["tickers"]))
    assert same["experiment"]["results_dir"] == base  # passing the default universe is not a variant


def test_synthetic_fallback_is_relabelled_and_redirected(cfg, monkeypatch):
    """If any ticker falls back to synthetic data, the run is flagged SYNTHETIC and moved away from results/."""
    fake = [{"ticker": "SPY", "is_synthetic": True}]
    monkeypatch.setattr(ex, "load_universe", lambda c, force_synthetic=False: ({"SPY": None}, fake))
    c = ex.build_config(deep_update(cfg, {}))
    ex.load_data(c)
    assert c["_synthetic"] and not c["_is_default_full_run"]
    assert c["experiment"]["results_dir"].endswith("_synthetic_fallback")


def test_reset_outputs_only_touches_generated_subfolders(tmp_path):
    for sub in ("metrics", "figures", "predictions", "reports", "models", "keep_me"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "x.txt").write_text("x")
    ex.reset_outputs(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep_me"]


# ---------------------------------------------------------------------------------------------
# rule-based key findings
# ---------------------------------------------------------------------------------------------


def _tables():
    cls = pd.DataFrame([
        {"ticker": "A", "split": "test", "model_key": "majority_class", "model": "Majority class", "accuracy": 0.5, "f1_macro": 0.2},
        {"ticker": "A", "split": "test", "model_key": "xgboost", "model": "XGBoost", "accuracy": 0.52, "f1_macro": 0.3},
    ])
    base = {"ticker": "A", "threshold": np.nan}
    trading = pd.DataFrame([
        {**base, "strategy": "Buy & Hold", "kind": "baseline", "model_key": "buy_and_hold", "sharpe": 0.5,
         "cumulative_return": 0.4, "max_drawdown": -0.3332852676923494, "exposure": 1.0, "n_trades": 1},
        # a clone of buy-and-hold
        {**base, "strategy": "Random Forest (conf.)", "kind": "ml_confidence", "model_key": "random_forest", "sharpe": 0.5,
         "cumulative_return": 0.4, "max_drawdown": -0.3332852676923494, "exposure": 1.0, "n_trades": 1},
        # same drawdown up to floating-point noise, better Sharpe
        {**base, "strategy": "XGBoost (conf.)", "kind": "ml_confidence", "model_key": "xgboost", "sharpe": 0.6,
         "cumulative_return": 0.3, "max_drawdown": -0.3332852676923492, "exposure": 0.9, "n_trades": 7},
        {**base, "strategy": "MA Crossover (10/50)", "kind": "baseline", "model_key": "ma_crossover", "sharpe": 0.4,
         "cumulative_return": 0.2, "max_drawdown": -0.2, "exposure": 0.6, "n_trades": 9},
    ])
    boot = pd.DataFrame([
        {"ticker": "A", "strategy": "Random Forest (conf.)", "statistic": "sharpe_minus_buy_and_hold", "estimate": 0.0, "low": 0.0, "high": 0.0},
        {"ticker": "A", "strategy": "XGBoost (conf.)", "statistic": "sharpe_minus_buy_and_hold", "estimate": 0.1, "low": -0.2, "high": 0.4},
        {"ticker": "A", "strategy": "MA Crossover (10/50)", "statistic": "sharpe_minus_buy_and_hold", "estimate": -0.1, "low": -0.5, "high": -0.01},
    ])
    return cls, trading, boot


def test_key_findings_ignore_floating_point_ties_and_flag_clones():
    cls, trading, boot = _tables()
    text = "\n".join(key_findings(cls, trading, boot, None))
    assert "Sharpe ratio higher for 1, lower for 0" in text
    assert "maximum drawdown smaller for 0, larger for 0" in text  # 2e-16 difference is not a result
    assert "1 of 2 ML strategies were invested on every test day" in text and "Random Forest on A" in text


def test_key_findings_split_bootstrap_by_kind_and_report_identical_intervals():
    cls, trading, boot = _tables()
    text = "\n".join(key_findings(cls, trading, boot, None))
    assert "ML strategies: 0 of 2 entirely above 0, 0 entirely below 0, 1 include 0, 1 identical to buy-and-hold" in text
    assert "rule-based baselines: 0 of 1 entirely above 0, 1 entirely below 0, 0 include 0" in text
