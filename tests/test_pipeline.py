"""End-to-end tests of the holdout pipeline in src/experiment.py on SYNTHETIC prices.

The synthetic series only exercises the code paths; nothing here says anything about real
predictive performance. One train + evaluate run is shared by most tests (module fixture) so the
whole file stays well under the runtime budget. Nothing is written to disk (paths=None).
"""

from __future__ import annotations

import copy
import itertools
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from src import experiment as ex
from src import models
from src.backtester import BacktestConfig
from src.lstm_model import TORCH_AVAILABLE
from src.utils import fmt

QUIET = logging.getLogger("tests.pipeline.quiet")
QUIET.setLevel(logging.WARNING)

K = 3  # three_class: Down=0, Neutral=1, Up=2
CLS_COLS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss"]


def boosting_key() -> str:
    return "xgboost" if models.XGBOOST_AVAILABLE else "hist_gradient_boosting"


@pytest.fixture(scope="module")
def pcfg(small_cfg):
    """Private copy of the small config so no test can leak mutations into other modules."""
    return copy.deepcopy(small_cfg)


@pytest.fixture(scope="module")
def holdout(pcfg, prices):
    """One full holdout run: training (incl. LSTM + ensemble) and evaluation with robustness analyses."""
    tr = ex.train_holdout_ticker("TEST", prices, pcfg, paths=None, logger=QUIET, with_lstm=True, with_analysis=False)
    ev = ex.evaluate_holdout_ticker("TEST", prices, tr["predictions"], pcfg, logger=QUIET, with_analysis=True)
    return tr, ev


def _split_parts(pred):
    return pred[pred["split"] == "val"], pred[pred["split"] == "test"]


# ---------------------------------------------------------------------------------------------
# training stage: prediction frame
# ---------------------------------------------------------------------------------------------


def test_prediction_frame_contains_exactly_val_and_test_rows(holdout):
    """The saved predictions hold every validation and test decision row - and nothing else."""
    tr, _ = holdout
    pred = tr["predictions"]
    val, test = _split_parts(pred)
    assert set(pred["split"]) == {"val", "test"}
    assert len(val) == tr["split"]["val"]["rows"] > 0
    assert len(test) == tr["split"]["test"]["rows"] > 0
    assert len(pred) == len(val) + len(test)
    assert pred.index.is_unique and pred.index.is_monotonic_increasing
    assert not pred[["label", "fwd_ret", "period_ret"]].isna().any().any()
    assert set(pred["label"].unique()) <= {0, 1, 2}


def test_probability_columns_sum_to_one_for_every_model(holdout):
    """Each model (incl. LSTM, ensemble and majority class) has K probability columns forming a distribution."""
    tr, _ = holdout
    pred = tr["predictions"]
    keys = ex.models_in_predictions(pred)
    expected = {"majority_class", "logistic_regression", "random_forest", boosting_key(), "ensemble"}
    if TORCH_AVAILABLE:
        expected.add("lstm")
    assert set(keys) == expected
    for key in keys:
        P = pred[ex.proba_cols(key, K)].to_numpy()
        assert np.isfinite(P).all(), key
        assert (P >= 0).all() and (P <= 1 + 1e-9).all(), key
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-5, err_msg=key)


def test_ensemble_is_average_of_members_chosen_on_validation(holdout):
    """The ensemble is the equal-weight mean of >= 2 members, chosen as the lowest-validation-log-loss subset."""
    tr, _ = holdout
    pred = tr["predictions"]
    members = tr["tuning"]["ensemble"]["best_params"]["members"]
    assert len(members) >= 2 and "majority_class" not in members
    mean = np.mean([pred[ex.proba_cols(m, K)].to_numpy() for m in members], axis=0)
    np.testing.assert_allclose(pred[ex.proba_cols("ensemble", K)].to_numpy(), mean, atol=1e-9)

    val, _ = _split_parts(pred)
    table = tr["tuning"]["ensemble"]["cv_table"]
    assert table.loc[0, "members"] == "+".join(members)
    assert table["val_log_loss"].is_monotonic_increasing
    ens = val[ex.proba_cols("ensemble", K)].to_numpy()
    ens = ens / ens.sum(axis=1, keepdims=True)  # LSTM members are float32: remove rounding before sklearn
    chosen = log_loss(val["label"], ens, labels=[0, 1, 2])
    assert chosen == pytest.approx(table["val_log_loss"].min(), rel=1e-6)


def test_splits_are_chronological_and_purged(holdout, pcfg, prices):
    """train <= train_end < val <= val_end < test, and no validation label looks past val_end."""
    tr, _ = holdout
    pred = tr["predictions"]
    val, test = _split_parts(pred)
    train_end, val_end = pd.Timestamp(pcfg["split"]["train_end"]), pd.Timestamp(pcfg["split"]["val_end"])
    assert pd.Timestamp(tr["split"]["train"]["end"]) <= train_end
    assert val.index.min() > train_end and val.index.max() <= val_end
    assert test.index.min() > val_end
    assert val.index.max() < test.index.min()

    data, _, _ = ex.prepare_ticker_dataset(prices, pcfg)
    assert (data.loc[val.index, "label_end_date"] <= val_end).all()
    # purging removed exactly the labelled window rows whose label ends after val_end
    window = data[(data.index > train_end) & (data.index <= val_end) & data["label"].notna()]
    purged = window.index.difference(val.index)
    assert len(purged) > 0
    assert (window.loc[purged, "label_end_date"] > val_end).all()


def test_validation_predictions_ignore_test_period_prices(holdout, pcfg, prices):
    """Leakage check: rewriting every price AFTER val_end leaves all validation probabilities
    (and hence thresholds and ensemble members) unchanged, while test probabilities do change."""
    tr, _ = holdout
    val_end = pd.Timestamp(pcfg["split"]["val_end"])
    altered = prices.copy()
    after = altered.index > val_end
    factor = np.exp(np.cumsum(np.random.default_rng(99).normal(0.0, 0.02, after.sum())))
    for col in ("open", "high", "low", "close", "raw_close"):
        if col in altered:
            altered.loc[after, col] = altered.loc[after, col].to_numpy() * factor

    tr2 = ex.train_holdout_ticker("TEST", altered, pcfg, paths=None, logger=QUIET, with_lstm=True, with_analysis=False)
    p1, p2 = tr["predictions"], tr2["predictions"]
    v1, t1 = _split_parts(p1)
    v2, t2 = _split_parts(p2)
    assert v1.index.equals(v2.index)
    for key in ex.models_in_predictions(p1):
        cols = ex.proba_cols(key, K)
        np.testing.assert_allclose(v2[cols].to_numpy(), v1[cols].to_numpy(), rtol=0, atol=1e-6, err_msg=key)
    assert tr2["tuning"]["ensemble"]["best_params"] == tr["tuning"]["ensemble"]["best_params"]
    lr = ex.proba_cols("logistic_regression", K)
    assert not np.allclose(t1[lr].to_numpy(), t2.loc[t1.index, lr].to_numpy()), "perturbation must reach the test period"


# ---------------------------------------------------------------------------------------------
# evaluation stage: strategies, thresholds and metrics
# ---------------------------------------------------------------------------------------------


def test_every_strategy_is_backtested_on_the_same_test_dates(holdout):
    """Baselines and ML strategies are scored on identical test decision dates (fair comparison)."""
    tr, ev = holdout
    _, test = _split_parts(tr["predictions"])
    assert len(ev["backtests"]) == len(ev["strategies"]) >= 3
    for name, bt in ev["backtests"].items():
        assert bt.frame.index.equals(test.index), name
        assert ev["strategies"][name]["positions"].index.equals(test.index), name
        assert bt.metrics["n_days"] == len(test)
    starts = set(ev["trading"]["start"]); ends = set(ev["trading"]["end"])
    assert len(starts) == 1 and len(ends) == 1


def test_thresholds_come_from_the_configured_grid_or_none(holdout, pcfg):
    """Every ML model gets a validation-chosen P(Up) threshold from the grid (or None = argmax fallback);
    confidence strategies use it and argmax strategies use none."""
    tr, ev = holdout
    grid = pcfg["signals"]["threshold_grid"]
    ml_keys = [k for k in ex.models_in_predictions(tr["predictions"]) if k != "majority_class"]
    assert set(ev["thresholds"]) == set(ml_keys)
    for key, tau in ev["thresholds"].items():
        assert tau is None or any(np.isclose(tau, g) for g in grid), (key, tau)
    trading = ev["trading"]
    for _, row in trading[trading["kind"] == "ml_confidence"].iterrows():
        tau = ev["thresholds"][row["model_key"]]
        assert (pd.isna(row["threshold"]) and tau is None) or row["threshold"] == tau
    assert trading.loc[trading["kind"] == "ml_argmax", "threshold"].isna().all()
    assert trading.loc[trading["kind"] == "baseline", "threshold"].isna().all()


def test_thresholds_depend_only_on_validation_rows(holdout, pcfg, prices):
    """Scrambling the TEST probabilities cannot change the chosen thresholds (test set used once)."""
    tr, ev = holdout
    pred = tr["predictions"].copy()
    is_test = pred["split"] == "test"
    prob_cols = [c for c in pred.columns if "__p" in c]
    rng = np.random.default_rng(5)
    for key in ex.models_in_predictions(pred):
        cols = ex.proba_cols(key, K)
        pred.loc[is_test, cols] = rng.dirichlet([1.0] * K, size=int(is_test.sum()))
    assert not pred.loc[is_test, prob_cols].equals(tr["predictions"].loc[is_test, prob_cols])
    _, thresholds, _ = ex.build_test_strategies("TEST", prices, pred, pcfg, K)
    assert thresholds == ev["thresholds"]


def test_trading_metrics_are_finite_where_expected(holdout, pcfg):
    """Return, drawdown, exposure and turnover metrics are finite for every strategy; Sharpe is
    finite exactly when the strategy is ever invested (zero exposure -> zero volatility -> N/A)."""
    tr, ev = holdout
    trading = ev["trading"]
    always = ["final_equity", "cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown",
              "exposure", "turnover_annual", "n_trades", "costs_paid"]
    assert np.isfinite(trading[always].to_numpy(dtype=float)).all()
    assert trading["max_drawdown"].between(-1.0, 0.0).all()
    assert trading["exposure"].between(0.0, 1.0).all()
    invested = trading["exposure"] > 0
    assert np.isfinite(trading.loc[invested, "sharpe"]).all()
    assert np.isfinite(trading.loc[invested, "sortino"]).all()
    assert trading.loc[~invested, "sharpe"].isna().all()
    assert (trading.loc[~invested, "cumulative_return"] == 0).all()

    # Buy & Hold: always invested, a single entry trade, return = (1 - cost) * prod(1 + R) - 1
    _, test = _split_parts(tr["predictions"])
    bh = trading.set_index("strategy").loc["Buy & Hold"]
    c = BacktestConfig.from_config(pcfg).cost_rate
    assert bh["exposure"] == 1.0 and bh["n_trades"] == 1
    assert bh["cumulative_return"] == pytest.approx((1 - c) * np.prod(1 + test["period_ret"].to_numpy()) - 1, rel=1e-10)


def test_classification_metrics_match_the_saved_predictions(holdout):
    """Test-period accuracy in the metrics table equals argmax accuracy recomputed from the prediction frame."""
    tr, ev = holdout
    _, test = _split_parts(tr["predictions"])
    cls = ev["classification"]
    cls_test = cls[cls["split"] == "test"].set_index("model_key")
    for key in ex.models_in_predictions(tr["predictions"]):
        acc = (test[ex.proba_cols(key, K)].to_numpy().argmax(axis=1) == test["label"].to_numpy()).mean()
        assert cls_test.loc[key, "accuracy"] == pytest.approx(acc), key
        assert np.isfinite(cls_test.loc[key, "log_loss"])
    pers_acc = (test["persistence__pred"] == test["label"]).mean()
    assert cls_test.loc["persistence", "accuracy"] == pytest.approx(pers_acc)
    assert np.isnan(cls_test.loc["persistence", "log_loss"])  # persistence outputs labels, not probabilities


def test_comparison_table_reports_na_classification_for_rule_baselines(holdout):
    """Buy & Hold and the MA crossover are trading rules, not classifiers: their classification
    columns are N/A, whereas every ML strategy carries its model's test classification metrics."""
    _, ev = holdout
    comp = ex.comparison_table(ev["classification"], ev["trading"])
    assert len(comp) == len(ev["trading"])
    rules = comp[comp["strategy"].eq("Buy & Hold") | comp["strategy"].str.startswith("MA Crossover")]
    assert len(rules) == 2
    assert rules[CLS_COLS].isna().all().all()
    assert all(fmt(v) == "N/A" for v in rules["accuracy"])
    assert np.isfinite(rules["sharpe"]).all()

    ml = comp[comp["kind"].isin(["ml_confidence", "ml_argmax"])]
    assert len(ml) > 0
    assert np.isfinite(ml[CLS_COLS].to_numpy(dtype=float)).all()
    # the (conf.) and (argmax) strategy of one model share the same underlying classifier metrics
    ml = ml.assign(model=ml["strategy"].str.replace(r" \((conf\.|argmax)\)$", "", regex=True))
    assert (ml.groupby("model")["accuracy"].nunique() == 1).all()


def test_robustness_outputs_are_consistent(holdout, pcfg):
    """With positions fixed, higher transaction costs can only lower cumulative return; at zero cost
    Buy & Hold earns exactly the compounded period return."""
    tr, ev = holdout
    for key in ("cost_sensitivity", "threshold_sensitivity", "regimes", "bootstrap", "calibration"):
        assert key in ev and len(ev[key]) > 0, key
    cs = ev["cost_sensitivity"]
    assert sorted(cs["cost_bps"].unique()) == sorted(pcfg["robustness"]["cost_bps_grid"])
    for strategy, part in cs.groupby("strategy"):
        part = part.sort_values("cost_bps")
        assert part["cumulative_return"].is_monotonic_decreasing, strategy
    _, test = _split_parts(tr["predictions"])
    bh0 = cs[(cs["strategy"] == "Buy & Hold") & (cs["cost_bps"] == 0)]["cumulative_return"].iloc[0]
    assert bh0 == pytest.approx(np.prod(1 + test["period_ret"].to_numpy()) - 1, rel=1e-10)
    b = ev["bootstrap"].dropna(subset=["low", "high"])
    assert (b["low"] <= b["high"]).all()
    assert ev["calibration"]["ece_p_up"].between(0, 1).all()


def test_xgboost_fallback_is_reported_under_its_own_name(pcfg, prices, monkeypatch):
    """Without xgboost the pipeline trains HistGradientBoosting and never labels it 'XGBoost'."""
    monkeypatch.setattr(models, "XGBOOST_AVAILABLE", False)
    tr = ex.train_holdout_ticker("TEST", prices, pcfg, paths=None, logger=QUIET, with_lstm=False, with_analysis=False)
    pred = tr["predictions"]
    keys = ex.models_in_predictions(pred)
    assert "hist_gradient_boosting" in keys and "xgboost" not in keys
    ev = ex.evaluate_holdout_ticker("TEST", prices, pred, pcfg, logger=QUIET, with_analysis=False)
    names = list(ev["strategies"])
    assert "HistGradientBoosting (XGBoost fallback) (conf.)" in names
    assert not any(n.startswith("XGBoost") for n in names)


# ---------------------------------------------------------------------------------------------
# selection rules used on the validation period
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def threshold_case():
    """40 validation days. P(Up) is 0.8 on days 0-9 (all positive returns), 0.6 on days 10-19
    (all negative returns) and 0.4 on days 20-39 (zero-mean noise). Costs are zero."""
    idx = pd.bdate_range("2020-01-01", periods=40)
    p_up = np.r_[np.full(10, 0.8), np.full(10, 0.6), np.full(20, 0.4)]
    proba = np.column_stack([(1 - p_up) / 2, (1 - p_up) / 2, p_up])
    ret = np.r_[np.tile([0.010, 0.004], 5), np.tile([-0.008, -0.002], 5), np.tile([0.003, -0.003], 10)]
    period_ret = pd.Series(ret, index=idx)
    bt_cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    scfg = {"threshold_grid": [0.3, 0.5, 0.65, 0.7, 0.9], "mode": "long_only", "selection_metric": "sharpe", "min_exposure": 0.05}
    return proba, idx, period_ret, bt_cfg, scfg


def _sharpe(r):
    r = np.asarray(r, dtype=float)
    return r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std(ddof=1) > 0 else np.nan


def test_select_threshold_maximises_validation_sharpe_ties_to_smallest(threshold_case):
    """0.65 and 0.70 select the same profitable days -> equal, maximal Sharpe -> the smaller (0.65)
    wins; 0.90 never trades (exposure 0) and is ineligible."""
    proba, idx, period_ret, bt_cfg, scfg = threshold_case
    tau, table = ex.select_threshold(proba, idx, period_ret, 2, 0, scfg, bt_cfg)
    assert tau == 0.65
    assert list(table["threshold"]) == scfg["threshold_grid"]
    t = table.set_index("threshold")
    assert t.loc[0.65, "sharpe"] == t.loc[0.7, "sharpe"]
    assert t.loc[0.9, "exposure"] == 0.0 and np.isnan(t.loc[0.9, "sharpe"])
    # independent Sharpe of the zero-cost long-only strategy for each threshold
    for thr in (0.3, 0.5, 0.65, 0.7):
        expected = _sharpe(np.where(proba[:, 2] >= thr, period_ret.to_numpy(), 0.0))
        assert t.loc[thr, "sharpe"] == pytest.approx(expected, rel=1e-9)
    assert t.loc[0.65, "sharpe"] > t[["sharpe"]].drop(index=[0.65, 0.7, 0.9])["sharpe"].max()


def test_select_threshold_enforces_minimum_exposure(threshold_case):
    """With min_exposure = 30% the 25%-exposure thresholds are excluded; the best remaining one is
    chosen; if nothing qualifies the function returns None (argmax fallback)."""
    proba, idx, period_ret, bt_cfg, scfg = threshold_case
    tau, table = ex.select_threshold(proba, idx, period_ret, 2, 0, {**scfg, "min_exposure": 0.30}, bt_cfg)
    eligible = table[table["exposure"] >= 0.30]
    assert tau == 0.5
    assert tau == eligible.loc[eligible["sharpe"].idxmax(), "threshold"]
    tau_none, _ = ex.select_threshold(proba, idx, period_ret, 2, 0, {**scfg, "min_exposure": 1.01}, bt_cfg)
    assert tau_none is None


def test_select_ensemble_returns_min_validation_log_loss_subset():
    """Brute-force check: the chosen subset (size >= 2, ML models only) has the lowest validation
    log loss of all subsets; the table lists every subset once, sorted ascending."""
    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, K, size=n)
    onehot = np.eye(K)[y]

    def noisy_model(skill, conc):
        return skill * onehot + (1 - skill) * rng.dirichlet([conc] * K, size=n)

    probas = {  # deliberately not in ML_ORDER; 'majority_class' must be ignored
        "xgboost": {"val": rng.dirichlet([1.0] * K, size=n)},          # uninformative
        "logistic_regression": {"val": noisy_model(0.35, 0.5)},
        "lstm": {"val": noisy_model(0.15, 1.0)},
        "random_forest": {"val": noisy_model(0.35, 0.5)},
        "majority_class": {"val": np.tile(np.bincount(y, minlength=K) / n, (n, 1))},
    }
    members, table = ex.select_ensemble(probas, y, K)

    ml = [k for k in ex.ML_ORDER if k in probas]
    subsets = [s for r in range(2, len(ml) + 1) for s in itertools.combinations(ml, r)]
    losses = {"+".join(s): log_loss(y, np.mean([probas[k]["val"] for k in s], axis=0), labels=list(range(K))) for s in subsets}
    best = min(losses, key=losses.get)

    assert "+".join(members) == best
    assert len(members) >= 2 and "majority_class" not in members
    assert members == [k for k in ex.ML_ORDER if k in members]  # canonical order
    assert len(table) == len(subsets) == 11
    assert set(table["members"]) == set(losses)
    assert table["val_log_loss"].is_monotonic_increasing
    for _, row in table.iterrows():
        assert row["val_log_loss"] == pytest.approx(losses[row["members"]], rel=1e-9)
    assert "xgboost" not in members, "an uninformative member should not survive validation selection"
