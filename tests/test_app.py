"""Headless tests of the Streamlit dashboard (skipped when Streamlit is not installed).

The app reads the committed results in ``results/``; no network access is needed because the live
training button is never pressed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
METRICS = ROOT / "results" / "metrics"

pytestmark = pytest.mark.skipif(not (METRICS / "run_summary.json").exists(), reason="no saved experiment results")


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_renders_every_tab_without_errors():
    at = _app()
    assert len(at.tabs) == 6
    assert not at.error
    assert len(at.dataframe) >= 5  # comparison, backtest, classification, explorer, walk-forward tables


@pytest.mark.parametrize("ticker", json.loads((METRICS / "run_summary.json").read_text())["tickers"] if (METRICS / "run_summary.json").exists() else [])
def test_default_backtest_reproduces_the_saved_test_metrics(ticker):
    """With the pipeline's own settings the live backtest must equal the saved trading metrics."""
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.selectbox[0].set_value(ticker).run()
    assert not at.exception
    trading = pd.read_csv(METRICS / "trading_metrics_test.csv")
    saved = trading[(trading["ticker"] == ticker) & (trading["strategy"] == "XGBoost (conf.)")].iloc[0]
    table = at.dataframe[1].value  # interactive-backtest metrics table (first row = model strategy)
    assert table.iloc[0]["Sharpe"] == f"{saved['sharpe']:.2f}"
    assert table.iloc[0]["Trades"] == int(saved["n_trades"])
    assert any("pipeline's own settings" in s.value for s in at.success)


def test_changing_threshold_and_costs_recomputes_the_backtest():
    at = _app()
    before = at.dataframe[1].value.iloc[0].to_dict()
    tau = [s for s in at.slider if s.label.startswith("P(Up) threshold")][0]
    tau.set_value(0.55).run()
    cost = [s for s in at.slider if s.label.startswith("One-way cost")][0]
    cost.set_value(40).run()
    assert not at.exception
    after = at.dataframe[1].value.iloc[0].to_dict()
    assert after != before
    assert any("post-hoc" in i.value for i in at.info)


def test_every_model_and_mode_runs():
    at = _app()
    for model in ["logistic_regression", "random_forest", "xgboost", "lstm", "ensemble"]:
        at.selectbox(key="bt_model").set_value(model).run()
        assert not at.exception, model
    at.segmented_control(key="bt_mode").set_value("long_short").run()
    [t for t in at.toggle if t.label.startswith("Volatility-targeted")][0].set_value(True).run()
    at.segmented_control(key="bt_period").set_value("val").run()
    assert not at.exception
