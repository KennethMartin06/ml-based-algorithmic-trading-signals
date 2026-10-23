"""Interactive demo dashboard for the ML-based algorithmic trading signals project.

Run locally:   streamlit run app/streamlit_app.py
Hosted:        Streamlit Community Cloud, main file path ``app/streamlit_app.py``

Every number is computed by the project's own ``src`` modules from the saved out-of-sample
predictions in ``results/`` (no hand-typed results). The "Train live" tab runs the real pipeline
end to end on freshly downloaded Yahoo Finance data.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtester import (  # noqa: E402
    BacktestConfig,
    positions_from_probabilities,
    run_backtest,
    volatility_target_positions,
)
from src.evaluation import classification_metrics, reliability_curve  # noqa: E402
from src.experiment import models_in_predictions, prepare_ticker_dataset, proba_cols, select_threshold  # noqa: E402
from src.models import XGBOOST_AVAILABLE, display_name, fit_estimator, predict_proba_full  # noqa: E402
from src.preprocessing import chronological_split  # noqa: E402
from src.reporting import key_findings  # noqa: E402
from src.utils import class_names_for, load_config, load_json  # noqa: E402
from src.visualization import COLORS  # noqa: E402

RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
CLASS_NAMES = class_names_for("three_class")
K, UP, DOWN = 3, 2, 0
BH_COLOR, MA_COLOR = COLORS["buy_and_hold"], COLORS["ma_crossover"]
REPO_URL = "https://github.com/KennethMartin06/ml-based-algorithmic-trading-signals"

st.set_page_config(page_title="ML Trading Signals", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------------------------
# data access (cached)
# ---------------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    path = METRICS / name
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json(name: str):
    return load_json(METRICS / name)


@st.cache_data(show_spinner=False)
def holdout_predictions(ticker: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "predictions" / f"{ticker}_holdout_predictions.csv", index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def holdout_positions(ticker: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "predictions" / f"{ticker}_holdout_positions.csv", index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def walk_forward_predictions(ticker: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "predictions" / f"{ticker}_walk_forward_predictions.csv", index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def config() -> dict:
    return load_config()


def color(key: str) -> str:
    return COLORS.get(key, "#52514e")


def pct(x, digits: int = 1) -> str:
    return "N/A" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 3) -> str:
    return "N/A" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def metrics_row(name: str, m: dict) -> dict:
    return {
        "Strategy": name,
        "Cum. return": pct(m["cumulative_return"]),
        "Ann. return": pct(m["annualized_return"]),
        "Ann. volatility": pct(m["annualized_volatility"]),
        "Sharpe": num(m["sharpe"], 2),
        "Sortino": num(m["sortino"], 2),
        "Max drawdown": pct(m["max_drawdown"]),
        "Days invested": pct(m["exposure"], 0),
        "Trades": int(m["n_trades"]),
        "Costs paid": f"${m['costs_paid']:,.0f}",
    }


def line_chart(series: dict[str, tuple[str, pd.Series]], title: str, yaxis: str, pct_axis: bool = False, height: int = 420) -> go.Figure:
    fig = go.Figure()
    for name, (key, s) in series.items():
        is_bh = key == "buy_and_hold"
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values * (100 if pct_axis else 1), name=name, mode="lines",
            line=dict(color=color(key), width=2.4 if is_bh else 1.8, dash="dash" if is_bh else "solid"),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}" + ("%" if pct_axis else "") + f"<extra>{name}</extra>",
        ))
    fig.update_layout(title=title, yaxis_title=yaxis, height=height, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=70, b=30, l=10, r=10))
    return fig


# ---------------------------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------------------------

cfg = config()
summary = read_json("run_summary.json")
tickers = summary["tickers"]
thresholds = read_json("selected_thresholds.json")

with st.sidebar:
    st.header("📈 ML Trading Signals")
    ticker = st.selectbox("Instrument", tickers, help="SPY = S&P 500 ETF; AAPL = Apple; MSFT = Microsoft")
    st.markdown(
        "**Machine Learning Based Algorithmic Trading Signals: A Comparative Study Against Technical Baseline Strategies**\n\n"
        "Mini-project, B.Tech CSE (AI & ML), Manipal Institute of Technology.\n\n"
        "Ishan Singh · Kenneth Martin\n\n"
        "Mentors: Dr. Narendra V G, Dr. Ashwath Rao B"
    )
    st.caption(f"Data: Yahoo Finance daily OHLCV, {cfg['data']['start']} to {cfg['data']['end']}. "
               f"Results generated by `python -m scripts.run_experiment` on real data "
               f"(synthetic data: {summary['synthetic_data']}).")
    st.caption("Academic research demo. Backtests are hypothetical and are not investment advice.")

st.title("Machine-learning trading signals vs technical baselines")
st.caption("Can supervised ML (Logistic Regression, Random Forest, XGBoost, LSTM, ensemble) turn daily market data into "
           "better trading signals than buy-and-hold or a moving-average crossover, after costs and without look-ahead?")

tabs = st.tabs(["Overview", "Interactive backtest", "Predictions", "Walk-forward & robustness", "Train a model live", "Figures & docs"])


# ---------------------------------------------------------------------------------------------
# 1. overview
# ---------------------------------------------------------------------------------------------

with tabs[0]:
    cls = read_csv("classification_metrics.csv")
    trading = read_csv("trading_metrics_test.csv")
    boot = read_csv("bootstrap_confidence_intervals.csv")
    wf = read_csv("walk_forward_trading.csv")
    ds = read_json("dataset_summary.json")[ticker]
    test_cls = cls[(cls["split"] == "test") & (cls["ticker"] == ticker)].set_index("model_key")
    tr = trading[trading["ticker"] == ticker].set_index("strategy")
    conf = tr[tr["kind"] == "ml_confidence"]
    best_ml = conf["sharpe"].idxmax()
    ml_cls = test_cls.drop(index=["majority_class", "persistence"], errors="ignore")
    best_acc_key = ml_cls["accuracy"].idxmax()

    st.subheader(f"{ticker}: headline results on the untouched test period ({ds['split']['test']['start']} to {ds['split']['test']['end']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buy & Hold Sharpe", num(tr.loc["Buy & Hold", "sharpe"], 2), border=True)
    clone = conf.loc[best_ml, "exposure"] >= 1.0 and conf.loc[best_ml, "n_trades"] <= 1
    c2.metric("Best ML strategy Sharpe", num(conf.loc[best_ml, "sharpe"], 2),
              delta="= Buy & Hold (always long)" if clone
              else f"{conf.loc[best_ml, 'sharpe'] - tr.loc['Buy & Hold', 'sharpe']:+.2f} vs B&H",
              delta_color="off" if clone else "normal", border=True, help=best_ml)
    c3.metric("Best ML accuracy", pct(ml_cls.loc[best_acc_key, "accuracy"]), border=True, help=display_name(best_acc_key))
    c4.metric("Majority-class accuracy", pct(test_cls.loc["majority_class", "accuracy"]), border=True,
              help="Always predicting the most frequent class (Up): the minimum bar for a useful classifier")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Key findings (computed from the result tables)")
        for f in key_findings(cls, trading, boot, wf):
            st.markdown(f"- {f}")
    with right:
        st.markdown("#### How the system works")
        st.markdown(
            "1. **Data**: daily OHLCV from Yahoo Finance, cleaned, split/dividend-adjusted.\n"
            "2. **Features**: 42 point-in-time indicators (returns, SMA/EMA, RSI, MACD, Bollinger, ATR, volatility, volume, regimes).\n"
            "3. **Target**: direction of the next tradable return (Down / Neutral / Up, ±0.1% band).\n"
            "4. **Chronological split**: train 2011–2019 → validation 2020–2021 → test 2022–2025, with purging.\n"
            "5. **Models**: tuned with purged time-series cross-validation (log loss).\n"
            "6. **Signals**: long when P(Up) ≥ τ, with τ chosen on validation only.\n"
            "7. **Backtest**: next-open execution, 15 bps one-way costs, identical for every strategy.\n"
            "8. **Robustness**: 10-year walk-forward, regimes, costs, block-bootstrap confidence intervals."
        )
    st.markdown("#### Model comparison (test period)")
    comp = read_csv("model_comparison_test.csv")
    comp = comp[(comp["ticker"] == ticker) & comp["kind"].isin(["baseline", "ml_confidence"])]
    show = pd.DataFrame({
        "Strategy": comp["strategy"],
        "Accuracy": comp["accuracy"].map(lambda v: num(v)),
        "Macro precision": comp["precision_macro"].map(lambda v: num(v)),
        "Macro recall": comp["recall_macro"].map(lambda v: num(v)),
        "Macro F1": comp["f1_macro"].map(lambda v: num(v)),
        "Cum. return": comp["cumulative_return"].map(pct),
        "Sharpe": comp["sharpe"].map(lambda v: num(v, 2)),
        "Sortino": comp["sortino"].map(lambda v: num(v, 2)),
        "Max DD": comp["max_drawdown"].map(pct),
        "Trades": comp["n_trades"].astype(int),
    })
    st.dataframe(show, hide_index=True)
    st.caption("N/A = not applicable (rule-based strategy without class predictions). Macro = Down, Neutral and Up weighted equally.")


# ---------------------------------------------------------------------------------------------
# 2. interactive backtest
# ---------------------------------------------------------------------------------------------

with tabs[1]:
    pred = holdout_predictions(ticker)
    model_keys = [k for k in models_in_predictions(pred) if k != "majority_class"]
    st.markdown("Re-run the backtest live on the saved **out-of-sample** model probabilities. Change the decision rule, "
                "costs or sizing and every number below is recomputed by `src/backtester.py`.")
    c1, c2, c3 = st.columns(3)
    model_key = c1.selectbox("Model", model_keys, key="bt_model", index=model_keys.index("xgboost") if "xgboost" in model_keys else 0,
                             format_func=display_name)
    period = c2.segmented_control("Period", ["test", "val"], default="test", required=True, key="bt_period",
                                  format_func={"test": "Test 2022–2025", "val": "Validation 2020–2021"}.get)
    mode = c3.segmented_control("Positions", ["long_only", "long_short"], default="long_only", required=True, key="bt_mode",
                                format_func={"long_only": "Long / cash", "long_short": "Long / short"}.get)
    tau_val = thresholds[ticker].get(model_key)
    c4, c5, c6 = st.columns(3)
    use_argmax = c4.toggle("Argmax rule (no threshold)", value=tau_val is None,
                           help="Go long whenever Up is the most probable class, instead of requiring P(Up) ≥ τ.")
    tau = c4.slider("P(Up) threshold τ", 0.30, 0.70, value=float(tau_val or 0.50), step=0.005, format="%.3f", disabled=use_argmax,
                    help=f"The pipeline chose τ = {tau_val} on the validation period (maximum validation Sharpe).")
    cost = c5.slider("One-way cost per unit turnover (bps)", 0, 50, value=int(cfg["backtest"]["fee_bps"] + cfg["backtest"]["slippage_bps"]),
                     help="Fee + slippage. A round trip (buy then sell) costs twice this.")
    vol_target = c6.toggle("Volatility-targeted sizing (15% annual)", value=False,
                           help="Scale exposure by 15% / realised 20-day volatility (known at decision time), capped at 1×.")

    part = pred[pred["split"] == period]
    probs = part[proba_cols(model_key, K)].values
    pos = positions_from_probabilities(probs, part.index, UP, DOWN, None if use_argmax else tau, mode)
    bh_pos = pd.Series(1.0, index=part.index)
    if vol_target:
        pos = volatility_target_positions(pos, part["realized_vol"], 0.15)
        bh_pos = volatility_target_positions(bh_pos, part["realized_vol"], 0.15)
    bt_cfg = BacktestConfig.from_config(cfg, fee_bps=cost, slippage_bps=0.0)
    strat = run_backtest(pos, part["period_ret"], bt_cfg)
    bh = run_backtest(bh_pos, part["period_ret"], bt_cfg)
    name = f"{display_name(model_key)} ({'argmax' if use_argmax else f'τ = {tau:.3f}'})"
    curves = {name: (model_key, strat.frame["equity"] / bt_cfg.initial_capital), "Buy & Hold": ("buy_and_hold", bh.frame["equity"] / bt_cfg.initial_capital)}
    rows = [metrics_row(name, strat.metrics), metrics_row("Buy & Hold", bh.metrics)]
    if period == "test" and not vol_target:
        ma_col = [c for c in holdout_positions(ticker).columns if c.startswith("MA Crossover")]
        if ma_col:
            ma = run_backtest(holdout_positions(ticker)[ma_col[0]].reindex(part.index), part["period_ret"], bt_cfg)
            curves[ma_col[0]] = ("ma_crossover", ma.frame["equity"] / bt_cfg.initial_capital)
            rows.append(metrics_row(ma_col[0], ma.metrics))

    if not use_argmax and tau_val is not None and abs(tau - tau_val) < 1e-9 and cost == 15 and not vol_target and mode == "long_only":
        st.success(f"These are the pipeline's own settings (τ = {tau_val} selected on validation), so the numbers match the report.")
    elif period == "test":
        st.info("Exploring settings on the test period is post-hoc analysis. The study chose τ on validation only, before looking at the test period.")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.plotly_chart(line_chart(curves, f"{ticker}: portfolio value, net of costs", "Value (× initial capital)"))
    dd = {n: (k, (s / s.cummax().clip(lower=1.0) - 1)) for n, (k, s) in curves.items()}
    st.plotly_chart(line_chart(dd, "Drawdown from peak", "Drawdown (%)", pct_axis=True, height=300))

    p_up = pd.Series(probs[:, UP], index=part.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_up.index, y=p_up, mode="lines", name="P(Up)", line=dict(color=color(model_key), width=1.2),
                             hovertemplate="%{x|%d %b %Y}<br>P(Up) = %{y:.3f}<extra></extra>"))
    invested = strat.frame["position"] != 0
    fig.add_trace(go.Scatter(x=p_up.index[invested.values], y=p_up[invested.values], mode="markers", name="Invested next session",
                             marker=dict(color=color(model_key), size=4), hoverinfo="skip"))
    if not use_argmax:
        fig.add_hline(y=tau, line=dict(color="#0b0b0b", dash="dot"), annotation_text=f"τ = {tau:.3f}")
    fig.update_layout(title="Model probability of an Up move and the resulting decisions", yaxis_title="P(Up)", height=320,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=70, b=30, l=10, r=10))
    st.plotly_chart(fig)
    st.caption("Timing: the decision is made after the close of day t using data up to t. The order fills at the next open and the "
               "position is held until the following open.")


# ---------------------------------------------------------------------------------------------
# 3. predictions and classification
# ---------------------------------------------------------------------------------------------

with tabs[2]:
    pred = holdout_predictions(ticker)
    period = st.segmented_control("Period", ["test", "val"], default="test", required=True,
                                  format_func={"test": "Test 2022–2025", "val": "Validation 2020–2021"}.get, key="cls_period")
    part = pred[pred["split"] == period]
    rows = []
    for key in models_in_predictions(pred):
        p = part[proba_cols(key, K)].values
        m = classification_metrics(part["label"].values, p.argmax(1), CLASS_NAMES, p)
        rows.append({"Model": display_name(key), "Accuracy": m["accuracy"], "Balanced accuracy": m["balanced_accuracy"],
                     "Macro precision": m["precision_macro"], "Macro recall": m["recall_macro"], "Macro F1": m["f1_macro"],
                     "Log loss": m["log_loss"], "Brier": m["brier"]})
    m = classification_metrics(part["label"].values, part["persistence__pred"].values, CLASS_NAMES, None)
    rows.append({"Model": "Naive persistence", "Accuracy": m["accuracy"], "Balanced accuracy": m["balanced_accuracy"],
                 "Macro precision": m["precision_macro"], "Macro recall": m["recall_macro"], "Macro F1": m["f1_macro"],
                 "Log loss": np.nan, "Brier": np.nan})
    st.markdown(f"Computed live from {len(part)} saved predictions. **Majority class** is the baseline to beat. "
                "Log loss and Brier score are lower-is-better.")
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)

    model_key = st.selectbox("Model to inspect", [k for k in models_in_predictions(pred) if k != "majority_class"],
                             format_func=display_name, key="cls_model")
    p = part[proba_cols(model_key, K)].values
    m = classification_metrics(part["label"].values, p.argmax(1), CLASS_NAMES, p)
    c1, c2 = st.columns(2)
    cm = np.array(m["confusion_matrix"])
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = [[f"{cm[i, j]}<br>{share[i, j]:.0%}" for j in range(K)] for i in range(K)]
    fig = px.imshow(share, x=[f"Pred. {c}" for c in CLASS_NAMES], y=[f"Actual {c}" for c in CLASS_NAMES],
                    color_continuous_scale=["#f4f8fd", "#3987e5", "#0d366b"], zmin=0, zmax=1, aspect="auto")
    fig.update_traces(text=text, texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{z:.0%}<extra></extra>")
    fig.update_layout(title=f"Confusion matrix: {display_name(model_key)}", coloraxis_showscale=False, height=360, margin=dict(t=60, b=20))
    c1.plotly_chart(fig)
    curve = reliability_curve(p[:, UP], (part["label"].values == UP).astype(float), 10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#52514e", dash="dash")))
    fig.add_trace(go.Scatter(x=curve["mean_predicted"], y=curve["observed_frequency"], mode="lines+markers", name=display_name(model_key),
                             line=dict(color=color(model_key)), customdata=curve["count"],
                             hovertemplate="mean P(Up) %{x:.3f}<br>observed Up rate %{y:.3f}<br>%{customdata} days<extra></extra>"))
    fig.update_layout(title="Calibration of P(Up)", xaxis_title="Predicted P(Up)", yaxis_title="Observed share of Up days",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=360, margin=dict(t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    c2.plotly_chart(fig)
    st.markdown("**Latest predictions in this period**")
    tail = part.tail(15)
    explorer = pd.DataFrame({
        "Decision date": tail.index.date,
        "P(Down)": tail[f"{model_key}__p0"].round(3), "P(Neutral)": tail[f"{model_key}__p1"].round(3), "P(Up)": tail[f"{model_key}__p2"].round(3),
        "Predicted": [CLASS_NAMES[i] for i in tail[proba_cols(model_key, K)].values.argmax(1)],
        "Actual": [CLASS_NAMES[i] for i in tail["label"].astype(int)],
        "Next-period return": tail["fwd_ret"].map(lambda v: f"{v * 100:+.2f}%"),
    })
    st.dataframe(explorer, hide_index=True)


# ---------------------------------------------------------------------------------------------
# 4. walk-forward and robustness
# ---------------------------------------------------------------------------------------------

with tabs[3]:
    wfp = walk_forward_predictions(ticker)
    st.markdown(f"**Walk-forward ({wfp['year'].min()}–{wfp['year'].max()})**: each year the models are re-tuned and retrained on "
                "all earlier data, thresholds are chosen on the previous year, and the following year is traded out-of-sample.")
    bt_cfg = BacktestConfig.from_config(cfg)
    curves = {}
    for col in [c for c in wfp.columns if c.endswith("__pos")]:
        key = col[: -len("__pos")]
        res = run_backtest(wfp[col], wfp["period_ret"], bt_cfg)
        curves[f"{display_name(key)} (conf.)"] = (key, res.frame["equity"] / bt_cfg.initial_capital)
    bh = run_backtest(pd.Series(1.0, index=wfp.index), wfp["period_ret"], bt_cfg)
    curves["Buy & Hold"] = ("buy_and_hold", bh.frame["equity"] / bt_cfg.initial_capital)
    st.plotly_chart(line_chart(curves, f"{ticker}: walk-forward out-of-sample portfolio value (recomputed live)", "Value (× initial capital)"))
    wft = read_csv("walk_forward_trading.csv")
    wft = wft[wft["ticker"] == ticker]
    st.dataframe(pd.DataFrame({"Strategy": wft["strategy"], "Cum. return": wft["cumulative_return"].map(pct),
                               "Sharpe": wft["sharpe"].map(lambda v: num(v, 2)), "Max DD": wft["max_drawdown"].map(pct),
                               "Days invested": wft["exposure"].map(lambda v: pct(v, 0)), "Trades": wft["n_trades"]}), hide_index=True)
    yearly = read_csv("walk_forward_yearly.csv")
    yearly = yearly[(yearly["ticker"] == ticker) & yearly["sharpe"].notna()]
    fig = px.line(yearly, x="year", y="sharpe", color="strategy", markers=True,
                  color_discrete_map={s: color(k) for s, k in zip(yearly["strategy"], yearly["color_key"])})
    fig.update_layout(title="Sharpe ratio per test year (N/A years with fewer than 20 invested days omitted)", yaxis_title="Sharpe ratio",
                      xaxis_title="Test year", height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    b = read_csv("bootstrap_confidence_intervals.csv")
    b = b[(b["ticker"] == ticker) & (b["statistic"] == "sharpe_minus_buy_and_hold")]
    fig = go.Figure(go.Scatter(x=b["estimate"], y=b["strategy"], mode="markers", marker=dict(size=9, color="#2a78d6"),
                               error_x=dict(type="data", symmetric=False, array=b["high"] - b["estimate"], arrayminus=b["estimate"] - b["low"]),
                               hovertemplate="%{y}<br>Δ Sharpe %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#0b0b0b", dash="dot"))
    fig.update_layout(title="Sharpe(strategy) − Sharpe(Buy & Hold), 95% block-bootstrap CI", xaxis_title="Difference in Sharpe ratio",
                      height=380, margin=dict(t=60, l=10))
    c1.plotly_chart(fig)
    c1.caption("An interval crossing 0 means no reliable difference from buy-and-hold at this sample size.")
    cost = read_csv("robustness_cost_sensitivity.csv")
    cost = cost[(cost["ticker"] == ticker) & (cost["color_key"] != "persistence")]
    fig = px.line(cost, x="cost_bps", y="sharpe", color="strategy", markers=True,
                  color_discrete_map={s: color(k) for s, k in zip(cost["strategy"], cost["color_key"])})
    fig.update_layout(title="Transaction-cost sensitivity (positions fixed)", xaxis_title="One-way cost per unit turnover (bps)",
                      yaxis_title="Test Sharpe ratio", height=380, legend=dict(title=None))
    c2.plotly_chart(fig)
    with st.expander("Execution-timing and target-definition variants"):
        v = read_csv("robustness_variants.csv")
        v = v[v["ticker"] == ticker].pivot_table(index="strategy", columns="variant", values="sharpe", sort=False).round(3)
        st.dataframe(v)
        st.caption("Test Sharpe ratio when trading at the same close (optimistic) or using a binary Up/Down target. "
                   "Tabular models + ensemble, reduced grids, no LSTM.")


# ---------------------------------------------------------------------------------------------
# 5. live training
# ---------------------------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(symbol: str, start: str, end: str):
    from src.data_loader import drop_unfinished_session, load_ticker

    cache = Path(tempfile.gettempdir()) / "ml_trading_signals_cache"
    # no file cache: an intraday download must not be reused after the close (st.cache_data expires hourly)
    prices, meta = load_ticker(symbol, start, end, raw_dir=cache, use_cache=False, allow_synthetic=False, max_retries=2)
    prices, dropped = drop_unfinished_session(prices)
    meta = meta.to_dict()
    meta["dropped_unfinished_session"] = dropped
    meta["last_date"] = prices.index[-1].strftime("%Y-%m-%d")
    return prices, meta


@st.cache_data(show_spinner=False)
def train_live(symbol: str, end: str, model_name: str, params: tuple, prices: pd.DataFrame) -> dict:
    params = dict(params)
    timings = {}
    t = time.perf_counter()
    data, fcols, report = prepare_ticker_dataset(prices, cfg)
    split = chronological_split(data, cfg["split"]["train_end"], cfg["split"]["val_end"], purge=True)
    timings["features, targets, purged split"] = time.perf_counter() - t
    X, y = data[fcols].to_numpy(float), data["label"]
    t = time.perf_counter()
    model = fit_estimator(model_name, params, data.loc[split.train, fcols].to_numpy(float), y.loc[split.train].astype(int).values, K, 42)
    timings["model training"] = time.perf_counter() - t
    t = time.perf_counter()
    pv = predict_proba_full(model, data.loc[split.val, fcols].to_numpy(float), K)
    pt = predict_proba_full(model, data.loc[split.test, fcols].to_numpy(float), K)
    bcfg = BacktestConfig.from_config(cfg)
    tau, _ = select_threshold(pv, split.val, data.loc[split.val, "period_ret"], UP, DOWN, cfg["signals"], bcfg)
    pos = positions_from_probabilities(pt, split.test, UP, DOWN, tau, cfg["signals"]["mode"])
    strat = run_backtest(pos, data.loc[split.test, "period_ret"], bcfg)
    bh = run_backtest(pd.Series(1.0, index=split.test), data.loc[split.test, "period_ret"], bcfg)
    y_test = y.loc[split.test].astype(int).values
    cm_model = classification_metrics(y_test, pt.argmax(1), CLASS_NAMES, pt)
    maj = np.bincount(y.loc[split.train].astype(int), minlength=K).argmax()
    cm_major = classification_metrics(y_test, np.full(len(y_test), maj), CLASS_NAMES, None)
    latest = predict_proba_full(model, X[-1:], K)[0]
    timings["threshold selection + backtest"] = time.perf_counter() - t
    return {"split": split.describe(), "n_features": len(fcols), "tau": tau, "strategy": strat, "bh": bh, "model_cls": cm_model,
            "major_cls": cm_major, "latest_date": data.index[-1], "latest_proba": latest, "timings": timings, "rows": len(data)}


with tabs[4]:
    st.markdown("Runs the **actual pipeline** end to end: download fresh data from Yahoo Finance → validate and adjust → "
                "42 features → targets → purged chronological split → train → choose τ on validation → backtest the test period. "
                "The test period runs from 2022 to the most recent trading day, so it includes data the report never saw.")
    best = read_json("best_params.json")[ticker]
    options = ["logistic_regression", "random_forest"] + (["xgboost"] if XGBOOST_AVAILABLE else [])
    c1, c2 = st.columns([1, 2])
    model_name = c1.radio("Model", options, format_func=display_name, index=len(options) - 1, key="live_model")
    use_tuned = c1.toggle("Use the hyperparameters tuned in the study", value=True,
                          help="Values selected by purged time-series cross-validation (results/metrics/best_params.json)")
    tuned = {k: v for k, v in best.get(model_name, {}).items()}
    with c2:
        if model_name == "logistic_regression":
            params = {"C": st.select_slider("C (inverse regularisation)", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                                            value=tuned.get("C", 0.001), disabled=use_tuned)}
        elif model_name == "random_forest":
            params = {"n_estimators": 300, "max_features": "sqrt",
                      "max_depth": st.slider("max_depth", 2, 15, int(tuned.get("max_depth", 3)), disabled=use_tuned),
                      "min_samples_leaf": st.slider("min_samples_leaf", 5, 200, int(tuned.get("min_samples_leaf", 80)), disabled=use_tuned)}
        else:
            params = {k: tuned[k] for k in ("subsample", "colsample_bytree", "min_child_weight", "reg_lambda") if k in tuned}
            params["max_depth"] = st.slider("max_depth", 1, 8, int(tuned.get("max_depth", 2)), disabled=use_tuned)
            params["learning_rate"] = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                                                       value=tuned.get("learning_rate", 0.03), disabled=use_tuned)
            params["n_estimators"] = st.slider("n_estimators", 50, 600, int(tuned.get("n_estimators", 100)), step=50, disabled=use_tuned)
        params["class_weight"] = None if not st.toggle("Balanced class weights", value=tuned.get("class_weight") == "balanced",
                                                       disabled=use_tuned) else "balanced"
        if use_tuned:
            params = {k: v for k, v in tuned.items()}
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    if st.button(f"▶ Download {ticker} and train {display_name(model_name)}", type="primary"):
        with st.status("Running the pipeline…", expanded=True) as status:
            try:
                st.write(f"Downloading {ticker} {cfg['data']['start']} → {end} from Yahoo Finance…")
                t0 = time.perf_counter()
                prices, meta = fetch_prices(ticker, cfg["data"]["start"], end)
                st.write(f"✓ {len(prices)} completed trading days, {meta['first_date']} → {meta['last_date']} "
                         f"(source: {meta['source']}; {meta['validation']['duplicate_dates_removed']} duplicates, "
                         f"{meta['validation']['rows_with_missing_prices_removed']} missing rows removed)")
                if meta["dropped_unfinished_session"]:
                    st.write("• Today's session is still open in New York, so its incomplete bar was left out.")
                st.write("Engineering features, building targets, splitting, training…")
                res = train_live(ticker, end, model_name, tuple(sorted(params.items(), key=lambda kv: kv[0])), prices)
                for step, sec in res["timings"].items():
                    st.write(f"✓ {step}: {sec:.2f} s")
                status.update(label=f"Pipeline finished in {time.perf_counter() - t0:.1f} s", state="complete", expanded=False)
                st.session_state["live_result"] = (ticker, model_name, res)
            except Exception as exc:  # network / Yahoo rate limit
                status.update(label="Pipeline failed", state="error")
                st.error(f"Could not complete the run: {exc}. Yahoo Finance sometimes rate-limits cloud servers; "
                         "wait a minute and try again. The other tabs use the saved results and still work.")
    live = st.session_state.get("live_result")
    if live and live[0] == ticker and live[1] == model_name:
        res = live[2]
        sp = res["split"]
        st.caption(f"Train {sp['train']['start']} → {sp['train']['end']} ({sp['train']['rows']} days) · "
                   f"validation {sp['val']['start']} → {sp['val']['end']} ({sp['val']['rows']}) · "
                   f"test {sp['test']['start']} → {sp['test']['end']} ({sp['test']['rows']}) · {res['n_features']} features")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", pct(res["model_cls"]["accuracy"]), delta=f"{(res['model_cls']['accuracy'] - res['major_cls']['accuracy']) * 100:+.1f} pp vs majority", border=True)
        m2.metric("Test macro F1", num(res["model_cls"]["f1_macro"]), border=True)
        m3.metric("Strategy Sharpe", num(res["strategy"].metrics["sharpe"], 2),
                  delta=f"{res['strategy'].metrics['sharpe'] - res['bh'].metrics['sharpe']:+.2f} vs B&H", border=True)
        m4.metric("τ chosen on validation", "argmax" if res["tau"] is None else f"{res['tau']:.3f}", border=True)
        curves = {f"{display_name(model_name)} (live)": (model_name, res["strategy"].frame["equity"] / 100_000),
                  "Buy & Hold": ("buy_and_hold", res["bh"].frame["equity"] / 100_000)}
        st.plotly_chart(line_chart(curves, f"{ticker}: test-period portfolio value from the model trained just now", "Value (× initial capital)"))
        st.dataframe(pd.DataFrame([metrics_row(f"{display_name(model_name)} (live)", res["strategy"].metrics), metrics_row("Buy & Hold", res["bh"].metrics)]),
                     hide_index=True)
        pr = res["latest_proba"]
        decision = "invest (long)" if res["tau"] is not None and pr[UP] >= res["tau"] else ("invest (long)" if res["tau"] is None and pr.argmax() == UP else "stay in cash")
        st.info(f"**Latest model output**: after the close of {res['latest_date'].date()}: P(Down) {pr[0]:.3f} · P(Neutral) {pr[1]:.3f} · "
                f"P(Up) {pr[2]:.3f} → rule says *{decision}* at the next open. This is a research output, not investment advice.")


# ---------------------------------------------------------------------------------------------
# 6. figures and docs
# ---------------------------------------------------------------------------------------------

with tabs[5]:
    figs = sorted((RESULTS / "figures").glob("*.png"))
    general = [f for f in figs if not any(f.name.startswith(t + "_") for t in tickers)]
    own = [f for f in figs if f.name.startswith(ticker + "_")]
    choice = st.selectbox("Figure (generated by the pipeline)", general + own, format_func=lambda p: p.stem.replace("_", " "))
    st.image(str(choice), width="stretch")
    for title, doc in (("Model comparison and discussion", "docs/model_comparison.md"), ("Methodology", "docs/methodology.md"),
                       ("Limitations", "docs/limitations.md")):
        with st.expander(title):
            st.markdown((ROOT / doc).read_text(encoding="utf-8"))
    st.caption(f"Source code: {REPO_URL}")
