"""Column-width (IEEE two-column, 3.4 in) figures for the mini-project report.

All figures are drawn from the saved experiment outputs in ``results/``; nothing is typed in by hand.
Fonts follow the IEEE template: Times New Roman, 8 pt labels.

    python report/report_figures.py      ->  report/figures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtester import BacktestConfig, run_backtest  # noqa: E402
from src.utils import load_config, load_json  # noqa: E402
from src.visualization import COLORS  # noqa: E402

RES = ROOT / "results"
MET = RES / "metrics"
OUT = Path(__file__).resolve().parent / "figures"
COL_W = 3.4  # inches, IEEE column width on A4 with this template
TEXT, GRID = "#0b0b0b", "#dddcd7"
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
         "lstm": "LSTM", "ensemble": "Ensemble", "buy_and_hold": "Buy & Hold", "ma_crossover": "MA crossover",
         "persistence": "Naive persistence", "majority_class": "Majority class"}
ML = ["logistic_regression", "random_forest", "xgboost", "lstm", "ensemble"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300, "figure.dpi": 150,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": "#3d3c39",
    })


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return path


def c(key: str) -> str:
    return COLORS.get(key, "#52514e")


# ---------------------------------------------------------------------------------------------


def fig_pipeline() -> Path:
    """System architecture as a vertical flow chart."""
    steps = [
        ("Yahoo Finance daily OHLCV, 2010–2025", "SPY, AAPL, MSFT; validation, split/dividend adjustment"),
        ("Feature engineering (42 point-in-time features)", "returns, SMA/EMA, RSI, MACD, Bollinger, ATR, volume, regimes"),
        ("Target construction", "next-open return; Down / Neutral / Up with ±0.1% band"),
        ("Purged chronological split", "train 2011–19 · validation 2020–21 · test 2022–25"),
        ("Model training and tuning", "LR · RF · XGBoost (purged time-series CV) · LSTM · ensemble"),
        ("Signal generation", "long if P(Up) ≥ τ, τ selected on validation"),
        ("Cost-aware backtest", "next-open fills, 15 bps per unit turnover, baselines"),
        ("Evaluation and robustness", "classification + trading metrics, walk-forward, bootstrap"),
    ]
    h, gap = 0.62, 0.22
    total = len(steps) * h + (len(steps) - 1) * gap
    fig, ax = plt.subplots(figsize=(COL_W, total * 0.62 + 0.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total)
    ax.axis("off")
    for i, (title, sub) in enumerate(steps):
        y = total - (i + 1) * h - i * gap
        fill = "#e9f1fb" if i not in (4, 5) else "#fdf0e9"
        ax.add_patch(FancyBboxPatch((0.2, y), 9.6, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec="#52514e", lw=0.6))
        ax.text(5, y + h * 0.64, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=TEXT)
        ax.text(5, y + h * 0.27, sub, ha="center", va="center", fontsize=6.5, color="#3d3c39")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5, y), (5, y - gap), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#52514e"))
    return save(fig, "fig_pipeline")


def fig_prices(cfg: dict) -> Path:
    tickers = load_json(MET / "run_summary.json")["tickers"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    pal = {"SPY": "#2a78d6", "AAPL": "#eb6834", "MSFT": "#1baf7a"}
    for t in tickers:
        raw = next((ROOT / "data" / "raw").glob(f"{t}_*.csv"), None)
        if raw is None:  # fall back to the saved holdout predictions (test period only)
            continue
        px = pd.read_csv(raw, index_col=0, parse_dates=True)["Adj Close"]
        ax.plot(px.index, px / px.iloc[0], label=t, color=pal.get(t, "#52514e"), lw=0.9)
    tr_end, va_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["val_end"])
    ax.axvspan(tr_end, va_end, color="#fdf0e9", lw=0, zorder=0)
    ax.axvspan(va_end, pd.Timestamp(cfg["data"]["end"]), color="#e9f7f1", lw=0, zorder=0)
    ax.text(pd.Timestamp("2014-06-01"), 0.97, "train", transform=ax.get_xaxis_transform(), fontsize=7, va="top", ha="center")
    ax.text(tr_end + (va_end - tr_end) / 2, 0.97, "val.", transform=ax.get_xaxis_transform(), fontsize=7, va="top", ha="center")
    ax.text(va_end + pd.Timedelta(days=730), 0.97, "test", transform=ax.get_xaxis_transform(), fontsize=7, va="top", ha="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Growth of 1 USD (log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right", ncol=3)
    return save(fig, "fig_prices")


def fig_accuracy_ci() -> Path:
    """Accuracy minus majority-class accuracy with block-bootstrap 95% CI (test)."""
    b = pd.read_csv(MET / "bootstrap_confidence_intervals.csv")
    b = b[b["statistic"] == "accuracy_minus_majority_class"].copy()
    tickers = list(dict.fromkeys(b["ticker"]))
    models = [NAMES[k] if k != "ensemble" else "Ensemble (soft vote)" for k in ML]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    offsets = np.linspace(-0.28, 0.28, len(tickers))
    pal = {"SPY": "#2a78d6", "AAPL": "#eb6834", "MSFT": "#1baf7a"}
    for off, t in zip(offsets, tickers):
        sub = b[b["ticker"] == t].set_index("strategy").reindex(models)
        y = np.arange(len(models)) + off
        ax.errorbar(sub["estimate"] * 100, y, xerr=[(sub["estimate"] - sub["low"]) * 100, (sub["high"] - sub["estimate"]) * 100],
                    fmt="o", ms=3, color=pal[t], elinewidth=0.8, capsize=1.5, label=t)
    ax.axvline(0, color=TEXT, lw=0.7, ls="--")
    ax.set_yticks(range(len(models)), [m.replace(" (soft vote)", "") for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy minus majority-class accuracy (pp)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.grid(axis="y", visible=False)
    return save(fig, "fig_accuracy_ci")


def fig_confusion() -> Path:
    from src.evaluation import classification_metrics

    tickers = load_json(MET / "run_summary.json")["tickers"]
    fig, axes = plt.subplots(1, len(tickers), figsize=(COL_W, 1.35))
    names = ["D", "N", "U"]
    for ax, t in zip(np.atleast_1d(axes), tickers):
        p = pd.read_csv(RES / "predictions" / f"{t}_holdout_predictions.csv", index_col=0)
        p = p[p["split"] == "test"]
        prob = p[[f"xgboost__p{k}" for k in range(3)]].values
        cm = np.array(classification_metrics(p["label"].values, prob.argmax(1), ["Down", "Neutral", "Up"], prob)["confusion_matrix"])
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.5, color="white" if share[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_title(t, fontsize=7.5)
        ax.tick_params(length=0)
    np.atleast_1d(axes)[0].set_ylabel("Actual")
    for ax in np.atleast_1d(axes):
        ax.set_xlabel("Predicted")
    fig.tight_layout(w_pad=0.6)
    return save(fig, "fig_confusion_xgb")


def fig_equity(cfg: dict, ticker: str = "SPY") -> Path:
    eq = pd.read_csv(RES / "predictions" / f"{ticker}_holdout_equity_curves.csv", index_col=0, parse_dates=True)
    cap = cfg["backtest"]["initial_capital"]
    show = [("Buy & Hold", "buy_and_hold"), ("MA Crossover (10/50)", "ma_crossover"), ("XGBoost (conf.)", "xgboost"),
            ("LSTM (conf.)", "lstm"), ("Ensemble (soft vote) (conf.)", "ensemble"), ("Naive persistence", "persistence")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for col, key in show:
        if col in eq:
            is_bh = key == "buy_and_hold"
            ax.plot(eq.index, eq[col] / cap, color=c(key), lw=1.3 if is_bh else 0.9, ls="--" if is_bh else "-",
                    label=NAMES[key], zorder=3 if is_bh else 2)
    ax.axhline(1, color=TEXT, lw=0.5)
    ax.set_ylabel("Portfolio value (× initial)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    return save(fig, f"fig_equity_{ticker.lower()}")


def fig_sharpe_ci() -> Path:
    b = pd.read_csv(MET / "bootstrap_confidence_intervals.csv")
    tr = pd.read_csv(MET / "trading_metrics_test.csv")
    kinds = tr.set_index(["ticker", "strategy"])["kind"]
    b = b[b["statistic"] == "sharpe_minus_buy_and_hold"].copy()
    b["kind"] = [kinds.get((t, s), "") for t, s in zip(b["ticker"], b["strategy"])]
    b = b[b["kind"].isin(["ml_confidence", "baseline"])]
    order = ["MA Crossover (10/50)"] + [f"{n} (conf.)" for n in ["Logistic Regression", "Random Forest", "XGBoost", "LSTM", "Ensemble (soft vote)"]]
    tickers = list(dict.fromkeys(b["ticker"]))
    pal = {"SPY": "#2a78d6", "AAPL": "#eb6834", "MSFT": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    offsets = np.linspace(-0.28, 0.28, len(tickers))
    for off, t in zip(offsets, tickers):
        sub = b[b["ticker"] == t].set_index("strategy").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(sub["estimate"], y, xerr=[sub["estimate"] - sub["low"], sub["high"] - sub["estimate"]], fmt="o", ms=3,
                    color=pal[t], elinewidth=0.8, capsize=1.5, label=t)
    ax.axvline(0, color=TEXT, lw=0.7, ls="--")
    ax.set_yticks(range(len(order)), [o.replace(" (conf.)", "").replace(" (soft vote)", "").replace(" (10/50)", " 10/50") for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe(strategy) − Sharpe(Buy & Hold), test period")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.grid(axis="y", visible=False)
    return save(fig, "fig_sharpe_ci")


def fig_walk_forward(cfg: dict) -> Path:
    tickers = load_json(MET / "run_summary.json")["tickers"]
    bt = BacktestConfig.from_config(cfg)
    fig, axes = plt.subplots(len(tickers), 1, figsize=(COL_W, 3.6), sharex=True)
    handles = {}
    for ax, t in zip(np.atleast_1d(axes), tickers):
        w = pd.read_csv(RES / "predictions" / f"{t}_walk_forward_predictions.csv", index_col=0, parse_dates=True)
        series = {"buy_and_hold": run_backtest(pd.Series(1.0, index=w.index), w["period_ret"], bt).frame["equity"]}
        for k in ML:
            if f"{k}__pos" in w:
                series[k] = run_backtest(w[f"{k}__pos"], w["period_ret"], bt).frame["equity"]
        for k, s in series.items():
            is_bh = k == "buy_and_hold"
            (h,) = ax.plot(s.index, s / bt.initial_capital, color=c(k), lw=1.2 if is_bh else 0.8, ls="--" if is_bh else "-", zorder=3 if is_bh else 2)
            handles[NAMES[k]] = h
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(t, fontsize=7.5, loc="left", pad=2)
        ax.set_ylabel("Value (log)")
    np.atleast_1d(axes)[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.4)
    return save(fig, "fig_walk_forward")


def fig_importance(ticker: str = "SPY", top: int = 10) -> Path:
    from feature_names import readable

    p = pd.read_csv(MET / "feature_importance" / f"{ticker}_xgboost_permutation.csv").head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.barh([readable(f) for f in p["feature"]], p["importance_mean"] * 1000, xerr=p["importance_std"] * 1000, color="#2a78d6", height=0.65,
            error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": "#52514e"})
    ax.axvline(0, color=TEXT, lw=0.6)
    ax.set_xlabel("Increase in validation log loss when permuted (×10⁻³)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=6.5)
    return save(fig, f"fig_importance_{ticker.lower()}")


def build_all() -> list[Path]:
    style()
    cfg = load_config()
    return [fig_pipeline(), fig_prices(cfg), fig_accuracy_ci(), fig_confusion(), fig_equity(cfg, "SPY"), fig_sharpe_ci(),
            fig_walk_forward(cfg), fig_importance("SPY")]


if __name__ == "__main__":
    for p in build_all():
        print("wrote", p.relative_to(ROOT))
