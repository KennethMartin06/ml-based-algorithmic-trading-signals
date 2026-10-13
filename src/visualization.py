"""Publication-quality figures (matplotlib, static PNG) generated from computed results only.

Colour rule: every strategy/model keeps the same colour in every figure (colour follows the
entity, never its rank). Hues come from a colour-blind-validated categorical palette in fixed order;
the passive benchmark (buy-and-hold) is drawn in neutral dark grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .features import bollinger, macd, rsi, sma  # noqa: E402

TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# fixed categorical order (validated palette, light mode)
COLORS = {
    "xgboost": "#2a78d6",
    "hist_gradient_boosting": "#2a78d6",
    "lstm": "#eb6834",
    "random_forest": "#1baf7a",
    "logistic_regression": "#eda100",
    "ensemble": "#e87ba4",
    "ma_crossover": "#008300",
    "persistence": "#4a3aa7",
    "buy_and_hold": "#52514e",
    "majority_class": "#a3a29c",
}
SPLIT_SHADES = {"train": "#eef4fc", "val": "#fdf0e9", "test": "#e9f7f1"}
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_ramp", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": TEXT_2,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 150,
        }
    )


setup_style()


def color_for(key: str) -> str:
    return COLORS.get(key, TEXT_2)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _date_axis(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _identical_groups(curves: dict[str, tuple[str, pd.Series]]) -> list[list[str]]:
    """Groups of strategies whose series are numerically identical (they overlap on the chart)."""
    names, groups, used = list(curves), [], set()
    for i, a in enumerate(names):
        if a in used:
            continue
        g = [a]
        for b in names[i + 1:]:
            if b not in used and np.allclose(curves[a][1].values, curves[b][1].values, equal_nan=True):
                g.append(b)
                used.add(b)
        if len(g) > 1:
            groups.append(g)
    return groups


def _note_overlaps(ax, curves):
    groups = _identical_groups(curves)
    if groups:
        txt = "Overlapping (identical) lines: " + "; ".join(" = ".join(g) for g in groups)
        ax.text(0.0, -0.13, txt, transform=ax.transAxes, fontsize=7.5, color=TEXT_2, va="top", wrap=True)


def _shade_splits(ax, split_ranges: dict | None):
    if not split_ranges:
        return
    for name, (start, end) in split_ranges.items():
        if start is None or end is None:
            continue
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=SPLIT_SHADES.get(name, GRID), zorder=0, lw=0)
        ax.text(pd.Timestamp(start), 1.0, f" {name}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=TEXT_2)


# ---------------------------------------------------------------------------------------------
# data figures
# ---------------------------------------------------------------------------------------------


def plot_price_history(close_by_ticker: dict[str, pd.Series], split_ranges: dict, path, synthetic: bool = False):
    n = len(close_by_ticker)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, squeeze=False)
    for ax, (ticker, close) in zip(axes[:, 0], close_by_ticker.items()):
        _shade_splits(ax, split_ranges)
        ax.plot(close.index, close.values, color="#2a78d6", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("Adj. close (USD, log)")
        ax.set_title(f"{ticker}{' (SYNTHETIC)' if synthetic else ''}", loc="left")
    _date_axis(axes[-1, 0])
    fig.suptitle("Adjusted daily close with train / validation / test periods", x=0.01, ha="left", fontsize=13, color=TEXT)
    return _save(fig, path)


def plot_technical_indicators(df: pd.DataFrame, ticker: str, path, start: str | None = None, end: str | None = None):
    close = df["close"]
    s20, s50 = sma(close, 20), sma(close, 50)
    bb = bollinger(close, 20, 2.0)
    r = rsi(close, 14)
    m = macd(close)
    sl = slice(start, end)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = axes[0]
    ax.fill_between(bb.loc[sl].index, bb["bb_lower"].loc[sl], bb["bb_upper"].loc[sl], color="#cde2fb", lw=0, label="Bollinger band (20, 2σ)")
    ax.plot(close.loc[sl], color=TEXT, lw=1.2, label="Adj. close")
    ax.plot(s20.loc[sl], color="#2a78d6", lw=1.4, label="SMA 20")
    ax.plot(s50.loc[sl], color="#eb6834", lw=1.4, label="SMA 50")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", ncol=4)
    ax.set_title(f"{ticker}: technical indicators used as model features", loc="left")
    axes[1].plot(r.loc[sl], color="#4a3aa7", lw=1.3)
    axes[1].axhline(70, color=TEXT_2, lw=0.8, ls="--")
    axes[1].axhline(30, color=TEXT_2, lw=0.8, ls="--")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("RSI (14)")
    axes[2].bar(m["macd_hist"].loc[sl].index, m["macd_hist"].loc[sl], color=np.where(m["macd_hist"].loc[sl] >= 0, "#1baf7a", "#e34948"), width=1.0, label="Histogram")
    axes[2].plot(m["macd"].loc[sl], color="#2a78d6", lw=1.2, label="MACD")
    axes[2].plot(m["macd_signal"].loc[sl], color="#eb6834", lw=1.2, label="Signal")
    axes[2].set_ylabel("MACD (12, 26, 9)")
    axes[2].legend(loc="upper left", ncol=3)
    _date_axis(axes[2])
    return _save(fig, path)


def plot_class_distribution(dist: pd.DataFrame, path):
    """dist columns: ticker, split, class, frequency."""
    tickers = list(dict.fromkeys(dist["ticker"]))
    splits = ["train", "val", "test"]
    classes = list(dict.fromkeys(dist["class"]))
    class_colors = {"Down": "#e34948", "Neutral": "#a3a29c", "Up": "#1baf7a"}
    fig, axes = plt.subplots(1, len(tickers), figsize=(3.6 * len(tickers), 3.4), sharey=True, squeeze=False)
    width = 0.8 / len(classes)
    for ax, t in zip(axes[0], tickers):
        sub = dist[dist["ticker"] == t]
        for k, cls in enumerate(classes):
            vals = [sub[(sub["split"] == s) & (sub["class"] == cls)]["frequency"].sum() for s in splits]
            x = np.arange(len(splits)) + (k - (len(classes) - 1) / 2) * width
            ax.bar(x, vals, width=width * 0.92, color=class_colors.get(cls, TEXT_2), label=cls)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=TEXT_2)
        ax.set_xticks(range(len(splits)), ["Train", "Validation", "Test"])
        ax.set_title(t, loc="left")
    axes[0, 0].set_ylabel("Share of labelled days")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Target class distribution by split", x=0.01, ha="left", fontsize=13, color=TEXT)
    return _save(fig, path)


# ---------------------------------------------------------------------------------------------
# model figures
# ---------------------------------------------------------------------------------------------


def plot_confusion_matrices(cms: dict[str, np.ndarray], class_names: list[str], title: str, path):
    n = len(cms)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.1 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (name, cm) in zip(axes.flat, cms.items()):
        cm = np.asarray(cm, dtype=float)
        row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(row_norm, cmap=BLUE_RAMP, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{int(cm[i, j])}\n{row_norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if row_norm[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    return _save(fig, path)


def plot_classification_comparison(cls: pd.DataFrame, path, metrics=("accuracy", "balanced_accuracy", "f1_macro")):
    """cls columns: ticker, model_key, model, + metric columns."""
    tickers = list(dict.fromkeys(cls["ticker"]))
    labels = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "Macro F1"}
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.8 * len(metrics)), squeeze=False)
    models = list(dict.fromkeys(cls["model_key"]))
    width = 0.8 / len(models)
    for ax, metric in zip(axes[:, 0], metrics):
        for k, mk in enumerate(models):
            vals = [cls[(cls["ticker"] == t) & (cls["model_key"] == mk)][metric].mean() for t in tickers]
            x = np.arange(len(tickers)) + (k - (len(models) - 1) / 2) * width
            name = cls[cls["model_key"] == mk]["model"].iloc[0]
            ax.bar(x, vals, width=width * 0.9, color=color_for(mk), label=name)
        ax.set_xticks(range(len(tickers)), tickers)
        ax.set_ylabel(labels.get(metric, metric))
        lo = np.nanmin(cls[metric]) if len(cls) else 0
        ax.set_ylim(max(0, lo - 0.08), min(1, np.nanmax(cls[metric]) + 0.05))
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Test-period classification metrics by model", x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_feature_importance(imp: pd.DataFrame, title: str, path, top_n: int = 20, value_col="importance_mean", err_col="importance_std"):
    d = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.2))
    ax.barh(d["feature"], d[value_col], xerr=d[err_col] if err_col in d else None, color="#2a78d6", height=0.7,
            error_kw={"ecolor": TEXT_2, "elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color=TEXT_2, lw=0.8)
    ax.set_xlabel("Increase in validation log loss when the feature is permuted")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_reliability(curves: dict[str, pd.DataFrame], title: str, path, keys: dict[str, str] | None = None):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=TEXT_2, ls="--", lw=1, label="Perfect calibration")
    for name, c in curves.items():
        if c is None or c.empty:
            continue
        key = (keys or {}).get(name, name)
        ax.plot(c["mean_predicted"], c["observed_frequency"], marker="o", ms=4, color=color_for(key), label=name)
    ax.set_xlabel("Mean predicted P(Up)")
    ax.set_ylabel("Observed frequency of Up")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_lstm_history(history: list[dict], title: str, path):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", label="Training loss")
    ax.plot(h["epoch"], h["val_loss"], color="#eb6834", label="Validation loss")
    best = h.loc[h["val_loss"].idxmin(), "epoch"]
    ax.axvline(best, color=TEXT_2, ls="--", lw=1)
    ax.text(best, ax.get_ylim()[1], f" best epoch {best}", va="top", fontsize=8, color=TEXT_2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted cross-entropy")
    ax.legend()
    ax.set_title(title, loc="left")
    return _save(fig, path)


# ---------------------------------------------------------------------------------------------
# trading figures
# ---------------------------------------------------------------------------------------------


def plot_equity_curves(curves: dict[str, tuple[str, pd.Series]], title: str, path, initial_capital: float = 100_000):
    """curves: display name -> (color key, equity series)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    order = sorted(curves, key=lambda n: curves[n][0] == "buy_and_hold")  # benchmark drawn last, on top
    for name in order:
        key, eq = curves[name]
        is_bh = key == "buy_and_hold"
        ax.plot(eq.index, eq / initial_capital, color=color_for(key), ls="--" if is_bh else "-", lw=2.0 if is_bh else 1.5,
                label=name, zorder=3 if is_bh else 2)
    ax.axhline(1.0, color=TEXT_2, lw=0.8)
    ax.set_ylabel("Portfolio value (× initial capital)")
    handles, lbls = ax.get_legend_handles_labels()
    idx = [lbls.index(n) for n in curves]  # legend in the caller's order
    ax.legend([handles[i] for i in idx], [lbls[i] for i in idx], ncol=2, loc="upper left")
    ax.set_title(title, loc="left")
    _date_axis(ax)
    _note_overlaps(ax, curves)
    return _save(fig, path)


def plot_drawdowns(curves: dict[str, tuple[str, pd.Series]], title: str, path):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for name, (key, dd) in curves.items():
        ls = "--" if key == "buy_and_hold" else "-"
        ax.plot(dd.index, dd * 100, color=color_for(key), ls=ls, lw=1.4, label=name)
    ax.set_ylabel("Drawdown from peak (%)")
    ax.legend(ncol=3, loc="lower left")
    ax.set_title(title, loc="left")
    _date_axis(ax)
    _note_overlaps(ax, curves)
    return _save(fig, path)


def plot_metric_bars(trading: pd.DataFrame, metric: str, ylabel: str, title: str, path, pct: bool = False):
    """trading columns: ticker, strategy, color_key, metric..."""
    tickers = list(dict.fromkeys(trading["ticker"]))
    strategies = list(dict.fromkeys(trading["strategy"]))
    width = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for k, s in enumerate(strategies):
        sub = trading[trading["strategy"] == s]
        key = sub["color_key"].iloc[0]
        vals = [sub[sub["ticker"] == t][metric].mean() * (100 if pct else 1) for t in tickers]
        x = np.arange(len(tickers)) + (k - (len(strategies) - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.9, color=color_for(key), label=s, hatch="//" if key == "buy_and_hold" else None, edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xticks(range(len(tickers)), tickers)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_cost_sensitivity(df: pd.DataFrame, path):
    """df columns: ticker, strategy, color_key, cost_bps, sharpe."""
    # naive persistence (~490 trades) falls below -2 and would flatten every other line; it stays in the tables
    df = df[df["color_key"] != "persistence"]
    tickers = list(dict.fromkeys(df["ticker"]))
    fig, axes = plt.subplots(1, len(tickers), figsize=(4.2 * len(tickers), 3.8), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tickers):
        sub = df[df["ticker"] == t]
        for s in dict.fromkeys(sub["strategy"]):
            d = sub[sub["strategy"] == s].sort_values("cost_bps")
            key = d["color_key"].iloc[0]
            is_bh = key == "buy_and_hold"
            ax.plot(d["cost_bps"], d["sharpe"], marker="o", ms=4, color=color_for(key), ls="--" if is_bh else "-",
                    lw=2.0 if is_bh else 1.5, zorder=3 if is_bh else 2, label=s)
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_xlabel("One-way cost per unit turnover (bps; round trip = 2×)")
        ax.set_title(t, loc="left")
    axes[0, 0].set_ylabel("Test Sharpe ratio")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.suptitle("Transaction-cost sensitivity (positions fixed, costs varied; naive persistence omitted, see report table)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_threshold_sensitivity(df: pd.DataFrame, chosen: dict, title: str, path):
    """df columns: model_key, model, threshold, sharpe, exposure (TEST period, post-hoc analysis)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for mk in dict.fromkeys(df["model_key"]):
        d = df[df["model_key"] == mk]
        axes[0].plot(d["threshold"], d["sharpe"], marker="o", ms=3, color=color_for(mk), label=d["model"].iloc[0])
        axes[1].plot(d["threshold"], d["exposure"] * 100, marker="o", ms=3, color=color_for(mk))
        if mk in chosen and chosen[mk] is not None:
            axes[0].axvline(chosen[mk], color=color_for(mk), lw=0.8, ls=":")
    axes[0].set_xlabel("P(Up) threshold τ")
    axes[0].set_ylabel("Test Sharpe ratio")
    axes[1].set_xlabel("P(Up) threshold τ")
    axes[1].set_ylabel("Days invested (%)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title, loc="left")
    return _save(fig, path)


def plot_walk_forward(wf: pd.DataFrame, metric: str, ylabel: str, title: str, path):
    """wf columns: ticker, year, strategy, color_key, metric."""
    tickers = list(dict.fromkeys(wf["ticker"]))
    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 2.9 * len(tickers)), sharex=True, squeeze=False)
    for ax, t in zip(axes[:, 0], tickers):
        sub = wf[wf["ticker"] == t]
        for s in dict.fromkeys(sub["strategy"]):
            d = sub[sub["strategy"] == s].sort_values("year")
            key = d["color_key"].iloc[0]
            ax.plot(d["year"], d[metric], marker="o", ms=4, color=color_for(key), ls="--" if key == "buy_and_hold" else "-", label=s)
        ax.axhline(0 if metric != "accuracy" else np.nan, color=TEXT_2, lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(t, loc="left")
    axes[-1, 0].set_xlabel("Test year (expanding-window walk-forward)")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_regime_heatmap(df: pd.DataFrame, title: str, path, value_col: str = "sharpe"):
    """df: rows = strategy, columns = regime, values = metric."""
    fig, ax = plt.subplots(figsize=(1.6 * df.shape[1] + 3, 0.45 * df.shape[0] + 1.4))
    vals = df.values.astype(float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    ax.imshow(vals, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, "N/A" if not np.isfinite(vals[i, j]) else f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=0)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.set_title(title, loc="left")
    return _save(fig, path)
