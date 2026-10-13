"""Markdown research report generated exclusively from the saved metric files.

Every number in the report is read from ``results/metrics``; every "key finding" is computed from
those tables by explicit rules, so the text cannot drift from the actual experiment output.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import fmt, load_json, resolve_path


def _md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = formats.get(c)
            if f == "pct":
                cells.append(fmt(v, 3, pct=True))
            elif f == "int":
                cells.append("N/A" if pd.isna(v) else f"{int(v)}")
            elif isinstance(f, int):
                cells.append(fmt(v, f))
            elif isinstance(v, (float, np.floating)):
                cells.append(fmt(v, 3))
            else:
                cells.append("N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.stat().st_size > 5:
        return pd.read_csv(path)
    return None


TOL = 1e-9  # metric differences smaller than this are floating-point noise, not a result


def _same_as_buy_and_hold(row) -> bool:
    """A long-only strategy that is invested on every day with a single (entry) trade holds exactly the
    buy-and-hold position, so all of its metrics equal buy-and-hold's."""
    return bool(row["exposure"] >= 1.0 - 1e-12 and row["n_trades"] <= 1)


def _compare_counts(rows: pd.DataFrame, bench: pd.DataFrame, metric: str, higher_is_better: bool = True) -> tuple[int, int]:
    """(# strictly better, # strictly worse) than the benchmark of the same ticker, beyond TOL."""
    better = worse = 0
    for _, r in rows.iterrows():
        a, b = r[metric], bench.loc[r["ticker"], metric]
        if pd.isna(a) or pd.isna(b):
            continue
        diff = (a - b) if higher_is_better else (b - a)
        better += diff > TOL
        worse += diff < -TOL
    return int(better), int(worse)


def key_findings(cls: pd.DataFrame, trading: pd.DataFrame, boot: pd.DataFrame | None, wf: pd.DataFrame | None) -> list[str]:
    """Objective statements derived from the result tables by fixed rules (with a float tolerance)."""
    out = []
    test = cls[cls["split"] == "test"]
    ml_keys = ["logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting", "lstm", "ensemble"]
    for t in dict.fromkeys(test["ticker"]):
        sub = test[(test["ticker"] == t) & test["model_key"].isin(ml_keys)]
        maj = test[(test["ticker"] == t) & (test["model_key"] == "majority_class")]
        if len(sub) and len(maj):
            best = sub.loc[sub["f1_macro"].idxmax()]
            best_acc = sub.loc[sub["accuracy"].idxmax()]
            out.append(
                f"**{t}** - highest test macro-F1: {best['model']} ({best['f1_macro']:.3f}); highest test accuracy: "
                f"{best_acc['model']} ({best_acc['accuracy']:.3f}) vs majority-class accuracy {maj['accuracy'].iloc[0]:.3f} "
                f"(macro-F1 {maj['f1_macro'].iloc[0]:.3f})."
            )
    conf = trading[trading["kind"] == "ml_confidence"]
    bh = trading[trading["strategy"] == "Buy & Hold"].set_index("ticker")
    if len(conf) and len(bh):
        n = len(conf)
        clones = conf[conf.apply(_same_as_buy_and_hold, axis=1)]
        s_b, s_w = _compare_counts(conf, bh, "sharpe")
        r_b, r_w = _compare_counts(conf, bh, "cumulative_return")
        d_b, d_w = _compare_counts(conf, bh, "max_drawdown")  # drawdowns are negative: larger = smaller loss
        out.append(f"Test period, {n} ML confidence-threshold strategies (models x tickers) vs buy-and-hold: Sharpe ratio higher for {s_b}, "
                   f"lower for {s_w}; cumulative return higher for {r_b}, lower for {r_w}; maximum drawdown smaller for {d_b}, larger for {d_w} "
                   f"(the remainder are equal within {TOL:g}).")
        if len(clones):
            names = ", ".join(f"{r['strategy'].replace(' (conf.)', '')} on {r['ticker']}" for _, r in clones.iterrows())
            out.append(f"{len(clones)} of {n} ML strategies were invested on every test day and are therefore identical to buy-and-hold "
                       f"({names}): the threshold selected on the validation period kept them always long.")
    ma = trading[trading["model_key"] == "ma_crossover"]
    if len(ma) and len(bh):
        ma_b, _ = _compare_counts(ma, bh, "sharpe")
        out.append(f"The MA-crossover baseline had a higher test Sharpe ratio than buy-and-hold for {ma_b} of {len(ma)} tickers.")
    if boot is not None and len(boot):
        d = boot[boot["statistic"] == "sharpe_minus_buy_and_hold"].merge(
            trading[["ticker", "strategy", "kind"]], on=["ticker", "strategy"], how="left")
        if len(d):
            parts = []
            for label, g in (("ML strategies", d[d["kind"] == "ml_confidence"]), ("rule-based baselines", d[d["kind"] == "baseline"])):
                if not len(g):
                    continue
                ident = (g["low"].abs() < 1e-12) & (g["high"].abs() < 1e-12)
                h = g[~ident]
                txt = (f"{label}: {int((h['low'] > 0).sum())} of {len(g)} entirely above 0, {int((h['high'] < 0).sum())} entirely below 0, "
                       f"{int(((h['low'] <= 0) & (h['high'] >= 0)).sum())} include 0")
                if ident.any():
                    txt += f", {int(ident.sum())} identical to buy-and-hold (zero difference)"
                parts.append(txt)
            out.append("Block-bootstrap 95% intervals for Sharpe(strategy) - Sharpe(buy-and-hold) - " + "; ".join(parts)
                       + ". An interval that includes 0 means no reliable difference at this sample size.")
        a = boot[boot["statistic"] == "accuracy_minus_majority_class"]
        if len(a):
            out.append(f"Accuracy minus majority-class accuracy (block bootstrap): {int((a['low'] > 0).sum())} of {len(a)} model/ticker intervals lie "
                       f"entirely above 0, {int((a['high'] < 0).sum())} entirely below 0, the rest include 0.")
    if wf is not None and len(wf):
        conf_wf = wf[wf["strategy"].str.endswith("(conf.)")]
        bh_wf = wf[wf["strategy"] == "Buy & Hold"].set_index("ticker")
        if len(conf_wf) and len(bh_wf):
            w_b, w_w = _compare_counts(conf_wf, bh_wf, "sharpe")
            r_b, _ = _compare_counts(conf_wf, bh_wf, "cumulative_return")
            out.append(f"Walk-forward ({wf['start'].min()[:4]}-{wf['end'].max()[:4]}, annual retraining): {w_b} of {len(conf_wf)} ML strategies had a "
                       f"higher out-of-sample Sharpe ratio than buy-and-hold ({w_w} lower); {r_b} had a higher cumulative return.")
    return out


def generate_report(results_root: str | Path, cfg: dict) -> Path:
    root = resolve_path(results_root)
    m = root / "metrics"
    summary = load_json(m / "run_summary.json") if (m / "run_summary.json").exists() else {}
    # describe the configuration the results were produced with, not the current CLI configuration
    cfg = summary.get("config", cfg)
    stages = summary.get("stage_seconds")  # None for staged (train/evaluate) runs
    manifest = load_json(m / "data_manifest.json")
    ds = load_json(m / "dataset_summary.json")
    best = load_json(m / "best_params.json")
    thresholds = load_json(m / "selected_thresholds.json")
    cls = pd.read_csv(m / "classification_metrics.csv")
    trading = pd.read_csv(m / "trading_metrics_test.csv")
    comp = pd.read_csv(m / "model_comparison_test.csv")
    dist = _read(m / "class_distribution.csv")
    boot = _read(m / "bootstrap_confidence_intervals.csv")
    wf_ran = stages is None or "walk_forward" in stages
    wf_tr = _read(m / "walk_forward_trading.csv") if wf_ran else None
    wf_cls = _read(m / "walk_forward_classification.csv") if wf_ran else None
    cost = _read(m / "robustness_cost_sensitivity.csv")
    vt = _read(m / "robustness_vol_target.csv")
    variants = _read(m / "robustness_variants.csv") if (stages is None or "sensitivity_variants" in stages) else None
    abl = _read(m / "feature_ablation.csv")
    cal = _read(m / "calibration.csv")
    regimes = _read(m / "robustness_regimes.csv")
    synthetic = bool(summary.get("synthetic_data", False)) or any(x.get("is_synthetic") for x in manifest)

    L = []
    L.append("# Experiment Report: ML-Based Algorithmic Trading Signals")
    L.append("")
    if synthetic:
        L.append("> **WARNING - SYNTHETIC DATA.** This run used a generated random walk, not market data. "
                 "Nothing in this report describes real market performance.")
    else:
        L.append("> Real market data (Yahoo Finance via `yfinance`). Historical backtest results are not a guarantee of future "
                 "performance and are not investment advice.")
    L.append("")
    L.append(f"Generated: {date.today().isoformat()} | experiment `{summary.get('experiment', cfg['experiment']['name'])}` | "
             f"total runtime {summary.get('total_seconds', 'N/A')} s")
    L.append("")

    # 1. configuration
    L.append("## 1. Experiment configuration")
    L.append("")
    conf_rows = pd.DataFrame([
        ("Universe", ", ".join(cfg["data"]["tickers"])),
        ("Target", f"{cfg['target']['scheme']}, horizon {cfg['target']['horizon']} day(s), neutral band ±{cfg['target']['neutral_threshold']:.2%}"),
        ("Execution convention", cfg["execution"]["convention"]),
        ("Signal mode", cfg["signals"]["mode"]),
        ("Threshold selection", f"maximise validation {cfg['signals']['selection_metric']} (min exposure {cfg['signals']['min_exposure']:.0%})"),
        ("Costs", f"{cfg['backtest']['fee_bps']} bps fee + {cfg['backtest']['slippage_bps']} bps slippage per unit turnover"),
        ("Initial capital", f"{cfg['backtest']['initial_capital']:,}"),
        ("Annualisation / risk-free", f"{cfg['backtest']['annualization']} days / {cfg['backtest']['risk_free_rate']:.2%}"),
        ("Tuning", f"purged TimeSeriesSplit, {cfg['models']['tuning']['cv_splits']} folds, scored by log loss"),
        ("Random seed", cfg["experiment"]["seed"]),
    ], columns=["Setting", "Value"])
    L.append(_md_table(conf_rows))
    env = summary.get("environment", {})
    if env:
        L.append("")
        L.append("Software: " + ", ".join(f"{k} {v}" for k, v in env.items() if k not in ("platform",)))
    L.append("")

    # 2. data
    L.append("## 2. Data sources and dates")
    L.append("")
    man = pd.DataFrame([{"Ticker": x["ticker"], "Source": x["source"] + (" (SYNTHETIC)" if x["is_synthetic"] else ""), "Rows": x["rows"],
                         "First date": x["first_date"], "Last date": x["last_date"], "SHA-256 (raw CSV)": x["sha256"][:16] + "…"} for x in manifest])
    L.append(_md_table(man, {"Rows": "int"}))
    L.append("")
    L.append("`source = cache` means the file in `data/raw` written by an earlier yfinance download was reused. "
             "Prices are split- and dividend-adjusted (factor `Adj Close / Close` applied to OHLC).")
    L.append("")

    # 3. observations, features, splits
    L.append("## 3. Observations, features and chronological splits")
    L.append("")
    rows = []
    for t, d in ds.items():
        sp = d["split"]
        rows.append({"Ticker": t, "Rows after warm-up": d["dataset"]["rows"], "Features": d["n_features"],
                     "Train": f"{sp['train']['start']} → {sp['train']['end']} ({sp['train']['rows']})",
                     "Validation": f"{sp['val']['start']} → {sp['val']['end']} ({sp['val']['rows']})",
                     "Test": f"{sp['test']['start']} → {sp['test']['end']} ({sp['test']['rows']})"})
    L.append(_md_table(pd.DataFrame(rows), {"Rows after warm-up": "int", "Features": "int"}))
    L.append("")
    L.append("Rows whose label window crosses a split boundary are purged. The first rows of the download are consumed by indicator warm-up (252-day windows).")
    L.append("")

    if dist is not None:
        L.append("### Class distribution")
        L.append("")
        piv = dist.pivot_table(index=["ticker", "split"], columns="class", values="frequency").reset_index()
        piv["split"] = pd.Categorical(piv["split"], ["train", "val", "test"])
        piv = piv.sort_values(["ticker", "split"])
        L.append(_md_table(piv, {c: "pct" for c in piv.columns if c not in ("ticker", "split")}))
        L.append("")

    # 4. models
    L.append("## 4. Models and selected hyperparameters")
    L.append("")
    L.append("Hyperparameters were selected on training data only (purged time-series CV); the LSTM architecture and the ensemble "
             "members on the validation period; decision thresholds on the validation period.")
    L.append("")
    rows = []
    for t, models in best.items():
        for k, p in models.items():
            if isinstance(p, dict):
                shown = {kk: vv for kk, vv in p.items() if kk not in ("seed", "shuffle_train_batches", "max_epochs", "patience", "batch_size")}
                rows.append({"Ticker": t, "Model": k, "Selected parameters": ", ".join(f"{kk}={vv}" for kk, vv in shown.items()),
                             "Threshold τ (val)": thresholds.get(t, {}).get(k)})
    L.append(_md_table(pd.DataFrame(rows), {"Threshold τ (val)": 3}))
    L.append("")

    # 5. classification
    L.append("## 5. Classification metrics (test period)")
    L.append("")
    L.append("Macro averages weight Down, Neutral and Up equally. Log loss and Brier score use predicted probabilities "
             "(lower is better); naive persistence produces hard labels only.")
    L.append("")
    tc = cls[cls["split"] == "test"][["ticker", "model", "n", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss", "brier"]]
    L.append(_md_table(tc, {"n": "int"}))
    L.append("")

    # 6. trading
    L.append("## 6. Trading metrics (test period, net of costs)")
    L.append("")
    L.append("ML strategies use the confidence rule (long if P(Up) ≥ τ, τ chosen on validation); \"argmax\" variants trade the predicted class directly.")
    L.append("")
    tt = trading[["ticker", "strategy", "threshold", "cumulative_return", "annualized_return", "annualized_volatility", "sharpe", "sortino",
                  "max_drawdown", "win_rate", "exposure", "turnover_annual", "n_trades", "costs_paid"]]
    L.append(_md_table(tt, {"cumulative_return": "pct", "annualized_return": "pct", "annualized_volatility": "pct", "max_drawdown": "pct",
                            "win_rate": "pct", "exposure": "pct", "n_trades": "int", "costs_paid": 0, "threshold": 3, "turnover_annual": 1}))
    L.append("")

    # 7. comparison
    L.append("## 7. Comparison with baselines (averaged over tickers)")
    L.append("")
    main = comp[comp["kind"].isin(["baseline", "ml_confidence"])]
    agg = main.groupby("strategy", sort=False)[["accuracy", "f1_macro", "cumulative_return", "sharpe", "sortino", "max_drawdown", "turnover_annual", "n_trades"]].mean().reset_index()
    L.append(_md_table(agg, {"cumulative_return": "pct", "max_drawdown": "pct", "n_trades": 1, "turnover_annual": 1}))
    L.append("")
    if boot is not None and len(boot):
        L.append("### Block-bootstrap 95% confidence intervals")
        L.append("")
        L.append(f"Circular block bootstrap (block {cfg['robustness']['bootstrap']['block_len']} days, {cfg['robustness']['bootstrap']['n_boot']} resamples) of daily test returns / prediction outcomes.")
        L.append("")
        b = boot[["ticker", "strategy", "statistic", "estimate", "low", "high"]]
        L.append(_md_table(b))
        L.append("")

    # 8. walk-forward
    if wf_tr is not None:
        L.append("## 8. Walk-forward evaluation (expanding window, annual retraining)")
        L.append("")
        L.append(f"For each test year Y in {cfg['walk_forward']['first_test_year']}-{cfg['walk_forward']['last_test_year']}: train on all years before Y-1, "
                 "select thresholds / early-stop on Y-1, test on Y. Hyperparameters are re-tuned in every fold with reduced grids. "
                 "Out-of-sample years are concatenated into one continuous backtest.")
        L.append("")
        w = wf_tr[["ticker", "strategy", "start", "end", "cumulative_return", "annualized_return", "sharpe", "sortino", "max_drawdown", "exposure", "n_trades"]]
        L.append(_md_table(w, {"cumulative_return": "pct", "annualized_return": "pct", "max_drawdown": "pct", "exposure": "pct", "n_trades": "int"}))
        L.append("")
        if wf_cls is not None:
            L.append(_md_table(wf_cls[["ticker", "model", "period", "n", "accuracy", "balanced_accuracy", "f1_macro", "log_loss"]], {"n": "int"}))
            L.append("")

    # 9. robustness
    L.append("## 9. Robustness analyses")
    L.append("")
    if cost is not None:
        L.append("### Transaction-cost sensitivity (test Sharpe ratio)")
        L.append("")
        L.append("Columns are the total one-way cost (fee + slippage) charged per unit of turnover; entering and later exiting a "
                 "position (a round trip) costs twice this. Positions are held fixed; only the cost assumption changes.")
        L.append("")
        piv = cost.pivot_table(index=["ticker", "strategy"], columns="cost_bps", values="sharpe", sort=False).reset_index()
        piv.columns = [f"{c} bps" if isinstance(c, (int, float, np.integer, np.floating)) else c for c in piv.columns]
        L.append(_md_table(piv))
        L.append("")
    if vt is not None:
        L.append("### Fixed vs volatility-targeted position sizing (test)")
        L.append("")
        L.append(_md_table(vt, {"cumulative_return": "pct", "annualized_return": "pct", "annualized_volatility": "pct", "max_drawdown": "pct", "turnover_annual": 1}))
        L.append("")
    if variants is not None:
        L.append("### Sensitivity to execution timing and target definition (tabular models + ensemble, reduced grids, no LSTM)")
        L.append("")
        L.append("`next_open` = signal after the close of day t, filled at the next open (default). `market-on-close` = idealised fill at the "
                 "same close that generated the signal (optimistic). `binary` = Up/Down target without a neutral band. Accuracy is not "
                 "comparable between 3-class and 2-class targets; compare each model with the majority-class row of the same variant.")
        L.append("")
        for metric, title in (("sharpe", "Test Sharpe ratio"), ("accuracy", "Test accuracy"), ("f1_macro", "Test macro-F1")):
            piv = variants.pivot_table(index=["ticker", "strategy"], columns="variant", values=metric, sort=False).reset_index()
            L.append(f"**{title}**")
            L.append("")
            L.append(_md_table(piv))
            L.append("")
    if abl is not None:
        L.append("### Feature-group ablation (XGBoost, tuned parameters fixed)")
        L.append("")
        L.append(_md_table(abl, {"n_features": "int", "test_cumulative_return": "pct", "threshold": 3}))
        L.append("")
    if cal is not None:
        L.append("### Calibration of P(Up) on the test period")
        L.append("")
        L.append(_md_table(cal, {"mean_p_up": "pct", "observed_up_rate": "pct"}))
        L.append("")
    if regimes is not None:
        L.append("### Sharpe ratio by regime (test period)")
        L.append("")
        piv = regimes.pivot_table(index=["ticker", "strategy"], columns="regime", values="sharpe", sort=False).reset_index()
        L.append(_md_table(piv))
        L.append("")
        L.append("Regime Sharpe ratios are computed on non-contiguous subsets of days and are descriptive only.")
        L.append("")

    # 10. figures
    L.append("## 10. Figures")
    L.append("")
    figs = sorted((root / "figures").glob("*.png"))
    for f in figs:
        L.append(f"- [{f.stem.replace('_', ' ')}](../figures/{f.name})")
    L.append("")
    key_figs = ["price_history.png", "classification_comparison.png", "sharpe_comparison.png", "cost_sensitivity.png", "walk_forward_sharpe.png"]
    for name in key_figs:
        if (root / "figures" / name).exists():
            L.append(f"![{name}](../figures/{name})")
            L.append("")

    # 11. findings
    L.append("## 11. Key findings (rule-based, computed from the tables above)")
    L.append("")
    for f in key_findings(cls, trading, boot, wf_tr):
        L.append(f"- {f}")
    L.append("")

    # 12. limitations + reproducibility
    L.append("## 12. Limitations")
    L.append("")
    L.append("- One test period (2022-2025) and three liquid US instruments chosen with hindsight; results may not generalise.")
    L.append("- Daily direction is close to unpredictable; small differences in accuracy are within sampling noise (see bootstrap intervals).")
    L.append("- Costs are a flat bps assumption; no market impact, borrow fees, taxes, dividends timing or partial fills.")
    L.append("- The optional RAG / news-sentiment extension was not implemented (no point-in-time news archive without paid APIs).")
    L.append("- See `docs/limitations.md` for the full discussion.")
    L.append("")
    L.append("## 13. Reproducibility")
    L.append("")
    L.append("```bash")
    L.append("pip install -r requirements.txt")
    L.append("python -m scripts.run_experiment            # full experiment (this report)")
    L.append("python -m scripts.run_experiment --quick    # SPY-only smoke run -> results_quick/")
    L.append("python -m pytest -q")
    L.append("```")
    L.append("")
    L.append("Raw downloads are cached in `data/raw/` and fingerprinted (SHA-256) in `results/metrics/data_manifest.json`. "
             "Yahoo Finance may revise adjusted prices; compare checksums to confirm identical inputs.")
    L.append("")

    out = root / "reports" / "experiment_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    # Only the full default real-data experiment updates the README (never quick, partial or synthetic runs).
    if summary.get("is_default_full_run") and not synthetic and root.name == "results":
        update_readme(resolve_path("README.md"), readme_results_block(root, cfg))
    return out


# ---------------------------------------------------------------------------------------------
# README results block
# ---------------------------------------------------------------------------------------------

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def readme_results_block(root: Path, cfg: dict) -> str:
    """Markdown summary of the main experiment, built only from saved metric files."""
    m = root / "metrics"
    comp = pd.read_csv(m / "model_comparison_test.csv")
    cls = pd.read_csv(m / "classification_metrics.csv")
    trading = pd.read_csv(m / "trading_metrics_test.csv")
    manifest = load_json(m / "data_manifest.json")
    boot = _read(m / "bootstrap_confidence_intervals.csv")
    wf = _read(m / "walk_forward_trading.csv")
    synthetic = any(x.get("is_synthetic") for x in manifest)
    bh = trading[trading["strategy"] == "Buy & Hold"]
    L = []
    if synthetic:
        L.append("> **SYNTHETIC DATA - these are not market results.**")
        L.append("")
    L.append(f"All numbers below are generated from `results/metrics/` by `python -m scripts.generate_report` "
             f"(real Yahoo Finance data; test period **{bh['start'].min()} → {bh['end'].max()}**, "
             f"{int(bh['n_days'].iloc[0])} trading days per ticker; next-open execution; "
             f"{cfg['backtest']['fee_bps'] + cfg['backtest']['slippage_bps']} bps per unit turnover; {cfg['signals']['mode'].replace('_', '-')}). "
             "Full tables: [results/reports/experiment_report.md](results/reports/experiment_report.md).")
    L.append("")
    L.append("ML rows use the confidence rule (long if P(Up) ≥ τ, τ chosen on validation). Classification metrics are "
             "test-set accuracy and **macro**-averaged precision / recall / F1 over Down, Neutral, Up. N/A = not applicable "
             "(rule-based strategy, no class predictions).")
    L.append("")
    test_cls = cls[cls["split"] == "test"]
    for t in dict.fromkeys(comp["ticker"]):
        sub = comp[(comp["ticker"] == t) & comp["kind"].isin(["baseline", "ml_confidence"])].copy()
        maj = test_cls[(test_cls["ticker"] == t) & (test_cls["model_key"] == "majority_class")]
        rows = []
        for _, r in sub.iterrows():
            rows.append({"Strategy": r["strategy"], "Accuracy": r["accuracy"], "Precision": r["precision_macro"], "Recall": r["recall_macro"],
                         "F1": r["f1_macro"], "Cum. return": r["cumulative_return"], "Sharpe": r["sharpe"], "Sortino": r["sortino"],
                         "Max DD": r["max_drawdown"], "Turnover/yr": r["turnover_annual"], "Trades": r["n_trades"]})
            if r["strategy"] == "Naive persistence" and len(maj):
                mm = maj.iloc[0]
                rows.append({"Strategy": "Majority class (classifier only)", "Accuracy": mm["accuracy"], "Precision": mm["precision_macro"],
                             "Recall": mm["recall_macro"], "F1": mm["f1_macro"], "Cum. return": np.nan, "Sharpe": np.nan, "Sortino": np.nan,
                             "Max DD": np.nan, "Turnover/yr": np.nan, "Trades": np.nan})
        L.append(f"#### {t}")
        L.append("")
        L.append(_md_table(pd.DataFrame(rows), {"Cum. return": "pct", "Max DD": "pct", "Trades": "int", "Turnover/yr": 1, "Sharpe": 2, "Sortino": 2}))
        L.append("")
    if wf is not None and len(wf):
        L.append(f"#### Walk-forward, out-of-sample {wf['start'].min()[:4]}–{wf['end'].max()[:4]} (annual retraining)")
        L.append("")
        piv_s = wf.pivot_table(index="strategy", columns="ticker", values="sharpe", sort=False)
        piv_r = wf.pivot_table(index="strategy", columns="ticker", values="cumulative_return", sort=False)
        rows = []
        for strat in piv_s.index:
            row = {"Strategy": strat}
            for t in piv_s.columns:
                row[f"{t} Sharpe"] = piv_s.loc[strat, t]
                row[f"{t} cum. return"] = piv_r.loc[strat, t]
            rows.append(row)
        L.append(_md_table(pd.DataFrame(rows), {c: ("pct" if "return" in c else 2) for c in rows[0] if c != "Strategy"}))
        L.append("")
    L.append("#### Key findings (computed by fixed rules from the tables)")
    L.append("")
    for f in key_findings(cls, trading, boot, wf):
        L.append(f"- {f}")
    L.append("")
    L.append("#### Selected figures")
    L.append("")
    first = cfg["data"]["tickers"][0]
    for name, caption in (("classification_comparison.png", "Test-period classification metrics"),
                          ("sharpe_comparison.png", "Test-period Sharpe ratio by strategy"),
                          (f"{first}_equity_curves.png", f"{first} equity curves (test, net of costs)"),
                          ("walk_forward_sharpe.png", "Walk-forward Sharpe ratio per year"),
                          ("cost_sensitivity.png", "Transaction-cost sensitivity")):
        if (root / "figures" / name).exists():
            L.append(f"**{caption}**")
            L.append("")
            L.append(f"![{caption}](results/figures/{name})")
            L.append("")
    return "\n".join(L)


def update_readme(readme: Path, block: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(head + README_START + "\n" + block + "\n" + README_END + tail, encoding="utf-8")
    return True
