# Screenshot capture notes

All images in this folder are **real screenshots of the project's Streamlit dashboard** (`app/streamlit_app.py`),
running locally at `http://localhost:8501`. They were captured on **24 October 2026** with a headless Chromium
browser (`agent-browser`) at a 1600 × 900 viewport. Nothing in them is mocked up or edited.

| File | Dashboard tab | What is visible |
|---|---|---|
| `01_overview.png` | Overview | Headline test-period results for SPY, key findings, "how the system works" |
| `02_backtest.png` | Interactive backtest | XGBoost on SPY with the pipeline's own settings (τ = 0.375, 15 bps), results table, portfolio-value chart |
| `02b_backtest_drawdown.png` | Interactive backtest (scrolled) | Portfolio value and drawdown charts |
| `03_predictions.png` | Predictions | Test-period classification metrics computed from the 1,001 saved predictions |
| `04_walkforward.png` | Walk-forward & robustness | SPY walk-forward out-of-sample portfolio value 2016–2025 |
| `05_train_live.png` | Train a model live | Result of pressing "Download SPY and train XGBoost" (see below) |
| `06_figures.png` | Figures & docs | Figure browser |

## The live training run (`05_train_live.png`)

The "Train a model live" tab downloads fresh daily data from Yahoo Finance and runs the real pipeline end to end.
It was run once during the capture with **SPY, XGBoost, "use the hyperparameters tuned in the study" on**. The
dashboard reported:

- Pipeline finished in 8.2 s
- Train 2011-01-31 → 2019-12-27 (2,243 days) · validation 2020-01-02 → 2021-12-29 (503) · test 2022-01-03 → 2026-10-21 (1,205) · 42 features
- Test accuracy 47.1% (−1.9 pp vs majority) · test macro F1 0.254
- Strategy Sharpe 0.68 (−0.03 vs buy-and-hold) · τ chosen on validation 0.375
- Results table: XGBoost (live) Sharpe 0.68, buy-and-hold 0.71

These numbers come from that single live run, and slide 16 of the deck quotes them. They are **not** part of the
reported study results in `results/`, which end on 2025-12-31. Rerunning the tab on another day will give different
numbers, because the test window extends to the latest trading day.
