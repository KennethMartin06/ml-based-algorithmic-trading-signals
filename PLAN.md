# Implementation Plan

Source: `../ML_Based_Algorithmic_Trading_Signals_Synopsis.docx` (MIT Manipal mini-project
synopsis, Ishan Singh & Kenneth Martin, mentors Dr. Narendra VG and Dr. Ashwath Rao B).

## Requirements extracted from the synopsis

| Synopsis section | Requirement |
|---|---|
| 5.1 Problem formulation | For each day *t*, use only information available at or before *t* to predict the direction of the next trading period; map the output to Buy / Hold / Sell. Strict chronological splits. |
| 5.2 Data | Daily OHLCV for liquid equities / indices from Yahoo Finance (`yfinance`); clean missing values, adjust for corporate actions, sort chronologically, check duplicates; time-based train/validation/test split. |
| 5.3 Features | SMA, EMA, RSI, MACD, Bollinger Bands, returns, rolling volatility, volume features, lagged values, all point-in-time. |
| 5.4 Baselines | Moving-average crossover and buy-and-hold under the same capital, period, costs and sizing. |
| 5.5 XGBoost | Next-period direction classifier; hyperparameters tuned on train/validation only; probability → signal with a confidence threshold (low confidence → Hold). |
| 5.6 LSTM | Rolling sequences of technical features, chronological windows, training-set normalisation only, unseen future test period. |
| 5.7 Backtest | Position sizing policy, transaction costs, slippage, execution timing; cumulative/annualised return, Sharpe, Sortino, max drawdown, volatility, win rate, turnover + accuracy/precision/recall/F1. |
| 5.8 RAG sentiment | Stretch goal: time-stamped news → financial sentiment → features → compare with technical-only. |
| 5.9 Robustness | Multiple assets and regimes; bootstrap CIs or repeated walk-forward evaluation. |

The professor's feedback on the previous version — "not enough machine learning" — is the main driver:
the ML side must be substantial, tuned, validated and compared honestly.

## Components to implement (today)

1. Data acquisition with caching, validation, corporate-action adjustment and a data manifest (SHA-256).
2. Point-in-time feature engineering (~40 scale-invariant features in 6 groups) with truncation-invariance tests.
3. Execution-aligned targets (three-class Down/Neutral/Up with a configurable neutral band; binary variant).
4. Chronological train/validation/test split with label-overlap purging.
5. Models: Logistic Regression, Random Forest, XGBoost (HistGradientBoosting fallback, clearly labelled),
   PyTorch LSTM, validation-selected soft-voting ensemble; majority-class and persistence baselines.
6. Hyperparameter tuning with purged `TimeSeriesSplit` CV (log-loss), LSTM tuning on validation loss.
7. Probability-threshold signals (threshold chosen on validation), argmax "always active" comparison.
8. Cost-aware backtester (fees + slippage in bps, next-open execution, long-only default, long/short option,
   volatility-target sizing option).
9. Baselines: buy-and-hold, MA crossover (10/50), naive persistence.
10. Expanding-window walk-forward evaluation (annual retraining, per-fold re-tuning).
11. Interpretability: permutation importance on validation, impurity/gain importance, mutual information,
    confusion matrices, reliability diagrams, Brier / log loss.
12. Robustness: multiple assets, volatility & trend regimes, per-year results, transaction-cost
    sensitivity, threshold sensitivity, feature-group ablation, circular block-bootstrap CIs.
13. Automated pipeline (`python -m scripts.run_experiment`), report generator, pytest suite, docs.

## Optional / stretch

- RAG news-sentiment extension (synopsis 5.8): **not implemented** today. There is no free, reliable,
  point-in-time historical news archive covering 2010–2025 that can be retrieved without paid APIs,
  and fabricating sentiment would be scientifically invalid. Documented as future work with a
  leakage-safe design in `docs/limitations.md`.
- Probability calibration *fitting* (only evaluation-side calibration analysis is done).

## Known limitations (planned, to be confirmed by results)

- Daily direction of liquid US large caps is close to a random walk; low predictive power is expected.
- Backtests ignore market impact, borrow costs, taxes and partial fills.
- Universe chosen with hindsight (current mega-caps) → survivorship bias.
- Yahoo adjusted prices are back-adjusted; only scale-invariant features are used to limit this effect.
