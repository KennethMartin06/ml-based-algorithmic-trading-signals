# Methodology

This document specifies exactly what the code does. All symbols refer to daily bars of one instrument;
*t* indexes trading days. Every setting below lives in `config.yaml`.

## 1. Data

* **Source:** Yahoo Finance via `yfinance` (`auto_adjust=False`), daily OHLCV + `Adj Close`, 2010-01-01 to
  2025-12-31, for SPY, AAPL and MSFT. The universe was fixed before any model was evaluated.
* **Cleaning** (`src/data_loader.py`): sort by date, drop duplicate dates, drop rows with missing or
  non-positive prices, widen High/Low that contradict Open/Close. Every action is counted in
  `results/metrics/data_manifest.json`; nothing is imputed.
* **Corporate actions:** with `f_t = AdjClose_t / Close_t`, adjusted prices are
  `O'_t = f_t·O_t`, `H'_t = f_t·H_t`, `L'_t = f_t·L_t`, `C'_t = AdjClose_t`.
  Returns across splits and dividends are therefore correct, and intraday ratios (C/O, H/L) are unchanged.
* **Reproducibility:** downloads are cached in `data/raw/` and fingerprinted with SHA-256.

## 2. Decision timing and execution convention

A decision for day *t* is made **after the close of day t** using information from days ≤ *t*.

| Convention | Fill | Return earned by the decision of day *t* |
|---|---|---|
| `next_open` (default) | open of *t+1* | `R_t = O'_{t+2} / O'_{t+1} − 1` (held until the next decision is filled) |
| `close` (sensitivity only) | close of *t* | `R_t = C'_{t+1} / C'_t − 1` |

The `close` convention assumes the same closing price can be used both to compute the signal and to trade,
which is optimistic. It is reported only as a sensitivity analysis.

## 3. Target

For horizon *h* (default 1) the forward return is aligned with the execution convention:

* `next_open`: `r_t = O'_{t+1+h} / O'_{t+1} − 1`
* `close`: `r_t = C'_{t+h} / C'_t − 1` (the formula `Close[t+1]/Close[t] − 1` for *h* = 1)

**Three-class label** with neutral band θ (default 0.001 = 0.1%, a modelling assumption):

```
y_t = 0 (Down)    if r_t < −θ
y_t = 1 (Neutral) if −θ ≤ r_t ≤ θ
y_t = 2 (Up)      if r_t > θ
```

**Binary label** (sensitivity analysis): `y_t = 1 (Up) if r_t > 0 else 0 (Down)`.

The label of row *t* depends on prices up to row `t + k`, where `k = h + 1` (next_open) or `k = h` (close).
The date of row `t + k` is stored as `label_end_date`. Rows without a complete future window have NaN labels and
are never used. The label is a research target. It is not a promise of profitability.

## 4. Features (42, all point-in-time)

Computed by `src/features.py` with pandas/NumPy only. No third-party indicator library is used.
Indicators use trailing windows (`rolling(..., min_periods=window)`) or causal recursions
(`ewm(adjust=False)`). Warm-up rows (up to 252 days) are dropped, never imputed. Every feature is
scale-invariant, because back-adjusted price *levels* would encode future dividends and splits.
The full dictionary is in `results/metrics/feature_dictionary.csv`.

| Group | Features | Definition (C = adjusted close) |
|---|---|---|
| Returns (14) | `ret_1`, `log_ret_1` | `C_t/C_{t−1} − 1`, `ln(C_t/C_{t−1})` |
| | `ret_lag_{1,2,3,5}` | `ret_1` lagged k days |
| | `cum_ret_{5,10,20,60}` | `C_t/C_{t−w} − 1` (momentum) |
| | `overnight_ret`, `intraday_ret` | `O_t/C_{t−1} − 1`, `C_t/O_t − 1` |
| | `hl_range`, `close_location` | `(H−L)/C`, `(C−L)/(H−L)` |
| Technical (15) | `close_sma_{5,10,20,50}`, `close_ema_{5,10,20,50}` | `C/MA − 1` |
| | `sma_10_50_ratio` | `SMA10/SMA50 − 1` (continuous MA-crossover state) |
| | `rsi_14` | Wilder RSI(14)/100 |
| | `macd_norm`, `macd_signal_norm`, `macd_hist_norm` | MACD(12,26,9) components / C |
| | `bb_pctb`, `bb_width` | Bollinger(20, 2σ, population σ) %B and width |
| Volatility (6) | `atr_norm` | Wilder ATR(14)/C |
| | `vol_{5,20,60}`, `vol_ratio_5_60` | rolling σ of log returns, short/long ratio |
| | `ret_z` | `ret_1 / vol_20` (volatility-normalised return) |
| Volume (3) | `rel_volume_{5,20}`, `volume_z_20` | `V/SMA_w(V)`, 20-day z-score of `ln(1+V)` |
| Regime (4) | `vol_regime` | percentile of `vol_20` within the trailing 252 days |
| | `trend_regime`, `trend_up` | `C/SMA200 − 1`, `1[SMA50 > SMA200]` |
| | `drawdown_252` | `C / max_{252}(C) − 1` |

Formulas: `EMA_t = α·x_t + (1−α)·EMA_{t−1}` with `α = 2/(span+1)`; Wilder smoothing uses `α = 1/n`.
`RSI = 100 − 100/(1 + avgGain/avgLoss)`, `TR_t = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|)`.

## 5. Splits and leakage controls

| Split | Dates (decision days) | Used for |
|---|---|---|
| Train | first valid day (2011) – 2019-12-31 | fitting; hyperparameter CV (tabular models); LSTM early stopping and architecture choice on its final 20% (purged) |
| Validation | 2020-01-01 – 2021-12-31 | ensemble members, decision thresholds, permutation importance; *no model is fitted or early-stopped on it* |
| Test | 2022-01-01 – 2025-12-31 | final evaluation only |

Leakage controls:

1. **Chronological splits only.** Observations are never shuffled across time.
2. **Purging.** A row is removed from its split if its `label_end_date` falls after the split's end date. For example,
   the label of 2019-12-31 needs the open of 2020-01-03, so that row is not used for training.
3. **Purged CV.** Hyperparameters are tuned with `TimeSeriesSplit(n_splits=4, gap=k)` inside the training period only.
4. **Train-only scaling.** Logistic Regression scales inside a `Pipeline`, so the scaler is refitted on each CV
   training fold. The LSTM scaler is fitted on rows up to the last training decision day.
5. **The test set is touched once**, after every choice (hyperparameters, architecture, ensemble members,
   thresholds) is frozen. Test-period threshold curves are reported as *post-hoc* sensitivity only.
6. **Point-in-time features**, verified by truncation-invariance tests (`tests/test_no_leakage.py`).
7. The next day's realised return is never an input. `ret_1` is today's return, which is known at the close of *t*.

## 6. Models

| Model | Implementation | Search space (train-period purged CV, log loss) |
|---|---|---|
| Logistic Regression | `Pipeline(StandardScaler, LogisticRegression(lbfgs, multinomial))` | C ∈ {0.001, 0.01, 0.1, 1}, class_weight ∈ {None, balanced} |
| Random Forest | `RandomForestClassifier(300 trees, max_features=sqrt)` | max_depth ∈ {3, 6, 10}, min_samples_leaf ∈ {20, 80}, class_weight |
| XGBoost | `XGBClassifier(multi:softprob, hist, subsample 0.8, colsample 0.8, min_child_weight 5)` | max_depth ∈ {2, 3, 5}, learning_rate ∈ {0.03, 0.1}, n_estimators ∈ {100, 300}, class_weight (via sample weights) |
| LSTM | PyTorch: LSTM(1 layer) → dropout 0.2 → linear; Adam (lr 1e-3, wd 1e-4), batch 64, cross-entropy (class-weighted when `balanced`); early stopping (patience 8, max 60 epochs, best weights restored) on the **unweighted** log loss of a purged *early-stopping slice* = the last 20% of the training period | hidden ∈ {16, 32} × window ∈ {10, 20} × class_weight ∈ {None, balanced}, chosen by log loss on the same early-stopping slice (walk-forward folds tune class_weight only, hidden 32 and window 20 fixed) |
| Ensemble | equal-weight average of class probabilities | member subset (≥ 2 models) with the lowest validation log loss |
| Majority class | training-set class frequencies | — |
| Naive persistence | predicts tomorrow's class = today's close-to-close class | — |

*Why log loss for tuning:* the trading rule uses probabilities, so a proper scoring rule is a better selection
criterion than accuracy. With a dominant Up class, accuracy can be maximised by predicting Up every day.

*Class imbalance:* class frequencies are reported per split. Balanced class weighting is a tuned
hyperparameter rather than a fixed choice. SMOTE is deliberately not used: synthetic interpolation between
time-ordered samples would create observations that never existed.

*LSTM windows:* for decision day *t* the input is the scaled feature matrix rows *t−L+1 … t*, and the target is
`y_t`. This gives the same labels and the same test dates as the tabular models, so all models are compared on
identical days. Only the order of training mini-batches is shuffled. Each window is self-contained and all training
windows precede the early-stopping slice.

*Why the LSTM early-stops inside the training period:* if it early-stopped on the validation period, its
validation predictions would be optimistically biased. Ensemble selection and threshold selection would then
favour it unfairly over the tabular models, whose validation predictions are genuinely out-of-sample. The slice is
separated from the fitting rows by the label horizon (purged), and the scaler is fitted on training rows only.

## 7. Signals

Given class probabilities `p_t = (P(Down), P(Neutral), P(Up))`:

* **Confidence rule (primary):** long if `P(Up) ≥ τ`, otherwise flat (Hold). In `long_short` mode the strategy is
  also short if `P(Down) ≥ τ` and `P(Down) > P(Up)`.
* **Argmax rule (always active):** long if the predicted class is Up (short if Down in long/short mode).
* τ is chosen from {0.30, 0.325, …, 0.65} by maximising the **validation-period** Sharpe ratio, subject to at
  least 5% of validation days invested. Ties go to the smallest τ. If no τ qualifies, the argmax rule is used.
* **Volatility targeting (robustness):** position × `min(1, 15% / (vol_20 · √252))`. `vol_20` is known at *t*.

Baselines use the same timing:

* **Buy & Hold:** long from the first test decision.
* **MA crossover:** long when `SMA10_t > SMA50_t` (closes ≤ *t*), else flat.
* **Naive persistence:** long if `C_t > C_{t−1}`.

## 8. Backtester

Inputs are the position `p_t` and the return `R_t`, both indexed by decision day. With `c = (fee + slippage)/10⁴`
(default 10 + 5 bps):

```
cost_t = c · |p_t − p_{t−1}| · E_{t−1}          (p_{−1} = 0: every strategy starts in cash)
E_t    = (E_{t−1} − cost_t) · (1 + p_t · R_t)
r_t    = E_t / E_{t−1} − 1                       (net daily return)
```

Costs are charged once per unit of turnover (one-way). A round trip (0 → 1 → 0) therefore costs 2c (30 bps by
default), and a reversal from −1 to +1 also costs 2c. A strategy that never trades pays nothing. NaN positions
become flat (0), so a missing prediction can never create a long position. Positions are not force-liquidated at
the end. Initial capital is 100,000.

## 9. Metrics

**Classification** (test period, argmax predictions): accuracy, balanced accuracy, precision/recall/F1 with
**macro** averaging (each class weighted equally) and weighted averaging, per-class scores, confusion matrix,
multiclass log loss, multiclass Brier score `mean_i Σ_k (p_ik − 1[y_i = k])²`, and ECE of P(Up).

**Trading** (from the backtested net returns `r_t`, 252 days per year, risk-free rate 0 by default):

| Metric | Definition |
|---|---|
| Cumulative return | `E_T / E_0 − 1` |
| Annualised return | `(E_T/E_0)^{252/n} − 1` |
| Annualised volatility | `std(r_t) · √252` |
| Sharpe ratio | `mean(r_t − r_f) / std(r_t − r_f) · √252` (NaN if σ = 0) |
| Sortino ratio | `mean(r_t − r_f) / sqrt(mean(min(r_t − r_f, 0)²)) · √252` |
| Maximum drawdown | `min_t (E_t / max_{s≤t} E_s − 1)`, with E_0 included in the running max |
| Calmar | annualised return / abs(max drawdown) |
| Win rate | share of invested days with a positive net return |
| Trade win rate | share of *completed* round trips (entry → exit, including both costs) with a positive return; a position still open at the end is excluded |
| Exposure | share of days with a non-zero position |
| Turnover | `Σ |Δp_t|` (total) and `Σ |Δp_t| / n · 252` (annual) |
| Number of trades | days on which the position changed |
| Costs paid | `Σ cost_t` in currency |

## 10. Robustness and statistics

* **Assets:** SPY, AAPL, MSFT.
* **Walk-forward:** for each test year *Y* = 2016 … 2025, train on years ≤ *Y*−2 (tabular CV and the LSTM's
  purged early-stopping slice both lie inside this period), validate on *Y*−1 (thresholds, ensemble members),
  and test on *Y*. Each fold re-tunes with reduced grids. Out-of-sample years are concatenated into one
  continuous backtest.
* **Regimes:** high vs low volatility (vol_20 above the *training* median), up vs down trend (C > SMA200),
  and calendar years. A regime or yearly Sharpe ratio is reported only when the strategy was invested on at
  least 20 days of that slice. Otherwise it is N/A, because a Sharpe ratio over a handful of days is only a sign indicator.
* **Transaction costs:** 0–40 bps one-way per unit turnover, with positions held fixed.
* **Threshold sensitivity:** test Sharpe ratio over the τ grid (post-hoc, not used for selection).
* **Execution and target sensitivity:** market-on-close execution and the binary target.
* **Feature ablation:** XGBoost on nested feature groups and on the top-15 features by training-set mutual information.
* **Uncertainty:** circular block bootstrap (block 20 days, 2,000 resamples) gives 95% percentile intervals for
  each Sharpe ratio, for the paired difference Sharpe(strategy) − Sharpe(buy & hold), and for the paired accuracy
  difference against the majority class. Blocks preserve short-range dependence. The intervals are approximate,
  and no formal multiple-testing correction is applied.
