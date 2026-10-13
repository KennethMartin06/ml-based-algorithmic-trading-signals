# Model Comparison

All numbers below come from the final full run (`python -m scripts.run_experiment`, default `config.yaml`),
saved in `results/metrics/`. Test period: 2022-01-03 → 2025-12-29 (1,001 trading days per ticker). Execution is at
the next open, costs are 15 bps one-way per unit of turnover, and strategies are long-only. The complete tables are
in [`results/reports/experiment_report.md`](../results/reports/experiment_report.md).

## 1. Roles of the models

| Model | Type | What it can learn | Why it is in the study |
|---|---|---|---|
| Buy & Hold | passive benchmark | nothing (always invested) | the bar any active strategy must clear |
| MA crossover 10/50 | rule-based technical | nothing; a fixed rule on two moving averages | the classical technical baseline from the synopsis |
| Naive persistence | rule-based | nothing; "tomorrow = today" | tests whether simple daily momentum exists |
| Majority class | naive classifier | class frequencies only | the floor for classification metrics |
| Logistic Regression | linear probabilistic classifier | additive (linear) effects of the 42 features | interpretable, strongly regularised ML baseline |
| Random Forest | bagged decision trees | non-linear effects and feature interactions | robust non-linear learner with built-in importance |
| XGBoost | gradient-boosted trees | non-linear effects, interactions, sequential error correction | the primary classifier of the synopsis (section 5.5) |
| LSTM | recurrent neural network | temporal patterns across 10–20 day windows of features | the sequence model of the synopsis (section 5.6) |
| Ensemble | soft vote of selected models | averages the probability estimates of its members | tests whether combining models reduces variance |

## 2. What the tuning selected

The tabular models were tuned with purged time-series CV on 2011–2019, and the LSTM on a purged slice at the end of
that period. In every case the search chose the **most regularised / lowest-capacity** option:

| Model | Selected for SPY / AAPL / MSFT |
|---|---|
| Logistic Regression | C = 0.001 (strongest regularisation in the grid), no class weighting, for all three |
| Random Forest | max_depth = 3 for all three; min_samples_leaf 80 / 20 / 80; no class weighting |
| XGBoost | max_depth = 2, learning_rate = 0.03, 100 trees (smallest model in the grid), no class weighting, for all three |
| LSTM | 32 / 16 / 16 hidden units, 10-day windows, no class weighting |
| Ensemble members | LR + LSTM / RF + XGBoost / LR + RF |

Two things follow. First, cross-validation itself finds very little exploitable structure: when more complex models
are allowed, the out-of-fold log loss gets worse. Second, the selected models stay close to the class prior. They
predict Up on 71–100% of test days and **never predict Neutral**. With no class weighting the Neutral class
(4–13% of days, depending on ticker and period) is never the most probable class. Balanced class weighting was available to every model
and was rejected by the log-loss criterion in all 12 model/ticker cases.

## 3. Classification results (test period)

| Ticker | Model | Accuracy | Balanced acc. | Macro-F1 | Log loss |
|---|---|---|---|---|---|
| SPY | Majority class | 0.495 | 0.333 | 0.221 | 0.943 |
| SPY | Logistic Regression | **0.497** | 0.335 | 0.226 | **0.928** |
| SPY | Random Forest | 0.494 | 0.333 | 0.220 | 0.930 |
| SPY | XGBoost | 0.472 | 0.322 | **0.250** | 0.937 |
| SPY | LSTM | 0.493 | 0.334 | 0.238 | 0.932 |
| SPY | Ensemble | 0.493 | 0.332 | 0.221 | 0.929 |
| AAPL | Majority class | **0.495** | 0.333 | 0.221 | 0.889 |
| AAPL | Logistic Regression | 0.473 | 0.324 | 0.276 | 0.889 |
| AAPL | Random Forest | 0.493 | 0.335 | 0.260 | 0.889 |
| AAPL | XGBoost | 0.489 | 0.334 | 0.280 | 0.891 |
| AAPL | LSTM | 0.485 | 0.334 | **0.294** | 0.900 |
| AAPL | Ensemble | 0.494 | 0.336 | 0.269 | 0.889 |
| MSFT | Majority class | 0.502 | 0.333 | 0.223 | 0.879 |
| MSFT | Logistic Regression | **0.510** | **0.351** | **0.326** | 0.876 |
| MSFT | Random Forest | 0.500 | 0.343 | 0.313 | 0.877 |
| MSFT | XGBoost | 0.507 | 0.347 | 0.316 | **0.875** |
| MSFT | LSTM | 0.500 | 0.345 | 0.324 | 0.883 |
| MSFT | Ensemble | 0.505 | 0.347 | 0.320 | 0.876 |

Observations:

* The best accuracy of any ML model is within **one percentage point** of the majority-class baseline for every
  ticker. The paired block-bootstrap 95% intervals for "accuracy minus majority-class accuracy" include zero for
  14 of 15 model/ticker pairs. The exception is SPY XGBoost, which is *significantly worse*
  (−2.3 pp, interval −4.0 to −0.8 pp).
* The ML models do improve **probabilistic** accuracy slightly. Their test log loss is at or below the
  majority-class prior in most cases (e.g. SPY LR 0.928 vs 0.943). They are also well calibrated: the expected
  calibration error of P(Up) is 0.004–0.058.
* Macro-F1 is higher than the baseline's (0.22) only because the models occasionally predict Down. They never
  predict Neutral, so the Neutral class always has F1 = 0.
* No model family dominates. The LSTM had the best macro-F1 on AAPL, Logistic Regression on MSFT and XGBoost on
  SPY, but the differences are small relative to their sampling uncertainty.

## 4. Trading results (test period, net of costs)

Each ML strategy goes long when P(Up) ≥ τ, with τ chosen on the 2020–2021 validation period.

| Ticker | Strategy | τ | Cum. return | Sharpe | Max DD | Days invested | Trades |
|---|---|---|---|---|---|---|---|
| SPY | **Buy & Hold** | – | 51.4% | **0.661** | −26.4% | 100% | 1 |
| SPY | MA crossover | – | 23.3% | 0.519 | **−14.0%** | 67% | 27 |
| SPY | Logistic Regression | 0.425 | 51.4% | 0.661 | −26.4% | 100% | 1 |
| SPY | Random Forest | 0.300 | 51.4% | 0.661 | −26.4% | 100% | 1 |
| SPY | XGBoost | 0.375 | 45.2% | 0.604 | −26.5% | 98% | 35 |
| SPY | LSTM | 0.375 | 48.6% | 0.636 | −26.4% | 100% | 3 |
| SPY | Ensemble | 0.300 | 51.4% | 0.661 | −26.4% | 100% | 1 |
| AAPL | Buy & Hold | – | 52.4% | 0.506 | −33.3% | 100% | 1 |
| AAPL | MA crossover | – | 24.0% | 0.393 | **−25.4%** | 56% | 31 |
| AAPL | Logistic Regression | 0.450 | 43.2% | 0.461 | −35.2% | 93% | 113 |
| AAPL | **Random Forest** | 0.400 | **61.3%** | **0.557** | −34.8% | 99% | 31 |
| AAPL | XGBoost | 0.325 | 54.5% | 0.519 | −33.5% | 99% | 25 |
| AAPL | LSTM | 0.400 | 38.5% | 0.425 | −35.7% | 98% | 37 |
| AAPL | Ensemble | 0.350 | 56.3% | 0.529 | −33.3% | 99% | 21 |
| MSFT | Buy & Hold | – | 50.4% | 0.510 | −34.7% | 100% | 1 |
| MSFT | MA crossover | – | 11.6% | 0.246 | **−27.9%** | 55% | 28 |
| MSFT | Logistic Regression | 0.375 | 46.6% | 0.488 | −34.6% | 98% | 23 |
| MSFT | Random Forest | 0.300 | 50.4% | 0.510 | −34.7% | 100% | 1 |
| MSFT | **XGBoost** | 0.400 | **51.8%** | **0.519** | −34.7% | 98% | 29 |
| MSFT | LSTM | 0.300 | 50.4% | 0.510 | −34.7% | 100% | 1 |
| MSFT | Ensemble | 0.425 | 28.4% | 0.367 | −36.1% | 95% | 49 |

* **5 of the 15 ML strategies are identical to buy-and-hold** (LR, RF and the ensemble on SPY; RF and the LSTM on
  MSFT). On the 2020–2021 validation period, a strong bull market apart from the brief COVID crash, the
  Sharpe-maximising threshold was one that never leaves the market.
* Of the remaining 10, **four** had a higher test Sharpe ratio than buy-and-hold: AAPL RF (+0.051), AAPL ensemble
  (+0.023), AAPL XGBoost (+0.013) and MSFT XGBoost (+0.010). The paired block-bootstrap 95% intervals for these
  differences all include zero. SPY XGBoost is the only ML strategy whose interval lies entirely below zero
  (−0.057, interval −0.128 to −0.005).
* The MA crossover had the **smallest drawdowns** on every ticker, because it sits out sustained declines. It also
  had the lowest Sharpe ratio of the non-trivial strategies on MSFT and a lower cumulative return than buy-and-hold
  on every ticker: the 2022–2025
  recoveries were sharp, and it re-entered late.
* Naive persistence lost 43–48% on every ticker. Most of that is transaction costs from ~490 trades: its Sharpe
  ratio is about +0.25 to +0.31 before costs (see the cost-sensitivity table), so daily momentum is not
  exploitable after realistic costs.

## 5. Walk-forward evaluation (2016–2025, annual retraining)

| Ticker | Buy & Hold Sharpe | Best ML strategy (Sharpe) | Worst ML strategy (Sharpe) | MA crossover Sharpe |
|---|---|---|---|---|
| SPY | **0.896** | Random Forest (0.702) | Logistic Regression (0.339) | 0.744 |
| AAPL | **0.981** | Random Forest (0.895) | XGBoost (0.415) | 0.875 |
| MSFT | **1.032** | Logistic Regression (0.759) | XGBoost (0.418) | 0.705 |

Over ten consecutive out-of-sample years, **none of the 15 ML strategies matched buy-and-hold** on Sharpe ratio or
on cumulative return. Retraining every year with a fresh validation year makes the confidence threshold swing
between "always invested" (τ = 0.30) and "rarely invested" (τ ≥ 0.55) from one year to the next. That produces
many trades (69–359 over ten years) and misses large parts of the bull market. Out-of-sample accuracy was
again below the majority class for every model (e.g. SPY: 0.464–0.497 vs 0.504). Exception: MSFT Logistic
Regression had a much smaller maximum drawdown (−22.9% vs −36.5%) at a lower Sharpe ratio (0.759 vs 1.032).

## 6. Robustness

* **Feature groups (XGBoost):** the feature groups change test log loss only in the third decimal. On SPY the 14
  return features alone beat the full 42-feature set (0.934 vs 0.937). On AAPL and MSFT the full set is marginally
  better (0.891 vs 0.893, 0.875 vs 0.878). Adding technical indicators, volume or regime features does not
  consistently help. Selecting the top-15 features by training-set mutual information is about as good as using all.
* **Feature importance:** the features the models rely on differ completely between tickers. Top permutation
  importance (validation) was `drawdown_252` and `macd_norm` for SPY, `intraday_ret` and `cum_ret_5` for AAPL, and
  `atr_norm` and `rsi_14` for MSFT. All importances are tiny (≤ 0.0045 increase in log loss). This instability is
  what one expects when there is no stable signal.
* **Execution timing:** under the optimistic market-on-close convention, several strategies change by large
  amounts (e.g. AAPL Random Forest Sharpe 0.584 → 0.011, Logistic Regression −0.013 → 0.393) with no consistent
  direction. Results depend heavily on details of the label definition.
* **Binary target:** removing the neutral band does not change the conclusion. Binary-target ML accuracy is
  0.49–0.55, against a majority-class accuracy of 0.53–0.54.
* **Hyperparameter grid:** with the reduced walk-forward grids, the AAPL Logistic Regression test Sharpe falls
  from 0.461 to −0.013 (see `robustness_variants.csv`). Between two otherwise identical full runs, a fourth-decimal
  change in MSFT XGBoost's CV log loss (0.9172 vs 0.9171) moved its validation threshold from 0.425 to 0.400 and its
  test Sharpe from 0.33 to 0.52. Trading results of weak classifiers are **highly sensitive to small modelling
  details**. Any single number in the tables should be read together with its bootstrap interval.
* **Volatility targeting:** scaling exposure to a 15% volatility target reduces maximum drawdowns by 21–44% for every
  strategy, ML and buy-and-hold alike (SPY −26.4% → −20.8%, AAPL −33.3% → −20.5%, MSFT −34.7% → −19.5% for
  buy-and-hold). Risk management, not prediction,
  drives this improvement.
* **Transaction costs:** buy-and-hold and the always-invested ML strategies are insensitive to costs. Between 0 and
  40 bps, the trading ML strategies lose 0.02–0.40 Sharpe (most: AAPL Logistic Regression, 113 trades) and the MA
  crossover loses 0.16–0.24.

## 7. Conclusions

1. On liquid US large caps at a daily horizon, five supervised learners (linear, two tree ensembles, an LSTM and a
   soft-voting ensemble) **did not predict next-day direction better than always predicting the most common
   class**. The differences are within sampling noise.
2. Because the predictions are close to the prior, the confidence-threshold strategies mostly **track buy-and-hold**.
   They either collapse to it exactly, or deviate in ways that are not statistically distinguishable from it. The
   few positive differences are small, unstable across walk-forward years, and sensitive to tuning details.
3. The **sequence model did not outperform the tabular models**. With honest early stopping inside the training
   period, the LSTM's probabilities converge to the same near-prior predictions.
4. Rule-based **MA crossover reduced drawdowns but not risk-adjusted returns**. Naive daily momentum was destroyed
   by transaction costs.
5. This is a scientifically valid negative result, consistent with weak-form market efficiency at this horizon
   for these instruments. It shows the value of the evaluation framework: under a leakage-free protocol with
   realistic costs, a model that "looks smart" does not produce an edge. Promising directions are longer horizons,
   cross-sectional (relative-value) targets across many stocks, and genuinely new information such as
   point-in-time news sentiment (the synopsis's stretch objective).
