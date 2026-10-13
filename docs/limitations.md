# Limitations

A historical backtest is a controlled experiment on one realised path of prices. The points below limit how far
the results in `results/` can be generalised. None of the results is investment advice.

## Markets and data

* **Non-stationarity.** Relationships between indicators and next-day returns change over time (regimes,
  market structure, monetary policy). A model trained on 2011–2019 is tested on 2022–2025, after the 2020–2021
  COVID shock and the 2022 rate-hiking cycle. The walk-forward evaluation, with annual retraining, partly
  addresses this, but it cannot remove it.
* **Weak signal at a daily horizon.** Daily returns of liquid US large caps have near-zero autocorrelation.
  A classifier that barely beats the majority class is the *expected* outcome, not a defect of the
  implementation. Small accuracy differences (a few tenths of a percentage point) are within sampling noise
  (see the bootstrap intervals).
* **Limited sample.** About 2,200 training days, 500 validation days and 1,000 test days per instrument. Daily
  observations are autocorrelated in volatility, so the effective sample size is smaller still. Sharpe ratios
  measured over four years have wide confidence intervals (often ±0.8 or more).
* **Small, hindsight-selected universe (survivorship bias).** SPY, AAPL and MSFT are among today's most
  successful instruments. Choosing them in 2026 builds in knowledge that they survived and grew. Results on
  delisted or weaker stocks could differ substantially.
* **Data source.** Yahoo Finance data is free but unaudited. Adjusted prices are *back-adjusted* and can be
  revised retroactively, so the input checksums are recorded. Only scale-invariant features are used, which
  prevents future corporate-action information from leaking through price levels. Volume figures are
  consolidated and split-adjusted by Yahoo.
* **No fundamental, macro, order-book or news data.** Only OHLCV-derived information is used.

## Modelling

* **Overfitting risk.** Each model searches a small grid, and thresholds are tuned on a 2-year validation period.
  With many models × thresholds × assets, some strategy will look good by chance. The study therefore reports
  every strategy (not only the best), all variants, and bootstrap intervals. No multiple-testing correction
  (e.g. a deflated Sharpe ratio) is applied, so any single "winner" should be treated with suspicion.
* **Validation-period dependence.** Thresholds are selected for validation Sharpe. If the validation period is a
  strong bull market (2020–2021), the best rule on validation is often "always invested", and the ML strategy
  collapses to buy-and-hold. This is a legitimate outcome of the protocol, and it shows that confidence
  thresholds do not transfer reliably across regimes.
* **Label definition.** The ±0.1% neutral band, the 1-day horizon and the next-open target are modelling
  choices. Other definitions (see the binary-target and market-on-close variants) give different numbers.
* **LSTM capacity vs data.** A small single-layer LSTM is used deliberately, because larger networks overfit a
  few thousand daily samples. The LSTM's results are specific to this architecture, tuning budget and seed.
  CPU training is deterministic for a fixed seed, but results may differ slightly across library versions.
* **Numerical reproducibility.** Re-running on the same machine and library versions gives bit-identical results
  (fixed seeds, deterministic CPU training; verified for XGBoost across separate processes). Another machine,
  thread count or library version can change probabilities in late decimal places. Because the strategies are
  driven by thresholds on nearly flat probabilities, such tiny changes can flip a validation-selected threshold
  and visibly change a strategy's test numbers (example in `docs/model_comparison.md`, section 6).
* **Feature importance is not causality.** Permutation importance and mutual information describe what the
  fitted models rely on in this sample, not economic mechanisms.
* **Calibration is evaluated, not corrected.** Reliability diagrams and ECE are reported. No calibration
  model is fitted.

## Backtesting assumptions

* **Transaction costs** are a flat 10 bps fee + 5 bps slippage per unit of turnover. This is illustrative: actual
  costs depend on the broker, order size, spread and venue. Results are shown for 0–40 bps.
* **No market impact, no partial fills, no borrow costs or short-sale constraints** (the default is long-only),
  no taxes, and no cash interest (the risk-free rate defaults to 0).
* **Execution at the open.** Fills at the official open price assume that an opening-auction order fills in full
  at that price. Gaps and auction imbalances can make real fills worse.
* **Dividends.** They are reinvested implicitly through adjusted prices, consistently for all strategies.
* **Single-asset strategies.** There is no portfolio construction across assets, and no leverage beyond 1×.

## Backtest vs live trading

Live trading adds latency, data errors in real time, outages, behavioural constraints and capacity limits. A
strategy that is profitable in a backtest can fail live, and a backtest that already fails to beat buy-and-hold
after costs gives no reason to expect live outperformance.

## Optional RAG / news-sentiment extension (not implemented)

The synopsis lists a Retrieval-Augmented Generation sentiment layer as a stretch objective. It was **not
implemented**, because a correct version needs a time-stamped historical news archive covering 2010–2025. That
archive is not freely available with reliable publication timestamps, and fabricated or look-ahead-contaminated
sentiment would invalidate the comparison. A leakage-safe design for future work:

1. Collect articles with source and publication timestamp (e.g. GDELT, SEC EDGAR 8-K filings, licensed archives).
2. Index them by ticker and date in a vector store. Retrieve only documents published **before** the decision
   time (close of day *t*, exchange time zone).
3. Score retrieved text with a financial sentiment model (e.g. FinBERT) and aggregate it into daily features
   (mean sentiment, article count, sentiment dispersion).
4. Lag the features by one bar if timestamps are uncertain, and add them to the technical feature set.
5. Compare technical-only vs technical + sentiment models with the identical chronological protocol, splits,
   thresholds and backtester.
