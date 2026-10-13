# Data

## Source

All market data comes from **Yahoo Finance**, downloaded with the open-source
[`yfinance`](https://github.com/ranaroussi/yfinance) package. No paid API, key or
terminal is required.

| Setting | Value (see `config.yaml`) |
|---|---|
| Instruments | `SPY` (S&P 500 ETF), `AAPL`, `MSFT` |
| Frequency | daily OHLCV |
| Period | 2010-01-01 to 2025-12-31 (inclusive) |
| Fields | Open, High, Low, Close, Adj Close, Volume |

## Layout

```
data/
└── raw/            # cached downloads: <TICKER>_<start>_<end>.csv  (git-ignored)
    └── synthetic/  # only created by --synthetic development runs (git-ignored)
```

The raw CSVs are **not committed**: they are re-created on the first run, and
Yahoo's terms restrict redistribution. Each run records the row count, date range
and SHA-256 checksum of every input file in
`results/metrics/data_manifest.json`, so you can confirm you are using identical
inputs. (Yahoo occasionally revises adjusted prices; a different checksum means
the downloaded data changed.)

```bash
python -m scripts.download_data            # download / reuse the cache
python -m scripts.download_data --refresh  # force a fresh download
```

## Processing (in `src/data_loader.py`)

1. Sort by date, drop duplicate dates (keep the last record).
2. Drop rows with missing or non-positive prices. Nothing is imputed, and every
   removal is counted in the manifest's `validation` report.
3. Widen High/Low where they are inconsistent with Open/Close. This corrects a
   data error and is also counted.
4. Adjust for splits and dividends: `factor = Adj Close / Close` is applied to
   Open, High and Low, and the adjusted close is used as Close.

## Synthetic data

`--synthetic` (or `data.allow_synthetic_fallback: true` when a download fails and
no cache exists) generates a deterministic regime-switching random walk. It exists
only for offline development and unit tests. Outputs go to `results_synthetic*/`,
every report and figure is labelled **SYNTHETIC**, and it is never used for the
reported results.
