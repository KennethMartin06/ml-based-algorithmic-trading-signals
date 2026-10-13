"""Unit tests for the technical indicators and the feature matrix (src/features.py).

The primitive indicators are checked against values worked out by hand on tiny series (or against
an explicit, loop-based re-implementation of the recursion), so a regression in a window length,
smoothing constant or ddof convention is caught. The feature-matrix tests then check the shape,
warm-up and bookkeeping properties the rest of the pipeline relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    atr,
    bollinger,
    compute_features,
    ema,
    feature_groups,
    feature_table,
    macd,
    rsi,
    sma,
    true_range,
    wilder_smooth,
)

NAN = np.nan


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Guard: feature tests must never download market data."""

    def _blocked(*args, **kwargs):  # pragma: no cover - only hit if a test misbehaves
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr("src.data_loader.download_ticker", _blocked)


def _s(values) -> pd.Series:
    return pd.Series(values, dtype=float)


def _ewm_reference(x: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """Explicit recursive EMA: e_0 = x_0 (first non-NaN value), e_t = (1-alpha) e_{t-1} + alpha x_t."""
    out = np.full(len(x), np.nan)
    prev, seen = None, 0
    for i, xi in enumerate(x):
        if np.isnan(xi):
            continue
        prev = xi if prev is None else (1.0 - alpha) * prev + alpha * xi
        seen += 1
        if seen >= min_periods:
            out[i] = prev
    return out


# ---------------------------------------------------------------------------------------------
# primitive indicators on hand-checkable series
# ---------------------------------------------------------------------------------------------


def test_sma_matches_hand_values():
    """SMA(3) of 1..6 is the mean of each trailing triple; the first window-1 rows are NaN."""
    out = sma(_s([1, 2, 3, 4, 5, 6]), 3)
    np.testing.assert_allclose(out.to_numpy(), [NAN, NAN, 2.0, 3.0, 4.0, 5.0], equal_nan=True)


@pytest.mark.parametrize("window", [2, 5, 10, 20])
def test_sma_warmup_is_exactly_window_minus_one_rows(window):
    """With min_periods=window, exactly the first window-1 rows are NaN and none afterwards."""
    out = sma(_s(np.arange(1.0, 41.0)), window)
    assert out.iloc[: window - 1].isna().all()
    assert out.iloc[window - 1 :].notna().all()


def test_ema_matches_hand_recursion():
    """EMA(span=3): alpha = 2/(3+1) = 0.5, seeded at the first value, NaN until span rows exist.

    e = 1, 1.5, 2.25, 3.125, 4.0625 -> the first two are masked by min_periods=3.
    """
    out = ema(_s([1, 2, 3, 4, 5]), 3)
    np.testing.assert_allclose(out.to_numpy(), [NAN, NAN, 2.25, 3.125, 4.0625], equal_nan=True)


@pytest.mark.parametrize("span", [5, 12, 26])
def test_ema_equals_explicit_recursion_on_price_series(prices, span):
    """On a long price series EMA equals the loop recursion with alpha = 2/(span+1)."""
    close = prices["close"].iloc[:200]
    expected = _ewm_reference(close.to_numpy(), 2.0 / (span + 1.0), span)
    np.testing.assert_allclose(ema(close, span).to_numpy(), expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("window", [3, 14])
def test_wilder_smoothing_uses_alpha_one_over_n(prices, window):
    """Wilder smoothing is the recursive EMA with alpha = 1/window (not 2/(window+1))."""
    x = prices["close"].pct_change().abs().iloc[:120]
    expected = _ewm_reference(x.to_numpy(), 1.0 / window, window)
    np.testing.assert_allclose(wilder_smooth(x, window).to_numpy(), expected, rtol=1e-12, equal_nan=True)


def test_rsi_matches_hand_values():
    """RSI(3) on a 6-day series, worked by hand with Wilder averages (alpha = 1/3).

    deltas = [., 1, -0.5, 1, 0.5, -1]
    avg gain = 1, 2/3, 7/9, 37/54, 37/81      avg loss = 0, 1/6, 1/9, 2/27, 31/81
    RS(t=3) = 7, RS(t=4) = 37/4, RS(t=5) = 37/31 ;  RSI = 100 - 100 / (1 + RS)
    Rows 0-2 are warm-up (the first difference is undefined, then 3 observations are needed).
    """
    out = rsi(_s([10, 11, 10.5, 11.5, 12, 11]), window=3)
    expected = [NAN, NAN, NAN, 87.5, 100 - 100 / (1 + 37 / 4), 100 - 100 / (1 + 37 / 31)]
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-12, equal_nan=True)


def test_rsi_is_100_for_strictly_increasing_prices():
    """No losing day in the window -> average loss is 0 -> RSI is defined as 100 after warm-up."""
    out = rsi(_s(np.linspace(100, 150, 60)), window=14)
    assert out.iloc[:14].isna().all()
    np.testing.assert_allclose(out.iloc[14:].to_numpy(), 100.0)


def test_rsi_is_50_for_flat_prices():
    """No gains and no losses -> the 0/0 case is mapped to the neutral value 50."""
    out = rsi(_s(np.full(40, 42.0)), window=14)
    assert out.iloc[:14].isna().all()
    np.testing.assert_allclose(out.iloc[14:].to_numpy(), 50.0)


def test_rsi_bounded_on_real_like_series(prices):
    """RSI stays inside [0, 100] on a realistic random-walk series."""
    out = rsi(prices["close"]).dropna()
    assert len(out) > 0
    assert out.between(0.0, 100.0).all()


def test_macd_matches_hand_values():
    """MACD(fast=2, slow=3, signal=2) on 1..6, worked by hand.

    EMA2 (alpha 2/3) = 1, 5/3, 23/9, 95/27, 365/81, 1337/243
    EMA3 (alpha 1/2) = 1, 3/2, 9/4, 25/8, 65/16, 161/32
    line = EMA2 - EMA3, NaN until `slow` rows exist  -> 11/36, 85/216, 575/1296, 3661/7776
    signal = EMA2 of the line seeded at its first value, NaN until 2 line values exist.
    """
    out = macd(_s([1, 2, 3, 4, 5, 6]), fast=2, slow=3, signal=2)
    line = [NAN, NAN, 11 / 36, 85 / 216, 575 / 1296, 3661 / 7776]
    sig = [NAN, NAN, NAN, 59 / 162, 811 / 1944, 10566 / 23328]
    np.testing.assert_allclose(out["macd"].to_numpy(), line, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(out["macd_signal"].to_numpy(), sig, rtol=1e-12, equal_nan=True)


def test_macd_histogram_is_line_minus_signal(prices):
    """macd_hist == macd - macd_signal on every row (NaN where either input is NaN)."""
    out = macd(prices["close"], 12, 26, 9)
    np.testing.assert_allclose(
        out["macd_hist"].to_numpy(), (out["macd"] - out["macd_signal"]).to_numpy(), rtol=1e-12, equal_nan=True
    )
    assert out["macd"].iloc[:25].isna().all() and out["macd"].iloc[25:].notna().all()
    assert out["macd_signal"].iloc[:33].isna().all() and out["macd_signal"].iloc[33:].notna().all()


def test_macd_line_equals_difference_of_reference_emas(prices):
    """The MACD line is EMA_fast - EMA_slow of the full recursion (fast EMA masked until `slow` rows)."""
    close = prices["close"].iloc[:150]
    x = close.to_numpy()
    fast = _ewm_reference(x, 2 / 13, 1)
    slow = _ewm_reference(x, 2 / 27, 26)
    np.testing.assert_allclose(macd(close)["macd"].to_numpy(), fast - slow, rtol=1e-10, equal_nan=True)


def test_bollinger_uses_population_std():
    """Bollinger(3, 2) on 1..5: every window has mean c-1 and population sd sqrt(2/3) (ddof=0)."""
    out = bollinger(_s([1, 2, 3, 4, 5]), window=3, num_std=2.0)
    sd = np.sqrt(2.0 / 3.0)
    mid = np.array([NAN, NAN, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(out["bb_mid"].to_numpy(), mid, equal_nan=True)
    np.testing.assert_allclose(out["bb_upper"].to_numpy(), mid + 2 * sd, equal_nan=True)
    np.testing.assert_allclose(out["bb_lower"].to_numpy(), mid - 2 * sd, equal_nan=True)
    # close is always one unit above the mid band: %B = (1 + 2 sd) / (4 sd)
    np.testing.assert_allclose(out["bb_pctb"].iloc[2:].to_numpy(), (1 + 2 * sd) / (4 * sd))
    np.testing.assert_allclose(out["bb_width"].to_numpy(), 4 * sd / mid, equal_nan=True)
    # and it is NOT the sample (ddof=1) standard deviation
    sd_sample = np.sqrt(1.0)
    assert not np.isclose(out["bb_upper"].iloc[2], 2.0 + 2 * sd_sample)


def test_bollinger_ddof0_on_price_series(prices):
    """Upper band == SMA + k * np.std(window, ddof=0) on every row of a real-like series."""
    close = prices["close"].iloc[:80]
    out = bollinger(close, window=20, num_std=2.0)
    for t in range(19, len(close)):
        w = close.iloc[t - 19 : t + 1].to_numpy()
        assert out["bb_upper"].iloc[t] == pytest.approx(w.mean() + 2.0 * w.std(ddof=0), rel=1e-10)


def test_true_range_includes_gaps():
    """TR = max(H-L, |H-C_prev|, |L-C_prev|); the overnight gap dominates on gap days.

    row1: gap up   max(5, |110-100|=10, |105-100|=5)  = 10
    row2: gap down max(5, |100-108|=8,  |95-108|=13)  = 13
    row3:          max(3, |100-96|=4,   |97-96|=1)    = 4
    row4:          max(3, |102-99|=3,   |99-99|=0)    = 3
    """
    high, low, close = _s([101, 110, 100, 100, 102]), _s([99, 105, 95, 97, 99]), _s([100, 108, 96, 99, 101])
    np.testing.assert_allclose(true_range(high, low, close).to_numpy(), [NAN, 10, 13, 4, 3], equal_nan=True)


def test_true_range_on_tiny_prices(tiny_prices):
    """Hand values for the 6-day conftest frame (first row has no previous close)."""
    tr = true_range(tiny_prices["high"], tiny_prices["low"], tiny_prices["close"])
    np.testing.assert_allclose(tr.to_numpy(), [NAN, 3.0, 3.0, 3.0, 3.0, 2.0], equal_nan=True)


def test_atr_is_wilder_smoothed_true_range():
    """ATR(2) with alpha = 1/2 seeded at the first TR: 10 -> 11.5 -> 7.75 -> 5.375 (row 1 is warm-up)."""
    high, low, close = _s([101, 110, 100, 100, 102]), _s([99, 105, 95, 97, 99]), _s([100, 108, 96, 99, 101])
    np.testing.assert_allclose(atr(high, low, close, window=2).to_numpy(), [NAN, NAN, 11.5, 7.75, 5.375], equal_nan=True)


# ---------------------------------------------------------------------------------------------
# the feature matrix
# ---------------------------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "ret_1", "log_ret_1", "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
    "cum_ret_5", "cum_ret_10", "cum_ret_20", "cum_ret_60", "overnight_ret", "intraday_ret",
    "hl_range", "close_location",
    "close_sma_5", "close_sma_10", "close_sma_20", "close_sma_50",
    "close_ema_5", "close_ema_10", "close_ema_20", "close_ema_50", "sma_10_50_ratio", "rsi_14",
    "macd_norm", "macd_signal_norm", "macd_hist_norm", "bb_pctb", "bb_width",
    "atr_norm", "vol_5", "vol_20", "vol_60", "vol_ratio_5_60", "ret_z",
    "rel_volume_5", "rel_volume_20", "volume_z_20",
    "vol_regime", "trend_regime", "trend_up", "drawdown_252",
]

# Number of leading NaN rows each feature must have with the default configuration. Each value is
# derived from the indicator definition (e.g. a lag-k return of a return needs 1 + k rows; vol_regime
# ranks vol_20, which itself needs 20 rows, over 252 rows -> 20 + 252 - 1 = 271).
EXPECTED_WARMUP = {
    "ret_1": 1, "log_ret_1": 1, "ret_lag_1": 2, "ret_lag_2": 3, "ret_lag_3": 4, "ret_lag_5": 6,
    "cum_ret_5": 5, "cum_ret_10": 10, "cum_ret_20": 20, "cum_ret_60": 60,
    "overnight_ret": 1, "intraday_ret": 0, "hl_range": 0, "close_location": 0,
    "close_sma_5": 4, "close_sma_10": 9, "close_sma_20": 19, "close_sma_50": 49,
    "close_ema_5": 4, "close_ema_10": 9, "close_ema_20": 19, "close_ema_50": 49,
    "sma_10_50_ratio": 49, "rsi_14": 14, "macd_norm": 25, "macd_signal_norm": 33, "macd_hist_norm": 33,
    "bb_pctb": 19, "bb_width": 19, "atr_norm": 14,
    "vol_5": 5, "vol_20": 20, "vol_60": 60, "vol_ratio_5_60": 60, "ret_z": 20,
    "rel_volume_5": 4, "rel_volume_20": 19, "volume_z_20": 19,
    "vol_regime": 271, "trend_regime": 199, "trend_up": 199, "drawdown_252": 251,
}


@pytest.fixture(scope="module")
def feats(prices, cfg) -> pd.DataFrame:
    return compute_features(prices, cfg["features"])


def test_feature_matrix_has_expected_42_columns(feats, prices):
    """Default config yields exactly the 42 documented features, indexed like the price frame."""
    assert list(feats.columns) == EXPECTED_COLUMNS
    assert feats.shape == (len(prices), 42)
    assert feats.index.equals(prices.index)


def test_default_cfg_equals_no_cfg(feats, prices):
    """compute_features without a config uses the same defaults as config.yaml."""
    pd.testing.assert_frame_equal(compute_features(prices), feats)


def test_feature_matrix_has_no_infinite_values(feats):
    """Division-by-zero cases are converted to NaN; no +/-inf may reach the models."""
    assert not np.isinf(feats.to_numpy()).any()


def test_each_feature_warmup_matches_its_window(feats):
    """Each feature is NaN exactly for its theoretical warm-up and complete afterwards."""
    for col, k in EXPECTED_WARMUP.items():
        s = feats[col]
        assert s.iloc[:k].isna().all(), f"{col}: expected NaN in the first {k} rows"
        assert s.iloc[k:].notna().all(), f"{col}: unexpected NaN after its {k}-row warm-up"


def test_no_nan_after_regime_warmup(feats, cfg):
    """After the 252-day regime window every feature is defined, except vol_regime which ranks the
    20-day volatility and therefore needs 20 + 252 - 1 rows; from then on the row is complete."""
    rw = cfg["features"]["regime_window"]
    assert rw == 252
    assert not feats.drop(columns="vol_regime").iloc[rw:].isna().any().any()
    full_warmup = 20 + rw - 1
    assert not feats.iloc[full_warmup:].isna().any().any()
    assert feats.iloc[full_warmup - 1].isna().any()  # and the warm-up is not longer than necessary


def test_feature_value_ranges(feats):
    """Bounded features stay in their mathematical ranges."""
    f = feats.dropna()
    assert f["rsi_14"].between(0, 1).all()
    assert f["close_location"].between(0, 1).all()
    assert f["vol_regime"].between(0, 1).all()
    assert set(f["trend_up"].unique()) <= {0.0, 1.0}
    assert (f["drawdown_252"] <= 1e-12).all()
    assert (f["hl_range"] >= 0).all() and (f["atr_norm"] > 0).all()


def test_feature_definitions_spot_check(feats, prices):
    """A few features recomputed directly from OHLCV at an arbitrary row."""
    t = 400
    o, h, l, c = (prices[k].to_numpy() for k in ("open", "high", "low", "close"))
    row = feats.iloc[t]
    assert row["ret_1"] == pytest.approx(c[t] / c[t - 1] - 1)
    assert row["log_ret_1"] == pytest.approx(np.log(c[t] / c[t - 1]))
    assert row["ret_lag_2"] == pytest.approx(c[t - 2] / c[t - 3] - 1)
    assert row["cum_ret_20"] == pytest.approx(c[t] / c[t - 20] - 1)
    assert row["overnight_ret"] == pytest.approx(o[t] / c[t - 1] - 1)
    assert row["intraday_ret"] == pytest.approx(c[t] / o[t] - 1)
    assert row["hl_range"] == pytest.approx((h[t] - l[t]) / c[t])
    assert row["close_sma_20"] == pytest.approx(c[t] / c[t - 19 : t + 1].mean() - 1)
    assert row["drawdown_252"] == pytest.approx(c[t] / c[t - 251 : t + 1].max() - 1)
    assert row["macd_hist_norm"] == pytest.approx(row["macd_norm"] - row["macd_signal_norm"])


def test_close_location_is_half_when_high_equals_low(tiny_prices):
    """A zero-range day (High == Low) gets the neutral close location 0.5 instead of NaN/inf."""
    df = tiny_prices.copy()
    df.loc[df.index[2], ["open", "high", "low", "close"]] = 102.0
    out = compute_features(df)
    assert out["close_location"].iloc[2] == 0.5
    assert np.isfinite(out["close_location"]).all()


def test_feature_groups_cover_every_column_exactly_once(feats):
    """Each model feature belongs to exactly one documented group."""
    groups = feature_groups(list(feats.columns))
    flat = [name for names in groups.values() for name in names]
    assert len(flat) == len(set(flat)), "a feature is listed in more than one group"
    assert sorted(flat) == sorted(feats.columns)
    assert set(groups) == {"returns", "technical", "volatility", "volume", "regime"}


def test_feature_table_has_one_row_per_feature(feats):
    """The documentation table lists every feature once, with a group, definition and rationale."""
    table = feature_table()
    assert list(table.columns) == ["feature", "group", "definition", "rationale"]
    assert not table["feature"].duplicated().any()
    assert set(feats.columns) <= set(table["feature"])
    sub = table.set_index("feature").loc[list(feats.columns)]
    assert (sub["definition"].str.len() > 0).all() and (sub["rationale"].str.len() > 0).all()
    groups = feature_groups(list(feats.columns))
    for g, names in groups.items():
        assert (sub.loc[names, "group"] == g).all()
