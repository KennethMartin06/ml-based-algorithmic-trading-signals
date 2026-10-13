"""Point-in-time technical feature engineering (pandas/NumPy only).

Every feature at row *t* is a function of OHLCV rows ``<= t`` only. All indicators use trailing
windows (``rolling``) or causal recursions (``ewm(adjust=False)``), so truncating the input series at
any date leaves earlier feature values unchanged; ``tests/test_no_leakage.py`` verifies this
property. Indicator warm-up periods produce NaN (``min_periods`` = window) and those rows are dropped
later instead of being imputed.

All model features are scale-invariant (returns, ratios, normalised oscillators). Raw price
levels are deliberately excluded: Yahoo's adjusted prices are back-adjusted with future dividends
and splits, so a level-based feature would carry information about later corporate actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# primitive indicators (exposed for unit tests and notebooks)
# ---------------------------------------------------------------------------------------------


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over the trailing ``window`` observations (inclusive of t)."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, alpha = 2/(span+1), recursive (no future data)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing = EMA with alpha = 1/window (used by RSI and ATR)."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_smooth(gain, window)
    avg_loss = wilder_smooth(loss, window)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)            # no losses in window -> RSI = 100
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # flat series -> neutral 50
    return out.where(avg_gain.notna() & avg_loss.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (EMA_fast - EMA_slow), signal line (EMA of MACD) and histogram."""
    line = close.ewm(span=fast, adjust=False, min_periods=slow).mean() - close.ewm(
        span=slow, adjust=False, min_periods=slow
    ).mean()
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands with population standard deviation (ddof=0, the conventional definition)."""
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = mid + num_std * sd, mid - num_std * sd
    width = upper - lower
    pctb = (close - lower) / width.replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pctb": pctb, "bb_width": width / mid})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # first row has no previous close
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    return wilder_smooth(true_range(high, low, close), window)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / sd.replace(0.0, np.nan)


def drawdown_from_peak(close: pd.Series, window: int) -> pd.Series:
    """Close relative to its trailing ``window``-day maximum (<= 0)."""
    return close / close.rolling(window, min_periods=window).max() - 1.0


# ---------------------------------------------------------------------------------------------
# feature matrix
# ---------------------------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, tuple[str, str]] = {}
FEATURE_GROUPS: dict[str, list[str]] = {}


def _register(group: str, name: str, description: str, rationale: str) -> None:
    FEATURE_DESCRIPTIONS[name] = (description, rationale)
    FEATURE_GROUPS.setdefault(group, [])
    if name not in FEATURE_GROUPS[group]:
        FEATURE_GROUPS[group].append(name)


def compute_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Build the model feature matrix from adjusted OHLCV (columns open/high/low/close/volume).

    Returns a DataFrame indexed like ``df``. Warm-up rows contain NaN.
    """
    cfg = cfg or {}
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"].astype(float)
    F: dict[str, pd.Series] = {}

    # --- A. price and return features -------------------------------------------------------
    ret = c.pct_change()
    log_ret = np.log(c / c.shift(1))
    F["ret_1"] = ret
    _register("returns", "ret_1", "Close-to-close return of day t", "Most recent price change (short-term reversal/momentum)")
    F["log_ret_1"] = log_ret
    _register("returns", "log_ret_1", "Log return ln(C_t/C_{t-1})", "Additive, more symmetric version of the daily return")
    for k in cfg.get("return_lags", [1, 2, 3, 5]):
        F[f"ret_lag_{k}"] = ret.shift(k)
        _register("returns", f"ret_lag_{k}", f"Return of day t-{k}", "Lagged returns let models detect short-horizon autocorrelation")
    for w in cfg.get("cumulative_windows", [5, 10, 20, 60]):
        F[f"cum_ret_{w}"] = c / c.shift(w) - 1.0
        _register("returns", f"cum_ret_{w}", f"Cumulative return over the last {w} days", "Price momentum / rate of change")
    F["overnight_ret"] = o / c.shift(1) - 1.0
    _register("returns", "overnight_ret", "Open_t / Close_{t-1} - 1", "Overnight gap: news arriving outside trading hours")
    F["intraday_ret"] = c / o - 1.0
    _register("returns", "intraday_ret", "Close_t / Open_t - 1 (signed open-close range)", "Intraday buying/selling pressure")
    F["hl_range"] = (h - l) / c
    _register("returns", "hl_range", "(High_t - Low_t) / Close_t", "Intraday range, a high-frequency volatility proxy")
    rng = (h - l).replace(0.0, np.nan)
    F["close_location"] = ((c - l) / rng).fillna(0.5)
    _register("returns", "close_location", "(Close-Low)/(High-Low), 0.5 when High=Low", "Where the day closed inside its range (stochastic-oscillator idea)")

    # --- B. technical indicators -----------------------------------------------------------
    for w in cfg.get("sma_windows", [5, 10, 20, 50]):
        F[f"close_sma_{w}"] = c / sma(c, w) - 1.0
        _register("technical", f"close_sma_{w}", f"Close / SMA_{w} - 1", "Distance from the trend (mean-reversion or trend-following cue)")
    for w in cfg.get("ema_windows", [5, 10, 20, 50]):
        F[f"close_ema_{w}"] = c / ema(c, w) - 1.0
        _register("technical", f"close_ema_{w}", f"Close / EMA_{w} - 1", "Like the SMA ratio but weights recent prices more")
    F["sma_10_50_ratio"] = sma(c, 10) / sma(c, 50) - 1.0
    _register("technical", "sma_10_50_ratio", "SMA_10 / SMA_50 - 1", "Continuous version of the MA-crossover baseline signal")
    F["rsi_14"] = rsi(c, cfg.get("rsi_window", 14)) / 100.0
    _register("technical", "rsi_14", "RSI(14) / 100 (Wilder smoothing)", "Overbought / oversold oscillator")
    mcfg = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    m = macd(c, mcfg["fast"], mcfg["slow"], mcfg["signal"])
    F["macd_norm"] = m["macd"] / c
    F["macd_signal_norm"] = m["macd_signal"] / c
    F["macd_hist_norm"] = m["macd_hist"] / c
    _register("technical", "macd_norm", "MACD(12,26) / Close", "Trend strength, price-normalised to be comparable over time")
    _register("technical", "macd_signal_norm", "MACD signal line (EMA9 of MACD) / Close", "Smoothed trend strength")
    _register("technical", "macd_hist_norm", "(MACD - signal) / Close", "Trend acceleration; sign change = classic MACD crossover")
    bcfg = cfg.get("bollinger", {"window": 20, "num_std": 2.0})
    b = bollinger(c, bcfg["window"], bcfg["num_std"])
    F["bb_pctb"] = b["bb_pctb"]
    F["bb_width"] = b["bb_width"]
    _register("technical", "bb_pctb", "%B = (Close - lower band) / (upper - lower)", "Position inside the Bollinger envelope (z-score-like)")
    _register("technical", "bb_width", "(upper - lower) / middle band", "Volatility squeeze / expansion")

    # --- C. volatility -----------------------------------------------------------------------
    F["atr_norm"] = atr(h, l, c, cfg.get("atr_window", 14)) / c
    _register("volatility", "atr_norm", "ATR(14) / Close", "Average true range including gaps, as % of price")
    vols = {}
    for w in cfg.get("volatility_windows", [5, 20, 60]):
        vols[w] = log_ret.rolling(w, min_periods=w).std()
        F[f"vol_{w}"] = vols[w]
        _register("volatility", f"vol_{w}", f"Rolling std of daily log returns over {w} days", "Realised volatility at several horizons")
    vw = sorted(vols)
    if len(vw) >= 2:
        F[f"vol_ratio_{vw[0]}_{vw[-1]}"] = vols[vw[0]] / vols[vw[-1]]
        _register("volatility", f"vol_ratio_{vw[0]}_{vw[-1]}", "Short / long realised volatility", "Volatility expanding (>1) or contracting (<1)")
    mid_w = 20 if 20 in vols else vw[0]
    F["ret_z"] = ret / vols[mid_w]
    _register("volatility", "ret_z", f"ret_1 / vol_{mid_w}", "Volatility-normalised return: how unusual today's move was")

    # --- D. volume ---------------------------------------------------------------------------
    for w in cfg.get("volume_windows", [5, 20]):
        F[f"rel_volume_{w}"] = v / sma(v, w).replace(0.0, np.nan)
        _register("volume", f"rel_volume_{w}", f"Volume / SMA_{w}(Volume)", "Relative volume: participation behind the move")
    F["volume_z_20"] = rolling_zscore(np.log1p(v), 20)
    _register("volume", "volume_z_20", "20-day z-score of log(1+Volume)", "Abnormal trading activity")

    # --- E. market regime --------------------------------------------------------------------
    rw = cfg.get("regime_window", 252)
    tw = cfg.get("trend_window", 200)
    F["vol_regime"] = vols[mid_w].rolling(rw, min_periods=rw).rank(pct=True)
    _register("regime", "vol_regime", f"Percentile of vol_{mid_w} within the trailing {rw} days", "Low/high volatility regime")
    F["trend_regime"] = c / sma(c, tw) - 1.0
    _register("regime", "trend_regime", f"Close / SMA_{tw} - 1", "Long-term trend regime (bull vs bear)")
    F["trend_up"] = (sma(c, 50) > sma(c, tw)).astype(float).where(sma(c, tw).notna())
    _register("regime", "trend_up", f"1 if SMA_50 > SMA_{tw} else 0", "Binary 'golden cross' trend state")
    F["drawdown_252"] = drawdown_from_peak(c, rw)
    _register("regime", "drawdown_252", f"Close / trailing {rw}-day max - 1", "Depth of the current drawdown (stress regime)")

    out = pd.DataFrame(F, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_groups(columns: list[str] | None = None) -> dict[str, list[str]]:
    """Feature names per group (computes a dummy frame once to populate the registry)."""
    if not FEATURE_GROUPS:
        idx = pd.bdate_range("2000-01-03", periods=300)
        dummy = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(1, 2, 300), "volume": 1.0}, index=idx
        )
        compute_features(dummy)
    if columns is None:
        return {g: list(v) for g, v in FEATURE_GROUPS.items()}
    return {g: [c for c in v if c in columns] for g, v in FEATURE_GROUPS.items()}


def feature_table() -> pd.DataFrame:
    """Feature dictionary (name, group, definition, rationale) used in the documentation."""
    groups = feature_groups()
    rows = []
    for g, names in groups.items():
        for n in names:
            d, r = FEATURE_DESCRIPTIONS[n]
            rows.append({"feature": n, "group": g, "definition": d, "rationale": r})
    return pd.DataFrame(rows)
