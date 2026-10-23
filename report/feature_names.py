"""Readable names for the model features (used in the report text and Fig. 8 instead of code identifiers)."""

from __future__ import annotations

import re

_FIXED = {
    "ret_1": "1-day return",
    "log_ret_1": "1-day log return",
    "overnight_ret": "overnight return",
    "intraday_ret": "intraday return",
    "hl_range": "high–low range",
    "close_location": "close location in daily range",
    "sma_10_50_ratio": "SMA10/SMA50 ratio",
    "rsi_14": "RSI(14)",
    "macd_norm": "MACD line (normalized)",
    "macd_signal_norm": "MACD signal line (normalized)",
    "macd_hist_norm": "MACD histogram (normalized)",
    "bb_pctb": "Bollinger %B",
    "bb_width": "Bollinger band width",
    "atr_norm": "ATR(14)/close",
    "ret_z": "volatility-normalized return",
    "volume_z_20": "20-day volume z-score",
    "vol_regime": "volatility percentile (252 days)",
    "trend_regime": "close/SMA200",
    "trend_up": "SMA50 > SMA200",
    "drawdown_252": "drawdown from 252-day high",
}
_PATTERNS = [
    (r"ret_lag_(\d+)", "return lagged {} days"),
    (r"cum_ret_(\d+)", "{}-day momentum"),
    (r"close_sma_(\d+)", "close/SMA{}"),
    (r"close_ema_(\d+)", "close/EMA{}"),
    (r"vol_ratio_(\d+)_(\d+)", "volatility ratio {}/{} days"),
    (r"vol_(\d+)", "{}-day volatility"),
    (r"rel_volume_(\d+)", "relative volume ({} days)"),
]


def readable(feature: str) -> str:
    if feature in _FIXED:
        return _FIXED[feature]
    for pat, fmt in _PATTERNS:
        m = re.fullmatch(pat, feature)
        if m:
            return fmt.format(*m.groups())
    return feature.replace("_", " ")
