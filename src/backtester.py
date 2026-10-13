"""Vectorised, cost-aware daily backtester.

Timing convention
-----------------
``positions[t]`` is the target exposure decided after the close of day t (using information up to
the close of t). ``period_returns[t]`` is the asset return earned by a position decided at t,
as defined by the execution convention in :mod:`src.targets` (default: Open[t+1] -> Open[t+2]).
The backtester never needs to shift anything itself: both inputs are indexed by decision date.

Accounting for period t (E = equity, p = position, c = (fee_bps + slippage_bps) / 10_000):

    cost_t   = c * |p_t - p_{t-1}| * E_{t-1}           (charged once, when the position changes)
    E_t      = (E_{t-1} - cost_t) * (1 + p_t * R_t)
    r_net_t  = E_t / E_{t-1} - 1

The portfolio starts flat (p_{-1} = 0), so a strategy that is long on its first day pays one entry
cost; a strategy that never trades pays nothing. Positions are not force-liquidated at the end.
Missing (NaN) positions are treated as flat - a missing prediction can never become a long trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    annualization: int = 252
    risk_free_rate: float = 0.0
    missing_position: str = "flat"  # or "hold" (carry the previous position forward)

    @property
    def cost_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "BacktestConfig":
        b = cfg.get("backtest", {})
        params = dict(
            initial_capital=b.get("initial_capital", 100_000.0),
            fee_bps=b.get("fee_bps", 10.0),
            slippage_bps=b.get("slippage_bps", 5.0),
            annualization=b.get("annualization", 252),
            risk_free_rate=b.get("risk_free_rate", 0.0),
            missing_position=b.get("missing_position", "flat"),
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    trades: pd.DataFrame | None = None


# ---------------------------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------------------------


def drawdown_series(equity: pd.Series, initial_capital: float | None = None) -> pd.Series:
    """Drawdown relative to the running peak (includes the initial capital as the first peak)."""
    eq = equity.astype(float)
    peak = eq.cummax()
    if initial_capital is not None:
        peak = np.maximum(peak, initial_capital)
    return eq / peak - 1.0


def max_drawdown(equity: pd.Series, initial_capital: float | None = None) -> float:
    if len(equity) == 0:
        return float("nan")
    return float(drawdown_series(equity, initial_capital).min())


def annualized_return(net_returns: pd.Series, annualization: int = 252) -> float:
    r = net_returns.dropna()
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r.values))
    if growth <= 0:
        return -1.0
    return growth ** (annualization / len(r)) - 1.0


def annualized_volatility(net_returns: pd.Series, annualization: int = 252) -> float:
    r = net_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(annualization))


def _daily_rf(risk_free_rate: float, annualization: int) -> float:
    return (1.0 + risk_free_rate) ** (1.0 / annualization) - 1.0


def sharpe_ratio(net_returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    """Annualised Sharpe ratio of daily excess returns; NaN when volatility is zero/undefined."""
    r = net_returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - _daily_rf(risk_free_rate, annualization)
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(annualization))


def sortino_ratio(net_returns: pd.Series, risk_free_rate: float = 0.0, annualization: int = 252) -> float:
    """Annualised Sortino ratio: mean excess return / downside deviation (target = risk-free)."""
    r = net_returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - _daily_rf(risk_free_rate, annualization)
    downside = np.sqrt(np.mean(np.minimum(excess.values, 0.0) ** 2))
    if downside < 1e-12:
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(annualization))


# ---------------------------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------------------------


def clean_positions(positions: pd.Series, missing: str = "flat") -> pd.Series:
    pos = positions.astype(float)
    if missing == "hold":
        pos = pos.ffill()
    return pos.fillna(0.0)


def run_backtest(positions: pd.Series, period_returns: pd.Series, cfg: BacktestConfig | None = None) -> BacktestResult:
    """Simulate a daily-rebalanced single-asset strategy (see module docstring for accounting)."""
    cfg = cfg or BacktestConfig()
    rets = period_returns.astype(float)
    pos = clean_positions(positions.reindex(rets.index), cfg.missing_position)
    valid = rets.notna()
    rets, pos = rets[valid], pos[valid]

    p = pos.values
    R = rets.values
    prev = np.concatenate([[0.0], p[:-1]])
    turnover = np.abs(p - prev)
    c = cfg.cost_rate

    n = len(p)
    equity = np.empty(n)
    cost_cash = np.empty(n)
    e_prev = cfg.initial_capital
    for i in range(n):
        cost_cash[i] = c * turnover[i] * e_prev
        equity[i] = (e_prev - cost_cash[i]) * (1.0 + p[i] * R[i])
        e_prev = equity[i]
    start_equity = np.concatenate([[cfg.initial_capital], equity[:-1]])
    net = equity / start_equity - 1.0
    gross = p * R

    frame = pd.DataFrame(
        {
            "position": p,
            "asset_return": R,
            "turnover": turnover,
            "gross_return": gross,
            "cost": cost_cash,
            "net_return": net,
            "equity": equity,
        },
        index=rets.index,
    )
    frame["drawdown"] = drawdown_series(frame["equity"], cfg.initial_capital).values
    trades = round_trip_trades(frame, cfg.cost_rate)
    metrics = compute_trading_metrics(frame, cfg, trades)
    return BacktestResult(frame=frame, metrics=metrics, trades=trades)


def round_trip_trades(frame: pd.DataFrame, cost_rate: float = 0.0) -> pd.DataFrame:
    """Contiguous runs of non-zero exposure with the same sign, with their compounded net return.

    The entry cost is already inside the first day's net return. The cost of closing a trade is
    charged on the following row (together with any new position), so for a closed trade the
    closing part ``cost_rate * |p_last|`` is attributed to the trade itself. A position still open
    at the end of the sample is reported with ``closed = False``.
    """
    pos = frame["position"].values
    net = frame["net_return"].values
    sign = np.sign(pos)
    rows, start = [], None
    for i in range(len(pos) + 1):
        s = sign[i] if i < len(pos) else 0.0
        if start is not None and (s != sign[start]):
            seg = slice(start, i)
            closed = i < len(pos)
            growth = float(np.prod(1.0 + net[seg]))
            if closed:
                growth *= 1.0 - cost_rate * abs(pos[i - 1])
            rows.append(
                {
                    "entry_date": frame.index[start],
                    "exit_date": frame.index[i - 1],
                    "direction": "long" if sign[start] > 0 else "short",
                    "days": i - start,
                    "net_return": growth - 1.0,
                    "closed": closed,
                }
            )
            start = None
        if start is None and i < len(pos) and s != 0:
            start = i
    return pd.DataFrame(rows, columns=["entry_date", "exit_date", "direction", "days", "net_return", "closed"])


def compute_trading_metrics(frame: pd.DataFrame, cfg: BacktestConfig, trades: pd.DataFrame | None = None) -> dict:
    """Trading metrics computed from the backtested net portfolio returns."""
    n = len(frame)
    net = frame["net_return"]
    ann = cfg.annualization
    if n == 0:
        return {"n_days": 0}
    final_equity = float(frame["equity"].iloc[-1])
    active = frame["position"] != 0
    n_trades = int((frame["turnover"] > 0).sum())
    total_turnover = float(frame["turnover"].sum())
    mdd = max_drawdown(frame["equity"], cfg.initial_capital)
    ann_ret = annualized_return(net, ann)
    out = {
        "n_days": int(n),
        "start": frame.index[0].strftime("%Y-%m-%d"),
        "end": frame.index[-1].strftime("%Y-%m-%d"),
        "final_equity": final_equity,
        "cumulative_return": final_equity / cfg.initial_capital - 1.0,
        "annualized_return": ann_ret,
        "annualized_volatility": annualized_volatility(net, ann),
        "sharpe": sharpe_ratio(net, cfg.risk_free_rate, ann),
        "sortino": sortino_ratio(net, cfg.risk_free_rate, ann),
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd < 0 else float("nan"),
        "exposure": float(active.mean()),
        "win_rate": float((net[active] > 0).mean()) if active.any() else float("nan"),
        "n_trades": n_trades,
        "turnover_total": total_turnover,
        "turnover_annual": total_turnover / n * ann,
        "costs_paid": float(frame["cost"].sum()),
        "costs_pct_initial": float(frame["cost"].sum()) / cfg.initial_capital,
    }
    closed = trades[trades["closed"]] if trades is not None and len(trades) else None
    out["open_position_at_end"] = bool(trades is not None and len(trades) and not trades["closed"].iloc[-1])
    if closed is not None and len(closed):
        # completed round trips only; a still-open position has no realised trade outcome yet
        out["n_round_trips"] = int(len(closed))
        out["trade_win_rate"] = float((closed["net_return"] > 0).mean())
        out["avg_holding_days"] = float(closed["days"].mean())
    else:
        out["n_round_trips"] = 0
        out["trade_win_rate"] = float("nan")
        out["avg_holding_days"] = float("nan")
    return out


# ---------------------------------------------------------------------------------------------
# signal construction
# ---------------------------------------------------------------------------------------------


def positions_from_probabilities(
    proba: np.ndarray,
    index: pd.Index,
    up_idx: int,
    down_idx: int = 0,
    threshold: float | None = None,
    mode: str = "long_only",
) -> pd.Series:
    """Convert class probabilities into target positions.

    * ``threshold is None`` (argmax / always-active): long when the predicted class is Up; in
      long/short mode short when it is Down; otherwise flat.
    * ``threshold = tau`` (confidence-based): long when P(Up) >= tau; in long/short mode short when
      P(Down) >= tau and P(Down) > P(Up); otherwise flat (Hold).
    Rows with NaN probabilities produce a flat position.
    """
    proba = np.asarray(proba, dtype=float)
    p_up, p_down = proba[:, up_idx], proba[:, down_idx]
    nan_rows = np.isnan(proba).any(axis=1)
    if threshold is None:
        pred = np.where(nan_rows, -1, np.nan_to_num(proba, nan=-1).argmax(axis=1))
        long_ = pred == up_idx
        short = (pred == down_idx) if mode == "long_short" else np.zeros_like(long_)
    else:
        long_ = (p_up >= threshold) & ~nan_rows
        short = ((p_down >= threshold) & (p_down > p_up) & ~nan_rows) if mode == "long_short" else np.zeros_like(long_)
        long_ = long_ & ~short
    pos = np.where(long_, 1.0, np.where(short, -1.0, 0.0))
    return pd.Series(pos, index=index, name="position")


def volatility_target_positions(
    positions: pd.Series, realized_daily_vol: pd.Series, annual_target: float = 0.15, annualization: int = 252, max_leverage: float = 1.0
) -> pd.Series:
    """Scale a {-1,0,1} signal by target_vol / realised vol (known at decision time), capped."""
    ann_vol = realized_daily_vol.reindex(positions.index) * np.sqrt(annualization)
    scale = (annual_target / ann_vol).clip(upper=max_leverage).fillna(0.0)
    return (positions * scale).rename("position")
