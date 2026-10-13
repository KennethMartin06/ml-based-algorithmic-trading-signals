"""Unit tests for the cost-aware backtester (src/backtester.py).

Accounting under test (c = cost rate, p = position, R = period return, E = equity):

    cost_t = c * |p_t - p_{t-1}| * E_{t-1},   E_t = (E_{t-1} - cost_t) * (1 + p_t * R_t),   p_{-1} = 0

Both ``positions[t]`` and ``period_returns[t]`` are indexed by the decision date t, so the
backtester must never shift either series. Every expected value below is derived by hand or with
an independent formula, never by calling the code under test a second time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    BacktestConfig,
    annualized_return,
    annualized_volatility,
    clean_positions,
    drawdown_series,
    max_drawdown,
    positions_from_probabilities,
    round_trip_trades,
    run_backtest,
    sharpe_ratio,
    sortino_ratio,
    volatility_target_positions,
)
from src.baselines import buy_and_hold_positions
from src.targets import period_returns

E0 = 100_000.0


def _cfg(cost_bps: float = 0.0, **kw) -> BacktestConfig:
    """Backtest config whose whole cost is ``cost_bps`` (all fee, no slippage)."""
    return BacktestConfig(initial_capital=kw.pop("initial_capital", E0), fee_bps=cost_bps, slippage_bps=0.0, **kw)


def _series(values, start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------


def test_cost_rate_combines_fee_and_slippage_in_bps():
    """cost_rate = (fee_bps + slippage_bps) / 10_000, i.e. 10 + 5 bps -> 0.0015 per unit turnover."""
    assert BacktestConfig(fee_bps=10, slippage_bps=5).cost_rate == pytest.approx(0.0015)


def test_from_config_reads_backtest_section_and_overrides(cfg):
    """from_config takes values from config.yaml's ``backtest`` block; keyword overrides win."""
    b = cfg["backtest"]
    bc = BacktestConfig.from_config(cfg)
    assert bc.initial_capital == b["initial_capital"]
    assert bc.cost_rate == pytest.approx((b["fee_bps"] + b["slippage_bps"]) / 10_000)
    assert bc.missing_position == b.get("missing_position", "flat")
    assert BacktestConfig.from_config(cfg, fee_bps=0.0, slippage_bps=0.0).cost_rate == 0.0


# ---------------------------------------------------------------------------------------------
# buy-and-hold and never-trade sanity checks
# ---------------------------------------------------------------------------------------------


def test_buy_and_hold_zero_cost_equals_compounded_asset_return(rng):
    """With zero costs a permanently long strategy earns exactly prod(1 + R) - 1."""
    R = _series(rng.normal(0.0005, 0.01, 300))
    res = run_backtest(buy_and_hold_positions(R.index), R, _cfg(0.0))
    expected = float(np.prod(1.0 + R.values) - 1.0)
    assert res.metrics["cumulative_return"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert res.frame["equity"].iloc[-1] == pytest.approx(E0 * (1.0 + expected), rel=1e-12)
    assert res.metrics["costs_paid"] == 0.0


def test_buy_and_hold_with_costs_pays_exactly_one_entry_cost(rng):
    """Starting flat, buy-and-hold pays one entry cost c*E0 and nothing afterwards, so
    final equity = E0 * (1 - c) * prod(1 + R)."""
    c_bps = 15.0
    c = c_bps / 10_000
    R = _series(rng.normal(0.0003, 0.012, 250))
    res = run_backtest(buy_and_hold_positions(R.index), R, _cfg(c_bps))
    f = res.frame
    assert f["cost"].iloc[0] == pytest.approx(c * E0)
    assert (f["cost"].iloc[1:] == 0.0).all()
    assert res.metrics["costs_paid"] == pytest.approx(c * E0)
    assert res.metrics["n_trades"] == 1
    assert f["equity"].iloc[-1] == pytest.approx(E0 * (1.0 - c) * np.prod(1.0 + R.values), rel=1e-12)


def test_buy_and_hold_on_next_open_returns_telescopes_to_open_ratio(tiny_prices):
    """End-to-end with src.targets.period_returns (next_open): buying at the first decision fills
    at Open[1] and the last usable period ends at Open[-1], so zero-cost equity = E0*Open[-1]/Open[1].
    The last two decision rows have no complete return and must be dropped."""
    R = period_returns(tiny_prices, "next_open")
    res = run_backtest(buy_and_hold_positions(tiny_prices.index), R, _cfg(0.0))
    o = tiny_prices["open"]
    assert len(res.frame) == len(tiny_prices) - 2
    assert res.frame["equity"].iloc[-1] == pytest.approx(E0 * o.iloc[-1] / o.iloc[1], rel=1e-12)


def test_buy_and_hold_on_synthetic_prices_matches_closed_form(prices):
    """On ~3 years of fixture prices the engine's loop agrees with the closed-form compounding."""
    R = period_returns(prices, "next_open")
    res = run_backtest(buy_and_hold_positions(prices.index), R, _cfg(10.0))
    valid = R.dropna()
    assert res.metrics["n_days"] == len(valid)
    expected = E0 * (1 - 0.001) * np.prod(1.0 + valid.values)
    assert res.metrics["final_equity"] == pytest.approx(expected, rel=1e-9)


def test_never_trading_strategy_is_inert(rng):
    """A strategy that is always flat pays no cost, keeps equity at E0, makes no trades and has an
    undefined (NaN) Sharpe/Sortino because its return volatility is exactly zero."""
    R = _series(rng.normal(0.0, 0.02, 120))
    res = run_backtest(pd.Series(0.0, index=R.index), R, _cfg(15.0))
    f, m = res.frame, res.metrics
    assert (f["cost"] == 0.0).all()
    assert np.allclose(f["equity"].values, E0, rtol=0, atol=0)
    assert m["n_trades"] == 0
    assert m["n_round_trips"] == 0
    assert m["exposure"] == 0.0
    assert m["cumulative_return"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert np.isnan(m["sharpe"])
    assert np.isnan(m["sortino"])
    assert np.isnan(m["win_rate"])  # no active days


# ---------------------------------------------------------------------------------------------
# transaction-cost accounting
# ---------------------------------------------------------------------------------------------


def test_costs_follow_hand_example_including_reversals():
    """Hand-computed path with c = 1% and E0 = 1000.

    t  p   R      |dp|  cost                       E_t
    0  +1  +0.10  1     0.01*1*1000     = 10        990     * 1.1 = 1089
    1  +1   0.00  0     0                           1089
    2  -1  +0.10  2     0.01*2*1089     = 21.78     1067.22 * 0.9 = 960.498
    3  +1  -0.10  2     0.01*2*960.498  = 19.20996  941.28804 * 0.9 = 847.159236
    4   0  +0.05  1     0.01*1*847.159236           838.68764364 * 1.0

    A reversal (+1 -> -1 or -1 -> +1) is two units of turnover and costs 2c * E_{t-1}.
    """
    pos = _series([1, 1, -1, 1, 0])
    R = _series([0.10, 0.0, 0.10, -0.10, 0.05])
    res = run_backtest(pos, R, _cfg(100.0, initial_capital=1000.0))
    f = res.frame
    np.testing.assert_allclose(f["turnover"].values, [1, 0, 2, 2, 1])
    np.testing.assert_allclose(f["cost"].values, [10.0, 0.0, 21.78, 19.20996, 8.47159236], rtol=1e-12)
    np.testing.assert_allclose(f["equity"].values, [1089.0, 1089.0, 960.498, 847.159236, 838.68764364], rtol=1e-12)
    assert res.metrics["n_trades"] == 4  # four days on which the position changed
    assert res.metrics["costs_paid"] == pytest.approx(10.0 + 21.78 + 19.20996 + 8.47159236)


def test_cost_is_charged_only_on_position_changes(rng):
    """For an arbitrary path: cost_t = c*|p_t - p_{t-1}|*E_{t-1} (p_{-1}=0, E_{-1}=E0), so days
    without a position change cost nothing."""
    c_bps = 20.0
    c = c_bps / 10_000
    pos = _series(rng.choice([-1.0, 0.0, 0.5, 1.0], size=200))
    R = _series(rng.normal(0.0, 0.01, 200))
    f = run_backtest(pos, R, _cfg(c_bps)).frame
    dp = np.abs(np.diff(np.concatenate([[0.0], pos.values])))
    e_prev = np.concatenate([[E0], f["equity"].values[:-1]])
    np.testing.assert_allclose(f["cost"].values, c * dp * e_prev, rtol=1e-12, atol=1e-9)
    assert (f["cost"].values[dp == 0] == 0.0).all()
    assert (f["cost"].values[dp > 0] > 0.0).all()


def test_equity_is_consistent_with_net_returns(rng):
    """E_t == E_{t-1} * (1 + net_t) for every day, hence final equity == E0 * prod(1 + net)."""
    pos = _series(rng.choice([-1.0, 0.0, 1.0], size=150))
    R = _series(rng.normal(0.0, 0.015, 150))
    f = run_backtest(pos, R, _cfg(15.0)).frame
    e_prev = np.concatenate([[E0], f["equity"].values[:-1]])
    np.testing.assert_allclose(f["equity"].values, e_prev * (1.0 + f["net_return"].values), rtol=1e-12)
    assert f["equity"].iloc[-1] == pytest.approx(E0 * np.prod(1.0 + f["net_return"].values), rel=1e-10)
    # gross return ignores costs: p_t * R_t
    np.testing.assert_allclose(f["gross_return"].values, pos.values * R.values)


# ---------------------------------------------------------------------------------------------
# drawdown
# ---------------------------------------------------------------------------------------------


def test_drawdown_on_known_equity_path_includes_initial_capital_as_first_peak():
    """Equity 90,110,99,120,60 from E0=100: running peaks are 100,110,110,120,120 (the initial
    capital is the first peak), so drawdowns are -10%, 0, -10%, 0, -50% and max DD is -50%."""
    eq = _series([90.0, 110.0, 99.0, 120.0, 60.0])
    dd = drawdown_series(eq, initial_capital=100.0)
    np.testing.assert_allclose(dd.values, [-0.1, 0.0, -0.1, 0.0, -0.5])
    assert max_drawdown(eq, initial_capital=100.0) == pytest.approx(-0.5)


def test_drawdown_without_initial_capital_uses_first_value_as_peak():
    """Without an initial capital the first observation is its own peak (drawdown 0 on day 0)."""
    eq = _series([90.0, 110.0, 99.0])
    np.testing.assert_allclose(drawdown_series(eq).values, [0.0, 0.0, -0.1])
    assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


def test_first_day_loss_shows_up_as_drawdown_in_backtest():
    """The engine measures drawdown against E0, so a loss on the very first day is a drawdown."""
    R = _series([-0.2, 0.1, 0.5])
    res = run_backtest(_series([1, 1, 1]), R, _cfg(0.0, initial_capital=100.0))
    # equity 80, 88, 132 -> peaks 100, 100, 132
    np.testing.assert_allclose(res.frame["drawdown"].values, [-0.2, -0.12, 0.0])
    assert res.metrics["max_drawdown"] == pytest.approx(-0.2)


# ---------------------------------------------------------------------------------------------
# timing / alignment and missing values
# ---------------------------------------------------------------------------------------------


def test_position_t_multiplies_period_return_t_not_t_plus_1():
    """A single long decision on day 1 must earn R[1] (= +10%), not R[2] (= -30%): both inputs are
    already aligned by decision date and the backtester must not shift them."""
    pos = _series([0, 1, 0, 0])
    R = _series([0.5, 0.10, -0.30, 0.20])
    f = run_backtest(pos, R, _cfg(0.0)).frame
    np.testing.assert_allclose(f["gross_return"].values, [0.0, 0.10, 0.0, 0.0])
    assert f["equity"].iloc[-1] == pytest.approx(E0 * 1.10)


def test_positions_are_aligned_by_date_label_not_by_order():
    """Positions are reindexed onto the return dates: a position given for a date with no return is
    ignored and a return date with no position is flat (never long)."""
    R = _series([0.05, 0.05, 0.05])
    d = R.index
    pos = pd.Series([1.0, 1.0], index=[d[1], d[1] + pd.Timedelta(days=30)])  # only d[1] overlaps
    f = run_backtest(pos, R, _cfg(0.0)).frame
    np.testing.assert_allclose(f["position"].values, [0.0, 1.0, 0.0])
    assert f["equity"].iloc[-1] == pytest.approx(E0 * 1.05)


def test_nan_positions_are_flat_with_no_long_exposure():
    """In 'flat' mode a NaN prediction becomes position 0: the huge +50% return on the NaN day is
    not earned, and leaving/re-entering the market is charged as two extra trades."""
    pos = _series([1.0, np.nan, 1.0])
    R = _series([0.01, 0.50, 0.01])
    res = run_backtest(pos, R, _cfg(10.0, missing_position="flat"))
    f = res.frame
    np.testing.assert_allclose(f["position"].values, [1.0, 0.0, 1.0])
    assert f["gross_return"].iloc[1] == 0.0
    assert res.metrics["n_trades"] == 3


def test_nan_positions_are_forward_filled_in_hold_mode():
    """In 'hold' mode a NaN carries the previous position forward; leading NaNs (nothing to carry)
    are still flat."""
    pos = _series([np.nan, 1.0, np.nan, -1.0, np.nan])
    R = _series([0.01, 0.02, 0.03, -0.01, -0.02])
    f = run_backtest(pos, R, _cfg(0.0, missing_position="hold")).frame
    np.testing.assert_allclose(f["position"].values, [0.0, 1.0, 1.0, -1.0, -1.0])
    np.testing.assert_allclose(clean_positions(pos, "flat").values, [0.0, 1.0, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(clean_positions(pos, "hold").values, [0.0, 1.0, 1.0, -1.0, -1.0])


def test_rows_with_nan_period_return_are_dropped():
    """Decision dates without a realised period return (e.g. the end of the sample) are removed
    from the simulation instead of being treated as a 0% return."""
    R = _series([0.01, np.nan, 0.02, np.nan])
    res = run_backtest(_series([1, 1, 1, 1]), R, _cfg(0.0))
    assert list(res.frame.index) == [R.index[0], R.index[2]]
    assert res.metrics["n_days"] == 2
    assert res.frame["equity"].iloc[-1] == pytest.approx(E0 * 1.01 * 1.02)


# ---------------------------------------------------------------------------------------------
# performance metrics vs manual formulas
# ---------------------------------------------------------------------------------------------


def test_risk_adjusted_metrics_match_manual_formulas(rng):
    """Sharpe, Sortino, annualised return and volatility against textbook formulas, with the annual
    risk-free rate converted geometrically to a daily rate: rf_d = (1 + rf)^(1/252) - 1."""
    r = pd.Series(rng.normal(0.0006, 0.011, 500))
    ann, rf = 252, 0.03
    rf_d = (1.0 + rf) ** (1.0 / ann) - 1.0
    x = r.values - rf_d

    sharpe = x.mean() / x.std(ddof=1) * np.sqrt(ann)
    downside = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))
    sortino = x.mean() / downside * np.sqrt(ann)
    ann_ret = np.prod(1.0 + r.values) ** (ann / len(r)) - 1.0
    ann_vol = r.values.std(ddof=1) * np.sqrt(ann)

    assert sharpe_ratio(r, rf, ann) == pytest.approx(sharpe, rel=1e-10)
    assert sortino_ratio(r, rf, ann) == pytest.approx(sortino, rel=1e-10)
    assert annualized_return(r, ann) == pytest.approx(ann_ret, rel=1e-10)
    assert annualized_volatility(r, ann) == pytest.approx(ann_vol, rel=1e-10)
    # the risk-free rate must actually lower the Sharpe ratio of the same returns
    assert sharpe_ratio(r, rf, ann) < sharpe_ratio(r, 0.0, ann)


def test_backtest_metrics_use_configured_risk_free_rate(rng):
    """run_backtest's Sharpe/Sortino are computed on its net returns with cfg.risk_free_rate."""
    pos = _series(rng.choice([0.0, 1.0], size=300))
    R = _series(rng.normal(0.0005, 0.01, 300))
    res = run_backtest(pos, R, _cfg(5.0, risk_free_rate=0.02))
    net = res.frame["net_return"]
    assert res.metrics["sharpe"] == pytest.approx(sharpe_ratio(net, 0.02, 252), rel=1e-12)
    assert res.metrics["sortino"] == pytest.approx(sortino_ratio(net, 0.02, 252), rel=1e-12)
    assert res.metrics["annualized_return"] == pytest.approx(annualized_return(net, 252), rel=1e-12)
    assert res.metrics["annualized_volatility"] == pytest.approx(annualized_volatility(net, 252), rel=1e-12)


def test_annualized_return_hand_values_and_edge_cases():
    """A constant 0.1% daily return over a year annualises to 1.001^252 - 1; a wipe-out returns -1;
    NaNs are ignored; too-short inputs give NaN for volatility and Sharpe."""
    assert annualized_return(pd.Series([0.001] * 252), 252) == pytest.approx(1.001 ** 252 - 1.0)
    assert annualized_return(pd.Series([0.001] * 126), 252) == pytest.approx(1.001 ** 252 - 1.0)
    assert annualized_return(pd.Series([0.5, -1.0, 0.2]), 252) == -1.0
    assert annualized_return(pd.Series([0.01, np.nan, 0.01]), 2) == pytest.approx(1.01 ** 2 - 1.0)
    assert np.isnan(annualized_return(pd.Series([], dtype=float)))
    assert np.isnan(annualized_volatility(pd.Series([0.01])))
    assert np.isnan(sharpe_ratio(pd.Series([0.01])))
    assert np.isnan(sortino_ratio(pd.Series([0.01, 0.02, 0.03])))  # no downside deviation


def test_win_rate_counts_only_active_days():
    """win_rate = share of days WITH exposure whose net return is positive; flat days are ignored.
    Active days: long +2% (win), long -1% (loss), short on -3% (win) -> 2/3. Exposure = 3/5."""
    pos = _series([1, 0, 1, -1, 0])
    R = _series([0.02, 0.05, -0.01, -0.03, -0.04])
    m = run_backtest(pos, R, _cfg(0.0)).metrics
    assert m["win_rate"] == pytest.approx(2.0 / 3.0)
    assert m["exposure"] == pytest.approx(3.0 / 5.0)


# ---------------------------------------------------------------------------------------------
# trades and turnover
# ---------------------------------------------------------------------------------------------


def _trade_frame() -> pd.DataFrame:
    pos = [0, 1, 1, 0, -1, -1, 1, 0.5, 0, -1]
    net = [0, 0.10, -0.05, 0, 0.02, 0.03, -0.01, 0.04, 0, 0.05]
    return pd.DataFrame({"position": pos, "net_return": net}, index=pd.bdate_range("2024-01-01", periods=10))


def test_round_trip_trades_on_hand_path_with_reversal():
    """Trades are maximal runs of same-sign exposure. On 0,1,1,0,-1,-1,1,0.5,0,-1:
    long[1..2], short[4..5], long[6..7] (the -1 -> +1 reversal splits trades; 1 -> 0.5 does not),
    and an open short[9..9] at the end. Each trade's return compounds its daily net returns."""
    f = _trade_frame()
    t = round_trip_trades(f)
    d = f.index
    assert list(t["direction"]) == ["long", "short", "long", "short"]
    assert list(t["entry_date"]) == [d[1], d[4], d[6], d[9]]
    assert list(t["exit_date"]) == [d[2], d[5], d[7], d[9]]
    assert list(t["days"]) == [2, 2, 2, 1]
    assert list(t["closed"]) == [True, True, True, False]  # the last short is still open at the end
    np.testing.assert_allclose(
        t["net_return"].values, [1.10 * 0.95 - 1, 1.02 * 1.03 - 1, 0.99 * 1.04 - 1, 0.05], rtol=1e-12
    )


def test_round_trip_trades_attribute_exit_cost_to_the_closed_trade():
    """The cost of closing a trade is charged on the next row; with cost_rate c it is attributed to the
    trade it closes as (1 - c*|p_last|). The open trade at the end pays no exit cost."""
    f = _trade_frame()
    c = 0.001
    t = round_trip_trades(f, cost_rate=c)
    expected = [1.10 * 0.95 * (1 - c) - 1, 1.02 * 1.03 * (1 - c) - 1, 0.99 * 1.04 * (1 - 0.5 * c) - 1, 0.05]
    np.testing.assert_allclose(t["net_return"].values, expected, rtol=1e-12)


def test_round_trip_trades_empty_when_always_flat():
    """A flat path has no trades but still returns a frame with the expected columns."""
    f = pd.DataFrame({"position": [0.0, 0.0], "net_return": [0.0, 0.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    t = round_trip_trades(f)
    assert t.empty
    assert list(t.columns) == ["entry_date", "exit_date", "direction", "days", "net_return", "closed"]


def test_buy_and_hold_has_no_completed_round_trip():
    """Buy-and-hold never exits: its single position is open at the end, so there is no completed round
    trip and the trade win rate is undefined (NaN) rather than a misleading 100%."""
    R = pd.Series([0.01, -0.02, 0.03], index=pd.bdate_range("2024-01-01", periods=3))
    m = run_backtest(pd.Series(1.0, index=R.index), R, _cfg(10.0)).metrics
    assert m["n_round_trips"] == 0
    assert m["open_position_at_end"] is True
    assert np.isnan(m["trade_win_rate"])


def test_turnover_trade_count_and_annual_turnover():
    """On the hand path the position changes on 7 days with total turnover
    1+1+1+2+0.5+0.5+1 = 7 over 10 days, so turnover_annual = 7/10*252; 3 completed round trips of
    2 days each (the final short is still open and is not a completed round trip)."""
    pos = pd.Series(_trade_frame()["position"].values, index=_trade_frame().index, dtype=float)
    R = pd.Series(0.001, index=pos.index)
    m = run_backtest(pos, R, _cfg(10.0)).metrics
    assert m["n_trades"] == 7
    assert m["turnover_total"] == pytest.approx(7.0)
    assert m["turnover_annual"] == pytest.approx(7.0 / 10.0 * 252)
    assert m["n_round_trips"] == 3
    assert m["avg_holding_days"] == pytest.approx(2.0)
    assert m["open_position_at_end"] is True


# ---------------------------------------------------------------------------------------------
# signal construction
# ---------------------------------------------------------------------------------------------

# three-class probabilities, columns = [Down, Neutral, Up]
_P3_ARGMAX = np.array(
    [
        [0.20, 0.30, 0.50],  # Up most likely
        [0.60, 0.30, 0.10],  # Down most likely
        [0.20, 0.50, 0.30],  # Neutral most likely
        [np.nan, np.nan, np.nan],  # missing prediction
        [0.10, np.nan, 0.90],  # partially missing prediction
    ]
)


def test_positions_from_probabilities_argmax_long_only():
    """Argmax rule, long-only: long only when Up is the predicted class; Down/Neutral/NaN -> flat."""
    idx = pd.bdate_range("2024-01-01", periods=len(_P3_ARGMAX))
    pos = positions_from_probabilities(_P3_ARGMAX, idx, up_idx=2, down_idx=0, threshold=None, mode="long_only")
    np.testing.assert_array_equal(pos.values, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert pos.index.equals(idx) and pos.name == "position"


def test_positions_from_probabilities_argmax_long_short():
    """Argmax rule, long/short: Up -> +1, Down -> -1, Neutral -> 0, any NaN in the row -> 0."""
    idx = pd.bdate_range("2024-01-01", periods=len(_P3_ARGMAX))
    pos = positions_from_probabilities(_P3_ARGMAX, idx, up_idx=2, down_idx=0, threshold=None, mode="long_short")
    np.testing.assert_array_equal(pos.values, [1.0, -1.0, 0.0, 0.0, 0.0])


# rows chosen to exercise each branch of the confidence rule with tau = 0.40
_P3_THRESH = np.array(
    [
        [0.20, 0.30, 0.50],  # P(Up) >= tau                         -> long
        [0.60, 0.30, 0.10],  # P(Down) >= tau, P(Down) > P(Up)      -> short (LS) / flat (LO)
        [0.25, 0.40, 0.35],  # neither reaches tau                  -> flat
        [0.45, 0.10, 0.45],  # both >= tau but P(Down) == P(Up)     -> long (no strict dominance)
        [0.50, 0.05, 0.45],  # both >= tau and P(Down) > P(Up)      -> short (LS) / long (LO)
        [np.nan, 0.10, 0.90],  # NaN row                            -> flat
    ]
)


def test_positions_from_probabilities_threshold_long_only():
    """Confidence rule, long-only: long iff P(Up) >= tau (the Down probability is irrelevant)."""
    idx = pd.bdate_range("2024-01-01", periods=len(_P3_THRESH))
    pos = positions_from_probabilities(_P3_THRESH, idx, up_idx=2, down_idx=0, threshold=0.40, mode="long_only")
    np.testing.assert_array_equal(pos.values, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0])


def test_positions_from_probabilities_threshold_long_short():
    """Confidence rule, long/short: short iff P(Down) >= tau and P(Down) > P(Up) (short wins over
    long in that case); otherwise long iff P(Up) >= tau; NaN rows flat."""
    idx = pd.bdate_range("2024-01-01", periods=len(_P3_THRESH))
    pos = positions_from_probabilities(_P3_THRESH, idx, up_idx=2, down_idx=0, threshold=0.40, mode="long_short")
    np.testing.assert_array_equal(pos.values, [1.0, -1.0, 0.0, 1.0, -1.0, 0.0])


def test_positions_from_probabilities_binary_scheme():
    """Binary scheme (Down=0, Up=1): the same rules apply with up_idx=1."""
    proba = np.array([[0.3, 0.7], [0.6, 0.4], [np.nan, np.nan], [0.52, 0.48]])
    idx = pd.bdate_range("2024-01-01", periods=4)
    lo = positions_from_probabilities(proba, idx, up_idx=1, down_idx=0, threshold=0.55, mode="long_only")
    ls = positions_from_probabilities(proba, idx, up_idx=1, down_idx=0, threshold=0.55, mode="long_short")
    am = positions_from_probabilities(proba, idx, up_idx=1, down_idx=0, threshold=None, mode="long_short")
    np.testing.assert_array_equal(lo.values, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(ls.values, [1.0, -1.0, 0.0, 0.0])
    np.testing.assert_array_equal(am.values, [1.0, -1.0, 0.0, -1.0])


def test_volatility_target_positions_scales_and_caps():
    """Position * min(target / annualised vol, max_leverage); NaN vol -> 0 exposure.
    Annual vols 30%, 5%, 15%, 7.5%, NaN with a 15% target give scales 0.5, 3->1, 1, 2->1, 0."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    pos = pd.Series([1.0, 1.0, -1.0, -1.0, 1.0], index=idx)
    ann_vol = np.array([0.30, 0.05, 0.15, 0.075, np.nan])
    daily_vol = pd.Series(ann_vol / np.sqrt(252), index=idx)
    out = volatility_target_positions(pos, daily_vol, annual_target=0.15, annualization=252, max_leverage=1.0)
    np.testing.assert_allclose(out.values, [0.5, 1.0, -1.0, -1.0, 0.0])
    assert out.abs().max() <= 1.0 + 1e-12
    assert out.name == "position"
    levered = volatility_target_positions(pos, daily_vol, annual_target=0.15, annualization=252, max_leverage=2.0)
    np.testing.assert_allclose(levered.values, [0.5, 2.0, -1.0, -2.0, 0.0])
