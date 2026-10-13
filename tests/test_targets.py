"""Tests for src/targets.py: forward returns, labels and label-availability dates.

The ``tiny_prices`` fixture (conftest.py) is small enough to verify every number by hand:

    row   date         open    close
    0     2024-01-01   100.0   100.5
    1     2024-01-02   101.0   102.0
    2     2024-01-03   103.0   102.5
    3     2024-01-04   102.0   103.5
    4     2024-01-05   104.0   105.0
    5     2024-01-08   106.0   106.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.targets import (
    build_target_frame,
    class_distribution,
    down_class_index,
    forward_returns,
    label_end_offset,
    make_labels,
    period_returns,
    up_class_index,
)

NAN = np.nan


def _assert_values(actual: pd.Series, expected: list[float]) -> None:
    """Compare a Series to hand-computed values, treating NaN == NaN."""
    np.testing.assert_allclose(actual.to_numpy(dtype=float), np.array(expected, dtype=float),
                               rtol=0, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------------------------
# forward_returns: hand-checked on tiny_prices
# ---------------------------------------------------------------------------------------------

def test_forward_returns_next_open_horizon_1(tiny_prices):
    """next_open, h=1: r_t = Open[t+2] / Open[t+1] - 1; the last 2 rows have no label window."""
    r = forward_returns(tiny_prices, horizon=1, convention="next_open")
    expected = [103 / 101 - 1, 102 / 103 - 1, 104 / 102 - 1, 106 / 104 - 1, NAN, NAN]
    _assert_values(r, expected)
    assert r.name == "fwd_ret_1"
    assert r.index.equals(tiny_prices.index)


def test_forward_returns_next_open_horizon_2(tiny_prices):
    """next_open, h=2: r_t = Open[t+3] / Open[t+1] - 1; the last 3 rows are NaN."""
    r = forward_returns(tiny_prices, horizon=2, convention="next_open")
    expected = [102 / 101 - 1, 104 / 103 - 1, 106 / 102 - 1, NAN, NAN, NAN]
    _assert_values(r, expected)
    assert r.name == "fwd_ret_2"


def test_forward_returns_close_horizon_1(tiny_prices):
    """close, h=1: r_t = Close[t+1] / Close[t] - 1; only the last row is NaN."""
    r = forward_returns(tiny_prices, horizon=1, convention="close")
    expected = [102 / 100.5 - 1, 102.5 / 102 - 1, 103.5 / 102.5 - 1, 105 / 103.5 - 1,
                106.5 / 105 - 1, NAN]
    _assert_values(r, expected)


def test_forward_returns_close_horizon_2(tiny_prices):
    """close, h=2: r_t = Close[t+2] / Close[t] - 1; the last 2 rows are NaN."""
    r = forward_returns(tiny_prices, horizon=2, convention="close")
    expected = [102.5 / 100.5 - 1, 103.5 / 102 - 1, 105 / 102.5 - 1, 106.5 / 103.5 - 1, NAN, NAN]
    _assert_values(r, expected)


def test_next_open_return_does_not_use_close_of_decision_day(tiny_prices):
    """The next_open target must not depend on Close[t] or Open[t] (they are known at decision time
    but are not the fill price), so changing them leaves every forward return unchanged."""
    base = forward_returns(tiny_prices, 1, "next_open")
    perturbed = tiny_prices.copy()
    perturbed["close"] = perturbed["close"] * 3.0
    perturbed.loc[perturbed.index[0], "open"] = 1.0  # row 0 is only ever a decision row
    _assert_values(forward_returns(perturbed, 1, "next_open"), base.tolist())


@pytest.mark.parametrize("convention", ["next_open", "close"])
@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_trailing_nan_count_equals_label_end_offset(prices, horizon, convention):
    """Exactly the last ``label_end_offset`` rows lack a complete label window; all others are finite."""
    r = forward_returns(prices, horizon, convention)
    k = label_end_offset(horizon, convention)
    assert r.iloc[-k:].isna().all()
    assert r.iloc[:-k].notna().all()


@pytest.mark.parametrize("horizon", [0, -1])
def test_forward_returns_rejects_non_positive_horizon(tiny_prices, horizon):
    """A horizon below one trading day is meaningless and must raise."""
    with pytest.raises(ValueError):
        forward_returns(tiny_prices, horizon=horizon)


def test_forward_returns_rejects_unknown_convention(tiny_prices):
    """Unknown execution conventions must raise instead of silently defaulting."""
    with pytest.raises(ValueError):
        forward_returns(tiny_prices, horizon=1, convention="vwap")


def test_period_returns_is_one_period_forward_return(tiny_prices):
    """period_returns (used by the backtester) is the h=1 forward return under the same convention."""
    for convention in ("next_open", "close"):
        pr = period_returns(tiny_prices, convention)
        _assert_values(pr, forward_returns(tiny_prices, 1, convention).tolist())
        assert pr.name == "period_ret"


# ---------------------------------------------------------------------------------------------
# label_end_offset
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "horizon, convention, expected",
    [(1, "next_open", 2), (2, "next_open", 3), (5, "next_open", 6),
     (1, "close", 1), (2, "close", 2), (5, "close", 5)],
)
def test_label_end_offset_values(horizon, convention, expected):
    """next_open needs h+1 future rows (fill at t+1, exit at t+1+h); close needs h rows."""
    assert label_end_offset(horizon, convention) == expected


def test_label_end_offset_rejects_unknown_convention():
    """An unknown convention raises ValueError."""
    with pytest.raises(ValueError):
        label_end_offset(1, "open")


# ---------------------------------------------------------------------------------------------
# make_labels
# ---------------------------------------------------------------------------------------------

def test_three_class_labels_including_exact_boundaries():
    """Down if r < -theta, Neutral if |r| <= theta (boundaries inclusive), Up if r > theta."""
    theta = 0.01
    fwd = pd.Series([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])
    labels = make_labels(fwd, "three_class", theta)
    _assert_values(labels, [0, 1, 1, 1, 1, 1, 2])


def test_three_class_values_just_outside_band():
    """Returns a hair beyond +/-theta leave the Neutral class."""
    theta = 0.001
    eps = 1e-12
    fwd = pd.Series([-theta - eps, -theta, theta, theta + eps])
    _assert_values(make_labels(fwd, "three_class", theta), [0, 1, 1, 2])


def test_zero_threshold_makes_only_exact_zero_neutral():
    """theta = 0 is allowed: only r == 0 is Neutral."""
    fwd = pd.Series([-1e-9, 0.0, 1e-9])
    _assert_values(make_labels(fwd, "three_class", 0.0), [0, 1, 2])


def test_negative_threshold_raises():
    """A negative neutral band is invalid."""
    with pytest.raises(ValueError):
        make_labels(pd.Series([0.01]), "three_class", -0.001)


def test_binary_labels_zero_return_is_down():
    """Binary: Up = 1 only if r > 0; a zero return is Down = 0."""
    fwd = pd.Series([-0.03, 0.0, 1e-9, 0.02])
    _assert_values(make_labels(fwd, "binary"), [0, 0, 1, 1])


@pytest.mark.parametrize("scheme", ["three_class", "binary"])
def test_nan_forward_returns_stay_nan_labels(scheme):
    """Rows without a forward return get a NaN label (float dtype), never a class."""
    fwd = pd.Series([0.02, NAN, -0.02, NAN])
    labels = make_labels(fwd, scheme, 0.001)
    assert labels.dtype == float
    assert labels.name == "label"
    assert labels.isna().tolist() == [False, True, False, True]
    assert labels.index.equals(fwd.index)


@pytest.mark.parametrize("scheme, n_classes", [("three_class", 3), ("binary", 2)])
def test_every_valid_return_gets_exactly_one_class(prices, scheme, n_classes):
    """Labels partition all finite forward returns into the valid class ids 0..K-1."""
    fwd = forward_returns(prices, 1, "next_open")
    labels = make_labels(fwd, scheme, 0.001)
    assert labels.notna().sum() == fwd.notna().sum()
    assert set(labels.dropna().unique()) <= set(range(n_classes))


def test_make_labels_rejects_unknown_scheme():
    """An unknown target scheme raises ValueError."""
    with pytest.raises(ValueError):
        make_labels(pd.Series([0.01]), "five_class")


def test_class_indices():
    """Up is the last class and Down is class 0 for both schemes."""
    assert up_class_index("three_class") == 2
    assert up_class_index("binary") == 1
    assert down_class_index("three_class") == 0
    assert down_class_index("binary") == 0


# ---------------------------------------------------------------------------------------------
# build_target_frame
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("convention", ["next_open", "close"])
@pytest.mark.parametrize("horizon", [1, 2])
def test_build_target_frame_label_end_date(tiny_prices, horizon, convention):
    """label_end_date[t] is the date ``label_end_offset`` rows after t (NaT past the end of data)."""
    cfg = {"horizon": horizon, "scheme": "three_class", "neutral_threshold": 0.001}
    frame = build_target_frame(tiny_prices, cfg, convention)
    k = label_end_offset(horizon, convention)
    idx = tiny_prices.index
    for i in range(len(idx)):
        if i + k < len(idx):
            assert frame["label_end_date"].iloc[i] == idx[i + k]
        else:
            assert pd.isna(frame["label_end_date"].iloc[i])


def test_build_target_frame_hand_checked_dates(tiny_prices):
    """Concrete example: with next_open and h=1 the label of Monday 2024-01-01 is known on
    Wednesday 2024-01-03 (fill at Tuesday's open, exit at Wednesday's open)."""
    frame = build_target_frame(tiny_prices, {"horizon": 1}, "next_open")
    assert frame.loc[pd.Timestamp("2024-01-01"), "label_end_date"] == pd.Timestamp("2024-01-03")
    # Thursday 2024-01-04's label ends two *trading* rows later: the following Monday, not Saturday.
    assert frame.loc[pd.Timestamp("2024-01-04"), "label_end_date"] == pd.Timestamp("2024-01-08")


@pytest.mark.parametrize("convention", ["next_open", "close"])
@pytest.mark.parametrize("horizon", [1, 2])
def test_build_target_frame_columns(tiny_prices, horizon, convention):
    """fwd_ret is the h-period return, label its class, and period_ret is ALWAYS the 1-period return."""
    theta = 0.001
    cfg = {"horizon": horizon, "scheme": "three_class", "neutral_threshold": theta}
    frame = build_target_frame(tiny_prices, cfg, convention)
    assert list(frame.columns) == ["fwd_ret", "label", "label_end_date", "period_ret"]
    assert frame.index.equals(tiny_prices.index)
    fwd = forward_returns(tiny_prices, horizon, convention)
    _assert_values(frame["fwd_ret"], fwd.tolist())
    _assert_values(frame["label"], make_labels(fwd, "three_class", theta).tolist())
    _assert_values(frame["period_ret"], forward_returns(tiny_prices, 1, convention).tolist())


def test_build_target_frame_label_known_iff_label_end_date_known(prices):
    """A label exists exactly for the rows whose label window ends inside the data."""
    frame = build_target_frame(prices, {"horizon": 3, "scheme": "binary"}, "next_open")
    assert (frame["label"].notna() == frame["label_end_date"].notna()).all()
    valid = frame.dropna(subset=["label_end_date"])
    assert (valid["label_end_date"] > valid.index).all()


def test_build_target_frame_uses_defaults(tiny_prices):
    """Missing config keys fall back to h=1, three_class, theta=0.001."""
    frame = build_target_frame(tiny_prices, {}, "close")
    fwd = forward_returns(tiny_prices, 1, "close")
    _assert_values(frame["label"], make_labels(fwd, "three_class", 0.001).tolist())


def test_build_target_frame_rejects_invalid_inputs(tiny_prices):
    """Invalid convention or scheme raise ValueError."""
    with pytest.raises(ValueError):
        build_target_frame(tiny_prices, {"horizon": 1}, "midday")
    with pytest.raises(ValueError):
        build_target_frame(tiny_prices, {"horizon": 1, "scheme": "ternary"}, "next_open")


# ---------------------------------------------------------------------------------------------
# class_distribution
# ---------------------------------------------------------------------------------------------

def test_class_distribution_three_class_counts_and_frequencies():
    """Counts per class ignore NaN labels; frequencies sum to one."""
    labels = pd.Series([0, 1, 1, 2, 2, 2, NAN])
    dist = class_distribution(labels, "three_class")
    assert dist["class"].tolist() == ["Down", "Neutral", "Up"]
    assert dist["count"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(dist["frequency"], [1 / 6, 2 / 6, 3 / 6])
    assert dist["frequency"].sum() == pytest.approx(1.0)


def test_class_distribution_includes_empty_classes():
    """A class that never occurs is listed with count 0 instead of being omitted."""
    dist = class_distribution(pd.Series([2.0, 2.0, 0.0]), "three_class")
    assert dist["count"].tolist() == [1, 0, 2]
    np.testing.assert_allclose(dist["frequency"], [1 / 3, 0.0, 2 / 3])


def test_class_distribution_binary():
    """Binary scheme yields two rows named Down/Up."""
    dist = class_distribution(pd.Series([0, 1, 1, 1]), "binary")
    assert dist["class"].tolist() == ["Down", "Up"]
    assert dist["count"].tolist() == [1, 3]
    np.testing.assert_allclose(dist["frequency"], [0.25, 0.75])


def test_class_distribution_all_nan_has_zero_counts_and_nan_frequency():
    """With no valid labels the counts are zero and frequencies are undefined (NaN), not 0/0 errors."""
    dist = class_distribution(pd.Series([NAN, NAN]), "three_class")
    assert dist["count"].tolist() == [0, 0, 0]
    assert dist["frequency"].isna().all()


def test_class_distribution_rejects_unknown_scheme():
    """An unknown scheme raises ValueError."""
    with pytest.raises(ValueError):
        class_distribution(pd.Series([0, 1]), "multi")
