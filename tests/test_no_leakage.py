"""Look-ahead / leakage tests - the most important checks in the project.

A decision is taken after the close of day t, so everything a model or rule sees for row t must be a
function of data at or before t, while the *label* of row t is the only quantity allowed to look
forward (up to ``label_end_date``). These tests verify that contract directly:

1. features are truncation-invariant and unaffected by any change to future prices;
2. targets use exactly the future prices the execution convention implies, and ``label_end_date``
   marks the last one;
3. chronological splits and walk-forward folds are ordered, disjoint and purged;
4. LSTM windows and its scaler never contain rows after the decision / training cut-off;
5. rule-based baselines are causal;
6. the time-series CV used for tuning never validates on data older than its training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src import models as models_module
from src.baselines import ma_crossover_positions, persistence_positions
from src.data_loader import adjust_ohlc, generate_synthetic_ohlcv, validate_ohlcv
from src.features import compute_features
from src.preprocessing import assemble_dataset, build_sequences, chronological_split, walk_forward_folds
from src.targets import build_target_frame, forward_returns, label_end_offset

PRICE_COLS = ["open", "high", "low", "close", "volume"]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Guard: leakage tests run on deterministic synthetic data and must never download anything."""

    def _blocked(*args, **kwargs):  # pragma: no cover - only hit if a test misbehaves
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr("src.data_loader.download_ticker", _blocked)


def _perturb_rows(df: pd.DataFrame, rows, rng, cols=PRICE_COLS, scale: float = 0.2) -> pd.DataFrame:
    """Return a copy with the given rows' columns multiplied by random factors in [1-scale, 1+scale]."""
    out = df.copy()
    idx = df.index[rows]
    for col in cols:
        out.loc[idx, col] = out.loc[idx, col].to_numpy() * rng.uniform(1 - scale, 1 + scale, len(idx))
    return out


def _dataset(prices: pd.DataFrame, cfg: dict, convention: str = "next_open") -> pd.DataFrame:
    data, _ = assemble_dataset(compute_features(prices, cfg["features"]), build_target_frame(prices, cfg["target"], convention))
    return data


# ---------------------------------------------------------------------------------------------
# 1. features are point-in-time
# ---------------------------------------------------------------------------------------------

CUTS = [15, 40, 120, 252, 272, 300, 401, 555, 700, 781]


@pytest.fixture(scope="module")
def full_features(prices, cfg) -> pd.DataFrame:
    return compute_features(prices, cfg["features"])


@pytest.mark.parametrize("cut", CUTS)
def test_features_truncation_invariant(prices, cfg, full_features, cut):
    """Features computed on prices[:cut] equal the full-sample features on those rows.

    In particular the last row (the decision row ``cut-1``) is identical whether or not later data
    exists - so no indicator peeks ahead. Cuts cover the warm-up, the regime-window boundary and
    the end of the sample.
    """
    trunc = compute_features(prices.iloc[:cut], cfg["features"])
    assert list(trunc.columns) == list(full_features.columns)
    last, ref = trunc.iloc[-1].to_numpy(), full_features.iloc[cut - 1].to_numpy()
    assert np.allclose(last, ref, rtol=1e-10, atol=1e-12, equal_nan=True), (
        "features of the last row change when later data is appended: "
        + str(list(full_features.columns[~np.isclose(last, ref, equal_nan=True)]))
    )
    assert np.allclose(trunc.to_numpy(), full_features.iloc[:cut].to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True)


def test_truncation_covers_at_least_eight_cuts_with_valid_rows(full_features):
    """Sanity check on the test design: >= 8 cut points, several with a complete feature row."""
    assert len(CUTS) >= 8
    assert sum(full_features.iloc[c - 1].notna().all() for c in CUTS) >= 5


@pytest.mark.parametrize("t", [60, 251, 300, 520, 760])
def test_future_price_perturbation_does_not_change_past_features(prices, cfg, full_features, t, rng):
    """Randomly rescaling every OHLCV value strictly AFTER row t leaves all features at rows <= t intact."""
    shocked = _perturb_rows(prices, slice(t + 1, None), rng)
    out = compute_features(shocked, cfg["features"])
    assert np.allclose(out.iloc[: t + 1].to_numpy(), full_features.iloc[: t + 1].to_numpy(), rtol=1e-10, equal_nan=True)
    # the perturbation is real: later rows did change
    assert not np.allclose(out.iloc[t + 1 :].to_numpy(), full_features.iloc[t + 1 :].to_numpy(), equal_nan=True)


def test_single_future_spike_only_moves_later_rows(prices, cfg, full_features):
    """A 50% spike in the close of one day affects that day and later rows, never earlier ones."""
    t = 450
    spiked = prices.copy()
    spiked.iloc[t, spiked.columns.get_loc("close")] *= 1.5
    out = compute_features(spiked, cfg["features"])
    changed = ~np.isclose(out.to_numpy(), full_features.to_numpy(), equal_nan=True)
    first_changed_row = int(np.argmax(changed.any(axis=1)))
    assert first_changed_row == t


# ---------------------------------------------------------------------------------------------
# 2. targets look forward - by exactly the documented amount
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def target_cfg(cfg) -> dict:
    return {"horizon": 1, "scheme": "three_class", "neutral_threshold": 0.001}


def test_label_end_offset_values():
    """next_open needs h+1 future rows (entry at t+1, exit at t+1+h); close needs h."""
    assert label_end_offset(1, "next_open") == 2
    assert label_end_offset(3, "next_open") == 4
    assert label_end_offset(1, "close") == 1
    assert label_end_offset(5, "close") == 5
    with pytest.raises(ValueError):
        label_end_offset(1, "vwap")


def test_next_open_return_definition(prices):
    """r_t = Open[t+1+h] / Open[t+1] - 1 for the realistic next-open execution."""
    o = prices["open"].to_numpy()
    for h in (1, 3):
        fwd = forward_returns(prices, h, "next_open").to_numpy()
        t = np.arange(len(o) - h - 1)
        np.testing.assert_allclose(fwd[t], o[t + 1 + h] / o[t + 1] - 1, rtol=1e-12)
        assert np.isnan(fwd[-(h + 1) :]).all()


def test_close_return_definition(prices):
    """r_t = Close[t+h] / Close[t] - 1 for the market-on-close sensitivity variant."""
    c = prices["close"].to_numpy()
    fwd = forward_returns(prices, 1, "close").to_numpy()
    np.testing.assert_allclose(fwd[:-1], c[1:] / c[:-1] - 1, rtol=1e-12)
    assert np.isnan(fwd[-1])


def test_next_open_label_depends_on_exit_open(prices, target_cfg):
    """The label of row t is decided by Open[t+2] relative to Open[t+1]: moving only Open[t+2]
    produces Up, Neutral or Down - the target genuinely depends on the future."""
    t = 400
    entry = prices["open"].iloc[t + 1]
    labels = {}
    for factor, expected in [(1.01, 2.0), (1.0005, 1.0), (0.99, 0.0)]:
        p = prices.copy()
        p.iloc[t + 2, p.columns.get_loc("open")] = entry * factor
        labels[factor] = build_target_frame(p, target_cfg, "next_open")["label"].iloc[t]
        assert labels[factor] == expected


def test_next_open_label_ignores_prices_at_or_before_t(prices, target_cfg, rng):
    """Under next-open execution nothing up to and including day t enters r_t (the entry is the
    NEXT open), so rescaling rows <= t leaves fwd_ret[t] and label[t] unchanged."""
    t = 400
    base = build_target_frame(prices, target_cfg, "next_open")
    shocked = build_target_frame(_perturb_rows(prices, slice(0, t + 1), rng), target_cfg, "next_open")
    assert shocked["fwd_ret"].iloc[t] == base["fwd_ret"].iloc[t]
    assert shocked["label"].iloc[t] == base["label"].iloc[t]


def test_next_open_label_depends_only_on_the_two_opens(prices, target_cfg, rng):
    """Only Open[t+1] (entry) and Open[t+2] (exit) matter: High/Low/Close/Volume of those days and
    every row after t+2 are irrelevant, while changing the entry open changes the return."""
    t = 400
    base = build_target_frame(prices, target_cfg, "next_open")
    other_cols = _perturb_rows(prices, slice(t + 1, t + 3), rng, cols=["high", "low", "close", "volume"])
    later_rows = _perturb_rows(prices, slice(t + 3, None), rng)
    for p in (other_cols, later_rows):
        assert build_target_frame(p, target_cfg, "next_open")["fwd_ret"].iloc[t] == base["fwd_ret"].iloc[t]
    entry_moved = prices.copy()
    entry_moved.iloc[t + 1, entry_moved.columns.get_loc("open")] *= 1.03
    assert build_target_frame(entry_moved, target_cfg, "next_open")["fwd_ret"].iloc[t] != base["fwd_ret"].iloc[t]


def test_close_label_depends_on_close_t_and_t_plus_1(prices, target_cfg, rng):
    """Market-on-close: r_t uses Close[t] and Close[t+1] only; earlier rows and rows > t+1 are irrelevant."""
    t = 400
    base = build_target_frame(prices, target_cfg, "close")["fwd_ret"].iloc[t]
    for rows in (slice(0, t), slice(t + 2, None)):
        p = _perturb_rows(prices, rows, rng)
        assert build_target_frame(p, target_cfg, "close")["fwd_ret"].iloc[t] == base
    for row in (t, t + 1):
        p = prices.copy()
        p.iloc[row, p.columns.get_loc("close")] *= 1.02
        assert build_target_frame(p, target_cfg, "close")["fwd_ret"].iloc[t] != base


@pytest.mark.parametrize("convention", ["next_open", "close"])
@pytest.mark.parametrize("horizon", [1, 3])
def test_label_end_date_is_the_last_price_the_label_uses(prices, convention, horizon, rng):
    """label_end_date[t] = date of row t + offset. Changing anything strictly after that date
    leaves the label unchanged; changing the exit price ON that date changes the forward return."""
    tcfg = {"horizon": horizon, "scheme": "three_class", "neutral_threshold": 0.001}
    off = label_end_offset(horizon, convention)
    base = build_target_frame(prices, tcfg, convention)
    n = len(prices)
    np.testing.assert_array_equal(base["label_end_date"].iloc[: n - off].to_numpy(), prices.index[off:].to_numpy())
    assert base["label_end_date"].iloc[n - off :].isna().all()
    assert base["label"].iloc[n - off :].isna().all() and base["label"].iloc[: n - off].notna().all()

    exit_col = "open" if convention == "next_open" else "close"
    for t in (300, 500, 700):
        end_pos = prices.index.get_loc(base["label_end_date"].iloc[t])
        after = build_target_frame(_perturb_rows(prices, slice(end_pos + 1, None), rng), tcfg, convention)
        assert after["fwd_ret"].iloc[t] == base["fwd_ret"].iloc[t]
        assert after["label"].iloc[t] == base["label"].iloc[t]
        p = prices.copy()
        p.iloc[end_pos, p.columns.get_loc(exit_col)] *= 1.05
        assert build_target_frame(p, tcfg, convention)["fwd_ret"].iloc[t] != base["fwd_ret"].iloc[t]


def test_labels_are_never_computed_without_their_future(prices, target_cfg):
    """With data truncated at day t the labels of the last `offset` rows are NaN, not guessed."""
    t = 500
    out = build_target_frame(prices.iloc[: t + 1], target_cfg, "next_open")
    assert out["label"].iloc[-2:].isna().all()
    assert out["label"].iloc[:-2].notna().all()


# ---------------------------------------------------------------------------------------------
# 3. chronological split with purging
# ---------------------------------------------------------------------------------------------

TRAIN_END, VAL_END = "2016-08-31", "2017-03-31"


@pytest.mark.parametrize("convention", ["next_open", "close"])
def test_chronological_split_is_ordered_disjoint_and_purged(prices, cfg, convention):
    """train < validation < test in time, no shared dates, and with purge=True no training label
    window reaches past train_end and no validation label window reaches past val_end."""
    data = _dataset(prices, cfg, convention)
    split = chronological_split(data, TRAIN_END, VAL_END, purge=True)
    tr, va, te = split.train, split.val, split.test
    assert len(tr) and len(va) and len(te)
    assert tr.max() < va.min() and va.max() < te.min()
    assert tr.max() <= pd.Timestamp(TRAIN_END) < va.min()
    assert va.max() <= pd.Timestamp(VAL_END) < te.min()
    assert not (set(tr) & set(va)) and not (set(va) & set(te)) and not (set(tr) & set(te))
    assert (data.loc[tr, "label_end_date"] <= pd.Timestamp(TRAIN_END)).all()
    assert (data.loc[va, "label_end_date"] <= pd.Timestamp(VAL_END)).all()
    for idx in (tr, va, te):
        assert data.loc[idx, "label"].notna().all()
        assert idx.is_monotonic_increasing


@pytest.mark.parametrize("convention", ["next_open", "close"])
def test_purge_removes_exactly_the_boundary_rows(prices, cfg, convention):
    """Without purging, the last `offset` rows before each boundary (whose labels straddle it) are
    present; purging removes exactly those rows and nothing else."""
    data = _dataset(prices, cfg, convention)
    off = label_end_offset(cfg["target"]["horizon"], convention)
    purged = chronological_split(data, TRAIN_END, VAL_END, purge=True)
    unpurged = chronological_split(data, TRAIN_END, VAL_END, purge=False)
    for name, end in (("train", TRAIN_END), ("val", VAL_END)):
        a, b = getattr(purged, name), getattr(unpurged, name)
        dropped = b.difference(a)
        assert set(a) <= set(b)
        assert len(dropped) == off  # both boundaries are trading days in the synthetic calendar
        assert (data.loc[dropped, "label_end_date"] > pd.Timestamp(end)).all()
        assert (dropped <= pd.Timestamp(end)).all()
        assert list(dropped) == list(b[-off:])  # the rows right before the boundary
    # unpurged splits partition every labelled row
    labelled = data.index[data["label"].notna()]
    assert unpurged.train.append(unpurged.val).append(unpurged.test).equals(labelled)
    assert purged.test.equals(unpurged.test)


def test_split_rejects_inverted_boundaries(prices, cfg):
    """train_end must precede val_end."""
    data = _dataset(prices, cfg)
    with pytest.raises(ValueError):
        chronological_split(data, VAL_END, TRAIN_END)


# ---------------------------------------------------------------------------------------------
# 4. walk-forward folds
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def long_data(cfg) -> pd.DataFrame:
    """Seven years of synthetic prices so that several expanding walk-forward folds exist."""
    raw = generate_synthetic_ohlcv("WALK", "2008-01-01", "2014-12-31", seed=11)
    clean, _ = validate_ohlcv(raw, "WALK")
    return _dataset(adjust_ohlc(clean), cfg)


@pytest.mark.parametrize("val_years", [1, 2])
def test_walk_forward_folds_are_ordered_purged_and_expanding(long_data, val_years):
    """Per fold: train.max < val.min < test.min, the test block is exactly calendar year Y, the
    validation block covers the val_years before Y, train/val are purged, and the training window
    expands (same start, superset of the previous fold's training rows)."""
    first = 2010 + val_years  # features are complete from early 2009 -> first fold trains on 2009
    folds = walk_forward_folds(long_data, first, 2014, val_years=val_years, purge=True, min_train_rows=200)
    assert [y for y, _ in folds] == list(range(first, 2015))
    labelled = long_data.index[long_data["label"].notna()]
    prev_train = None
    for year, split in folds:
        tr, va, te = split.train, split.val, split.test
        assert tr.max() < va.min() and va.max() < te.min()
        assert (te.year == year).all()
        assert te.equals(labelled[labelled.year == year])  # nothing in the test year is dropped
        assert ((va.year >= year - val_years) & (va.year <= year - 1)).all()
        assert (tr.year <= year - val_years - 1).all()
        train_end = pd.Timestamp(f"{year - val_years - 1}-12-31")
        val_end = pd.Timestamp(f"{year - 1}-12-31")
        assert (long_data.loc[tr, "label_end_date"] <= train_end).all()
        assert (long_data.loc[va, "label_end_date"] <= val_end).all()
        assert not (set(tr) & set(va)) and not (set(va) & set(te))
        if prev_train is not None:
            assert tr[0] == prev_train[0]
            assert set(prev_train) < set(tr)
        prev_train = tr


def test_walk_forward_skips_folds_with_too_little_training_data(long_data):
    """Folds whose expanding training window is shorter than min_train_rows are not produced."""
    all_folds = walk_forward_folds(long_data, 2010, 2014, val_years=1, min_train_rows=1)
    folds = walk_forward_folds(long_data, 2010, 2014, val_years=1, min_train_rows=500)
    years = [y for y, _ in folds]
    assert 2010 not in [y for y, _ in all_folds]  # no labelled row before 2009 -> empty training set
    assert years == [y for y, s in all_folds if len(s.train) >= 500]
    assert 0 < len(folds) < len(all_folds)


# ---------------------------------------------------------------------------------------------
# 5. LSTM sequences and scaler
# ---------------------------------------------------------------------------------------------


def test_build_sequences_windows_end_at_the_decision_row():
    """The window for end position p is exactly X[p-L+1 : p+1]; its last row is row p itself."""
    X = np.arange(30 * 3, dtype=float).reshape(30, 3)
    L = 5
    positions = np.array([4, 5, 13, 29])
    out = build_sequences(X, positions, L)
    assert out.shape == (len(positions), L, 3)
    for k, p in enumerate(positions):
        np.testing.assert_array_equal(out[k], X[p - L + 1 : p + 1])
        np.testing.assert_array_equal(out[k, -1], X[p])


def test_build_sequences_rejects_short_history_and_out_of_range():
    """A window that would start before row 0 (p < L-1) or end past the data is refused."""
    X = np.zeros((20, 2))
    with pytest.raises(ValueError):
        build_sequences(X, np.array([3]), seq_len=5)
    with pytest.raises(ValueError):
        build_sequences(X, np.array([4, 20]), seq_len=5)
    assert build_sequences(X, np.array([4]), seq_len=5).shape == (1, 5, 2)


def test_build_sequences_ignore_rows_after_p():
    """Changing rows after p cannot change the window ending at p."""
    X = np.random.default_rng(0).normal(size=(25, 4))
    before = build_sequences(X, np.array([10]), 6)
    X2 = X.copy()
    X2[11:] += 100.0
    np.testing.assert_array_equal(build_sequences(X2, np.array([10]), 6), before)


@pytest.fixture()
def lstm_problem(rng):
    """Small chronological problem: rows 0-39 train, 42-49 validation, 50-59 'test'."""
    n, f = 60, 3
    X = rng.normal(loc=[0.0, 5.0, -3.0], scale=[1.0, 2.0, 0.5], size=(n, f))
    X[40:] += np.array([3.0, -4.0, 2.0])  # later regime has a different mean
    y = rng.integers(0, 3, n).astype(float)
    return X, y, np.arange(0, 40), np.arange(42, 50)


def _tiny_lstm():
    from src.lstm_model import LSTMSequenceClassifier

    return LSTMSequenceClassifier(n_classes=3, seq_len=5, hidden_size=4, max_epochs=2, patience=2, batch_size=16, seed=0)


def test_lstm_scaler_fitted_on_training_rows_only(lstm_problem):
    """The StandardScaler mean/variance equal those of rows 0..last training position, not the
    full matrix (which includes validation/test rows)."""
    pytest.importorskip("torch")
    X, y, train_pos, val_pos = lstm_problem
    m = _tiny_lstm().fit(X, y, train_pos, val_pos)
    last = train_pos.max()
    np.testing.assert_allclose(m.scaler.mean_, X[: last + 1].mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(m.scaler.var_, X[: last + 1].var(axis=0), rtol=1e-12)
    assert m.scaler.n_samples_seen_ == last + 1
    assert not np.allclose(m.scaler.mean_, X.mean(axis=0))


def test_lstm_scaler_unchanged_by_rows_after_training(lstm_problem):
    """Rewriting every feature row after the last training row leaves the fitted scaler identical."""
    pytest.importorskip("torch")
    X, y, train_pos, val_pos = lstm_problem
    a = _tiny_lstm().fit(X, y, train_pos, val_pos)
    X2 = X.copy()
    X2[train_pos.max() + 1 :] = X2[train_pos.max() + 1 :] * 10.0 + 50.0
    b = _tiny_lstm().fit(X2, y, train_pos, val_pos)
    np.testing.assert_array_equal(a.scaler.mean_, b.scaler.mean_)
    np.testing.assert_array_equal(a.scaler.scale_, b.scaler.scale_)


def test_lstm_fit_ignores_rows_after_validation(lstm_problem):
    """Training uses train windows and early stopping uses validation windows only: changing the
    features and labels of the test rows (after the last validation row) yields identical weights."""
    torch = pytest.importorskip("torch")
    X, y, train_pos, val_pos = lstm_problem
    a = _tiny_lstm().fit(X, y, train_pos, val_pos)
    X2, y2 = X.copy(), y.copy()
    X2[val_pos.max() + 1 :] += 25.0
    y2[val_pos.max() + 1 :] = (y2[val_pos.max() + 1 :] + 1) % 3
    b = _tiny_lstm().fit(X2, y2, train_pos, val_pos)
    sa, sb = a.net.state_dict(), b.net.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"parameter {k} depends on post-validation rows"
    np.testing.assert_array_equal(a.predict_proba(X, val_pos), b.predict_proba(X2, val_pos))


def test_lstm_prediction_at_p_ignores_later_rows(lstm_problem):
    """At inference the probability for decision row p depends only on rows p-L+1..p."""
    pytest.importorskip("torch")
    X, y, train_pos, val_pos = lstm_problem
    m = _tiny_lstm().fit(X, y, train_pos, val_pos)
    p = 52
    X2 = X.copy()
    X2[p + 1 :] = -X2[p + 1 :] * 7.0
    np.testing.assert_array_equal(m.predict_proba(X, [p]), m.predict_proba(X2, [p]))
    X3 = X.copy()
    X3[p] += 5.0  # sanity: the decision row itself does matter
    assert not np.allclose(m.predict_proba(X, [p]), m.predict_proba(X3, [p]))


# ---------------------------------------------------------------------------------------------
# 6. baselines are causal
# ---------------------------------------------------------------------------------------------

BASELINES = {
    "ma_crossover": lambda c: ma_crossover_positions(c, 10, 50),
    "ma_crossover_short": lambda c: ma_crossover_positions(c, 10, 50, allow_short=True),
    "persistence": lambda c: persistence_positions(c),
    "persistence_short": lambda c: persistence_positions(c, allow_short=True),
}


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_positions_truncation_invariant(prices, name):
    """Positions computed on close[:cut] equal the full-sample positions on those dates."""
    rule = BASELINES[name]
    full = rule(prices["close"])
    for cut in [2, 30, 50, 51, 120, 333, 600, 781]:
        pd.testing.assert_series_equal(rule(prices["close"].iloc[:cut]), full.iloc[:cut])


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_positions_ignore_future_prices(prices, name, rng):
    """Rescaling closes strictly after t leaves all positions at dates <= t unchanged."""
    rule = BASELINES[name]
    t = 300
    full = rule(prices["close"])
    shocked = _perturb_rows(prices, slice(t + 1, None), rng, cols=["close"])
    pd.testing.assert_series_equal(rule(shocked["close"]).iloc[: t + 1], full.iloc[: t + 1])


def test_ma_crossover_is_flat_during_warmup(prices):
    """Before SMA_50 exists the MA rule holds no position (never an accidental long)."""
    pos = ma_crossover_positions(prices["close"], 10, 50, allow_short=True)
    assert (pos.iloc[:49] == 0.0).all()


# ---------------------------------------------------------------------------------------------
# 7. time-series cross-validation used for tuning
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_splits,gap", [(4, label_end_offset(1, "next_open")), (3, label_end_offset(1, "close")), (5, 4)])
def test_time_series_split_validates_only_on_later_rows(n_splits, gap):
    """sklearn TimeSeriesSplit(n_splits, gap) as configured in models.py: every validation index is
    later than every training index, and at least `gap` rows separate them, so no training label
    window (t .. t+gap) overlaps a validation decision row."""
    X = np.zeros((300, 2))
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        assert tr.max() < va.min()
        assert va.min() - tr.max() - 1 >= gap
        assert tr.max() + gap < va.min()
        assert tr[0] == 0 and np.all(np.diff(tr) == 1)  # expanding, contiguous past window


def test_grid_search_uses_purged_forward_splits(monkeypatch, rng):
    """Record the splits that time_series_grid_search actually iterates over: it must build
    TimeSeriesSplit with the requested gap, and never validate on rows older than its training rows."""
    calls = []

    class RecordingTSS(TimeSeriesSplit):
        def split(self, X, y=None, groups=None):
            for tr, va in super().split(X, y, groups):
                calls.append((self.n_splits, self.gap, tr, va))
                yield tr, va

    monkeypatch.setattr(models_module, "TimeSeriesSplit", RecordingTSS)
    X = rng.normal(size=(240, 3))
    y = rng.integers(0, 3, 240)
    gap = label_end_offset(1, "next_open")
    best, table = models_module.time_series_grid_search(
        "logistic_regression", X, y, {"C": [0.1, 1.0], "class_weight": [None]}, None, n_classes=3, n_splits=3, gap=gap
    )
    assert len(table) == 2 and best["C"] in (0.1, 1.0)
    assert len(calls) == 2 * 3  # 2 grid points x 3 folds
    for n_splits, used_gap, tr, va in calls:
        assert (n_splits, used_gap) == (3, gap)
        assert tr.max() < va.min()
        assert va.min() - tr.max() - 1 == gap
