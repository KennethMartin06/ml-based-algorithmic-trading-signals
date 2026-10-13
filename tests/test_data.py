"""Tests for src/data_loader.py: column normalisation, validation, adjustment, synthetic data and
the cache -> yfinance -> synthetic acquisition chain.

No test touches the network or the project's real ``data/raw`` directory:
* an autouse fixture replaces the ``yfinance`` module with a stub that fails loudly if called;
* every ``load_ticker`` call uses pytest's ``tmp_path`` as ``raw_dir``;
* tests of ``load_ticker`` replace ``src.data_loader.download_ticker`` with an explicit fake.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data_loader as dl
from src.data_loader import (
    RAW_COLUMNS,
    DataAcquisitionError,
    adjust_ohlc,
    cache_file,
    file_sha256,
    flatten_yfinance_columns,
    generate_synthetic_ohlcv,
    load_ticker,
    load_universe,
    validate_ohlcv,
)
from src.utils import deep_update

TICKER = "XYZ"
START, END = "2020-01-01", "2020-03-31"


# ---------------------------------------------------------------------------------------------
# helpers and fixtures
# ---------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Install a stub ``yfinance`` module that fails if any test tries to download, and make the
    retry back-off instantaneous."""
    stub = types.ModuleType("yfinance")

    def _forbidden(*args, **kwargs):
        raise AssertionError("network access attempted in a unit test")

    stub.download = _forbidden
    monkeypatch.setitem(sys.modules, "yfinance", stub)
    monkeypatch.setattr(dl.time, "sleep", lambda *_: None)
    return stub


def make_raw(dates, close=None, adj_factor=1.0) -> pd.DataFrame:
    """Small, internally consistent raw (yfinance-style) OHLCV frame."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    n = len(idx)
    close = np.asarray(close if close is not None else 100.0 + np.arange(n), dtype=float)
    open_ = close - 0.5
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + 1.0,
            "Low": np.minimum(open_, close) - 1.0,
            "Close": close,
            "Adj Close": close * adj_factor,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def fake_download_frame() -> pd.DataFrame:
    """What a successful (fake) download returns: business days, with a 2% dividend adjustment so
    it is distinguishable from the synthetic generator's output (where Adj Close == Close)."""
    dates = pd.bdate_range(START, END)
    return make_raw(dates, adj_factor=0.98)


def failing_download(*args, **kwargs):
    raise DataAcquisitionError("simulated network failure")


def must_not_download(*args, **kwargs):
    raise AssertionError("download_ticker must not be called here")


# ---------------------------------------------------------------------------------------------
# flatten_yfinance_columns
# ---------------------------------------------------------------------------------------------

def _values_frame(tickers=("SPY",)) -> dict[str, np.ndarray]:
    """Distinct values per (ticker, field) so that a wrong selection is detectable."""
    return {t: np.arange(len(RAW_COLUMNS) * 3, dtype=float).reshape(3, -1) + 100 * k
            for k, t in enumerate(tickers)}


def test_flatten_price_ticker_multiindex():
    """yfinance >= 0.2.48 style (Price, Ticker) columns collapse to the Price level."""
    vals = _values_frame(("SPY",))["SPY"]
    cols = pd.MultiIndex.from_product([RAW_COLUMNS, ["SPY"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(vals, columns=cols)
    out = flatten_yfinance_columns(raw, "SPY")
    assert list(out.columns) == RAW_COLUMNS
    np.testing.assert_array_equal(out.to_numpy(), vals)


def test_flatten_ticker_price_multiindex():
    """group_by='ticker' style (Ticker, Price) columns collapse to the Price level."""
    vals = _values_frame(("SPY",))["SPY"]
    cols = pd.MultiIndex.from_product([["SPY"], RAW_COLUMNS], names=["Ticker", "Price"])
    raw = pd.DataFrame(vals, columns=cols)
    out = flatten_yfinance_columns(raw, "SPY")
    assert list(out.columns) == RAW_COLUMNS
    np.testing.assert_array_equal(out.to_numpy(), vals)


@pytest.mark.parametrize("order", ["price_ticker", "ticker_price"])
def test_flatten_selects_requested_ticker_from_multi_ticker_frame(order):
    """With several tickers in one frame, only the requested ticker's columns are returned."""
    vals = _values_frame(("SPY", "AAPL"))
    if order == "price_ticker":
        tuples = [(c, t) for c in RAW_COLUMNS for t in ("SPY", "AAPL")]
        data = np.column_stack([vals[t][:, i] for i, _ in enumerate(RAW_COLUMNS) for t in ("SPY", "AAPL")])
    else:
        tuples = [(t, c) for t in ("SPY", "AAPL") for c in RAW_COLUMNS]
        data = np.column_stack([vals[t][:, i] for t in ("SPY", "AAPL") for i, _ in enumerate(RAW_COLUMNS)])
    raw = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(tuples))
    out = flatten_yfinance_columns(raw, "AAPL")
    assert list(out.columns) == RAW_COLUMNS
    np.testing.assert_array_equal(out.to_numpy(), vals["AAPL"])


def test_flatten_flat_columns_unchanged_and_duplicates_dropped():
    """Single-level columns pass through; a duplicated column name keeps its first occurrence."""
    raw = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["Open", "Close", "Open"])
    out = flatten_yfinance_columns(raw, "SPY")
    assert list(out.columns) == ["Open", "Close"]
    assert out.iloc[0].tolist() == [1.0, 2.0]

    flat = pd.DataFrame(np.ones((2, len(RAW_COLUMNS))), columns=RAW_COLUMNS)
    pd.testing.assert_frame_equal(flatten_yfinance_columns(flat, "SPY"), flat)


def test_flatten_multiindex_without_ticker_falls_back_to_first_level():
    """If the ticker label is absent (e.g. renamed index symbol) the first level is used."""
    cols = pd.MultiIndex.from_product([RAW_COLUMNS, ["^GSPC"]])
    raw = pd.DataFrame(np.ones((2, len(RAW_COLUMNS))), columns=cols)
    out = flatten_yfinance_columns(raw, "SPX")
    assert list(out.columns) == RAW_COLUMNS


# ---------------------------------------------------------------------------------------------
# download_ticker (with the yfinance module stubbed; no network)
# ---------------------------------------------------------------------------------------------

def _install_fake_yf(monkeypatch, frame_factory, calls: list):
    stub = types.ModuleType("yfinance")

    def download(ticker, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        return frame_factory()

    stub.download = download
    monkeypatch.setitem(sys.modules, "yfinance", stub)


@pytest.mark.parametrize("tz", [None, "America/New_York"])
def test_download_ticker_normalises_yfinance_output(monkeypatch, tz):
    """download_ticker requests unadjusted data with an exclusive end one day after the inclusive
    config end, flattens MultiIndex columns and returns a tz-naive, midnight-normalised index."""
    idx = pd.date_range("2020-01-02", periods=3, freq="D", tz=tz)
    raw = make_raw(idx)
    raw.columns = pd.MultiIndex.from_product([raw.columns, [TICKER]], names=["Price", "Ticker"])
    calls: list = []
    _install_fake_yf(monkeypatch, lambda: raw, calls)

    out = dl.download_ticker(TICKER, "2020-01-01", "2020-01-31", max_retries=1)

    assert len(calls) == 1
    assert calls[0]["start"] == "2020-01-01"
    assert calls[0]["end"] == "2020-02-01"  # end_inclusive + 1 day
    assert calls[0]["auto_adjust"] is False
    assert list(out.columns) == RAW_COLUMNS
    assert out.index.tz is None
    assert out.index.name == "Date"
    assert list(out.index) == list(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]))


def test_download_ticker_missing_adj_close_falls_back_to_close(monkeypatch):
    """When Yahoo omits 'Adj Close' the loader copies Close (and logs that prices are unadjusted)."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=3)).drop(columns="Adj Close")
    _install_fake_yf(monkeypatch, lambda: raw, [])
    out = dl.download_ticker(TICKER, "2020-01-01", "2020-01-31", max_retries=1)
    assert list(out.columns) == RAW_COLUMNS
    np.testing.assert_array_equal(out["Adj Close"].to_numpy(), out["Close"].to_numpy())


@pytest.mark.parametrize(
    "factory",
    [lambda: pd.DataFrame(), lambda: make_raw(pd.bdate_range("2020-01-01", periods=3)).drop(columns="Volume")],
    ids=["empty", "missing_volume"],
)
def test_download_ticker_retries_then_raises(monkeypatch, factory):
    """Empty or schema-broken responses are retried ``max_retries`` times, then raise
    DataAcquisitionError (never return partial data)."""
    calls: list = []
    _install_fake_yf(monkeypatch, factory, calls)
    with pytest.raises(DataAcquisitionError):
        dl.download_ticker(TICKER, "2020-01-01", "2020-01-31", max_retries=3)
    assert len(calls) == 3


# ---------------------------------------------------------------------------------------------
# validate_ohlcv
# ---------------------------------------------------------------------------------------------

def test_validate_sorts_unsorted_input():
    """Out-of-order dates are sorted ascending and the report records that they were unsorted."""
    raw = make_raw(["2020-01-03", "2020-01-01", "2020-01-02"], close=[103.0, 101.0, 102.0])
    out, report = validate_ohlcv(raw, TICKER)
    assert out.index.is_monotonic_increasing
    assert out["Close"].tolist() == [101.0, 102.0, 103.0]
    assert report["was_sorted"] is False

    _, report_sorted = validate_ohlcv(out, TICKER)
    assert report_sorted["was_sorted"] is True


def test_validate_removes_duplicate_dates_keeping_last():
    """A repeated date keeps only its last record (the most recent revision of that bar)."""
    raw = make_raw(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
                   close=[100.0, 50.0, 102.0, 103.0])
    out, report = validate_ohlcv(raw, TICKER)
    assert report["duplicate_dates_removed"] == 1
    assert not out.index.has_duplicates
    assert out.loc[pd.Timestamp("2020-01-02"), "Close"] == 102.0
    assert len(out) == 3


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Adj Close"])
def test_validate_drops_rows_with_missing_prices(column):
    """A NaN in any price column removes the whole row (no imputation) and is counted."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=5))
    raw.iloc[2, raw.columns.get_loc(column)] = np.nan
    out, report = validate_ohlcv(raw, TICKER)
    assert report["rows_with_missing_prices_removed"] == 1
    assert raw.index[2] not in out.index
    assert len(out) == 4
    assert not out[["Open", "High", "Low", "Close", "Adj Close"]].isna().any().any()


def test_validate_drops_rows_with_non_positive_prices():
    """Zero or negative prices are data errors: those rows are dropped and counted."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=5))
    raw.iloc[1, raw.columns.get_loc("Close")] = 0.0
    raw.iloc[3, raw.columns.get_loc("Low")] = -1.0
    out, report = validate_ohlcv(raw, TICKER)
    assert report["rows_with_nonpositive_prices_removed"] == 2
    assert list(out.index) == [raw.index[0], raw.index[2], raw.index[4]]
    assert report["input_rows"] == 5
    assert report["output_rows"] == 3


def test_validate_repairs_high_low_inconsistencies():
    """High below max(Open, Close) or Low above min(Open, Close) is widened to the Open/Close
    range; consistent rows are left untouched."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=3), close=[100.0, 101.0, 102.0])
    # Row 1: High (95) below Close (101); row 2: Low (110) above Open (101.5).
    raw.iloc[1, raw.columns.get_loc("High")] = 95.0
    raw.iloc[2, raw.columns.get_loc("Low")] = 110.0
    out, report = validate_ohlcv(raw, TICKER)

    assert report["high_low_inconsistencies_repaired"] == 2
    assert out["High"].iloc[1] == 101.0                      # max(Open=100.5, Close=101)
    assert out["Low"].iloc[2] == 101.5                        # min(Open=101.5, Close=102)
    assert out["High"].iloc[0] == raw["High"].iloc[0]         # untouched
    assert out["Low"].iloc[0] == raw["Low"].iloc[0]
    assert (out["High"] >= out[["Open", "Close"]].max(axis=1)).all()
    assert (out["Low"] <= out[["Open", "Close"]].min(axis=1)).all()


def test_validate_raises_on_negative_volume():
    """Negative volume cannot be repaired sensibly, so validation refuses the data."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=3))
    raw.iloc[1, raw.columns.get_loc("Volume")] = -5.0
    with pytest.raises(DataAcquisitionError):
        validate_ohlcv(raw, TICKER)


def test_validate_counts_but_keeps_missing_and_zero_volume():
    """Missing or zero volume is reported but does not remove a price bar."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=4))
    raw.iloc[1, raw.columns.get_loc("Volume")] = np.nan
    raw.iloc[2, raw.columns.get_loc("Volume")] = 0.0
    out, report = validate_ohlcv(raw, TICKER)
    assert len(out) == 4
    assert report["missing_volume_rows"] == 1
    assert report["zero_volume_rows"] == 1


def test_validate_reports_calendar_gaps():
    """Calendar gaps between bars are measured; gaps longer than 7 days are counted."""
    raw = make_raw(["2020-01-01", "2020-01-02", "2020-01-20", "2020-01-21"])
    _, report = validate_ohlcv(raw, TICKER)
    assert report["max_calendar_gap_days"] == 18
    assert report["gaps_over_7_days"] == 1


def test_validate_does_not_mutate_input_and_clean_data_passes(raw_ohlcv):
    """Clean synthetic data passes with nothing removed or repaired, and the input is not modified."""
    before = raw_ohlcv.copy()
    out, report = validate_ohlcv(raw_ohlcv, "TEST")
    pd.testing.assert_frame_equal(raw_ohlcv, before)
    assert report["output_rows"] == report["input_rows"] == len(raw_ohlcv)
    assert report["duplicate_dates_removed"] == 0
    assert report["rows_with_missing_prices_removed"] == 0
    assert report["rows_with_nonpositive_prices_removed"] == 0
    assert report["high_low_inconsistencies_repaired"] == 0
    pd.testing.assert_frame_equal(out, raw_ohlcv, check_freq=False)


# ---------------------------------------------------------------------------------------------
# adjust_ohlc
# ---------------------------------------------------------------------------------------------

def test_adjust_ohlc_applies_adj_close_factor():
    """factor = Adj Close / Close multiplies Open/High/Low; close becomes Adj Close; raw close kept."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=4), close=[100.0, 102.0, 104.0, 106.0])
    raw["Adj Close"] = raw["Close"] * np.array([0.5, 0.5, 1.0, 1.0])  # e.g. a 2:1 split
    out = adjust_ohlc(raw)

    factor = (raw["Adj Close"] / raw["Close"]).to_numpy()
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "raw_close"]
    assert out.index.name == "date"
    np.testing.assert_allclose(out["open"], raw["Open"].to_numpy() * factor)
    np.testing.assert_allclose(out["high"], raw["High"].to_numpy() * factor)
    np.testing.assert_allclose(out["low"], raw["Low"].to_numpy() * factor)
    np.testing.assert_allclose(out["close"], raw["Adj Close"].to_numpy())
    np.testing.assert_allclose(out["raw_close"], raw["Close"].to_numpy())
    assert out["volume"].dtype == float


def test_adjust_ohlc_preserves_intraday_ratios():
    """Adjustment rescales a whole bar, so Close/Open, High/Low and High/Close ratios are unchanged."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=5), close=[50.0, 52.0, 51.0, 55.0, 54.0])
    raw["Adj Close"] = raw["Close"] * np.array([0.25, 0.4, 0.9, 0.97, 1.0])
    out = adjust_ohlc(raw)
    np.testing.assert_allclose(out["close"] / out["open"], raw["Close"] / raw["Open"], rtol=1e-12)
    np.testing.assert_allclose(out["high"] / out["low"], raw["High"] / raw["Low"], rtol=1e-12)
    np.testing.assert_allclose(out["high"] / out["close"], raw["High"] / raw["Close"], rtol=1e-12)


def test_adjust_ohlc_identity_when_no_corporate_actions():
    """With Adj Close == Close the adjusted OHLC equal the raw OHLC."""
    raw = make_raw(pd.bdate_range("2020-01-01", periods=3))
    out = adjust_ohlc(raw)
    for low_name, raw_name in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")]:
        np.testing.assert_allclose(out[low_name], raw[raw_name])


# ---------------------------------------------------------------------------------------------
# generate_synthetic_ohlcv
# ---------------------------------------------------------------------------------------------

def test_synthetic_is_deterministic_for_same_seed_and_ticker():
    """Same (ticker, seed) -> identical frame, so dev runs are reproducible."""
    a = generate_synthetic_ohlcv("AAA", START, END, seed=5)
    b = generate_synthetic_ohlcv("AAA", START, END, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_differs_across_tickers_and_seeds():
    """Different tickers (or seeds) get different paths, so synthetic tickers are not clones."""
    a = generate_synthetic_ohlcv("AAA", START, END, seed=5)
    b = generate_synthetic_ohlcv("BBB", START, END, seed=5)
    c = generate_synthetic_ohlcv("AAA", START, END, seed=6)
    assert not np.allclose(a["Close"], b["Close"])
    assert not np.allclose(a["Close"], c["Close"])


def test_synthetic_has_raw_schema_and_valid_bars():
    """Synthetic data mimics raw yfinance output: business days, RAW_COLUMNS, positive and
    internally consistent OHLC, non-negative volume."""
    df = generate_synthetic_ohlcv("AAA", START, END, seed=0)
    assert list(df.columns) == RAW_COLUMNS
    assert df.index.equals(pd.DatetimeIndex(pd.bdate_range(START, END), name="Date"))
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
    assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
    assert (df["Volume"] >= 0).all()


# ---------------------------------------------------------------------------------------------
# cache_file / file_sha256
# ---------------------------------------------------------------------------------------------

def test_cache_file_name_is_filesystem_safe(tmp_path):
    """Index symbols such as ^GSPC are mapped to a safe file name inside raw_dir."""
    p = cache_file(tmp_path, "^GSPC", START, END)
    assert p == tmp_path / f"IDX_GSPC_{START}_{END}.csv"
    assert cache_file(tmp_path, "BRK/B", START, END).name == f"BRK_B_{START}_{END}.csv"


def test_file_sha256_matches_hashlib(tmp_path):
    """The recorded checksum is the standard SHA-256 of the file's bytes."""
    p = tmp_path / "x.csv"
    p.write_bytes(b"Date,Close\n2020-01-01,1.0\n")
    assert file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# load_ticker: cache -> yfinance -> synthetic chain (raw_dir = tmp_path)
# ---------------------------------------------------------------------------------------------

def _assert_adjusted_equals(frame: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Loaded frame equals the validated+adjusted version of ``raw`` (CSV round-trip tolerance)."""
    expected = adjust_ohlc(validate_ohlcv(raw)[0])
    pd.testing.assert_frame_equal(frame, expected, check_freq=False, rtol=1e-12)


def test_load_ticker_cache_hit_does_not_download(tmp_path, monkeypatch):
    """(a) An existing cache file is used directly: source='cache' and no download is attempted."""
    raw = fake_download_frame()
    path = cache_file(tmp_path, TICKER, START, END)
    raw.to_csv(path)
    monkeypatch.setattr(dl, "download_ticker", must_not_download)

    frame, meta = load_ticker(TICKER, START, END, raw_dir=tmp_path)

    assert meta.source == "cache"
    assert meta.is_synthetic is False
    assert meta.sha256 == file_sha256(path)
    assert meta.rows == len(raw)
    assert meta.first_date == raw.index[0].strftime("%Y-%m-%d")
    assert meta.last_date == raw.index[-1].strftime("%Y-%m-%d")
    _assert_adjusted_equals(frame, raw)


def test_load_ticker_failure_without_cache_raises(tmp_path, monkeypatch):
    """(b) Download failure, no cache, synthetic not allowed -> DataAcquisitionError; nothing is
    written, so no fabricated data can appear silently."""
    monkeypatch.setattr(dl, "download_ticker", failing_download)
    with pytest.raises(DataAcquisitionError):
        load_ticker(TICKER, START, END, raw_dir=tmp_path, allow_synthetic=False)
    assert list(tmp_path.rglob("*.csv")) == []


def test_load_ticker_failure_with_synthetic_allowed_is_flagged(tmp_path, monkeypatch):
    """(c) Download failure + allow_synthetic -> deterministic synthetic data flagged
    is_synthetic=True / source='synthetic', stored outside the real cache location."""
    monkeypatch.setattr(dl, "download_ticker", failing_download)

    frame, meta = load_ticker(TICKER, START, END, raw_dir=tmp_path, allow_synthetic=True, seed=3)

    assert meta.source == "synthetic"
    assert meta.is_synthetic is True
    synth_path = cache_file(tmp_path / "synthetic", TICKER, START, END)
    assert synth_path.exists()
    assert Path(meta.file).resolve() == synth_path.resolve()
    assert meta.sha256 == file_sha256(synth_path)
    assert not cache_file(tmp_path, TICKER, START, END).exists()  # real cache not polluted
    _assert_adjusted_equals(frame, generate_synthetic_ohlcv(TICKER, START, END, seed=3))


def test_synthetic_fallback_never_masquerades_as_cached_real_data(tmp_path, monkeypatch):
    """After a synthetic fallback, a later successful run downloads real data (source='yfinance')
    instead of picking up the synthetic file as a 'cache' hit."""
    monkeypatch.setattr(dl, "download_ticker", failing_download)
    load_ticker(TICKER, START, END, raw_dir=tmp_path, allow_synthetic=True)

    monkeypatch.setattr(dl, "download_ticker", lambda *a, **k: fake_download_frame())
    _, meta = load_ticker(TICKER, START, END, raw_dir=tmp_path)
    assert meta.source == "yfinance"
    assert meta.is_synthetic is False


def test_load_ticker_force_synthetic(tmp_path, monkeypatch):
    """(d) force_synthetic skips both the cache and the downloader and flags the result synthetic."""
    real_cache = cache_file(tmp_path, TICKER, START, END)
    fake_download_frame().to_csv(real_cache)
    digest_before = file_sha256(real_cache)
    monkeypatch.setattr(dl, "download_ticker", must_not_download)

    frame, meta = load_ticker(TICKER, START, END, raw_dir=tmp_path, force_synthetic=True, seed=11)

    assert meta.source == "synthetic"
    assert meta.is_synthetic is True
    assert file_sha256(real_cache) == digest_before  # real cache left untouched
    _assert_adjusted_equals(frame, generate_synthetic_ohlcv(TICKER, START, END, seed=11))


def test_load_ticker_successful_download_is_cached_with_sha256(tmp_path, monkeypatch):
    """(e) A successful download is written to the cache file, its SHA-256 is recorded, and the
    next call is served from that cache with identical data."""
    calls: list = []

    def fake_download(ticker, start, end_inclusive, max_retries=3):
        calls.append((ticker, start, end_inclusive, max_retries))
        return fake_download_frame()

    monkeypatch.setattr(dl, "download_ticker", fake_download)
    frame, meta = load_ticker(TICKER, START, END, raw_dir=tmp_path, max_retries=2)

    path = cache_file(tmp_path, TICKER, START, END)
    assert calls == [(TICKER, START, END, 2)]
    assert meta.source == "yfinance"
    assert meta.is_synthetic is False
    assert path.exists()
    assert meta.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(meta.sha256) == 64
    assert meta.validation["output_rows"] == len(frame)
    _assert_adjusted_equals(frame, fake_download_frame())

    monkeypatch.setattr(dl, "download_ticker", must_not_download)
    frame2, meta2 = load_ticker(TICKER, START, END, raw_dir=tmp_path)
    assert meta2.source == "cache"
    assert meta2.sha256 == meta.sha256
    pd.testing.assert_frame_equal(frame2, frame, check_freq=False, rtol=1e-12)


def test_load_ticker_refresh_failure_falls_back_to_existing_cache(tmp_path, monkeypatch):
    """With use_cache=False the loader tries to refresh; if that fails it uses the cached file
    (source='cache') rather than synthetic data or an error."""
    raw = fake_download_frame()
    raw.to_csv(cache_file(tmp_path, TICKER, START, END))
    monkeypatch.setattr(dl, "download_ticker", failing_download)

    frame, meta = load_ticker(TICKER, START, END, raw_dir=tmp_path, use_cache=False, allow_synthetic=True)
    assert meta.source == "cache"
    assert meta.is_synthetic is False
    _assert_adjusted_equals(frame, raw)


def test_load_universe_passes_config_through(cfg, tmp_path, monkeypatch):
    """load_universe loads every requested ticker with the configured raw_dir/dates and returns
    one manifest entry per ticker."""
    monkeypatch.setattr(dl, "download_ticker", must_not_download)
    local_cfg = deep_update(cfg, {"data": {"raw_dir": str(tmp_path), "start": START, "end": END}})

    frames, manifest = load_universe(local_cfg, tickers=["AAA", "BBB"], force_synthetic=True)

    assert list(frames) == ["AAA", "BBB"]
    assert [m["ticker"] for m in manifest] == ["AAA", "BBB"]
    assert all(m["is_synthetic"] and m["source"] == "synthetic" for m in manifest)
    assert all(Path(m["file"]).resolve().is_relative_to(tmp_path.resolve()) for m in manifest)
    seed = cfg["experiment"]["seed"]
    _assert_adjusted_equals(frames["AAA"], generate_synthetic_ohlcv("AAA", START, END, seed=seed))
