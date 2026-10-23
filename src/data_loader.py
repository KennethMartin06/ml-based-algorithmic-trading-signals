"""Market-data acquisition (Yahoo Finance via yfinance), caching and validation.

Design rules
------------
* Real data comes from yfinance only. Downloads are cached in ``data/raw`` so experiments are
  reproducible and the same data is not fetched twice.
* If a download fails, the loader falls back to an existing cached file. It NEVER fabricates data
  silently: a deterministic synthetic series is produced only when the caller explicitly allows it,
  and the returned metadata (and every downstream report) is then flagged ``is_synthetic=True``.
* Prices are adjusted for splits and dividends using Yahoo's ``Adj Close``: the factor
  ``Adj Close / Close`` is applied to Open, High and Low so that all OHLC fields are on the same
  adjusted scale.
"""

from __future__ import annotations

import hashlib
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, resolve_path

LOGGER = get_logger()

RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class DataAcquisitionError(RuntimeError):
    """Raised when market data cannot be obtained from any allowed source."""


@dataclass
class DataMeta:
    ticker: str
    source: str  # "yfinance", "cache" or "synthetic"
    is_synthetic: bool
    rows: int = 0
    first_date: str = ""
    last_date: str = ""
    sha256: str = ""
    file: str = ""
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------------------------

def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise yfinance output to single-level columns.

    Recent yfinance versions return a ``(Price, Ticker)`` MultiIndex even for one ticker; older
    versions (or ``group_by='ticker'``) return ``(Ticker, Price)``. Both are handled.
    """
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            if ticker in set(values):
                df = df.xs(ticker, axis=1, level=level)
                break
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def download_ticker(ticker: str, start: str, end_inclusive: str, max_retries: int = 3) -> pd.DataFrame:
    """Download daily OHLCV (unadjusted + Adj Close) for one ticker from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # let load_ticker apply its cache / explicit-synthetic fallbacks
        raise DataAcquisitionError("yfinance is not installed (pip install yfinance)") from exc

    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise DataAcquisitionError(f"yfinance returned no rows for {ticker}")
            df = flatten_yfinance_columns(raw, ticker)
            missing = [c for c in RAW_COLUMNS if c not in df.columns]
            if "Adj Close" in missing and "Close" in df.columns:
                LOGGER.warning("%s: 'Adj Close' missing from yfinance output; prices are NOT adjusted", ticker)
                df["Adj Close"] = df["Close"]
                missing.remove("Adj Close")
            if missing:
                raise DataAcquisitionError(f"{ticker}: yfinance output missing columns {missing}")
            df = df[RAW_COLUMNS].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df.index.name = "Date"
            return df
        except Exception as exc:  # network errors, rate limits, schema changes
            last_error = exc
            LOGGER.warning("Download attempt %d/%d for %s failed: %s", attempt, max_retries, ticker, exc)
            time.sleep(min(2 ** attempt, 10))
    raise DataAcquisitionError(f"Could not download {ticker}: {last_error}")


def cache_file(raw_dir: str | Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return resolve_path(raw_dir) / f"{safe}_{start}_{end}.csv"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# validation and adjustment
# ---------------------------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame, ticker: str = "") -> tuple[pd.DataFrame, dict]:
    """Sort, de-duplicate and sanity-check raw OHLCV data.

    Rows with missing or non-positive prices are dropped (never imputed). High/Low values that are
    inconsistent with Open/Close are widened to the Open/Close range, which is a correction of a
    data error rather than imputation. Every action is counted in the returned report.
    """
    report: dict = {"input_rows": int(len(df))}
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    report["was_sorted"] = bool(out.index.is_monotonic_increasing)
    out = out.sort_index()

    dup_mask = out.index.duplicated(keep="last")
    report["duplicate_dates_removed"] = int(dup_mask.sum())
    out = out[~dup_mask]

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in out.columns]
    nan_mask = out[price_cols].isna().any(axis=1)
    report["rows_with_missing_prices_removed"] = int(nan_mask.sum())
    out = out[~nan_mask]

    nonpos_mask = (out[price_cols] <= 0).any(axis=1)
    report["rows_with_nonpositive_prices_removed"] = int(nonpos_mask.sum())
    out = out[~nonpos_mask]

    if "Volume" in out.columns:
        if (out["Volume"] < 0).any():
            raise DataAcquisitionError(f"{ticker}: negative volume found")
        report["missing_volume_rows"] = int(out["Volume"].isna().sum())
        report["zero_volume_rows"] = int((out["Volume"] == 0).sum())

    hi_bad = out["High"] < out[["Open", "Close"]].max(axis=1)
    lo_bad = out["Low"] > out[["Open", "Close"]].min(axis=1)
    report["high_low_inconsistencies_repaired"] = int((hi_bad | lo_bad).sum())
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)

    gaps = out.index.to_series().diff().dt.days
    report["max_calendar_gap_days"] = int(gaps.max()) if len(gaps) > 1 else 0
    report["gaps_over_7_days"] = int((gaps > 7).sum())
    report["output_rows"] = int(len(out))
    if report["gaps_over_7_days"]:
        LOGGER.warning("%s: %d calendar gaps longer than 7 days", ticker, report["gaps_over_7_days"])
    return out, report


def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Return lower-case adjusted OHLCV columns used by the rest of the pipeline.

    ``factor = Adj Close / Close`` captures split and dividend adjustments. Applying it to Open,
    High and Low keeps intraday relationships (e.g. Close/Open) unchanged while making returns
    across corporate actions correct. Volume is reported by Yahoo on a split-adjusted basis.
    """
    factor = df["Adj Close"] / df["Close"]
    out = pd.DataFrame(index=df.index)
    out["open"] = df["Open"] * factor
    out["high"] = df["High"] * factor
    out["low"] = df["Low"] * factor
    out["close"] = df["Adj Close"]
    out["volume"] = df["Volume"].astype(float)
    out["raw_close"] = df["Close"]
    out.index.name = "date"
    return out


# ---------------------------------------------------------------------------------------------
# synthetic fallback (development / unit tests only)
# ---------------------------------------------------------------------------------------------

def generate_synthetic_ohlcv(ticker: str, start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Deterministic regime-switching random walk with raw-yfinance-style columns.

    FOR DEVELOPMENT AND TESTING ONLY. It is not market data and results obtained on it must never be
    reported as market performance.
    """
    rng = np.random.default_rng(seed + zlib.crc32(ticker.encode()))
    dates = pd.bdate_range(start, end)
    n = len(dates)
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        regime[i] = regime[i - 1] if rng.random() > 0.01 else 1 - regime[i - 1]
    mu = np.where(regime == 0, 0.0005, -0.0003)
    sigma = np.where(regime == 0, 0.009, 0.02)
    log_ret = mu + sigma * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    overnight = 0.3 * sigma * rng.standard_normal(n)
    prev_close = np.concatenate([[100.0], close[:-1]])
    open_ = prev_close * np.exp(overnight)
    spread = np.abs(sigma * rng.standard_normal(n)) * close
    high = np.maximum(open_, close) + spread * 0.5
    low = np.minimum(open_, close) - spread * 0.5
    low = np.maximum(low, 0.01)
    volume = np.round(rng.lognormal(mean=15, sigma=0.3, size=n))
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume},
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


# ---------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------

def load_ticker(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str | Path = "data/raw",
    use_cache: bool = True,
    allow_synthetic: bool = False,
    force_synthetic: bool = False,
    max_retries: int = 3,
    seed: int = 0,
) -> tuple[pd.DataFrame, DataMeta]:
    """Load one ticker: cache -> yfinance -> (explicitly allowed) synthetic fallback.

    Returns the adjusted, validated OHLCV frame and a :class:`DataMeta` record.
    """
    ensure_dir(raw_dir)
    path = cache_file(raw_dir, ticker, start, end)

    if force_synthetic:
        raw = generate_synthetic_ohlcv(ticker, start, end, seed=seed)
        source = "synthetic"
        path = cache_file(Path(raw_dir) / "synthetic", ticker, start, end)
        ensure_dir(path.parent)
        raw.to_csv(path)
    elif use_cache and path.exists():
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        source = "cache"
    else:
        try:
            raw = download_ticker(ticker, start, end, max_retries=max_retries)
            raw.to_csv(path)
            source = "yfinance"
        except DataAcquisitionError as exc:
            if path.exists():
                LOGGER.warning("%s: download failed (%s); using cached file %s", ticker, exc, path.name)
                raw = pd.read_csv(path, index_col=0, parse_dates=True)
                source = "cache"
            elif allow_synthetic:
                LOGGER.error("%s: download failed and no cache exists. Using SYNTHETIC fallback data.", ticker)
                raw = generate_synthetic_ohlcv(ticker, start, end, seed=seed)
                source = "synthetic"
                path = cache_file(Path(raw_dir) / "synthetic", ticker, start, end)
                ensure_dir(path.parent)
                raw.to_csv(path)
            else:
                raise DataAcquisitionError(
                    f"{ticker}: download failed and no cached data at {path}. "
                    "Re-run with network access, or pass --synthetic for a clearly labelled dev run."
                ) from exc

    clean, report = validate_ohlcv(raw, ticker)
    adjusted = adjust_ohlc(clean)
    meta = DataMeta(
        ticker=ticker,
        source=source,
        is_synthetic=source == "synthetic",
        rows=len(adjusted),
        first_date=adjusted.index[0].strftime("%Y-%m-%d"),
        last_date=adjusted.index[-1].strftime("%Y-%m-%d"),
        sha256=file_sha256(path),
        file=path.relative_to(resolve_path(".")).as_posix() if path.is_relative_to(resolve_path(".")) else str(path),
        validation=report,
    )
    LOGGER.info("%s: %d rows %s -> %s (source=%s)", ticker, meta.rows, meta.first_date, meta.last_date, source)
    return adjusted, meta


def drop_unfinished_session(prices: pd.DataFrame, now_utc: pd.Timestamp | None = None) -> tuple[pd.DataFrame, bool]:
    """Drop today's bar while the US session is still open (before 16:15 New York time).

    During trading hours Yahoo's daily bar is incomplete - its "close" is only the latest price - so
    it must not be used as a completed day when computing features or labels. Returns the frame and
    whether a bar was removed.
    """
    now_ny = (now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")).tz_convert("America/New_York")
    session_closed = now_ny.hour > 16 or (now_ny.hour == 16 and now_ny.minute >= 15)
    if len(prices) and prices.index[-1].date() >= now_ny.date() and not session_closed:
        return prices.iloc[:-1], True
    return prices, False


def load_universe(cfg: dict, tickers: list[str] | None = None, force_synthetic: bool = False):
    """Load every configured ticker. Returns ``({ticker: frame}, [DataMeta dicts])``."""
    dcfg = cfg["data"]
    frames, manifest = {}, []
    for ticker in tickers or dcfg["tickers"]:
        df, meta = load_ticker(
            ticker,
            dcfg["start"],
            dcfg["end"],
            raw_dir=dcfg.get("raw_dir", "data/raw"),
            use_cache=dcfg.get("use_cache", True),
            allow_synthetic=dcfg.get("allow_synthetic_fallback", False),
            force_synthetic=force_synthetic,
            max_retries=dcfg.get("max_retries", 3),
            seed=cfg["experiment"].get("seed", 0),
        )
        frames[ticker] = df
        manifest.append(meta.to_dict())
    return frames, manifest
