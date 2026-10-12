"""Shared primitives for the conformal-selective-uq substrate.

The market-data layer: a fixed, published universe of liquid US large caps, a
trading calendar, point-in-time stamping and split handling. Every input is
public daily bar data from yfinance.

Conventions (point-in-time correctness is the whole game):
  * `known_at` = the bar date's 16:00 ET close, DST-correct via America/New_York,
    stored UTC. No-lookahead is `known_at <= decision_time`, never a hoped-for lag.
  * Trading calendar = XNYS (via the optional exchange_calendars package, else a
    weekday calendar) so a missing bar is disambiguated holiday/weekend
    (expected) vs a real coverage hole.
  * yfinance `auto_adjust=False` still returns split-back-adjusted OHLC; we UN-ADJUST
    to as-reported/raw using yfinance's own split column (self-consistent inversion)
    so no future-split lookahead is baked into a level feature, and derive
    split-adjusted RETURNS separately (the future-split factor cancels in a ratio).

NOT run directly; imported by panel.py / synthetic.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional: exchange_calendars gives the XNYS calendar; else a weekday calendar
    import exchange_calendars as xc  # type: ignore

    _HAVE_XC = True
except Exception:  # noqa: BLE001
    _HAVE_XC = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROC = DATA / "processed"
for _d in (RAW, INTERIM, PROC):
    _d.mkdir(parents=True, exist_ok=True)

# Persisted artifacts (all gitignored — *.parquet).
PRICES = INTERIM / "public_prices_raw.parquet"  # raw (as-reported) OHLCV + known_at
RETURNS = INTERIM / "public_returns.parquet"  # split-adj daily returns substrate
PANEL = PROC / "public_pit_panel.parquet"  # the PIT feature panel (panel.py output)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` through a temp file and an atomic rename.

    Drivers reuse caches by filename, so a run that fails part-way must never leave
    a partial or broken file under the real name. The temp name carries the PID, so
    two processes building the same cache do not write over each other.
    """
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# Window: a multi-year public span covering a clear bull leg + a drawdown, so the
# train-pre-drawdown -> test-drawdown shift (G-SHIFT) is real, not synthetic.
WIN_LO = "2018-01-01"
WIN_HI = "2024-12-31"
PULL_END = "2025-01-02"  # yfinance `end` is exclusive

NY = "America/New_York"


# --------------------------------------------------------------------------- #
# Transparent public universe                                                 #
# --------------------------------------------------------------------------- #
# A FIXED, published cross-section of liquid US large caps grouped by GICS sector:
# the demonstration universe. Membership is hardcoded, so it carries a known
# survivorship bias, stated openly. The semi-synthetic target (Layer 1a) is built
# on this universe, so survivorship does not threaten the *method* claims; the real
# forward-return task (Layer 1b) is a check on real data, not a backtest. Cap tier
# is derived PIT from trailing dollar volume, not a fundamentals feed, so the panel
# needs no paid data.
SECTORS: dict[str, list[str]] = {
    "tech": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN",
             "AMD", "INTC", "TXN", "QCOM", "IBM"],
    "comm": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
    "discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX"],
    "staples": ["WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "MO"],
    "health": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
    "financials": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "C"],
    "industrials": ["CAT", "HON", "UNP", "BA", "GE", "RTX", "DE", "LMT", "UPS", "MMM"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "materials": ["LIN", "APD", "SHW", "FCX", "NEM", "DOW"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "realestate": ["AMT", "PLD", "EQIX", "PSA", "O"],
}

# Public factor proxies (ETFs and indices), downloaded with the universe. SPY is the
# market leg; QQQ, MTUM and IWM feed the factor-beta features in panel.py.
# (ticker, role, is_yield)
FACTOR_PROXIES: list[tuple[str, str, bool]] = [
    ("SPY", "broad_market", False),
    ("QQQ", "broad_growth", False),
    ("IWM", "small_cap", False),
    ("MTUM", "momentum_factor", False),
    ("USMV", "lowvol_factor", False),
    ("VLUE", "value_factor", False),
    ("^VIX", "volatility", False),
    ("^TNX", "rates_10y", True),
]
MARKET_TICKER = "SPY"  # market-model market leg


def universe() -> pd.DataFrame:
    """One row per name: ticker, sector. The transparent public cross-section."""
    rows = [{"ticker": t, "sector": sec} for sec, ts in SECTORS.items() for t in ts]
    return pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)


def all_tickers() -> list[str]:
    """Universe names + factor proxies, de-duplicated, for the price pull."""
    names = universe()["ticker"].tolist()
    facs = [t for t, _r, _y in FACTOR_PROXIES]
    return sorted(set(names) | set(facs))


# --------------------------------------------------------------------------- #
# trading calendar + bitemporal stamping                                      #
# --------------------------------------------------------------------------- #
def nyse_sessions(lo: str = WIN_LO, hi: str = WIN_HI) -> pd.DatetimeIndex:
    """Tz-naive XNYS session dates in [lo, hi] (weekday fallback if no xcals)."""
    if _HAVE_XC:
        return xc.get_calendar("XNYS").sessions_in_range(lo, hi).tz_localize(None)
    rng = pd.bdate_range(lo, hi)
    return pd.DatetimeIndex(rng)


def known_at_utc(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Close bars become known at 16:00 ET on the bar date (DST-correct), UTC."""
    d = pd.to_datetime(dates)
    base = d.dt.normalize() if isinstance(d, pd.Series) else d.normalize()
    local = base + pd.Timedelta(hours=16)
    return (
        pd.Series(local)
        .dt.tz_localize(NY, nonexistent="shift_forward", ambiguous=True)
        .dt.tz_convert("UTC")
    )


def yyyymmdd(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.strftime("%Y%m%d").astype(int)


# --------------------------------------------------------------------------- #
# raw un-adjustment (invert yfinance split-back-adjustment, self-consistent)  #
# --------------------------------------------------------------------------- #
def unadjust_to_raw(h: pd.DataFrame) -> pd.DataFrame:
    """One symbol's yfinance history (auto_adjust=False, actions=True) ->
    as-reported OHLCV + split-adjusted close (for returns).

    raw_level[t] = adj_level[t] * Pi{split_ratio_i : split_date_i > t}.
    The split column is yfinance's own, so the inversion is exact in its frame.
    """
    h = h.sort_index()
    sp = h["Stock Splits"].replace(0.0, np.nan).fillna(1.0)
    rev_cumprod = sp[::-1].cumprod()[::-1]
    cum_future = rev_cumprod / sp  # product of split ratios strictly AFTER t
    out = pd.DataFrame(index=h.index)
    for c in ["Open", "High", "Low", "Close"]:
        out[c.lower()] = h[c] * cum_future
    out["volume"] = (h["Volume"] / cum_future).round()
    out["close_split_adj"] = h["Close"]  # already split-adj by yfinance
    out["stock_split"] = h["Stock Splits"]
    out["dividend"] = h.get("Dividends", 0.0)
    return out


def returns_wide() -> pd.DataFrame:
    """Wide (date x ticker) split-adjusted daily simple returns from RETURNS."""
    r = pd.read_parquet(RETURNS)
    return r.pivot(index="date", columns="ticker", values="ret").sort_index()
