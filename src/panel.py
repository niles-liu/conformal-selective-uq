"""Public PIT feature panel — the substrate for every layer.

Pulls daily OHLCV for the transparent public universe + factor proxies (yfinance,
no API key), persists a raw (as-reported) price panel + a split-adjusted returns
substrate, then builds a **point-in-time, no-lookahead** feature panel: every
feature is lagged so `known_at <= decision_time`.

Features (all LAGGED >= 1 session — known strictly before the decision date):
  * trailing total log-momentum over windows {21, 63, 126, 252}
  * idiosyncratic (beta-residualized vs SPY) momentum over {21, 63} + its
    systematic (market-beta) complement
  * rolling realized volatility {21, 63}
  * lagged market beta vs SPY (126d window)
  * distance below trailing 252d high (drawdown proxy)
  * trailing dollar-volume rank -> cap/liquidity tier (PIT, no fundamentals feed)
  * standardized factor exposures (rolling beta to QQQ, MTUM, IWM)

Group labels for conditional coverage (Mondrian groups): sector, cap_tier,
momentum_decile (per-date decile of mom_126).

Run: python src/panel.py            (pull prices + build panel)
     python src/panel.py --build    (rebuild panel from cached prices only)

The pull fails loudly if any ticker cannot be downloaded, since a silently smaller
universe would shift every downstream number. `--allow-missing` overrides that.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import (
    MARKET_TICKER,
    PANEL,
    PRICES,
    PULL_END,
    RETURNS,
    WIN_LO,
    all_tickers,
    known_at_utc,
    nyse_sessions,
    returns_wide,
    unadjust_to_raw,
    universe,
    write_parquet_atomic,
    yyyymmdd,
)

MOM_WINDOWS = [21, 63, 126, 252]
RESID_WINDOWS = [21, 63]
VOL_WINDOWS = [21, 63]
BETA_WINDOW = 126
BETA_MINP = 60
FACTOR_BETA_TICKERS = ["QQQ", "MTUM", "IWM"]
CHUNK = 40
MAX_RETRY = 4

# The ONLY columns a learner may see — the public PIT features. Generative
# internals (pub_signal, overlay, lambda_i, raw_tilt, y, weight, *_sigma) and the
# group labels are NEVER inputs; keeping this explicit is the leakage guard.
FEATURE_COLS = [
    "mom_21", "mom_63", "mom_126", "mom_252",
    "resid_mom_21", "sys_mom_21", "resid_mom_63", "sys_mom_63",
    "vol_21", "vol_63",
    "beta_mkt", "beta_qqq", "beta_mtum", "beta_iwm",
    "dist_252hi", "log_dollar_vol",
]
GROUP_COLS = ["sector", "cap_tier", "mom_decile"]


# --------------------------------------------------------------------------- #
# 1. price pull                                                               #
# --------------------------------------------------------------------------- #
def _download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(MAX_RETRY):
        try:
            df = yf.download(
                tickers,
                start=WIN_LO,
                end=PULL_END,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if df is not None and len(df):
                return df
        except Exception as e:  # noqa: BLE001
            print(f"    retry {attempt+1}/{MAX_RETRY} after error: {e}")
        time.sleep(2**attempt)
    return pd.DataFrame()


def _extract(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    multi = isinstance(df.columns, pd.MultiIndex)
    lvl0 = set(df.columns.get_level_values(0)) if multi else set()
    for t in tickers:
        try:
            sub = df[t] if (multi and t in lvl0) else (df if not multi else None)
            if sub is None:
                continue
            sub = sub.dropna(how="all")
            if sub.empty or "Close" not in sub.columns:
                continue
            raw = unadjust_to_raw(sub).dropna(subset=["close"])
            if raw.empty:
                continue
            raw = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
            raw["ticker"] = t
            out[t] = raw
        except Exception:  # noqa: BLE001
            continue
    return out


def pull(allow_missing: bool = False) -> None:
    tickers = all_tickers()
    chunks = [tickers[i : i + CHUNK] for i in range(0, len(tickers), CHUNK)]
    print(f"universe+proxies: {len(tickers)} tickers | {len(chunks)} chunks")
    got: dict[str, pd.DataFrame] = {}
    for i, ch in enumerate(chunks):
        df = _download(ch)
        part = _extract(df, ch) if len(df) else {}
        got.update(part)
        print(f"  chunk {i+1}/{len(chunks)}: {len(part)}/{len(ch)} tickers")
        time.sleep(0.6)
    if not got:
        raise SystemExit("no prices fetched; check the network connection and yfinance")

    missing = [t for t in tickers if t not in got]
    if missing:  # Yahoo sometimes drops a few names under load: one more pass for those
        print(f"  retrying {len(missing)} missing ticker(s): {', '.join(missing)}")
        df = _download(missing)
        got.update(_extract(df, missing) if len(df) else {})
        missing = [t for t in tickers if t not in got]
    if missing:
        msg = f"{len(missing)} of {len(tickers)} tickers failed to download: {', '.join(missing)}"
        if not allow_missing:
            raise SystemExit(
                f"{msg}\nNothing was written. A panel without these names would change every "
                "downstream number. Re-run later (Yahoo rate-limits bursts), or pass "
                "--allow-missing to build without them (results will then differ from the "
                "published ones).")
        print(f"  WARNING: {msg}; continuing because of --allow-missing")

    bars = pd.concat(got.values(), ignore_index=True)
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None).dt.normalize()
    sessions = set(nyse_sessions())
    bars = bars[bars["date"].isin(sessions)]

    bars["report_date"] = yyyymmdd(bars["date"])
    bars["known_at"] = known_at_utc(bars["date"]).values
    cols = ["report_date", "date", "known_at", "ticker", "open", "high", "low",
            "close", "volume"]
    write_parquet_atomic(bars[cols].sort_values(["ticker", "report_date"]), PRICES)

    # split-adjusted returns substrate (future-split factor cancels in the ratio)
    r = bars[["report_date", "date", "ticker", "close_split_adj", "volume"]].copy()
    r = r.sort_values(["ticker", "report_date"])
    r["ret"] = r.groupby("ticker")["close_split_adj"].pct_change()
    r["dollar_vol"] = r["close_split_adj"] * r["volume"]
    write_parquet_atomic(
        r.dropna(subset=["ret"])[["report_date", "date", "ticker", "ret", "dollar_vol"]], RETURNS)
    print(f"  wrote {PRICES.name} ({len(bars):,} bars), {RETURNS.name}")
    build()


# --------------------------------------------------------------------------- #
# 2. PIT features                                                             #
# --------------------------------------------------------------------------- #
def _rolling_beta(L: pd.DataFrame, m: pd.Series, window: int, minp: int) -> pd.DataFrame:
    """Point-in-time rolling market-model beta vs m, LAGGED one session."""
    var_m = m.rolling(window, min_periods=minp).var()
    mean_m = m.rolling(window, min_periods=minp).mean()
    mean_cm = L.mul(m, axis=0).rolling(window, min_periods=minp).mean()
    mean_c = L.rolling(window, min_periods=minp).mean()
    cov = mean_cm.sub(mean_c.mul(mean_m, axis=0))
    return cov.div(var_m, axis=0).shift(1)


def _stack(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    s = frame.stack().rename(name)
    s.index = s.index.set_names(["date", "ticker"])
    return s.reset_index()


def build() -> None:
    uni = universe()
    names = uni["ticker"].tolist()
    wide = returns_wide()  # date x ticker simple returns (universe + proxies)
    if MARKET_TICKER not in wide.columns:
        raise SystemExit(f"market proxy {MARKET_TICKER} missing from returns")

    L = np.log1p(wide.clip(lower=-0.95))
    m = L[MARKET_TICKER]
    beta_lag = _rolling_beta(L, m, BETA_WINDOW, BETA_MINP)
    eps = L.sub(beta_lag.mul(m, axis=0))  # idiosyncratic log return

    feats: list[pd.DataFrame] = []

    # trailing total log-momentum + idiosyncratic/systematic decomposition
    for w in MOM_WINDOWS:
        minp = max(5, w // 2)
        raw_mom = L.rolling(w, min_periods=minp).sum().shift(1)
        feats.append(_stack(raw_mom[names], f"mom_{w}"))
    for w in RESID_WINDOWS:
        minp = max(5, w // 2)
        resid = eps.rolling(w, min_periods=minp).sum().shift(1)
        raw = L.rolling(w, min_periods=minp).sum().shift(1)
        feats.append(_stack(resid[names], f"resid_mom_{w}"))
        feats.append(_stack((raw - resid)[names], f"sys_mom_{w}"))

    # realized volatility
    for w in VOL_WINDOWS:
        minp = max(5, w // 2)
        vol = L.rolling(w, min_periods=minp).std().shift(1) * np.sqrt(252)
        feats.append(_stack(vol[names], f"vol_{w}"))

    # lagged market beta
    feats.append(_stack(beta_lag[names], "beta_mkt"))

    # factor exposures (rolling beta to QQQ / MTUM / IWM, lagged)
    for ft in FACTOR_BETA_TICKERS:
        if ft in L.columns:
            fb = _rolling_beta(L, L[ft], BETA_WINDOW, BETA_MINP)
            feats.append(_stack(fb[names], f"beta_{ft.lower()}"))

    # distance below trailing 252d high (drawdown proxy), lagged
    lvl = (1.0 + wide[names]).cumprod()
    roll_hi = lvl.rolling(252, min_periods=60).max()
    dd = (lvl / roll_hi - 1.0).shift(1)
    feats.append(_stack(dd, "dist_252hi"))

    # trailing dollar-volume (size/liquidity proxy), lagged
    r = pd.read_parquet(RETURNS)
    dv = r.pivot(index="date", columns="ticker", values="dollar_vol").sort_index()
    dv_lag = dv[names].rolling(63, min_periods=20).mean().shift(1)
    feats.append(_stack(np.log(dv_lag), "log_dollar_vol"))

    # merge all features on (date, ticker)
    panel = feats[0]
    for f in feats[1:]:
        panel = panel.merge(f, on=["date", "ticker"], how="outer")
    panel = panel.merge(uni, on="ticker", how="left")

    # bitemporal stamp + report_date
    panel["report_date"] = yyyymmdd(panel["date"])
    panel["known_at"] = known_at_utc(panel["date"]).values

    # group labels --------------------------------------------------------- #
    # cap_tier: per-date tercile of trailing dollar-volume (PIT)
    panel["cap_tier"] = (
        panel.groupby("date")["log_dollar_vol"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                     labels=["small", "mid", "large"])
                   if s.notna().sum() >= 3 else pd.Series(["mid"] * len(s), index=s.index))
        .astype("object")
    )
    # momentum_decile: per-date decile of mom_126 (PIT)
    panel["mom_decile"] = (
        panel.groupby("date")["mom_126"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False,
                                     duplicates="drop")
                   if s.notna().sum() >= 10 else np.nan)
    )

    # keep rows with a full core feature set (drop the warmup head)
    core = [f"mom_{w}" for w in MOM_WINDOWS] + ["beta_mkt", "vol_21"]
    panel = panel.dropna(subset=core).sort_values(["date", "ticker"]).reset_index(drop=True)
    write_parquet_atomic(panel, PANEL)

    feat_cols = [c for c in panel.columns if c not in
                 ("date", "report_date", "known_at", "ticker", "sector",
                  "cap_tier", "mom_decile")]
    print("=== panel ===")
    print(f"  {len(panel):,} rows | {panel.ticker.nunique()} names "
          f"| {panel.date.nunique()} sessions "
          f"({panel.report_date.min()}..{panel.report_date.max()})")
    print(f"  {len(feat_cols)} features: {feat_cols}")
    print(f"  groups: {panel.sector.nunique()} sectors, cap_tier, mom_decile")
    print(f"  wrote {PANEL.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download public prices and build the PIT feature panel.")
    ap.add_argument("--build", action="store_true",
                    help="rebuild the panel from the cached prices only (no download)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="build even if some tickers fail to download (results will differ)")
    args = ap.parse_args()
    if args.build:
        build()
    else:
        pull(allow_missing=args.allow_missing)
