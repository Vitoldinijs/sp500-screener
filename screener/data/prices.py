"""Daily OHLCV prices for the whole universe.

Design goals
------------
* **One batched request set per day.** yfinance downloads many tickers per
  HTTP call, so ~500 names cost a handful of requests, not 500. This is the
  only thing we refresh daily.
* **Keyless.** Neither provider needs an API key, so nothing here consumes
  the Alpha Vantage quota.
* **Degrade, never crash.** If the batch download partially fails we fall
  back to Stooq for the missing names, and finally to the on-disk cache.

Returns a *long* frame: date | ticker | open | high | low | close | volume
with prices adjusted for splits/dividends.
"""
from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
CHUNK = 100          # tickers per yfinance batch call
STOOQ_MAX = 40       # cap per-ticker fallback work so a run can't hang


# --------------------------------------------------------------------------
# yfinance (primary)
# --------------------------------------------------------------------------
def _download_yfinance(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            raw = yf.download(
                chunk,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[prices] yfinance chunk {i//CHUNK} failed: {exc}")
            continue
        if raw is None or raw.empty:
            continue
        frames.append(_tidy_yf(raw, chunk))
        time.sleep(1.0)  # be polite between batches

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _tidy_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex (ticker, field) columns into long form."""
    out = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = [t for t in tickers if t in raw.columns.get_level_values(0)]
        for t in available:
            sub = raw[t].reset_index()
            sub.columns = [str(c).lower() for c in sub.columns]
            sub["ticker"] = t
            out.append(sub)
    else:  # single ticker => flat columns
        sub = raw.reset_index()
        sub.columns = [str(c).lower() for c in sub.columns]
        sub["ticker"] = tickers[0]
        out.append(sub)

    if not out:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={"index": "date"})
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


# --------------------------------------------------------------------------
# Stooq (fallback, per ticker)
# --------------------------------------------------------------------------
def _download_stooq(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import requests

    frames = []
    for t in tickers[:STOOQ_MAX]:
        sym = t.lower().replace("-", "-") + ".us"
        url = (
            f"https://stooq.com/q/d/l/?s={sym}&d1={start:%Y%m%d}"
            f"&d2={end:%Y%m%d}&i=d"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200 or "Date" not in resp.text[:200]:
                continue
            sub = pd.read_csv(io.StringIO(resp.text))
            sub.columns = [c.lower() for c in sub.columns]
            sub["ticker"] = t
            frames.append(sub)
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[df["close"] > 0]
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return normalise(pd.read_csv(path))
        except Exception as exc:  # noqa: BLE001
            print(f"[prices] cache unreadable ({exc})")
    return pd.DataFrame(columns=COLUMNS)


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    # round to keep the gzipped cache small
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].round(4)
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out.to_csv(path, index=False, compression="gzip")


def get_prices(
    tickers: list[str],
    cache_path: Path,
    lookback_days: int = 420,
    provider: str = "yfinance",
    allow_network: bool = True,
    asof: date | None = None,
) -> pd.DataFrame:
    """Fetch (or load) adjusted daily bars for `tickers`."""
    end = asof or date.today()
    start = end - timedelta(days=int(lookback_days * 1.6))  # calendar padding

    cached = load_cache(cache_path)
    if not allow_network:
        # Honour `asof` offline too. Without this clip an offline run would
        # hand back bars dated after the as-of date, which is lookahead: the
        # strategy would "see" prices it could not have known. Costs nothing
        # in normal operation (the cache ends today) but keeps replays honest.
        if not cached.empty:
            cached = cached[cached["date"] <= pd.Timestamp(end)]
        return cached

    fresh = pd.DataFrame(columns=COLUMNS)
    if provider == "yfinance":
        fresh = normalise(_download_yfinance(tickers, start, end))
    elif provider == "stooq":
        fresh = normalise(_download_stooq(tickers, start, end))

    got = set(fresh["ticker"].unique()) if not fresh.empty else set()
    missing = [t for t in tickers if t not in got]
    if missing and provider == "yfinance":
        print(f"[prices] {len(missing)} tickers missing; trying Stooq fallback")
        extra = normalise(_download_stooq(missing, start, end))
        if not extra.empty:
            fresh = pd.concat([fresh, extra], ignore_index=True)

    if fresh.empty:
        print("[prices] all providers failed; falling back to cache")
        return cached

    # Merge fresh over cache so partial outages never lose history.
    merged = pd.concat([cached, fresh], ignore_index=True)
    merged = normalise(merged)
    cutoff = pd.Timestamp(end) - pd.Timedelta(days=int(lookback_days * 1.6))
    merged = merged[merged["date"] >= cutoff]
    # Save the full merged window, but never *return* bars after the as-of
    # date. Saving first means a backfill run with an early `asof` can't
    # delete newer bars that are already cached.
    save_cache(merged, cache_path)
    return merged[merged["date"] <= pd.Timestamp(end)]


def to_wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """date-indexed DataFrame, one column per ticker."""
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="ticker", values=field, aggfunc="last").sort_index()
