"""Macroeconomic series from FRED, with publication lags applied.

Why this module is careful about dates
--------------------------------------
FRED stamps an observation with the date it DESCRIBES, not the date it was
published. October's CPI sits under 2026-10-01 but is not released until
mid-November, and it is revised afterwards. Reading the frame naively would
let a backtest act on 2026-10-05 using a number nobody could see until
2026-11-13 — a look-ahead so large it would make any result meaningless.

So every series carries an explicit `lag_days`, and `as_of()` only returns
observations whose publication date has actually passed. The lags are
deliberately generous: being a few days late with a macro reading costs
almost nothing, while being a day early is cheating.

Daily market-based series (Treasury yields, fed funds, VIX, WTI, the dollar
index) are published next business day and are not revised, so they carry a
small lag. Monthly survey data (CPI, unemployment) carries a large one.

Everything degrades to the on-disk cache: no API key, a failed request or a
throttled host leaves the pipeline running on the last good data rather than
crashing the daily job.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
SCHEMA = ["date", "series_id", "value"]


@dataclass(frozen=True)
class Series:
    series_id: str
    label: str
    lag_days: int          # calendar days from observation date to publication
    monthly: bool = False  # monthly series are carried forward between prints


# Chosen so the whole set comes from one provider with one key. Daily series
# first; the monthly ones are the ones with the punishing publication lags.
SERIES: tuple[Series, ...] = (
    Series("DGS10", "treasury_10y", 2),
    Series("DGS2", "treasury_2y", 2),
    Series("T10Y2Y", "yield_curve_10y_2y", 2),
    Series("DFF", "fed_funds", 2),
    Series("VIXCLS", "vix", 2),
    Series("DCOILWTICO", "oil_wti", 5),       # weekly revisions early on
    Series("DTWEXBGS", "dollar_index", 5),
    Series("BAMLH0A0HYM2", "high_yield_spread", 2),
    # Monthly. The lag covers the gap from the month being described to the
    # release, plus room for the first revision.
    Series("CPIAUCSL", "cpi", 45, monthly=True),
    Series("UNRATE", "unemployment", 35, monthly=True),
)

BY_ID = {s.series_id: s for s in SERIES}
BY_LABEL = {s.label: s for s in SERIES}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["series_id"] = out["series_id"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    out = out.drop_duplicates(["date", "series_id"], keep="last")
    return out[SCHEMA].sort_values(["series_id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def _fetch_one(series_id: str, start: date, end: date, api_key: str) -> pd.DataFrame:
    import requests

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    resp = requests.get(FRED_URL, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    payload = resp.json()
    rows = payload.get("observations") or []
    if not rows:
        return _empty()
    frame = pd.DataFrame(rows)[["date", "value"]]
    # FRED writes "." for a missing print (holidays, suspended series).
    frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    frame["series_id"] = series_id
    return normalise(frame)


def fetch_macro(start: date, end: date, api_key: str | None = None,
                series: tuple[Series, ...] = SERIES) -> pd.DataFrame:
    """Pull the configured series. Returns whatever succeeded, never raises."""
    api_key = api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("[macro] FRED_API_KEY not set; skipping refresh")
        return _empty()

    frames = []
    for spec in series:
        try:
            got = _fetch_one(spec.series_id, start, end, api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"[macro] {spec.series_id}: {exc}")
            continue
        if not got.empty:
            frames.append(got)
    if not frames:
        print("[macro] no series returned")
        return _empty()
    return normalise(pd.concat(frames, ignore_index=True))


def get_macro(cache_path: Path, start: date, end: date,
              api_key: str | None = None, allow_network: bool = True) -> pd.DataFrame:
    """Refresh the cache where possible and return the merged long frame."""
    cache_path = Path(cache_path)
    cached = _empty()
    if cache_path.exists():
        try:
            cached = normalise(pd.read_csv(cache_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[macro] unreadable cache ({exc}); starting fresh")

    fresh = fetch_macro(start, end, api_key) if allow_network else _empty()
    if fresh.empty:
        if cached.empty:
            print("[macro] no data available from cache or network")
        return cached

    merged = normalise(pd.concat([cached, fresh], ignore_index=True))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index=False)
    return merged


# ---------------------------------------------------------------------------
# point-in-time access
# ---------------------------------------------------------------------------
def as_of(frame: pd.DataFrame, asof: date) -> dict[str, float]:
    """Latest value per series that had actually been PUBLISHED by `asof`.

    This is the only function the rest of the code should use to read macro
    data. Reading `frame` directly would hand a backtest numbers that were
    still unpublished on the date being simulated.
    """
    if frame is None or frame.empty:
        return {}
    asof_ts = pd.Timestamp(asof)
    out: dict[str, float] = {}
    for series_id, chunk in frame.groupby("series_id"):
        spec = BY_ID.get(str(series_id))
        if spec is None:
            continue
        published = chunk["date"] + pd.Timedelta(days=spec.lag_days)
        visible = chunk[published <= asof_ts]
        if visible.empty:
            continue
        out[spec.label] = float(visible.sort_values("date")["value"].iloc[-1])
    return out


def history(frame: pd.DataFrame, label: str, asof: date | None = None) -> pd.Series:
    """Published history of one series as a date-indexed Series.

    Used for the regime layer, which needs changes over time rather than a
    single reading. Honours the same publication lag as `as_of`.
    """
    spec = BY_LABEL.get(label)
    if spec is None or frame is None or frame.empty:
        return pd.Series(dtype=float)
    chunk = frame[frame["series_id"] == spec.series_id].copy()
    if chunk.empty:
        return pd.Series(dtype=float)
    if asof is not None:
        published = chunk["date"] + pd.Timedelta(days=spec.lag_days)
        chunk = chunk[published <= pd.Timestamp(asof)]
    if chunk.empty:
        return pd.Series(dtype=float)
    out = chunk.set_index("date")["value"].sort_index()
    return out[~out.index.duplicated(keep="last")]


def default_start(end: date, lookback_days: int = 2000) -> date:
    return end - timedelta(days=lookback_days)
