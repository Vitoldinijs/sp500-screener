"""Fundamental ratios — refreshed WEEKLY, not daily.

Rationale: fundamentals only change on quarterly filings, so a daily pull of
500 names is pure waste and the main ban risk. The weekly job snapshots every
ticker's ratios with an `asof` date and appends to `fundamentals.csv`. The
screener always reads the *latest* snapshot at or before the trading date,
which gives an approximate point-in-time view that improves as history builds.

Fields (all optional; missing => NaN, handled by the scorer):
  pe_ratio, pb_ratio, ps_ratio, ev_ebitda, fcf_yield,
  roe, gross_margin, net_margin, net_debt_ebitda,
  revenue_growth, earnings_growth, market_cap,
  last_earnings_date, earnings_surprise_pct

`last_earnings_date` / `earnings_surprise_pct` feed the post-earnings-
announcement-drift (PEAD) signal in factors.py: the most recent *reported*
(not estimated) EPS surprise, and the date it was reported on. Scored only
while fresh — see the staleness gate in `factors.build_metrics`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda", "fcf_yield",
    "roe", "gross_margin", "net_margin", "net_debt_ebitda",
    "revenue_growth", "earnings_growth", "market_cap",
    "last_earnings_date", "earnings_surprise_pct",
]
SCHEMA = ["asof", "ticker", *FIELDS]


def _safe(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return np.nan


def _extract(info: dict) -> dict:
    """Map a yfinance `.info` dict into our normalised field set."""
    mc = _safe(info, "marketCap")
    ebitda = _safe(info, "ebitda")
    total_debt = _safe(info, "totalDebt")
    cash = _safe(info, "totalCash")
    net_debt = (total_debt - cash) if not (np.isnan(total_debt) or np.isnan(cash)) else np.nan
    ev = _safe(info, "enterpriseValue")
    ev_ebitda = _safe(info, "enterpriseToEbitda")
    if np.isnan(ev_ebitda) and not np.isnan(ev) and ebitda and ebitda > 0:
        ev_ebitda = ev / ebitda
    fcf = _safe(info, "freeCashflow")
    fcf_yield = (fcf / mc) if (not np.isnan(fcf) and mc and mc > 0) else np.nan
    ndte = (net_debt / ebitda) if (not np.isnan(net_debt) and ebitda and ebitda > 0) else np.nan

    return {
        "pe_ratio": _safe(info, "trailingPE", "forwardPE"),
        "pb_ratio": _safe(info, "priceToBook"),
        "ps_ratio": _safe(info, "priceToSalesTrailing12Months"),
        "ev_ebitda": ev_ebitda,
        "fcf_yield": fcf_yield,
        "roe": _safe(info, "returnOnEquity"),
        "gross_margin": _safe(info, "grossMargins"),
        "net_margin": _safe(info, "profitMargins"),
        "net_debt_ebitda": ndte,
        "revenue_growth": _safe(info, "revenueGrowth"),
        "earnings_growth": _safe(info, "earningsGrowth", "earningsQuarterlyGrowth"),
        "market_cap": mc,
    }


def _extract_earnings(tk) -> dict:
    """Most recent *reported* EPS surprise from a yfinance Ticker object.

    `get_earnings_dates()` returns both past reports and future estimates in
    one table; future rows have no "Reported EPS" yet. We want the newest
    row that has actually happened, so we filter to reported rows first,
    then take the latest — never the newest row from the raw table.
    """
    try:
        df = tk.get_earnings_dates(limit=8)
    except Exception:  # noqa: BLE001
        return {"last_earnings_date": np.nan, "earnings_surprise_pct": np.nan}
    if df is None or df.empty:
        return {"last_earnings_date": np.nan, "earnings_surprise_pct": np.nan}

    reported = df[df.get("Reported EPS").notna()] if "Reported EPS" in df.columns else df.iloc[0:0]
    if reported.empty:
        return {"last_earnings_date": np.nan, "earnings_surprise_pct": np.nan}

    row = reported.sort_index().iloc[-1]
    idx = reported.sort_index().index[-1]
    when = idx.tz_localize(None) if getattr(idx, "tzinfo", None) else idx
    surprise = row.get("Surprise(%)")
    return {
        "last_earnings_date": pd.Timestamp(when).date().isoformat(),
        "earnings_surprise_pct": float(surprise) if pd.notna(surprise) else np.nan,
    }


def fetch_fundamentals(tickers: list[str], asof: date | None = None) -> pd.DataFrame:
    """Pull current fundamentals for each ticker (weekly job)."""
    import yfinance as yf

    asof = asof or date.today()
    rows = []
    for t in tickers:
        rec = {"asof": asof.isoformat(), "ticker": t, **{f: np.nan for f in FIELDS}}
        tk = yf.Ticker(t)
        try:
            info = tk.info
            if info:
                rec.update(_extract(info))
        except Exception as exc:  # noqa: BLE001
            print(f"[fundamentals] {t}: {exc}")
        # Separate try/except: a scrape failure here must not cost us the
        # ratios we already got from `.info` above.
        try:
            rec.update(_extract_earnings(tk))
        except Exception as exc:  # noqa: BLE001
            print(f"[fundamentals] {t} earnings: {exc}")
        rows.append(rec)
    return pd.DataFrame(rows, columns=SCHEMA)


def append_snapshot(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Append this week's snapshot; keep one row per (asof, ticker)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["asof", "ticker"], keep="last")
    combined = combined.sort_values(["ticker", "asof"]).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def latest_asof(path: Path, on_or_before: date | None = None) -> pd.DataFrame:
    """Return the most recent snapshot per ticker at/<= on_or_before.

    This is the point-in-time read used by the scorer.
    """
    if not path.exists():
        return pd.DataFrame(columns=SCHEMA)
    df = pd.read_csv(path)
    df["asof_dt"] = pd.to_datetime(df["asof"], errors="coerce")
    if on_or_before is not None:
        df = df[df["asof_dt"] <= pd.Timestamp(on_or_before)]
    if df.empty:
        return pd.DataFrame(columns=SCHEMA)
    df = df.sort_values("asof_dt").groupby("ticker", as_index=False).last()
    return df.drop(columns=["asof_dt"])
