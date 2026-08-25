"""Fundamental ratios — refreshed WEEKLY, not daily.

Rationale: fundamentals only change on quarterly filings, so a daily pull of
500 names is pure waste and the main ban risk. The weekly job snapshots every
ticker's ratios with an `asof` date and appends to `fundamentals.csv`. The
screener always reads the *latest* snapshot at or before the trading date,
which gives an approximate point-in-time view that improves as history builds.

Fields (all optional; missing => NaN, handled by the scorer):
  pe_ratio, pb_ratio, ps_ratio, ev_ebitda, fcf_yield,
  roe, gross_margin, net_margin, net_debt_ebitda,
  revenue_growth, earnings_growth, market_cap
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


def fetch_fundamentals(tickers: list[str], asof: date | None = None) -> pd.DataFrame:
    """Pull current fundamentals for each ticker (weekly job)."""
    import yfinance as yf

    asof = asof or date.today()
    rows = []
    for t in tickers:
        rec = {"asof": asof.isoformat(), "ticker": t, **{f: np.nan for f in FIELDS}}
        try:
            info = yf.Ticker(t).info
            if info:
                rec.update(_extract(info))
        except Exception as exc:  # noqa: BLE001
            print(f"[fundamentals] {t}: {exc}")
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
