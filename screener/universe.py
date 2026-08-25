"""S&P 500 universe loading.

Live path: scrape the current constituents from Wikipedia (works on the
GitHub runner, which has internet). Fallback: a committed snapshot CSV so
the pipeline never hard-fails if the scrape breaks. Every successful scrape
refreshes that snapshot.

Note on survivorship bias: this uses the *current* index membership. The
paper-trading forward simulation is unaffected (it only ever trades names
that are in the index on the day of the trade), but historical backtests in
the learning module inherit a mild survivorship bias. This is documented in
the README; eliminating it would require a paid point-in-time constituents
feed.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .config import Config

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _clean_ticker(t: str) -> str:
    # Wikipedia uses BRK.B; yfinance wants BRK-B, stooq wants brk-b.us
    return str(t).strip().upper().replace(".", "-")


def fetch_from_wikipedia() -> pd.DataFrame:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (screener; +https://github.com)"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )
    df["ticker"] = df["ticker"].map(_clean_ticker)
    keep = ["ticker", "name", "sector", "industry"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["ticker"])
    return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def load_universe(cfg: Config, allow_network: bool = True) -> pd.DataFrame:
    """Return DataFrame[ticker, name, sector, industry].

    Tries Wikipedia first (and refreshes the snapshot); falls back to the
    committed CSV on any failure.
    """
    snapshot = cfg.path("universe_csv")
    if allow_network:
        try:
            df = fetch_from_wikipedia()
            if len(df) >= 400:  # sanity: index should be ~500 names
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(snapshot, index=False)
                return df
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            print(f"[universe] Wikipedia fetch failed ({exc}); using snapshot")

    if snapshot.exists():
        return pd.read_csv(snapshot)
    raise RuntimeError(
        "No S&P 500 universe available: Wikipedia fetch failed and no "
        f"snapshot at {snapshot}"
    )


def sector_map(universe: pd.DataFrame) -> dict[str, str]:
    return dict(zip(universe["ticker"], universe.get("sector", "Unknown")))
