"""S&P 500 universe loading and point-in-time membership history.

Live jobs refresh current constituents from Wikipedia and archive the dated
membership in state/universe_history.csv. Historical replay chooses the latest
archived constituent set dated on or before the simulated session. Before the
first archived set, replay falls back to the committed list and reports the
remaining survivorship-bias limitation.
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_COLUMNS = ["ticker", "name", "sector", "industry"]
HISTORY_COLUMNS = ["asof", *UNIVERSE_COLUMNS]


def _clean_ticker(t: str) -> str:
    # Wikipedia uses BRK.B; yfinance wants BRK-B, stooq wants brk-b.us
    return str(t).strip().upper().replace(".", "-")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if "ticker" not in df.columns:
        raise ValueError("Universe data must contain a ticker column.")
    out = df.copy()
    for column in UNIVERSE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["ticker"] = out["ticker"].map(_clean_ticker)
    out = out[
        out["ticker"].ne("") & out["ticker"].str.lower().ne("nan")
    ]
    return (
        out[UNIVERSE_COLUMNS]
        .drop_duplicates(subset=["ticker"], keep="last")
        .sort_values("ticker")
        .reset_index(drop=True)
    )


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
    return _normalise(df)


def append_snapshot(universe: pd.DataFrame, path: Path, asof: date | str) -> pd.DataFrame:
    """Persist one complete membership snapshot for a calendar date.

    A second run on the same date replaces that date's rows; other dates remain
    untouched. This makes retries safe and keeps the archive append-only by date.
    """
    snapshot_date = pd.Timestamp(asof).date().isoformat()
    new = _normalise(universe)
    if new.empty:
        raise ValueError("Refusing to archive an empty S&P 500 membership snapshot.")
    new.insert(0, "asof", snapshot_date)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if not old.empty and not {"asof", "ticker"}.issubset(old.columns):
            raise RuntimeError(f"Invalid universe-history schema at {path}.")
        for column in HISTORY_COLUMNS:
            if column not in old.columns:
                old[column] = pd.NA
        old = old[HISTORY_COLUMNS]
        combined = pd.concat([old, new[HISTORY_COLUMNS]], ignore_index=True)
    else:
        combined = new[HISTORY_COLUMNS]

    combined["asof_dt"] = pd.to_datetime(combined["asof"], errors="coerce")
    combined["ticker"] = combined["ticker"].map(_clean_ticker)
    combined = combined.dropna(subset=["asof_dt"])
    combined = combined[
        combined["ticker"].ne("") & combined["ticker"].str.lower().ne("nan")
    ]
    combined = combined.drop_duplicates(subset=["asof", "ticker"], keep="last")
    combined = combined.sort_values(["asof_dt", "ticker"])
    result = combined[HISTORY_COLUMNS].reset_index(drop=True)
    result.to_csv(path, index=False)
    return result


def historical_snapshot(path: Path, asof: date | str) -> pd.DataFrame | None:
    """Return the latest archived membership dated on/before *asof*.

    Returns None when no eligible historical snapshot exists. It never reads a
    snapshot from a future date.
    """
    path = Path(path)
    if not path.exists():
        return None
    history = pd.read_csv(path)
    if history.empty:
        return None
    if not {"asof", "ticker"}.issubset(history.columns):
        raise RuntimeError(f"Invalid universe-history schema at {path}.")

    history["asof_dt"] = pd.to_datetime(history["asof"], errors="coerce")
    target = pd.Timestamp(asof).normalize()
    eligible = history[
        history["asof_dt"].notna() & (history["asof_dt"] <= target)
    ]
    if eligible.empty:
        return None

    latest_date = eligible["asof_dt"].max()
    snapshot = eligible.loc[eligible["asof_dt"] == latest_date]
    return _normalise(snapshot)


def fetch_current_universe(cfg: Config, allow_network: bool = True) -> pd.DataFrame:
    """Return today's constituents, using the committed CSV as a fallback."""
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
        return _normalise(pd.read_csv(snapshot))
    raise RuntimeError(
        "No S&P 500 universe available: Wikipedia fetch failed and no "
        f"snapshot at {snapshot}"
    )


def load_universe(
    cfg: Config,
    allow_network: bool = True,
    asof: date | None = None,
) -> pd.DataFrame:
    """Return the constituent list known on the requested date.

    Live calls (asof=None) refresh the current index membership. Historical
    calls use the newest dated archive on or before the requested date and
    never query the current web page, avoiding future-membership leakage.
    """
    if asof is not None:
        history_path = cfg.path("universe_history_csv")
        archived = historical_snapshot(history_path, asof)
        if archived is not None:
            return archived
        # If there is no archived set old enough, do not query Wikipedia for a
        # historical date. Fall back to the committed list; the backtest report
        # makes this pre-archive survivorship-bias interval explicit.
        return fetch_current_universe(cfg, allow_network=False)
    return fetch_current_universe(cfg, allow_network=allow_network)


def sector_map(universe: pd.DataFrame) -> dict[str, str]:
    return dict(zip(universe["ticker"], universe.get("sector", "Unknown")))
