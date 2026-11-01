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
# `date_added` is the index-inclusion date Wikipedia publishes next to each
# constituent. It used to be parsed and thrown away, which is what made the
# backtest buy PLTR in early 2024 — nine months before it joined the index.
# Keeping it lets a replay exclude names that had not been added yet.
UNIVERSE_COLUMNS = ["ticker", "name", "sector", "industry", "date_added"]
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
    if "date_added" in out.columns:
        # Wikipedia carries a few footnote-laden or malformed dates; an
        # unparseable one becomes NaT and is treated as "always a member",
        # which is the conservative reading for a name already on the list.
        out["date_added"] = pd.to_datetime(
            out["date_added"].astype(str).str.slice(0, 10), errors="coerce"
        ).dt.date
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
            "Date added": "date_added",
            "Date first added": "date_added",
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


def members_as_of(universe: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Drop constituents that had not yet joined the index on `asof`.

    The dated archive is the authoritative source, but it only begins when
    this bot started running. Before that the committed list is all there
    is — and replaying today's list into the past is what let a 2024 session
    buy Palantir, which did not join the index until September 2024, or
    Vistra, which joined that May. Those are not merely survivors: they were
    added BECAUSE they had already risen, so letting the screener hold them
    from the start manufactures most of the backtest's return.

    `date_added` fixes that half of the problem exactly. The other half —
    companies that were members then and have since been dropped — cannot be
    recovered from a current snapshot, since they are simply absent from it.
    That residual bias is milder: removals are usually poor performers, and a
    screener that buys its top-ranked names would rarely have held them. It
    is reported in the backtest summary rather than papered over.

    A missing or unparseable `date_added` is treated as "already a member",
    the conservative reading for a name that is on the list at all.
    """
    if universe is None or universe.empty or "date_added" not in universe.columns:
        return universe
    added = pd.to_datetime(universe["date_added"], errors="coerce")
    eligible = added.isna() | (added <= pd.Timestamp(asof))
    return universe[eligible].reset_index(drop=True)


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
        # No archived set is old enough. Fall back to the committed list,
        # but filtered by each name's index-inclusion date so the replay
        # cannot hold a company before it was a constituent. The backtest
        # report still flags the residual (removed-member) bias.
        return members_as_of(fetch_current_universe(cfg, allow_network=False), asof)
    return fetch_current_universe(cfg, allow_network=allow_network)


def sector_map(universe: pd.DataFrame) -> dict[str, str]:
    return dict(zip(universe["ticker"], universe.get("sector", "Unknown")))
