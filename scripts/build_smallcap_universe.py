#!/usr/bin/env python3
"""Build the small/mid-cap candidate list: US-listed common stocks by size.

    python scripts/build_smallcap_universe.py --config config_smallcap.yml

Writes the config's `universe_csv` in the usual universe schema (ticker,
name, sector, industry, date_added). The list is deliberately WIDER than
the traded size band: the strategy's own filters judge market capitalisation
point-in-time every session, so a candidate list cut tightly to today's
$300M-$10B would quietly exclude names that are in the band on another day.

Sources, best first:
1. Nasdaq's stock screener — every NYSE/Nasdaq/NYSE American listing with
   market cap and sector in one request. The site guards against bots, so
   this can fail from a datacenter IP.
2. SEC's exchange ticker list, with sectors derived from each company's SIC
   code (cached, so only new companies cost a request). No market cap: the
   whole listed set is kept and the size filter does the rest.
3. If both fail, the previous file is left in place and the run says so.

Excluded throughout: OTC listings, preferreds, warrants, rights, units,
blank-check companies (SPACs), and funds/notes — none of them are the
operating businesses the fundamentals describe.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.config import Config  # noqa: E402
from screener.universe import UNIVERSE_COLUMNS  # noqa: E402

NASDAQ_URL = ("https://api.nasdaq.com/api/screener/stocks"
              "?tableonly=true&limit=25&offset=0&exchange={exchange}&download=true")
NASDAQ_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
SEC_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Candidate band, wider than the traded band on both sides (see docstring).
CANDIDATE_MIN_CAP = 150e6
CANDIDATE_MAX_CAP = 20e9
MIN_EXPECTED = 1000          # a sane US small/mid list is in the thousands

# Nasdaq's sector names -> the GICS-style names the rest of the system uses
# (sector caps, sector-neutral scoring, leverage overrides all key on them).
NASDAQ_SECTORS = {
    "Technology": "Information Technology",
    "Finance": "Financials",
    "Basic Materials": "Materials",
    "Telecommunications": "Communication Services",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Health Care": "Health Care",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

_NOT_COMMON = re.compile(
    r"\b(warrants?|rights?|units?|preferred|depositary shares? representing|"
    r"notes due|debentures|acquisition corp|blank check|trust preferred|"
    r"etf|fund)\b", re.IGNORECASE)
_SYMBOL = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")


def clean_symbol(raw: str) -> str | None:
    """Exchange symbol -> the yfinance form, or None for non-common shares."""
    s = str(raw or "").strip().upper().replace("/", "-").replace(".", "-")
    if "^" in s or "$" in s or not _SYMBOL.match(s):
        return None
    return s


def is_operating_common(name: str) -> bool:
    return not _NOT_COMMON.search(str(name or ""))


def parse_money(value) -> float:
    s = str(value or "").replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def sic_to_sector(sic) -> str:
    """Coarse SIC -> GICS-style sector. Only used on the SEC fallback path."""
    try:
        c = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if c == 6770:
        return "SPAC"
    ranges = [
        (100, 999, "Consumer Staples"), (1300, 1399, "Energy"),
        (1000, 1499, "Materials"), (1500, 1799, "Industrials"),
        (2000, 2199, "Consumer Staples"), (2200, 2399, "Consumer Discretionary"),
        (2400, 2499, "Materials"), (2500, 2599, "Consumer Discretionary"),
        (2600, 2699, "Materials"), (2700, 2799, "Communication Services"),
        (2830, 2836, "Health Care"), (2840, 2844, "Consumer Staples"),
        (2800, 2899, "Materials"), (2900, 2999, "Energy"),
        (3000, 3399, "Materials"), (3400, 3499, "Industrials"),
        (3570, 3579, "Information Technology"), (3500, 3599, "Industrials"),
        (3630, 3639, "Consumer Discretionary"), (3600, 3699, "Information Technology"),
        (3710, 3716, "Consumer Discretionary"), (3700, 3799, "Industrials"),
        (3840, 3851, "Health Care"), (3800, 3899, "Information Technology"),
        (3900, 3999, "Consumer Discretionary"), (4800, 4899, "Communication Services"),
        (4900, 4999, "Utilities"), (4000, 4799, "Industrials"),
        (5000, 5199, "Industrials"), (5400, 5499, "Consumer Staples"),
        (5912, 5912, "Consumer Staples"), (5200, 5999, "Consumer Discretionary"),
        (6500, 6553, "Real Estate"), (6798, 6798, "Real Estate"),
        (6000, 6799, "Financials"), (7370, 7379, "Information Technology"),
        (7300, 7399, "Industrials"), (7810, 7829, "Communication Services"),
        (7000, 7999, "Consumer Discretionary"), (8000, 8099, "Health Care"),
        (8700, 8799, "Industrials"),
    ]
    for lo, hi, sector in ranges:
        if lo <= c <= hi:
            return sector
    return "Unknown"


def from_nasdaq(session) -> pd.DataFrame:
    rows = []
    for exchange in ("nasdaq", "nyse", "amex"):
        resp = session.get(NASDAQ_URL.format(exchange=exchange),
                           headers=NASDAQ_HEADERS, timeout=60)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
        table = data.get("rows") or (data.get("table") or {}).get("rows") or []
        for r in table:
            rows.append({
                "symbol": r.get("symbol"), "name": r.get("name"),
                "market_cap": parse_money(r.get("marketCap")),
                "sector": r.get("sector") or "", "industry": r.get("industry") or "",
            })
        time.sleep(1.0)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ticker"] = df["symbol"].map(clean_symbol)
    df = df[df["ticker"].notna() & df["name"].map(is_operating_common)]
    df = df[(df["market_cap"] >= CANDIDATE_MIN_CAP) & (df["market_cap"] <= CANDIDATE_MAX_CAP)]
    df["sector"] = df["sector"].map(NASDAQ_SECTORS).fillna("Unknown")
    return df


def from_sec(session, user_agent: str, sic_cache: Path) -> pd.DataFrame:
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    payload = session.get(SEC_EXCHANGE_URL, headers=headers, timeout=60).json()
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    df = df[df["exchange"].isin(["Nasdaq", "NYSE"])].copy()
    df["ticker"] = df["ticker"].map(clean_symbol)
    df = df[df["ticker"].notna() & df["name"].map(is_operating_common)]

    cache = (pd.read_csv(sic_cache, dtype={"cik": int, "sic": str})
             if sic_cache.exists() else pd.DataFrame(columns=["cik", "sic", "sic_description"]))
    known = dict(zip(cache["cik"], zip(cache["sic"], cache["sic_description"])))
    fetched = []
    for i, cik in enumerate(sorted(set(df["cik"]) - set(known)), start=1):
        time.sleep(0.12)
        try:
            sub = session.get(SEC_SUBMISSIONS_URL.format(cik=int(cik)),
                              headers=headers, timeout=60).json()
        except Exception as exc:  # noqa: BLE001
            print(f"[universe] SIC for CIK {cik}: {exc}", file=sys.stderr)
            continue
        known[cik] = (str(sub.get("sic") or ""), str(sub.get("sicDescription") or ""))
        fetched.append({"cik": cik, "sic": known[cik][0], "sic_description": known[cik][1]})
        if i % 500 == 0:
            print(f"[universe] SIC codes fetched: {i}")
    if fetched:
        cache = pd.concat([cache, pd.DataFrame(fetched)], ignore_index=True)
        sic_cache.parent.mkdir(parents=True, exist_ok=True)
        cache.drop_duplicates("cik", keep="last").to_csv(sic_cache, index=False)

    df["sic"] = df["cik"].map(lambda c: known.get(c, ("", ""))[0])
    df["industry"] = df["cik"].map(lambda c: known.get(c, ("", ""))[1])
    df["sector"] = df["sic"].map(sic_to_sector)
    df = df[df["sector"] != "SPAC"]
    df["market_cap"] = float("nan")
    return df


def finalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates("ticker").copy()
    out["date_added"] = ""     # no index to join: membership is by size, daily
    return out[UNIVERSE_COLUMNS].sort_values("ticker").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config_smallcap.yml"))
    args = ap.parse_args(argv)
    cfg = Config.load(args.config)
    target = cfg.path("universe_csv")

    import requests
    session = requests.Session()
    out, source = None, None
    try:
        df = from_nasdaq(session)
        if len(df) >= MIN_EXPECTED:
            out, source = finalise(df), "nasdaq screener"
        else:
            print(f"[universe] Nasdaq returned only {len(df)} candidates",
                  file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[universe] Nasdaq screener failed: {exc}", file=sys.stderr)

    if out is None:
        ua = os.environ.get("SEC_USER_AGENT", "").strip()
        if ua:
            try:
                df = from_sec(session, ua, cfg.path("state_dir") / "sic_cache.csv")
                if len(df) >= MIN_EXPECTED:
                    out, source = finalise(df), "SEC exchange list + SIC sectors"
            except Exception as exc:  # noqa: BLE001
                print(f"[universe] SEC fallback failed: {exc}", file=sys.stderr)
        else:
            print("[universe] SEC_USER_AGENT not set: no SEC fallback", file=sys.stderr)

    if out is None:
        if target.exists():
            print(f"[universe] both sources failed; keeping the previous list "
                  f"({len(pd.read_csv(target))} names)")
            return 0
        print("[universe] ERROR: no universe could be built and none exists",
              file=sys.stderr)
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(f"[universe] {len(out)} candidates from {source} -> {target}")
    print("[universe] by sector: " + ", ".join(
        f"{k} {v}" for k, v in out["sector"].value_counts().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
