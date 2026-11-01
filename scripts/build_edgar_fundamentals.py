#!/usr/bin/env python3
"""Build weekly point-in-time fundamental snapshots from SEC filings.

    SEC_USER_AGENT="Your Name your@email" \\
    python scripts/build_edgar_fundamentals.py --start 2020-01-02

Writes state/cache/fundamentals_edgar.csv.gz in the live fundamentals schema,
one row per ticker per week, for a replay to use instead of the live yfinance
snapshots (which only begin in October 2026). See screener/data/edgar.py for
what is reconstructed and how.

SEC requires a User-Agent naming who is calling, with a contact address, and
refuses requests without one. It is read from SEC_USER_AGENT rather than
hard-coded so the contact is the repository owner's choice.

The run ends with a quality report, and fails rather than writing a file
when the result is grossly wrong — above all when market capitalisations
miss known values by a factor that only a split error produces.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.config import Config  # noqa: E402
from screener.data import edgar  # noqa: E402

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
OUTPUT = "state/cache/fundamentals_edgar.csv.gz"

SCHEMA = ["asof", "ticker", "pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
          "fcf_yield", "roe", "gross_margin", "net_margin", "net_debt_ebitda",
          "revenue_growth", "earnings_growth", "market_cap",
          "last_earnings_date", "earnings_surprise_pct"]

# Market capitalisations known well enough to catch a split error, which
# moves a number by the split ratio (4x, 10x, 50x), not by a few percent.
# NVDA and CMG and AVGO all split heavily after these dates, so a missing
# split adjustment shows here first.  (ticker, week-ending Sunday, $bn)
ANCHORS = [
    ("NVDA", "2023-07-02", 1045.0),   # $423 x ~2.47bn shares, pre 10:1 split
    ("AAPL", "2021-01-03", 2230.0),   # $132.7 x ~16.8bn
    ("MSFT", "2022-01-02", 2520.0),   # $336 x ~7.5bn
    ("CMG", "2023-12-31", 63.0),      # $2,287 x ~27.5m, pre 50:1 split
    ("AVGO", "2023-12-31", 530.0),    # $1,116 x ~470m, pre 10:1 split
]
ANCHOR_WARN = 0.25      # report a miss beyond 25%
ANCHOR_FAIL = 2.0       # a factor of two is a split or unit error, not noise


class SecClient:
    """Polite SEC access: declared identity, <10 requests/second, retries."""

    def __init__(self, user_agent: str, min_interval: float = 0.12):
        import requests
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._min_interval = min_interval
        self._last = 0.0

    def get_json(self, url: str, attempts: int = 4):
        for attempt in range(attempts):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._session.get(url, timeout=60)
            except Exception as exc:  # noqa: BLE001
                if attempt == attempts - 1:
                    raise RuntimeError(f"{url}: {exc}") from exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                if resp.status_code == 403 and attempt == 0:
                    print(f"[edgar] 403 from SEC for {url} — usually a missing "
                          "or rejected User-Agent; check SEC_USER_AGENT",
                          file=sys.stderr)
                if attempt == attempts - 1:
                    raise RuntimeError(f"{url}: HTTP {resp.status_code}")
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        return None


def cik_map(client: SecClient) -> dict[str, int]:
    payload = client.get_json(TICKERS_URL) or {}
    out = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).upper().replace(".", "-")
        if ticker:
            out.setdefault(ticker, int(row["cik_str"]))
    return out


def universe_tickers(cfg: Config) -> list[str]:
    tickers = set(pd.read_csv(cfg.path("universe_csv"))["ticker"].astype(str).str.upper())
    hist = cfg.path("universe_history_csv")
    if hist.exists():
        h = pd.read_csv(hist)
        if "ticker" in h.columns:
            tickers |= set(h["ticker"].dropna().astype(str).str.upper())
    return sorted(tickers)


def download_close_and_splits(tickers, start, end, chunk=50, attempts=3):
    """Split-adjusted but NOT dividend-adjusted closes, plus split events.

    The price cache the replay uses is adjusted for dividends too, which
    makes every past price lower than what traded by the dividends paid
    since. For momentum that washes out; for a market capitalisation it
    understates high-yield names by a quarter or more over six years and
    would make them look cheap. So the builder fetches its own closes.
    """
    import yfinance as yf
    closes, splits = {}, {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        raw = None
        for attempt in range(attempts):
            try:
                raw = yf.download(batch, start=start.isoformat(),
                                  end=(end + timedelta(days=1)).isoformat(),
                                  auto_adjust=False, actions=True,
                                  group_by="ticker", progress=False,
                                  threads=True, timeout=60)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[edgar] price batch {i // chunk + 1} attempt "
                      f"{attempt + 1} failed: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for t in batch:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t]
            else:
                sub = raw
            close = pd.to_numeric(sub.get("Close"), errors="coerce").dropna()
            if close.empty:
                continue
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            closes[t] = close[~close.index.duplicated(keep="last")].sort_index()
            s = sub.get("Stock Splits")
            if s is not None:
                s = pd.to_numeric(s, errors="coerce")
                s = s[s > 0]
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                splits[t] = s
            else:
                splits[t] = pd.Series(dtype=float)
        print(f"[edgar] prices {min(i + chunk, len(tickers))}/{len(tickers)}")
    return closes, splits


def weekly_rows(ticker: str, states: pd.DataFrame, close: pd.Series,
                splits: pd.Series, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Join statement states to weekly prices, point-in-time on both sides.

    Snapshots are dated on Sundays, like the live weekly job: built from
    Friday's close and from filings made by Friday, used by the sessions
    that follow.
    """
    if states.empty or close.empty:
        return pd.DataFrame()
    px = close.reindex(close.index.union(weeks)).sort_index().ffill(limit=6)
    frame = pd.DataFrame({"asof": weeks, "close": px.reindex(weeks).to_numpy()})
    frame = pd.merge_asof(frame.sort_values("asof"),
                          states.sort_values("available_from"),
                          left_on="asof", right_on="available_from",
                          direction="backward")
    stale = (frame["asof"] - frame["period_end"]).dt.days > edgar.STALE_DAYS
    frame.loc[stale | frame["period_end"].isna(), "shares"] = np.nan
    factor = [edgar.split_factor(splits, sf) for sf in frame["shares_filed"]]
    frame["market_cap"] = frame["close"] * frame["shares"] * np.asarray(factor)
    r = edgar.ratios(frame)
    r.loc[stale.to_numpy(), :] = np.nan
    r.insert(0, "ticker", ticker)
    r.insert(0, "asof", frame["asof"].dt.strftime("%Y-%m-%d"))
    r["last_earnings_date"] = frame["filed"].dt.strftime("%Y-%m-%d")
    r["earnings_surprise_pct"] = np.nan
    return r[SCHEMA]


def quality_report(out: pd.DataFrame, cfg: Config) -> list[str]:
    """Print coverage, filter impact, live overlap and anchors.

    Returns a list of fatal problems; empty means the file may be written.
    """
    fatal: list[str] = []
    fields = [c for c in SCHEMA if c not in ("asof", "ticker",
                                             "last_earnings_date",
                                             "earnings_surprise_pct")]
    year = out["asof"].str.slice(0, 4)
    print("\n[edgar] coverage by year (% of ticker-weeks with a value)")
    cov = out[fields].notna().groupby(year).mean().mul(100).round(0)
    print(cov.T.to_string())

    # How many names the live filter would drop as "fundamentals incomplete"
    # once the earnings block is set aside (it cannot be rebuilt from SEC).
    blocks = cfg["metric_blocks"]
    counted = [m for b in ("value", "quality", "growth") for m in blocks[b] if m in out]
    limit = int(cfg["filters"].get("max_missing_fundamentals", 3))
    missing = out[counted].isna().sum(axis=1)
    excluded = (missing > limit).groupby(year).mean().mul(100).round(1)
    print(f"\n[edgar] ticker-weeks failing 'fundamentals incomplete' "
          f"(> {limit} of {len(counted)} missing), %:")
    print("  " + "  ".join(f"{y}: {v}" for y, v in excluded.items()))
    if (excluded > 25).any():
        fatal.append("over a quarter of ticker-weeks would be excluded as "
                     "incomplete in some year — the reconstruction is too thin")

    # Overlap with the live yfinance snapshots: same tickers, nearby dates.
    live_path = cfg.path("fundamentals_csv")
    if live_path.exists():
        live = pd.read_csv(live_path)
        live["asof_dt"] = pd.to_datetime(live["asof"], errors="coerce")
        ours = out.assign(asof_dt=pd.to_datetime(out["asof"]))
        joined = pd.merge_asof(
            live.sort_values("asof_dt"), ours.sort_values("asof_dt"),
            on="asof_dt", by="ticker", direction="nearest",
            tolerance=pd.Timedelta(days=7), suffixes=("_yf", "_sec"))
        print(f"\n[edgar] against live yfinance snapshots "
              f"({joined['ticker'].nunique()} tickers, {len(joined)} rows)")
        print(f"  {'field':<16}{'median sec/yf':>15}{'rank corr':>11}{'n':>7}")
        for f in fields:
            a, b = joined.get(f"{f}_sec"), joined.get(f"{f}_yf")
            if a is None or b is None:
                continue
            both = pd.DataFrame({"a": a, "b": b}).dropna()
            both = both[(both["a"] > 0) & (both["b"] > 0)] if f != "earnings_growth" else both
            if len(both) < 30:
                continue
            ratio = float((both["a"] / both["b"]).median())
            corr = float(both["a"].rank().corr(both["b"].rank()))
            print(f"  {f:<16}{ratio:>15.2f}{corr:>11.2f}{len(both):>7}")
            if f == "market_cap" and not 0.8 <= ratio <= 1.25:
                fatal.append(f"market cap median sec/yf is {ratio:.2f}, not ~1")

    # Anchors: the decisive check for split handling.
    print("\n[edgar] known market caps ($bn)")
    misses = 0
    for ticker, week, expected in ANCHORS:
        row = out[(out["ticker"] == ticker) & (out["asof"] == week)]
        got = float(row["market_cap"].iloc[0]) / 1e9 if not row.empty else np.nan
        if not np.isfinite(got):
            print(f"  {ticker:<5} {week}: no value (expected ~{expected:,.0f})")
            continue
        off = got / expected
        flag = ""
        if off > ANCHOR_FAIL or off < 1 / ANCHOR_FAIL:
            flag, misses = "  <-- GROSS MISS (split or unit error)", misses + 1
        elif abs(off - 1) > ANCHOR_WARN:
            flag = "  <-- off by more than 25%"
        print(f"  {ticker:<5} {week}: {got:,.0f} vs ~{expected:,.0f}  "
              f"(x{off:.2f}){flag}")
    if misses >= 2:
        fatal.append(f"{misses} anchor market caps are off by more than "
                     f"{ANCHOR_FAIL:g}x — share counts and prices are on "
                     "different split bases")
    return fatal


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yml"))
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default=None)
    ap.add_argument("--output", default=str(ROOT / OUTPUT))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N tickers (for a quick trial)")
    args = ap.parse_args(argv)

    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent or "@" not in user_agent:
        print("[edgar] SEC_USER_AGENT is not set. SEC refuses anonymous "
              "requests: set it to a name and contact e-mail, e.g. "
              "'Jane Doe jane@example.com', as a repository secret.",
              file=sys.stderr)
        return 2

    cfg = Config.load(args.config)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (datetime.strptime(args.end, "%Y-%m-%d").date() if args.end
           else date.today() - timedelta(days=1))
    weeks = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=14),
                          pd.Timestamp(end), freq="W-SUN")
    # Statement states are needed from a year before the first snapshot so
    # the first weeks already have a full trailing year behind them.
    states_since = pd.Timestamp(start) - pd.Timedelta(days=400)

    client = SecClient(user_agent)
    ciks = cik_map(client)
    tickers = universe_tickers(cfg)
    if args.limit:
        tickers = tickers[:args.limit]
    mapped = [t for t in tickers if t in ciks]
    print(f"[edgar] {len(mapped)}/{len(tickers)} tickers have a CIK")
    unmapped = sorted(set(tickers) - set(mapped))
    if unmapped:
        print(f"[edgar] no CIK: {', '.join(unmapped[:20])}"
              + (" ..." if len(unmapped) > 20 else ""))
    if len(mapped) < 0.8 * len(tickers):
        print("[edgar] ERROR: under 80% of the universe maps to a CIK",
              file=sys.stderr)
        return 2

    closes, splits = download_close_and_splits(
        mapped, start - timedelta(days=30), end)
    print(f"[edgar] closes for {len(closes)}/{len(mapped)} tickers; "
          f"{sum(len(s) > 0 for s in splits.values())} with splits in range")

    frames, failed = [], []
    t0 = time.monotonic()
    for i, ticker in enumerate(mapped, start=1):
        try:
            payload = client.get_json(FACTS_URL.format(cik=ciks[ticker]))
            facts = edgar.parse_companyfacts(payload or {})
            facts = facts[facts["end"] >= states_since - pd.Timedelta(days=800)]
            states = edgar.filing_states(facts)
            if not states.empty:
                states = states[states["available_from"] >= states_since]
            rows = weekly_rows(ticker, states, closes.get(ticker, pd.Series(dtype=float)),
                               splits.get(ticker, pd.Series(dtype=float)), weeks)
            if rows.empty:
                failed.append(ticker)
            else:
                frames.append(rows)
        except Exception as exc:  # noqa: BLE001
            failed.append(ticker)
            print(f"[edgar] {ticker}: {exc}", file=sys.stderr)
        if i % 50 == 0 or i == len(mapped):
            print(f"[edgar] facts {i}/{len(mapped)} "
                  f"({time.monotonic() - t0:.0f}s, {len(failed)} without rows)")

    if not frames:
        print("[edgar] ERROR: no rows built", file=sys.stderr)
        return 2
    out = pd.concat(frames, ignore_index=True)
    if failed:
        print(f"[edgar] no usable history: {', '.join(failed[:30])}"
              + (" ..." if len(failed) > 30 else ""))

    fatal = quality_report(out, cfg)
    if fatal:
        for problem in fatal:
            print(f"[edgar] FATAL: {problem}", file=sys.stderr)
        print("[edgar] not writing the file", file=sys.stderr)
        return 3

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, compression="gzip")
    print(f"\n[edgar] wrote {len(out):,} rows, {out['ticker'].nunique()} tickers, "
          f"{out['asof'].min()} .. {out['asof'].max()} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
