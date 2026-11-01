"""Download the price history a backtest needs, once, into the local cache.

The variant sweep runs the same period six times. Letting each leg pull its
own decade of daily bars wastes an hour of runner time per leg and is a
reliable way to get rate-limited by Yahoo — which fails every leg with the
same unhelpful "no prices returned". This job does the download once and
leaves it in state/cache/prices.csv.gz for the legs to restore.

    python scripts/prefetch_prices.py --start 2016-01-04
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.config import Config  # noqa: E402
from screener import universe as universe_mod  # noqa: E402
from screener.data import macro as macro_mod  # noqa: E402
from screener.data import prices as price_mod  # noqa: E402


def _as_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yml"))
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default=None)
    ap.add_argument("--warmup-sessions", type=int, default=320)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    start = _as_date(args.start, date(2016, 1, 4))
    end = _as_date(args.end, date.today() - timedelta(days=1))

    # Match the backtest's own warm-up window so the legs need nothing more.
    warmup_days = int(math.ceil(args.warmup_sessions * 365.25 / 252.0)) + 30

    # The macro block needs more history than the price factors do. Betas are
    # a 104-week regression of weekly stock returns on weekly driver changes,
    # plus a 13-week trend window on top, so roughly 2.3 calendar years must
    # already exist BEFORE the first replay session — about twice the 320
    # trading sessions the price factors warm up on. Sizing the download from
    # the warm-up alone is why the macro block sat unavailable for the first
    # 165 sessions of the last run: the data simply was not there yet.
    macro_settings = dict(cfg.data.get("macro_factors") or {})
    beta_weeks = int(macro_settings.get("window_weeks", 104))
    trend_weeks = int(macro_settings.get("trend_lookback_weeks", 13))
    macro_days = int(math.ceil((beta_weeks + trend_weeks) * 7)) + 60

    lead_days = max(warmup_days, macro_days)
    download_start = start - timedelta(days=lead_days)
    print(f"[prefetch] lead-in {lead_days} days "
          f"(price warm-up {warmup_days}, macro betas {macro_days})")

    # Refresh the constituent list first. This is what carries `date_added`,
    # the index-inclusion date that stops a replay holding a company before
    # it was a member — the single largest source of bias in the last run.
    try:
        universe = universe_mod.fetch_current_universe(cfg, allow_network=True)
        print(f"[prefetch] universe refreshed: {len(universe)} constituents")
    except Exception as exc:  # noqa: BLE001
        print(f"[prefetch] universe refresh failed ({exc}); using committed CSV")
        universe = pd.read_csv(cfg.path("universe_csv"))

    if "date_added" in universe.columns:
        known = pd.to_datetime(universe["date_added"], errors="coerce").notna().sum()
        print(f"[prefetch] index-inclusion dates known for {known}/{len(universe)} names")
    else:
        print("[prefetch] WARNING: no date_added column; a replay cannot tell "
              "when a company joined the index and will hold today's "
              "constituents retroactively.", file=sys.stderr)

    tickers = set(universe["ticker"].dropna().astype(str).str.upper())

    history_path = cfg.path("universe_history_csv")
    if history_path.exists():
        hist = pd.read_csv(history_path)
        if "ticker" in hist.columns:
            tickers |= set(hist["ticker"].dropna().astype(str).str.upper())

    tickers.add(str(cfg.data.get("benchmark", "SPY")).upper())
    ordered = sorted(tickers)

    print(f"[prefetch] {len(ordered)} tickers, {download_start} -> {end}")
    cache_path = cfg.path("prices_cache")
    cached = price_mod.load_cache(cache_path)
    if not cached.empty:
        print(f"[prefetch] existing cache: {len(cached):,} rows")

    fresh = price_mod.normalise(
        price_mod._download_yfinance(
            ordered, download_start, end, timeout=180, chunk=40, attempts=3
        )
    )
    if fresh.empty:
        print("[prefetch] ERROR: no prices downloaded", file=sys.stderr)
        return 2

    merged = price_mod.normalise(pd.concat([cached, fresh], ignore_index=True))
    price_mod.save_cache(merged, cache_path)

    covered = merged.groupby("ticker")["date"].agg(["min", "max", "count"])
    print(f"[prefetch] cached {len(merged):,} rows for {len(covered):,} tickers")
    print(f"[prefetch] earliest {merged['date'].min().date()}, "
          f"latest {merged['date'].max().date()}")
    thin = int((covered["count"] < 100).sum())
    if thin:
        print(f"[prefetch] note: {thin} tickers have under 100 bars "
              "(recent listings or delisted names)")

    # Macro too. A replay runs without network by design, so the regime
    # layer and the macro block read this file and nothing else. Skip it and
    # both go silently inert — which would make the regime-off and macro-off
    # sweep legs identical to the baseline and the comparison worthless.
    macro_path = cfg.path("macro_csv")
    macro = macro_mod.get_macro(macro_path, download_start, end, allow_network=True)
    if macro.empty:
        print("[prefetch] WARNING: no macro data. The regime layer will sit at "
              "full exposure and the macro block will be absent, so the "
              "regime-off and macro-off legs will not differ from baseline. "
              "Check that FRED_API_KEY is set.", file=sys.stderr)
    else:
        by_series = macro.groupby("series_id")["date"].agg(["min", "max", "count"])
        print(f"[prefetch] macro: {len(macro):,} rows across "
              f"{len(by_series)} series -> {macro_path}")
        for sid, row in by_series.iterrows():
            print(f"    {sid:<14} {row['min'].date()} .. {row['max'].date()}  "
                  f"({int(row['count']):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
