"""Orchestration: the daily / weekly / monthly jobs.

Run from the repo root:

    python -m screener daily     # after the US close: screen, trade, report
    python -m screener weekly    # refresh fundamentals for the whole universe
    python -m screener monthly   # walk-forward weight tuning
    python -m screener report    # rebuild Portfolio.xlsx from existing state

The daily job is the only one that must run on schedule; the others are
maintenance.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmark as bmod
from . import excel_report, factors, learn, portfolio, scoring, universe
from .config import Config, ensure_dirs
from .data import alpha_vantage as av
from .data import fundamentals as fund_mod
from .data import prices as price_mod


# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            _log(f"could not read {path.name}: {exc}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# DAILY
# ---------------------------------------------------------------------------
def run_daily(cfg: Config, allow_network: bool = True,
              prices_long: pd.DataFrame | None = None,
              asof: date | None = None,
              build_report: bool = True) -> dict:
    ensure_dirs(cfg)
    strat = cfg.strategy
    dcfg = cfg.data

    # --- universe ---------------------------------------------------------
    uni = universe.load_universe(cfg, allow_network=allow_network)
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())
    bench = dcfg.get("benchmark", "SPY")
    fetch_list = tickers + ([bench] if bench not in tickers else [])
    _log(f"universe: {len(tickers)} tickers")

    # --- prices (the one daily network pull) ------------------------------
    if prices_long is None:
        prices_long = price_mod.get_prices(
            fetch_list,
            cache_path=cfg.path("prices_cache"),
            lookback_days=int(dcfg.get("price_lookback_days", 420)),
            provider=dcfg.get("price_provider", "yfinance"),
            allow_network=allow_network,
            asof=asof,
        )
    elif asof is not None:
        # Caller-supplied frame (backfill / replay): clip it to the as-of date
        # so a replay can never read bars from its own future.
        prices_long = prices_long[prices_long["date"] <= pd.Timestamp(asof)]
    if prices_long.empty:
        raise RuntimeError("no price data available — aborting without changes")

    close_wide = price_mod.to_wide(prices_long, "close")
    open_wide = price_mod.to_wide(prices_long, "open")
    data_date = close_wide.index.max()
    _log(f"prices through {data_date:%Y-%m-%d} ({len(close_wide.columns)} tickers)")

    # --- fundamentals (weekly file, read point-in-time) -------------------
    fund = fund_mod.latest_asof(cfg.path("fundamentals_csv"),
                                on_or_before=data_date.date())
    fund_asof = fund["asof"].max() if not fund.empty and "asof" in fund else None
    if fund.empty:
        _log("WARNING: no fundamentals yet — run `python -m screener weekly` first. "
             "Scoring will fall back to price factors only.")
    else:
        _log(f"fundamentals: {len(fund)} tickers, snapshot {fund_asof}")

    # --- state ------------------------------------------------------------
    state = portfolio.PortfolioState.load(
        cfg.path("portfolio_json"), float(strat["start_capital"])
    )
    already_ran = state.last_run_date == data_date.strftime("%Y-%m-%d")
    if already_ran:
        _log(f"state already processed {state.last_run_date}; rebuilding report only")

    # --- score ------------------------------------------------------------
    metrics = factors.build_metrics(prices_long, fund, uni, asof=data_date)
    metrics = metrics.drop(index=[bench], errors="ignore")
    scored = scoring.score(metrics, cfg)
    eligible_n = int(scored["eligible"].sum()) if "eligible" in scored else 0
    _log(f"scored {len(scored)} names, {eligible_n} passed the filters")

    shortlist = scoring.pick_candidate(
        scored, state.held_tickers, state.sector_counts, cfg
    )

    # --- Alpha Vantage verification of the single chosen name -------------
    quota = av.Quota(cfg.path("av_usage"),
                     budget=int(dcfg["alpha_vantage"].get("daily_budget", 22)))
    verdict = {"verdict": "unverified", "reason": "not attempted", "calls_used": 0}
    chosen = None
    if shortlist and dcfg["alpha_vantage"].get("enabled", True) and allow_network:
        for cand in shortlist[:3]:                    # winner, then runners-up
            cached = {}
            if not fund.empty:
                row = fund[fund["ticker"] == cand["ticker"]]
                if not row.empty:
                    cached = row.iloc[0].to_dict()
            verdict = av.verify_candidate(
                cand["ticker"], cand["last_close"], quota, cached=cached
            )
            _log(f"AV check {cand['ticker']}: {verdict['verdict']} — {verdict['reason']}")
            cand["av_verdict"] = verdict["verdict"]
            cand["av_reason"] = verdict["reason"]
            if verdict["verdict"] != "rejected":
                chosen = cand
                break
        if chosen is None:
            _log("all top candidates rejected by Alpha Vantage; skipping today's entry")
    elif shortlist:
        chosen = shortlist[0]
        chosen["av_verdict"] = "unverified"
        chosen["av_reason"] = "verification disabled or offline"

    # --- trade ------------------------------------------------------------
    ladder = portfolio.Ladder(cfg, state)
    if not already_ran:
        bar = pd.DataFrame({
            "open": open_wide.loc[data_date] if data_date in open_wide.index else np.nan,
            "close": close_wide.loc[data_date],
        })
        bar = bar.dropna(subset=["close"])
        name_map = dict(zip(uni["ticker"], uni.get("name", uni["ticker"])))
        summary = ladder.run_day(
            bar, data_date.date(),
            shortlist=[chosen] if chosen else [],
            name_map=name_map,
        )
        for e in summary["events"]:
            _log(f"  {e}")
        _log(f"cash ${summary['cash']:,.2f} · {summary['positions']} positions · "
             f"equity ${summary['equity']:,.2f}")
        if summary["entry_queued"]:
            _log(f"queued BUY {summary['entry_queued']} for next open")

        portfolio.append_trades(ladder.trades, cfg.path("trades_csv"))
        state.save(cfg.path("portfolio_json"))

        bench_series = bmod.benchmark_equity(
            prices_long, float(strat["start_capital"]),
            _first_equity_date(cfg, data_date), bench
        )
        portfolio.append_equity({
            "date": data_date.strftime("%Y-%m-%d"),
            "cash": round(state.cash, 2),
            "positions_value": round(state.positions_value(), 2),
            "equity": round(state.equity(), 2),
            "n_positions": len(state.positions),
            "benchmark_close": round(float(close_wide[bench].loc[data_date]), 4)
                if bench in close_wide.columns and data_date in close_wide.index else None,
            "benchmark_equity": round(float(bench_series.loc[data_date]), 2)
                if data_date in bench_series.index else None,
        }, cfg.path("equity_csv"))

    # --- persist the screen + build the report ---------------------------
    screen_out = scored.head(60).copy()
    screen_out.to_csv(cfg.path("screen_csv"))

    meta = {
        "run_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_date": data_date.strftime("%Y-%m-%d"),
        "universe_size": len(tickers),
        "eligible_count": eligible_n,
        "candidate": chosen["ticker"] if chosen else "—",
        "av_verdict": verdict.get("verdict", "—"),
        "av_reason": verdict.get("reason", ""),
        "av_calls": quota.used,
        "price_provider": dcfg.get("price_provider", ""),
        "fundamentals_asof": fund_asof or "—",
        "state_version": state.version,
    }
    if build_report:
        _build_report(cfg, state, scored, meta, close_wide)
    _append_run_log(cfg, meta, state)
    return meta


def _first_equity_date(cfg: Config, fallback: pd.Timestamp) -> str:
    eq = _load_csv(cfg.path("equity_csv"))
    if not eq.empty and "date" in eq:
        return str(eq["date"].min())
    return fallback.strftime("%Y-%m-%d")


def _build_report(cfg: Config, state, scored: pd.DataFrame, meta: dict,
                  close_wide: pd.DataFrame) -> Path:
    equity = _load_csv(cfg.path("equity_csv"))
    trades = _load_csv(cfg.path("trades_csv"))
    stats = portfolio.performance_stats(equity, trades)
    out = excel_report.build_report(
        cfg.path("excel_report"), state, equity, trades,
        scored.head(60), stats, meta, cfg, price_history=close_wide,
    )
    _log(f"report written: {out}")
    return out


def _append_run_log(cfg: Config, meta: dict, state) -> None:
    row = {**meta, "equity": round(state.equity(), 2),
           "cash": round(state.cash, 2), "positions": len(state.positions)}
    path = cfg.path("run_log")
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------------------
# WEEKLY
# ---------------------------------------------------------------------------
def run_weekly(cfg: Config, limit: int | None = None) -> dict:
    ensure_dirs(cfg)
    uni = universe.load_universe(cfg, allow_network=True)
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())
    if limit:
        tickers = tickers[:limit]
    _log(f"refreshing fundamentals for {len(tickers)} tickers "
         "(this is the slow weekly job)")

    snap = fund_mod.fetch_fundamentals(tickers)
    filled = int(snap[fund_mod.FIELDS].notna().any(axis=1).sum())
    combined = fund_mod.append_snapshot(snap, cfg.path("fundamentals_csv"))
    _log(f"got data for {filled}/{len(tickers)} tickers; "
         f"{len(combined)} total snapshot rows on file")
    return {"tickers": len(tickers), "filled": filled, "rows": len(combined)}


# ---------------------------------------------------------------------------
# MONTHLY
# ---------------------------------------------------------------------------
def run_monthly(cfg: Config, dry_run: bool = False) -> dict:
    ensure_dirs(cfg)
    uni = universe.load_universe(cfg, allow_network=True)
    tickers = sorted(uni["ticker"].astype(str).unique().tolist())
    prices_long = price_mod.get_prices(
        tickers, cache_path=cfg.path("prices_cache"),
        lookback_days=int(cfg.data.get("price_lookback_days", 420)),
        provider=cfg.data.get("price_provider", "yfinance"),
        allow_network=True,
    )
    rec = learn.run_learning(cfg, prices_long, uni, dry_run=dry_run)
    _log(f"learning decision: {rec['decision']} — {rec['reason']}")
    if rec["decision"] == "accepted":
        _log(f"weights {rec['weights_before']} -> {rec['weights_after']}")
    return rec


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="screener")
    ap.add_argument("command",
                    choices=["daily", "weekly", "monthly", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="use cached data only, make no network calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="monthly: evaluate but do not write new weights")
    ap.add_argument("--limit", type=int, default=None,
                    help="weekly: cap the number of tickers (for testing)")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    try:
        if args.command == "daily":
            run_daily(cfg, allow_network=not args.offline)
        elif args.command == "weekly":
            run_weekly(cfg, limit=args.limit)
        elif args.command == "monthly":
            run_monthly(cfg, dry_run=args.dry_run)
        elif args.command == "report":
            state = portfolio.PortfolioState.load(
                cfg.path("portfolio_json"), float(cfg.strategy["start_capital"])
            )
            screen = _load_csv(cfg.path("screen_csv"))
            if not screen.empty and screen.columns[0] not in ("ticker",):
                screen = screen.rename(columns={screen.columns[0]: "ticker"})
            if "ticker" in screen.columns:
                screen = screen.set_index("ticker")
            _build_report(cfg, state, screen, {
                "run_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "data_date": state.last_run_date or "—",
                "state_version": state.version,
            }, pd.DataFrame())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
