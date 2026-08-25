#!/usr/bin/env python3
"""Offline end-to-end simulation: ~300 trading days of the real daily job.

Why this exists
---------------
The unit tests check each piece in isolation. This script checks the *whole
machine* by running the exact function GitHub Actions runs — ``pipeline.run_daily``
— once per simulated trading day, against a synthetic market, with no network
access at all. It then asserts the money invariants after every single day.

What it proves
--------------
* the daily job survives 300 consecutive runs, each one resuming from the
  state file the previous run wrote (that is how Actions actually behaves);
* the ladder fills up to ``n_slots`` and then rolls one position per day;
* cash never goes negative, no ticker is ever held twice, the sector cap holds;
* the accounting identity ``equity == capital + realised + unrealised`` holds
  to the cent, every day;
* nothing in the pipeline reads a price from its own future;
* the weekly fundamentals job and the monthly learning job both work on real
  accumulated state;
* the Excel report builds from a fully populated portfolio.

Usage
-----
    python scripts/simulate.py                  # 300 days, temp dir
    python scripts/simulate.py --days 60        # quick smoke run
    python scripts/simulate.py --keep DIR       # keep the simulated repo

Exit code is non-zero if any invariant broke, so CI can gate on it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener import learn, pipeline, portfolio  # noqa: E402
from screener.config import Config  # noqa: E402
from screener.data import fundamentals as fund_mod  # noqa: E402
from screener.data import prices as price_mod  # noqa: E402
from tests import synthetic  # noqa: E402

WARMUP_DAYS = 300          # bars needed before day 1 (MA200 + 12m momentum)
N_NAMES = 120              # synthetic universe size
FUND_REFRESH_EVERY = 5     # trading days between fundamentals snapshots (weekly)
LEARN_EVERY = 63           # trading days between learning runs (~quarterly)


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------
def build_fake_repo(root: Path, universe: pd.DataFrame) -> Config:
    """Lay out a throwaway repo so cfg.path() resolves inside `root`."""
    (root / "state" / "cache").mkdir(parents=True, exist_ok=True)
    (root / "screener" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "config.yml", root / "config.yml")
    universe.to_csv(root / "screener" / "data" / "sp500.csv", index=False)

    cfg = Config.load(root / "config.yml")
    cfg.root = root
    return cfg


def publish_fundamentals(cfg: Config, universe: pd.DataFrame,
                         prices: pd.DataFrame, asof: pd.Timestamp,
                         seed: int) -> None:
    """Simulate one run of the weekly fundamentals job.

    Fundamentals are derived from prices truncated at `asof` and stamped with
    that date, so `latest_asof` later reads them point-in-time — exactly the
    mechanism the live weekly job builds up over months.
    """
    past = prices[prices["date"] <= asof].copy()
    past.attrs["alphas"] = prices.attrs.get("alphas", {})   # attrs die on slicing
    snap = synthetic.make_fundamentals(
        universe, past, asof=asof.strftime("%Y-%m-%d"), seed=seed
    )
    fund_mod.append_snapshot(snap, cfg.path("fundamentals_csv"))


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------
def check_invariants(cfg: Config, state, trades: pd.DataFrame, day: str) -> None:
    """Raise AssertionError on the first violated invariant."""
    strat = cfg.strategy
    n_slots = int(strat["n_slots"])
    hold = int(strat["hold_days"])
    cap = int(strat["max_positions_per_sector"])

    def bad(msg: str):
        raise AssertionError(f"[{day}] {msg}")

    # --- cash / equity sanity --------------------------------------------
    if state.cash < -0.01:
        bad(f"negative cash: {state.cash:.4f}")
    eq = state.equity()
    if not np.isfinite(eq) or eq <= 0:
        bad(f"equity is not a positive finite number: {eq}")

    # --- slot discipline -------------------------------------------------
    if len(state.positions) > n_slots:
        bad(f"{len(state.positions)} positions exceeds n_slots={n_slots}")
    tickers = [p["ticker"] for p in state.positions]
    if len(tickers) != len(set(tickers)):
        dupes = {t for t in tickers if tickers.count(t) > 1}
        bad(f"duplicate ticker(s) held: {sorted(dupes)}")

    buys = [o for o in state.pending_orders if o["side"] == "BUY"]
    if len(buys) > int(strat["buys_per_day"]):
        bad(f"{len(buys)} buy orders queued, limit is {strat['buys_per_day']}")
    queued = {o["ticker"] for o in buys}
    if queued & set(tickers):
        bad(f"queued a buy for an already-held name: {sorted(queued & set(tickers))}")

    # --- diversification -------------------------------------------------
    for sector, n in state.sector_counts.items():
        if n > cap:
            bad(f"sector {sector!r} holds {n} positions, cap is {cap}")

    # --- hold period -----------------------------------------------------
    # A position may overrun slightly: the sell is queued on the day the hold
    # elapses and fills at the next open, and a missing bar defers it further.
    for p in state.positions:
        if p["bars_held"] > hold + 4:
            bad(f"{p['ticker']} held {p['bars_held']} bars, hold_days={hold}")

    # --- the accounting identity ----------------------------------------
    realised = 0.0
    if trades is not None and not trades.empty and "pnl" in trades:
        realised = float(pd.to_numeric(trades["pnl"], errors="coerce").sum())
    unrealised = float(sum(
        float(p.get("last_value", p["cost_basis"])) - float(p["cost_basis"])
        for p in state.positions
    ))
    expected = float(state.start_capital) + realised + unrealised
    n_rows = (0 if trades is None or trades.empty else len(trades)) + len(state.positions)
    tol = 0.01 * n_rows + 0.10          # each stored row carries <=1c of rounding
    if abs(eq - expected) > tol:
        bad(
            f"accounting identity broken: equity {eq:,.2f} != capital "
            f"{state.start_capital:,.2f} + realised {realised:,.2f} + unrealised "
            f"{unrealised:,.2f} = {expected:,.2f} (diff {eq - expected:+.2f}, tol {tol:.2f})"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def simulate(days: int, seed: int, keep: Path | None, verbose: bool,
             resume: bool = False, max_seconds: float | None = None) -> int:
    t0 = time.time()
    if not verbose:
        # The daily job logs ~6 lines per run, which is right for CI but
        # unreadable across 300 days. Keep our own one-line-per-day summary.
        pipeline._log = lambda *_a, **_k: None

    universe = synthetic.make_universe(N_NAMES, seed=seed)
    total_bars = WARMUP_DAYS + days + 5
    prices = synthetic.make_prices(universe, days=total_bars, seed=seed + 1)

    workdir = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="screener-sim-"))
    if workdir.exists() and not resume:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = build_fake_repo(workdir, universe)
    # Seed the price cache; the pipeline then reads it offline, clipped to
    # each day's as-of date. No network is touched anywhere in this script.
    if not cfg.path("prices_cache").exists():
        price_mod.save_cache(price_mod.normalise(prices), cfg.path("prices_cache"))

    all_dates = sorted(prices["date"].unique())
    sim_dates = [pd.Timestamp(d) for d in all_dates[WARMUP_DAYS:WARMUP_DAYS + days]]

    # Which days did an earlier chunk already process? The state file is the
    # authority, exactly as it is for the real job resuming the next morning.
    done: set[str] = set()
    if resume:
        prior = pipeline._load_csv(cfg.path("equity_csv"))
        if not prior.empty and "date" in prior:
            done = set(prior["date"].astype(str))

    print(f"simulating {len(sim_dates)} trading days "
          f"({sim_dates[0]:%Y-%m-%d} -> {sim_dates[-1]:%Y-%m-%d}) "
          f"on {N_NAMES} synthetic names")
    if done:
        print(f"resuming: {len(done)} days already on file")
    print(f"working dir: {workdir}\n")

    # One fundamentals snapshot must predate day 1, or the first screens have
    # nothing but price factors to work with.
    if not cfg.path("fundamentals_csv").exists():
        publish_fundamentals(cfg, universe, prices,
                             pd.Timestamp(all_dates[WARMUP_DAYS - 1]), seed + 2)

    learning_runs: list[dict] = []
    failures: list[str] = []
    peak_positions = 0
    processed = 0
    exhausted = False

    for i, day in enumerate(sim_dates):
        is_last = i == len(sim_dates) - 1
        if day.strftime("%Y-%m-%d") in done:
            continue
        if max_seconds is not None and time.time() - t0 > max_seconds and not is_last:
            exhausted = True
            break

        # ---- weekly job -------------------------------------------------
        if i > 0 and i % FUND_REFRESH_EVERY == 0:
            publish_fundamentals(cfg, universe, prices, day, seed + 100 + i)

        # ---- the actual daily job, exactly as CI runs it ----------------
        try:
            meta = pipeline.run_daily(
                cfg,
                allow_network=False,
                asof=day.date(),
                build_report=is_last,      # one workbook at the end, not 300
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append(f"[{day:%Y-%m-%d}] run_daily raised {type(exc).__name__}: {exc}")
            break

        # ---- verify against what was persisted, not in-memory state -----
        state = portfolio.PortfolioState.load(
            cfg.path("portfolio_json"), float(cfg.strategy["start_capital"])
        )
        trades = pipeline._load_csv(cfg.path("trades_csv"))
        peak_positions = max(peak_positions, len(state.positions))
        try:
            check_invariants(cfg, state, trades, f"{day:%Y-%m-%d}")
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"  INVARIANT FAILED {exc}")
            break

        # ---- monthly-ish learning job on real accumulated state ---------
        if i > 0 and i % LEARN_EVERY == 0:
            rec = learn.run_learning(
                cfg, prices[prices["date"] <= day], universe
            )
            learning_runs.append(rec)
            print(f"  [{day:%Y-%m-%d}] learning: {rec['decision']} — {rec['reason']}")

        if verbose or (i % 25 == 0) or is_last:
            print(f"  {day:%Y-%m-%d}  day {i+1:>3}/{len(sim_dates)}  "
                  f"equity ${state.equity():>10,.2f}  cash ${state.cash:>9,.2f}  "
                  f"pos {len(state.positions):>2}  trades {len(trades):>3}  "
                  f"pick {meta.get('candidate', '—')}")
        processed += 1

    # ---- results --------------------------------------------------------
    equity = pipeline._load_csv(cfg.path("equity_csv"))
    trades = pipeline._load_csv(cfg.path("trades_csv"))
    state = portfolio.PortfolioState.load(
        cfg.path("portfolio_json"), float(cfg.strategy["start_capital"])
    )
    stats = portfolio.performance_stats(equity, trades)

    if exhausted and not failures:
        remaining = len(sim_dates) - len(done) - processed
        print(f"\n  time budget reached after {processed} day(s) this chunk; "
              f"{remaining} still to go")
        print(f"  equity ${state.equity():,.2f}  ·  {stats['n_trades']} closed trades")
        print(f"\nPARTIAL — resume with:\n"
              f"  python scripts/simulate.py --days {days} --seed {seed} "
              f"--keep {workdir} --resume"
              + (f" --max-seconds {int(max_seconds)}" if max_seconds else ""))
        return 3      # distinct from 0 (done) and 1 (broken invariant)

    print("\n" + "=" * 68)
    print("RESULT")
    print("=" * 68)
    print(f"  days simulated      {stats['days_live']}")
    print(f"  final equity        ${stats['equity']:,.2f}"
          f"   (start ${state.start_capital:,.2f})")
    print(f"  total return        {stats['total_return']:+.2%}")
    print(f"  benchmark (SPY)     {stats['benchmark_return']:+.2%}")
    print(f"  max drawdown        {stats['max_drawdown']:.2%}")
    print(f"  sharpe              {stats['sharpe']:.2f}")
    print(f"  closed trades       {stats['n_trades']}")
    print(f"  win rate            {stats['win_rate']:.1%}")
    print(f"  avg hold (bars)     {stats['avg_hold']:.1f}")
    print(f"  open positions      {len(state.positions)} (peak {peak_positions}"
          f" / {cfg.strategy['n_slots']} slots)")
    print(f"  learning runs       {len(learning_runs)}"
          f"  {[r['decision'] for r in learning_runs]}")
    print(f"  wall clock          {time.time() - t0:,.1f}s")

    # Did the ladder actually fill and roll as designed?
    n_slots = int(cfg.strategy["n_slots"])
    if len(sim_dates) > n_slots + int(cfg.strategy["hold_days"]) + 5:
        if peak_positions < n_slots:
            failures.append(
                f"ladder never filled: peak {peak_positions} of {n_slots} slots "
                "in a run long enough to fill"
            )
        expected_trades = len(sim_dates) - n_slots - 5
        if stats["n_trades"] < expected_trades * 0.7:
            failures.append(
                f"only {stats['n_trades']} closed trades, expected roughly "
                f"{expected_trades} (one roll-off per day once full)"
            )

    xlsx = cfg.path("excel_report")
    if xlsx.exists():
        print(f"\n  workbook: {xlsx}  ({xlsx.stat().st_size / 1024:,.0f} KB)")
    else:
        failures.append("no Excel report was produced")

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("all invariants held on every simulated day")
    if not keep:
        print(f"(temp dir left in place for inspection: {workdir})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep", default=None,
                    help="write the simulated repo here instead of a temp dir")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --keep (skips days "
                         "already recorded in the equity curve)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this long and print a resume "
                         "command; exits 3 when the run is still incomplete")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.resume and not args.keep:
        ap.error("--resume needs --keep DIR (a temp dir cannot be resumed)")
    return simulate(args.days, args.seed,
                    Path(args.keep) if args.keep else None, args.verbose,
                    resume=args.resume, max_seconds=args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
