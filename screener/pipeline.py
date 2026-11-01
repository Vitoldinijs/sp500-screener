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
from . import macro_factors as macro_fac
from . import regime as regime_mod
from .data import macro as macro_mod
from . import excel_report, evaluation, factors, learn, lifecycle, portfolio, scoring, universe
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
# Last reported cause of an absent macro block, so a replay logs each
# distinct reason once instead of once per simulated session.
_LAST_MACRO_REASON: str | None = None


def run_daily(cfg: Config, allow_network: bool = True,
              prices_long: pd.DataFrame | None = None,
              asof: date | None = None,
              build_report: bool = True,
              replay_cache: bool = False,
              derive_outcomes: bool = True) -> dict:
    ensure_dirs(cfg)
    strat = cfg.strategy
    dcfg = cfg.data

    # --- universe ---------------------------------------------------------
    uni = universe.load_universe(cfg, allow_network=allow_network, asof=asof)
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
    elif asof is not None and not replay_cache:
        # Caller-supplied frame (backfill / replay): clip it to the as-of date
        # so a replay can never read bars from its own future. With
        # replay_cache the clip moves to the wide frames below, so the long
        # frame keeps its identity and its pivot stays cached.
        prices_long = prices_long[prices_long["date"] <= pd.Timestamp(asof)]
    if prices_long.empty:
        raise RuntimeError("no price data available — aborting without changes")

    # A replay hands the same long frame back every session, so the pivot
    # is cached on it and clipping happens on the (cheap) wide result. A
    # live run passes replay_cache=False and behaves exactly as before.
    close_wide = price_mod.to_wide(prices_long, "close", cache=replay_cache)
    open_wide = price_mod.to_wide(prices_long, "open", cache=replay_cache)
    volume_wide = price_mod.to_wide(prices_long, "volume", cache=replay_cache)
    if asof is not None:
        clip = pd.Timestamp(asof)
        close_wide = close_wide.loc[close_wide.index <= clip]
        open_wide = open_wide.loc[open_wide.index <= clip]
        volume_wide = volume_wide.loc[volume_wide.index <= clip]
    data_date = close_wide.index.max()
    _log(f"prices through {data_date:%Y-%m-%d} ({len(close_wide.columns)} tickers)")

    # Archive live membership on the date for which we have market data.
    # A replay passes asof explicitly and must never manufacture past snapshots.
    if asof is None:
        universe.append_snapshot(uni, cfg.path("universe_history_csv"), data_date.date())

    # --- fundamentals (weekly file, read point-in-time) -------------------
    fund = fund_mod.latest_asof(cfg.path("fundamentals_csv"),
                                on_or_before=data_date.date())
    fund_asof = fund["asof"].max() if not fund.empty and "asof" in fund else None
    if fund.empty:
        _log("WARNING: no fundamentals yet — run `python -m screener weekly` first. "
             "Price factors can still be scored, but new entries are blocked until "
             "market-cap data is available.")
    else:
        _log(f"fundamentals: {len(fund)} tickers, snapshot {fund_asof}")

    # --- state ------------------------------------------------------------
    state = portfolio.PortfolioState.load(
        cfg.path("portfolio_json"), float(strat["start_capital"])
    )
    # Upgrade older committed states using the historical equity peak rather
    # than resetting the breaker to a lower balance after a drawdown.
    try:
        saved_peak = float(state.peak_equity or 0.0)
    except (TypeError, ValueError):
        saved_peak = 0.0
    if not np.isfinite(saved_peak) or saved_peak <= 0:
        history = _load_csv(cfg.path("equity_csv"))
        historical_peak = 0.0
        if not history.empty and "equity" in history:
            values = pd.to_numeric(history["equity"], errors="coerce").dropna()
            if not values.empty:
                historical_peak = float(values.max())
        state.peak_equity = max(
            float(state.start_capital), historical_peak, state.equity()
        )
        if state.peak_equity > 0:
            state.current_drawdown_pct = max(
                0.0, (state.peak_equity - state.equity()) / state.peak_equity
            )
        try:
            trigger = float(strat.get("max_portfolio_drawdown_pct", 0.20))
        except (TypeError, ValueError):
            trigger = 0.20
        if trigger > 0 and state.current_drawdown_pct >= trigger:
            state.drawdown_paused = True
            state.drawdown_pause_date = state.last_run_date or data_date.strftime("%Y-%m-%d")
    already_ran = state.last_run_date == data_date.strftime("%Y-%m-%d")
    if already_ran:
        _log(f"state already processed {state.last_run_date}; rebuilding report only")

    # --- macro ------------------------------------------------------------
    # Fetched once and used twice: per-stock sensitivities feed the scoring
    # table, and the same frame drives the regime layer below. Best-effort —
    # a FRED outage leaves the cached series in place, and with nothing at
    # all both consumers degrade rather than failing the run.
    asof_date = data_date.date() if hasattr(data_date, "date") else data_date
    macro_frame = macro_mod.get_macro(
        cfg.path("macro_csv"), macro_mod.default_start(asof_date), asof_date,
        allow_network=allow_network,
    )

    # --- score ------------------------------------------------------------
    metrics = factors.build_metrics(prices_long, fund, uni, asof=data_date,
                                    close_wide=close_wide, volume_wide=volume_wide)
    metrics = metrics.drop(index=[bench], errors="ignore")

    tilt = macro_fac.macro_tilt(close_wide.drop(columns=[bench], errors="ignore"),
                                macro_frame, asof_date, cfg)
    # Say WHY the macro block is absent, but only when the answer changes. A
    # replay hits this path once per simulated session, so logging every time
    # buries the run in thousands of identical lines — which is how the
    # reason went unread the first time. One line per distinct cause.
    global _LAST_MACRO_REASON
    if tilt.empty:
        reason = str(tilt.attrs.get("unavailable_reason") or "reason not reported")
        if reason != _LAST_MACRO_REASON:
            _log(f"macro sensitivities unavailable: {reason}; "
                 "composite renormalises without them")
            _LAST_MACRO_REASON = reason
    else:
        _LAST_MACRO_REASON = None
        metrics = metrics.join(tilt, how="left")
        _log(f"macro tilt for {int(tilt['macro_tilt'].notna().sum())} names")

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

    # --- market regime: how much may be invested at all --------------------
    # Assessed before trading so today's entry respects it. Reuses the macro
    # frame fetched above rather than hitting FRED a second time.
    bench_close = close_wide[bench] if bench in close_wide.columns else None
    assessment = regime_mod.assess(cfg, bench_close, macro_frame, asof_date)
    _log(f"regime: exposure {assessment.exposure:.0%} (score {assessment.score:.2f})"
         + (f" — {'; '.join(assessment.notes)}" if assessment.notes else ""))

    # --- trade ------------------------------------------------------------
    ladder = portfolio.Ladder(cfg, state)
    ladder.set_exposure(assessment.exposure)
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
            scored=scored,
        )
        for e in summary["events"]:
            _log(f"  {e}")
        _log(f"cash ${summary['cash']:,.2f} · {summary['positions']} positions · "
             f"equity ${summary['equity']:,.2f}")
        if summary["entry_queued"]:
            _log(f"queued BUY {summary['entry_queued']} for next open")
        if summary.get("rebalances_queued"):
            _log(f"queued {summary['rebalances_queued']} rank-weighted rebalances for next open")

        portfolio.append_trades(ladder.trades, cfg.path("trades_csv"))
        state.save(cfg.path("portfolio_json"))

        # Clip explicitly rather than relying on the caller. Only the value
        # at data_date is read today, but handing a point-in-time function a
        # frame that runs past the simulated date is a look-ahead waiting to
        # happen the first time someone reads more of the series.
        bench_long = prices_long[prices_long["date"] <= pd.Timestamp(data_date)]
        bench_series = bmod.benchmark_equity(
            bench_long, float(strat["start_capital"]),
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
    _append_decision_history(cfg, scored, state, data_date, chosen, shortlist, meta)
    # Both ledgers below are idempotent rebuilds from decision_history and
    # trades, and feed reporting only — no trading decision reads them. A
    # replay therefore gets an identical final state by deriving them once
    # at the end instead of re-reading a growing file every session.
    if not derive_outcomes:
        return meta
    outcome_summary = lifecycle.reconcile_signal_outcomes(
        cfg.path("decision_history_csv"),
        cfg.path("trades_csv"),
        cfg.path("signal_outcomes_csv"),
    )
    if outcome_summary["closed_trades"]:
        _log(
            "lifecycle outcomes: {}/{} linked; {} eligible for future evaluation".format(
                outcome_summary["linked_trades"],
                outcome_summary["closed_trades"],
                outcome_summary["learning_eligible_trades"],
            )
        )
    evaluation_report = evaluation.refresh_lifecycle_evaluation(
        cfg.path("signal_outcomes_csv"),
        cfg.path("lifecycle_evaluation_json"),
        cfg.learning,
    )
    _log(
        "lifecycle assessment: {} ({} evaluable trades)".format(
            evaluation_report["status"],
            evaluation_report["counts"]["evaluable_trades"],
        )
    )
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


def _append_decision_history(cfg: Config, scored: pd.DataFrame, state,
                            data_date: pd.Timestamp, chosen: dict | None,
                            shortlist: list[dict], meta: dict) -> None:
    """Append one auditable record per ticker and data date.

    This is a decision ledger, not a fixed-horizon return label. Later runs
    append new observations; lifecycle results can be joined by signal_id and
    trade_id when a position actually closes.
    """
    path = cfg.path("decision_history_csv")
    if scored.empty:
        return
    # Only the dedupe keys are needed here. Reading the whole ledger — tens
    # of columns and, by the end of a replay, hundreds of thousands of rows
    # — just to build a set of (date, ticker) pairs was both slow and the
    # source of the mixed-dtype warnings: pandas infers types per chunk, and
    # a column that is text in one chunk and all-NaN in another comes back
    # inconsistent. Naming the columns and their type removes both problems.
    existing = pd.DataFrame()
    if path.exists():
        try:
            existing = pd.read_csv(path, usecols=["data_date", "ticker"],
                                   dtype=str)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            _log(f"could not read decision history keys: {exc}")

    positions = {p.get("ticker"): p for p in state.positions}
    pending = {
        o.get("ticker"): o for o in state.pending_orders
        if o.get("side") in {"BUY", "SELL"}
    }
    shortlist_by_ticker = {x.get("ticker"): x for x in shortlist}
    chosen_ticker = chosen.get("ticker") if chosen else None
    eligible_count = int(scored.get("eligible", pd.Series(False, index=scored.index)).fillna(False).sum())
    records = []

    for ticker, row in scored.iterrows():
        block_values = {
            b: (None if pd.isna(row.get(f"block_{b}")) else float(row[f"block_{b}"]))
            for b in cfg["metric_blocks"]
        }
        present = [v for v in block_values.values() if v is not None]
        agreement = (
            sum(v > 0 for v in present) / len(present) if present else None
        )
        p = positions.get(ticker)
        order = pending.get(ticker)
        records.append({
            "data_date": data_date.strftime("%Y-%m-%d"),
            "ticker": str(ticker),
            "signal_id": f"{data_date.strftime('%Y-%m-%d')}:{ticker}",
            "name": row.get("name"),
            "sector": row.get("sector"),
            "last_close": _safe_float(row.get("last_close")),
            "eligible": bool(row.get("eligible", False)),
            "exclude_reason": row.get("exclude_reason", ""),
            "rank": _safe_float(row.get("rank")),
            "rank_percentile": (
                round((eligible_count - float(row["rank"])) / (eligible_count - 1), 6)
                if eligible_count > 1 and pd.notna(row.get("rank")) and bool(row.get("eligible", False))
                else (1.0 if eligible_count == 1 and bool(row.get("eligible", False)) else None)
            ),
            "composite": _safe_float(row.get("composite")),
            "blocks_used": _safe_float(row.get("blocks_used")),
            "factor_agreement_positive_share": agreement,
            "fundamental_coverage": (
                _safe_float(1 - float(row.get("missing_fundamentals", 0)) /
                            max(1, sum(len(cfg["metric_blocks"].get(b, []))
                                       for b in ("value", "quality", "growth", "earnings"))))
            ),
            **{f"block_{k}": v for k, v in block_values.items()},
            "is_chosen_candidate": ticker == chosen_ticker,
            "shortlisted": ticker in shortlist_by_ticker,
            "portfolio_action": (
                order.get("side") if order else
                ("HOLD_REVIEW" if p else ("BUY_CANDIDATE" if ticker in shortlist_by_ticker else "NO_ACTION"))
            ),
            "action_reason": (
                order.get("reason") if order else
                ("top eligible candidate after verification" if ticker == chosen_ticker else
                 row.get("exclude_reason", "") if not bool(row.get("eligible", False)) else "")
            ),
            "is_held": p is not None,
            "entry_date": p.get("entry_date") if p else None,
            "entry_price": _safe_float(p.get("entry_price")) if p else None,
            "cost_basis": _safe_float(p.get("cost_basis")) if p else None,
            "unrealised_pct": _safe_float(p.get("unrealised_pct")) if p else None,
            "bars_held_observed": _safe_float(p.get("bars_held")) if p else None,
            "last_signal_review": p.get("last_signal_review") if p else None,
            "forecast_confidence": None,
            "confidence_status": "uncalibrated",
            "portfolio_equity": round(state.equity(), 2),
            "portfolio_cash": round(state.cash, 2),
            "portfolio_positions": len(state.positions),
            "price_provider": meta.get("price_provider"),
            "fundamentals_asof": meta.get("fundamentals_asof"),
        })

    fresh = pd.DataFrame(records)

    # Per-metric columns, for research replays only. Block means hide which
    # metric is doing the work — the momentum block came back with an IC of
    # zero while carrying the largest weight, and a block-level number cannot
    # say whether six metrics are all dead or two good ones are being
    # cancelled by four bad ones. The live ledger is committed to git every
    # day, so this stays off there: ~25 extra columns a session would roughly
    # double its growth for no operational benefit.
    if bool(cfg.get("ledger_metric_columns", False)) and not fresh.empty:
        by_ticker = scored.reindex(fresh["ticker"].to_numpy())
        # The column set comes from the CONFIG, never from whichever z-scores
        # happen to exist this session. Which ones exist varies — z_macro_tilt
        # is absent while macro history is short, a sparse metric is skipped
        # below five values — and this file is appended to under a header
        # written once. A row with a different column set would land under
        # the wrong names without any error.
        # z_*: exactly what the model consumed — winsorized, sector-neutral
        # where configured, and multiplied by the configured direction, so
        # a positive IC always means "this metric works the way we use it".
        for block_metrics in cfg["metric_blocks"].values():
            for name in block_metrics:
                col = f"z_{name}"
                fresh[col] = (pd.to_numeric(by_ticker[col], errors="coerce").to_numpy()
                              if col in by_ticker.columns else np.nan)
        # m_*: raw values of metrics that sit in no block, kept to test
        # alternatives without changing any score.
        for name in cfg.get("diagnostic_metrics") or []:
            fresh[f"m_{name}"] = (pd.to_numeric(by_ticker[name], errors="coerce").to_numpy()
                                  if name in by_ticker.columns else np.nan)

    # Align every append to the header already on disk. Appending a CSV
    # does not check column names, so any drift in what a session produces
    # would silently shift values into the wrong columns.
    if path.exists():
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except (ValueError, OSError, pd.errors.ParserError,
                pd.errors.EmptyDataError):
            header = []
        if header:
            dropped = [c for c in fresh.columns if c not in header]
            if dropped:
                _log(f"decision history: {len(dropped)} column(s) not in the "
                     f"existing header were left out: {', '.join(dropped[:5])}")
            fresh = fresh.reindex(columns=header)

    if not existing.empty and {"data_date", "ticker"}.issubset(existing.columns):
        keys = set(zip(existing["data_date"].astype(str), existing["ticker"].astype(str)))
        fresh = fresh[
            ~fresh.apply(lambda r: (str(r["data_date"]), str(r["ticker"])) in keys, axis=1)
        ]
    if not fresh.empty:
        fresh.to_csv(path, mode="a", header=not path.exists(), index=False)


def _safe_float(value):
    try:
        out = float(value)
        return out if np.isfinite(out) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# WEEKLY
# ---------------------------------------------------------------------------
def run_weekly(cfg: Config, limit: int | None = None) -> dict:
    ensure_dirs(cfg)
    uni = universe.load_universe(cfg, allow_network=True)
    universe.append_snapshot(uni, cfg.path("universe_history_csv"), date.today())
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
    """Tune from the saved closed-trade ledger; no price download is needed."""
    ensure_dirs(cfg)
    # Exact entry-time factor snapshots and realised outcomes are already
    # committed in state/signal_outcomes.csv, so no network pull is needed.
    rec = learn.run_learning(cfg, pd.DataFrame(), pd.DataFrame(), dry_run=dry_run)
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
