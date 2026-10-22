#!/usr/bin/env python3
"""Historical, point-in-time replay of the production paper-trading strategy.

The replay calls the same daily pipeline used by the paper portfolio and
executes pending orders at the next session's open. Results are written outside
the temporary replay state; no live portfolio files are modified.

Examples:
    python scripts/backtest.py
    python scripts/backtest.py --start 2026-08-25 --end 2026-10-21
    python scripts/backtest.py --offline
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener import pipeline, portfolio, universe  # noqa: E402
from screener.config import Config  # noqa: E402
from screener.data import prices as price_mod  # noqa: E402


_ET = ZoneInfo("America/New_York")
TRADING_DAYS_PER_YEAR = 252


def _latest_completed_end() -> date:
    """Avoid including a partial daily bar while the US market is open."""
    now = datetime.now(_ET)
    if now.weekday() < 5 and now.hour < 17:
        return now.date() - timedelta(days=1)
    return now.date()


def _as_date(value, label: str) -> date:
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid {label} date {value!r}; use YYYY-MM-DD.") from exc


def _source_paths(config_path: Path) -> tuple[Config, Path, Path]:
    config_path = config_path.resolve()
    source_root = config_path.parent
    cfg = Config.load(config_path)
    universe_path = source_root / cfg["paths"]["universe_csv"]
    fundamentals_path = source_root / cfg["paths"]["fundamentals_csv"]
    return cfg, universe_path, fundamentals_path


def _fundamental_coverage(fundamentals_path: Path) -> tuple[date, date, int]:
    if not fundamentals_path.exists():
        raise RuntimeError(
            f"No point-in-time fundamental snapshots at {fundamentals_path}. "
            "Run the weekly fundamentals job before backtesting."
        )
    try:
        fund = pd.read_csv(fundamentals_path, usecols=["asof"])
        asof = pd.to_datetime(fund["asof"], errors="coerce").dropna()
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError("Could not read the fundamental snapshot history.") from exc
    if asof.empty:
        raise RuntimeError("The fundamental snapshot file contains no valid dates.")
    return asof.min().date(), asof.max().date(), int(asof.dt.date.nunique())


def _read_price_history(cfg: Config, tickers: list[str], start: date, end: date,
                        warmup_sessions: int, offline: bool) -> pd.DataFrame:
    # Momentum uses roughly 12 months of bars. Calendar padding includes
    # weekends/holidays and a buffer for missing market sessions.
    warmup_calendar_days = int(math.ceil(warmup_sessions * 365.25 / 252.0)) + 30
    download_start = start - timedelta(days=warmup_calendar_days)
    cache_path = cfg.path("prices_cache")
    cached = price_mod.load_cache(cache_path)
    if offline:
        prices = cached
        if prices.empty:
            raise RuntimeError(
                f"Offline mode requested but no local price cache exists at {cache_path}."
            )
    else:
        try:
            fresh = price_mod.normalise(
                price_mod._download_yfinance(tickers, download_start, end)
            )
        except Exception as exc:  # noqa: BLE001
            fresh = pd.DataFrame(columns=price_mod.COLUMNS)
            print(f"[backtest] Yahoo download failed: {exc}")
        prices = pd.concat([cached, fresh], ignore_index=True)
        prices = price_mod.normalise(prices) if not prices.empty else prices
        if prices.empty:
            raise RuntimeError("No historical prices returned by the provider or cache.")

    prices = price_mod.normalise(prices)
    prices = prices[
        (prices["date"] >= pd.Timestamp(download_start))
        & (prices["date"] <= pd.Timestamp(end))
        & prices["ticker"].isin(tickers)
    ].copy()
    if prices.empty:
        raise RuntimeError("No prices overlap the requested backtest/warm-up period.")
    return prices


def _curve_metrics(values: pd.Series, dates: pd.Series) -> dict:
    series = pd.to_numeric(values, errors="coerce")
    date_series = pd.to_datetime(dates, errors="coerce")
    frame = pd.DataFrame({"value": series, "date": date_series}).dropna()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        return {
            "start_value": None, "end_value": None, "total_return": None,
            "cagr": None, "max_drawdown": None, "annualized_volatility": None,
            "sharpe": None, "sortino": None, "calmar": None,
        }

    curve = frame["value"].astype(float)
    start_value, end_value = float(curve.iloc[0]), float(curve.iloc[-1])
    elapsed_days = max(0, int((frame["date"].iloc[-1] - frame["date"].iloc[0]).days))
    total_return = end_value / start_value - 1.0 if start_value > 0 else None
    cagr = (
        (end_value / start_value) ** (365.25 / elapsed_days) - 1.0
        if start_value > 0 and end_value > 0 and elapsed_days > 0 else None
    )
    drawdown = curve / curve.cummax() - 1.0
    daily_returns = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility = (
        float(daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(daily_returns) >= 2 else None
    )
    daily_sd = float(daily_returns.std(ddof=1)) if len(daily_returns) >= 2 else 0.0
    sharpe = (
        float(daily_returns.mean() / daily_sd * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(daily_returns) >= 20 and daily_sd > 0 else None
    )
    downside = np.minimum(daily_returns.to_numpy(dtype=float), 0.0)
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    )
    sortino = (
        float(daily_returns.mean() / downside_deviation * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(daily_returns) >= 20 and downside_deviation > 0 else None
    )
    max_drawdown = float(drawdown.min()) if len(drawdown) else None
    calmar = (
        float(cagr / abs(max_drawdown))
        if cagr is not None and max_drawdown is not None and max_drawdown < 0
        else None
    )
    return {
        "start_value": round(start_value, 2),
        "end_value": round(end_value, 2),
        "total_return": _finite(total_return),
        "cagr": _finite(cagr),
        "max_drawdown": _finite(max_drawdown),
        "annualized_volatility": _finite(volatility),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "calmar": _finite(calmar),
    }


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _relative_metrics(equity: pd.DataFrame) -> dict:
    required = {"date", "equity", "benchmark_equity"}
    if not required.issubset(equity.columns) or equity.empty:
        return {
            "excess_total_return": None, "beta": None,
            "return_correlation": None, "tracking_error": None,
            "information_ratio": None,
        }
    frame = equity[["date", "equity", "benchmark_equity"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame["benchmark_equity"] = pd.to_numeric(frame["benchmark_equity"], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if len(frame) < 2 or frame["equity"].iloc[0] <= 0 or frame["benchmark_equity"].iloc[0] <= 0:
        return {
            "excess_total_return": None, "beta": None,
            "return_correlation": None, "tracking_error": None,
            "information_ratio": None,
        }
    portfolio_total = float(frame["equity"].iloc[-1] / frame["equity"].iloc[0] - 1.0)
    benchmark_total = float(
        frame["benchmark_equity"].iloc[-1] / frame["benchmark_equity"].iloc[0] - 1.0
    )
    returns = frame[["equity", "benchmark_equity"]].pct_change().dropna()
    if len(returns) < 2:
        return {
            "excess_total_return": round(portfolio_total - benchmark_total, 8),
            "beta": None, "return_correlation": None,
            "tracking_error": None, "information_ratio": None,
        }
    p_ret = returns["equity"]
    b_ret = returns["benchmark_equity"]
    benchmark_variance = float(b_ret.var(ddof=1))
    beta = float(p_ret.cov(b_ret) / benchmark_variance) if benchmark_variance > 0 else None
    corr = float(p_ret.corr(b_ret)) if p_ret.std(ddof=1) > 0 and b_ret.std(ddof=1) > 0 else None
    active = p_ret - b_ret
    tracking = float(active.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    active_sd = float(active.std(ddof=1))
    information = (
        float(active.mean() / active_sd * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(active) >= 20 and active_sd > 0 else None
    )
    return {
        "excess_total_return": round(portfolio_total - benchmark_total, 8),
        "beta": _finite(beta),
        "return_correlation": _finite(corr),
        "tracking_error": _finite(tracking),
        "information_ratio": _finite(information),
    }


def _copy_replay_inputs(config_path: Path, workspace: Path) -> Config:
    config_path = config_path.resolve()
    source_root = config_path.parent
    raw_cfg = Config.load(config_path)
    destination_config = workspace / "config.yml"
    shutil.copy2(config_path, destination_config)
    for key in ("universe_csv", "fundamentals_csv"):
        relative = Path(raw_cfg["paths"][key])
        src = source_root / relative
        if not src.exists():
            if key == "universe_csv":
                raise RuntimeError(f"Missing local universe snapshot: {src}")
            continue
        dest = workspace / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    copied = Config.load(destination_config)
    copied.root = workspace
    return copied


def run_backtest(config_path: str | Path, prices_long: pd.DataFrame,
                 start: date | str, end: date | str,
                 output_dir: str | Path, warmup_sessions: int = 320,
                 verbose: bool = False) -> dict:
    """Replay the production daily pipeline in an isolated temporary workspace."""
    start_date, end_date = _as_date(start, "start"), _as_date(end, "end")
    if start_date > end_date:
        raise ValueError("Backtest start date must be on or before its end date.")
    if warmup_sessions < 252:
        raise ValueError("Use at least 252 warm-up sessions for annual momentum factors.")

    config_path = Path(config_path).resolve()
    template, universe_path, fundamentals_path = _source_paths(config_path)
    earliest_fundamental, latest_fundamental, snapshot_count = _fundamental_coverage(
        fundamentals_path
    )
    if start_date < earliest_fundamental:
        raise ValueError(
            f"Cannot start on {start_date}: the first committed point-in-time "
            f"fundamental snapshot is {earliest_fundamental}. Using current ratios "
            "for earlier dates would introduce look-ahead bias. Archive historical "
            "fundamentals first, then rerun."
        )

    if not universe_path.exists():
        raise RuntimeError(f"Missing local universe snapshot: {universe_path}")
    raw_universe = pd.read_csv(universe_path)
    if "ticker" not in raw_universe:
        raise RuntimeError("Universe snapshot must contain a ticker column.")
    benchmark_symbol = str(template.data.get("benchmark", "SPY")).upper()
    tickers = sorted(set(raw_universe["ticker"].dropna().astype(str).str.upper()) | {benchmark_symbol})

    prices = price_mod.normalise(prices_long)
    prices = prices[(prices["date"] <= pd.Timestamp(end_date)) & prices["ticker"].isin(tickers)]
    if prices.empty:
        raise RuntimeError("The supplied price frame has no data before the requested end date.")
    benchmark_prices = prices[
        (prices["ticker"] == benchmark_symbol)
        & (prices["date"] >= pd.Timestamp(start_date))
    ]
    market_dates = sorted(pd.to_datetime(benchmark_prices["date"].drop_duplicates()).tolist())
    if not market_dates:
        raise RuntimeError(
            f"No {benchmark_symbol} trading sessions between {start_date} and {end_date}."
        )
    actual_start, actual_end = market_dates[0].date(), market_dates[-1].date()
    if actual_start < earliest_fundamental:
        raise ValueError(
            f"First replay session {actual_start} predates the earliest point-in-time "
            f"fundamental snapshot ({earliest_fundamental})."
        )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    old_logger = pipeline._log
    if not verbose:
        pipeline._log = lambda *_args, **_kwargs: None
    try:
        with tempfile.TemporaryDirectory(prefix="sp500-backtest-") as temporary:
            cfg = _copy_replay_inputs(config_path, Path(temporary))
            for index, session in enumerate(market_dates, start=1):
                pipeline.run_daily(
                    cfg,
                    allow_network=False,
                    prices_long=prices,
                    asof=session.date(),
                    build_report=False,
                )
                if verbose and (index % 25 == 0 or index == len(market_dates)):
                    print(f"[backtest] {index}/{len(market_dates)} sessions through {session:%Y-%m-%d}")

            equity_path = cfg.path("equity_csv")
            equity = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame()
            trades_path = cfg.path("trades_csv")
            trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
            run_log = pd.read_csv(cfg.path("run_log")) if cfg.path("run_log").exists() else pd.DataFrame()
            state_path = cfg.path("portfolio_json")
            state = portfolio.PortfolioState.load(
                state_path, float(cfg.strategy["start_capital"])
            )
            final_equity = state.equity()
            initial_capital = float(cfg.strategy["start_capital"])
            portfolio_metrics = _curve_metrics(
                equity.get("equity", pd.Series(dtype=float)),
                equity.get("date", pd.Series(dtype=str)),
            )
            benchmark_metrics = _curve_metrics(
                equity.get("benchmark_equity", pd.Series(dtype=float)),
                equity.get("date", pd.Series(dtype=str)),
            )
            relative = _relative_metrics(equity)

            if not trades.empty:
                trade_returns = pd.to_numeric(trades.get("return_pct"), errors="coerce").dropna()
                trade_pnl = pd.to_numeric(trades.get("pnl"), errors="coerce").dropna()
                gross_profit = float(trade_pnl.clip(lower=0).sum())
                gross_loss = float(-trade_pnl.clip(upper=0).sum())
                win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else None
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
                average_trade_return = float(trade_returns.mean()) if len(trade_returns) else None
            else:
                win_rate = profit_factor = average_trade_return = None

            if not equity.empty:
                equity["positions_value"] = pd.to_numeric(
                    equity.get("positions_value", pd.Series(0.0, index=equity.index)),
                    errors="coerce",
                )
                equity["cash"] = pd.to_numeric(
                    equity.get("cash", pd.Series(np.nan, index=equity.index)),
                    errors="coerce",
                )
                equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
                mean_exposure = float((equity["positions_value"] / equity["equity"]).replace(
                    [np.inf, -np.inf], np.nan
                ).dropna().mean())
                mean_cash = float((equity["cash"] / equity["equity"]).replace(
                    [np.inf, -np.inf], np.nan
                ).dropna().mean())
                average_positions = float(pd.to_numeric(
                    equity.get("n_positions", pd.Series(dtype=float)), errors="coerce"
                ).mean())
            else:
                mean_exposure = mean_cash = average_positions = None

            fund = pd.read_csv(fundamentals_path)
            fund_dates = pd.to_datetime(fund.get("asof"), errors="coerce").dropna()
            fund_dates = fund_dates[fund_dates <= pd.Timestamp(actual_end)]
            historical_fund_tickers = int(
                fund.loc[
                    pd.to_datetime(fund["asof"], errors="coerce") <= pd.Timestamp(actual_end),
                    "ticker",
                ].nunique()
            ) if not fund_dates.empty and "ticker" in fund else 0
            prices_present = set(prices["ticker"].astype(str).unique())
            missing_tickers = sorted(set(raw_universe["ticker"].dropna().astype(str)) - prices_present)
            sessions = int(len(equity))
            elapsed_days = max(0, (actual_end - actual_start).days)
            limitations = [
                "The available universe is the current S&P 500 constituent snapshot; historical membership is not reconstructed, so survivorship bias remains.",
                "Fundamental values are only used from snapshots dated on or before each simulated session.",
                "Trades execute at the next session open and include the configured slippage and commission model.",
                "SPY is the buy-and-hold benchmark, marked on the same session dates.",
                "A successful historical simulation is not evidence of future profitability.",
            ]
            if sessions < 126:
                limitations.append(
                    f"Only {sessions} trading sessions were tested (<126); annualized metrics are exploratory."
                )
            if len(trades) < 30:
                limitations.append(
                    f"Only {len(trades)} trades closed; trade-level statistics have a small sample."
                )
            if missing_tickers:
                limitations.append(
                    f"{len(missing_tickers)} current-universe tickers had no price history in the downloaded window."
                )

            summary = {
                "schema_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "requested_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "actual_period": {
                    "start": actual_start.isoformat(),
                    "end": actual_end.isoformat(),
                    "calendar_days": elapsed_days,
                    "trading_sessions": sessions,
                },
                "initial_capital": round(initial_capital, 2),
                "final_equity": round(float(final_equity), 2),
                "strategy": {
                    **portfolio_metrics,
                    "closed_trades": int(len(trades)),
                    "win_rate": _finite(win_rate),
                    "profit_factor": _finite(profit_factor),
                    "average_trade_return": _finite(average_trade_return),
                    "average_exposure": _finite(mean_exposure),
                    "average_cash_weight": _finite(mean_cash),
                    "average_open_positions": _finite(average_positions),
                },
                "benchmark": {
                    "symbol": benchmark_symbol,
                    "method": "buy-and-hold from the first replay close; adjusted daily prices",
                    **benchmark_metrics,
                },
                "relative_to_benchmark": relative,
                "execution_assumptions": {
                    "fills": "next session open",
                    "slippage_bps_each_side": float(cfg.strategy.get("slippage_bps", 0.0)),
                    "commission_per_order": float(cfg.strategy.get("commission_per_trade", 0.0)),
                    "fractional_shares": bool(cfg.strategy.get("fractional_shares", True)),
                    "risk_free_rate": 0.0,
                    "annualization_sessions": TRADING_DAYS_PER_YEAR,
                },
                "data_coverage": {
                    "current_universe_tickers": int(len(raw_universe)),
                    "tickers_with_any_price_history": int(len(prices_present)),
                    "tickers_missing_price_history": missing_tickers,
                    "fundamental_snapshot_first_date": earliest_fundamental.isoformat(),
                    "fundamental_snapshot_latest_date_available": (
                        fund_dates.max().date().isoformat() if not fund_dates.empty else None
                    ),
                    "fundamental_snapshot_dates_available": snapshot_count,
                    "fundamental_tickers_with_snapshots_by_end": historical_fund_tickers,
                    "warmup_sessions_requested": int(warmup_sessions),
                    "price_history_sessions_before_backtest": int(
                        (prices.loc[
                            (prices["ticker"] == benchmark_symbol)
                            & (prices["date"] < pd.Timestamp(actual_start)), "date"
                        ].nunique())
                    ),
                },
                "diagnostics": {
                    "average_eligible_tickers": (
                        _finite(pd.to_numeric(run_log.get("eligible_count"), errors="coerce").mean())
                        if not run_log.empty and "eligible_count" in run_log else None
                    ),
                    "minimum_eligible_tickers": (
                        int(pd.to_numeric(run_log["eligible_count"], errors="coerce").min())
                        if not run_log.empty and "eligible_count" in run_log else None
                    ),
                    "annualized_metrics_reliable_sample": bool(sessions >= 126),
                    "trade_statistics_reliable_sample": bool(len(trades) >= 30),
                },
                "limitations": limitations,
            }

            equity.to_csv(out_dir / "equity_curve.csv", index=False)
            if trades.empty:
                (out_dir / "trades.csv").write_text(
                    "trade_id,ticker,entry_date,exit_date,pnl,return_pct,exit_reason\n",
                    encoding="utf-8",
                )
            else:
                trades.to_csv(out_dir / "trades.csv", index=False)
            (out_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    finally:
        pipeline._log = old_logger

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay the paper-trading strategy against a buy-and-hold SPY benchmark."
    )
    parser.add_argument("--config", default=str(ROOT / "config.yml"))
    parser.add_argument("--start", help="YYYY-MM-DD; defaults to the first point-in-time fundamental snapshot")
    parser.add_argument("--end", help="YYYY-MM-DD; defaults to the latest completed US market date")
    parser.add_argument("--output-dir", default="backtest-results/latest")
    parser.add_argument("--warmup-sessions", type=int, default=320)
    parser.add_argument("--offline", action="store_true", help="use only the local price cache")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        config_path = Path(args.config).resolve()
        cfg, universe_path, fundamentals_path = _source_paths(config_path)
        earliest_fundamental, latest_fundamental, _ = _fundamental_coverage(fundamentals_path)
        requested_start = _as_date(args.start, "start") if args.start else earliest_fundamental
        requested_end = _as_date(args.end, "end") if args.end else _latest_completed_end()
        now = datetime.now(_ET)
        if requested_end >= now.date() and now.weekday() < 5 and now.hour < 17:
            requested_end = now.date() - timedelta(days=1)
            print(
                f"[backtest] excluding today's incomplete session; using end date {requested_end}"
            )
        if requested_start < earliest_fundamental:
            raise ValueError(
                f"Requested start {requested_start} predates the first point-in-time "
                f"fundamental snapshot ({earliest_fundamental}). Current ratios will not "
                "be backfilled into the past because that would cause look-ahead bias."
            )
        if requested_end < requested_start:
            raise ValueError("End date must be on or after start date.")

        raw_universe = pd.read_csv(universe_path)
        benchmark = str(cfg.data.get("benchmark", "SPY")).upper()
        tickers = sorted(set(raw_universe["ticker"].dropna().astype(str).str.upper()) | {benchmark})
        prices = _read_price_history(
            cfg, tickers, requested_start, requested_end,
            args.warmup_sessions, args.offline,
        )
        summary = run_backtest(
            config_path, prices, requested_start, requested_end,
            args.output_dir, warmup_sessions=args.warmup_sessions,
            verbose=args.verbose,
        )
        print(json.dumps({
            "actual_period": summary["actual_period"],
            "strategy": summary["strategy"],
            "benchmark": summary["benchmark"],
            "relative_to_benchmark": summary["relative_to_benchmark"],
            "limitations": summary["limitations"],
            "output_dir": str(Path(args.output_dir).resolve()),
        }, ensure_ascii=False, indent=2))
        if summary["actual_period"]["trading_sessions"] < 126:
            print(
                "\nWARNING: short history; run a longer point-in-time dataset before "
                "making conclusions about strategy edge."
            )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[backtest] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
