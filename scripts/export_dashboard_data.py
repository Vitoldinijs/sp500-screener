"""Export committed state into one JSON file the dashboard site reads.

Run after `daily` / `weekly` / `monthly` (see the GitHub Actions workflows —
each job runs this and commits docs/data/dashboard.json alongside the state
it already commits). Reads only files already in `state/`; makes no network
calls, so it's cheap and safe to run every time.

    python -m scripts.export_dashboard_data
    python scripts/export_dashboard_data.py --config config.yml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener import portfolio  # noqa: E402
from screener.config import Config  # noqa: E402

MAX_EQUITY_ROWS = 500     # ~2 trading years; keeps the file (and its git
                           # diff) bounded as the bot runs for longer
MAX_TRADE_ROWS = 100
MAX_CANDIDATE_ROWS = 25
MAX_LEARNING_ROWS = 24    # 2 years of monthly decisions


def _clean(o):
    """Recursively make a value JSON-safe: no NaN/Inf, no numpy/pandas types.

    Plain json.dump would either crash on numpy scalars or, worse, silently
    emit a literal `NaN` token — which is NOT valid JSON and makes every
    browser's JSON.parse throw. Everything passes through this before it's
    written.
    """
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NaT:
        return None
    return o


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[export] could not read {path.name}: {exc}")
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _records(df: pd.DataFrame, n: int | None = None, tail: bool = True) -> list[dict]:
    if df.empty:
        return []
    d = df.tail(n) if (n and tail) else (df.head(n) if n else df)
    return d.to_dict(orient="records")


def build_payload(cfg: Config) -> dict:
    strat = cfg.strategy

    state = portfolio.PortfolioState.load(
        cfg.path("portfolio_json"), float(strat["start_capital"])
    )
    equity = _read_csv(cfg.path("equity_csv"))
    trades = _read_csv(cfg.path("trades_csv"))
    run_log = _read_csv(cfg.path("run_log"))
    screen = _read_csv(cfg.path("screen_csv"))
    weights_hist = _read_csv(cfg.path("weights_history"))
    lifecycle_evaluation = _read_json(cfg.path("lifecycle_evaluation_json"))

    stats = portfolio.performance_stats(equity, trades)

    # --- positions -----------------------------------------------------
    pos_fields = ("ticker", "name", "sector", "entry_date", "entry_price",
                  "shares", "cost_basis", "last_price", "last_value",
                  "unrealised_pct", "bars_held", "entry_score",
                  "allocation_weight", "average_entry_price", "last_rank_percentile")
    positions = [{k: p.get(k) for k in pos_fields} for p in state.positions]
    positions.sort(key=lambda p: p.get("entry_date") or "", reverse=True)

    pend_fields = ("ticker", "name", "sector", "notional", "entry_score",
                   "rank", "reason", "av_verdict", "av_reason")
    pending = [{k: o.get(k) for k in pend_fields}
               for o in state.pending_orders if o.get("side") == "BUY"]
    pending_rebalances = [
        {k: o.get(k) for k in ("ticker", "action", "target_value", "current_value",
                               "notional", "allocation_weight", "reason")}
        for o in state.pending_orders if o.get("side") == "REBALANCE"
    ]

    # --- top candidates from the latest screen --------------------------
    top_candidates = []
    if not screen.empty:
        idx_col = screen.columns[0]
        s = screen.rename(columns={idx_col: "ticker"}) if idx_col != "ticker" else screen
        if "composite" in s.columns:
            s = s.sort_values("composite", ascending=False)
        keep = ["ticker", "name", "sector", "last_close", "composite", "rank",
                "eligible"] + [c for c in s.columns if c.startswith("block_")]
        keep = [c for c in keep if c in s.columns]
        top_candidates = _records(s[keep], MAX_CANDIDATE_ROWS, tail=False)

    # --- latest run metadata --------------------------------------------
    latest_run = run_log.iloc[-1].to_dict() if not run_log.empty else {}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": {
            "run_time": latest_run.get("run_time"),
            "data_date": latest_run.get("data_date") or state.last_run_date,
            "universe_size": latest_run.get("universe_size"),
            "eligible_count": latest_run.get("eligible_count"),
            "candidate": latest_run.get("candidate"),
            "av_verdict": latest_run.get("av_verdict"),
            "fundamentals_asof": latest_run.get("fundamentals_asof"),
        },
        "strategy": {
            "name": strat.get("name"),
            "start_capital": strat.get("start_capital"),
            "n_slots": strat.get("n_slots"),
            "max_positions_per_sector": strat.get("max_positions_per_sector"),
            "cash_reserve_pct": strat.get("cash_reserve_pct", 0.10),
            "allocation_min_multiplier": strat.get("allocation_min_multiplier", 0.60),
            "allocation_max_multiplier": strat.get("allocation_max_multiplier", 1.80),
        },
        "stats": stats,
        "portfolio": {
            "cash": round(state.cash, 2),
            "equity": round(state.equity(), 2),
            "n_positions": len(state.positions),
            "n_slots": int(strat.get("n_slots", 0)),
            "positions": positions,
            "pending_orders": pending,
            "pending_rebalances": pending_rebalances,
            "sector_counts": state.sector_counts,
        },
        "equity_curve": _records(equity, MAX_EQUITY_ROWS),
        "recent_trades": list(reversed(_records(trades, MAX_TRADE_ROWS))),
        "top_candidates": top_candidates,
        "factor_weights": cfg.weights,
        "lifecycle_evaluation": lifecycle_evaluation,
        "learning": {
            "enabled": cfg.learning.get("enabled"),
            "mode": cfg.learning.get("mode"),
            "recent_decisions": list(reversed(_records(weights_hist, MAX_LEARNING_ROWS))),
        },
    }
    return _clean(payload)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None,
                     help="override output path (default: paths.dashboard_json)")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    payload = build_payload(cfg)

    out = Path(args.out) if args.out else cfg.path("dashboard_json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[export] wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
