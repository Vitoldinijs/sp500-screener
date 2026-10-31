"""Archive the live paper portfolio and start a fresh one.

Needed when a change to position sizing makes the existing book
incompatible with the new rules — a set of 21 positions at 4.5% of equity
each cannot be rebalanced toward a 20% target without ~$33k that does not
exist, so the ladder would churn against unreachable targets forever.

What is reset (portfolio state — tied to the old sizing rules):
    portfolio.json, equity_curve.csv, trades.csv, signal_outcomes.csv,
    lifecycle_evaluation.json

What is KEPT (observations about the market — independent of our book):
    fundamentals.csv, universe_history.csv, decision_history.csv,
    weights_history.csv, run_log.csv, last_screen.csv

Everything reset is copied to state/archive/<date>/ first. Nothing is
deleted, so the old track record stays available for comparison.

    python scripts/reset_portfolio.py --dry-run
    python scripts/reset_portfolio.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.config import Config  # noqa: E402
from screener.portfolio import PortfolioState  # noqa: E402

# (config path key, header to write back for CSVs)
RESET_FILES = [
    ("portfolio_json", None),
    ("equity_csv", "date,cash,positions_value,equity,n_positions,"
                   "benchmark_close,benchmark_equity"),
    ("trades_csv", None),          # header preserved from the existing file
    ("signal_outcomes_csv", None),
    ("lifecycle_evaluation_json", None),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--capital", type=float, default=None,
                    help="starting capital; defaults to strategy.start_capital")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    capital = float(args.capital if args.capital is not None
                    else cfg.strategy["start_capital"])
    stamp = date.today().isoformat()
    archive = ROOT / "state" / "archive" / stamp

    print(f"archive -> {archive.relative_to(ROOT)}")
    print(f"fresh capital: ${capital:,.2f}\n")

    for key, header in RESET_FILES:
        try:
            path = cfg.path(key)
        except Exception:
            continue
        if not path.exists():
            print(f"  skip    {key}: not present")
            continue

        print(f"  reset   {path.relative_to(ROOT)}")
        if args.dry_run:
            continue

        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archive / path.name)

        if path.suffix == ".csv":
            # Keep the real header from the archived file where we have one,
            # so a schema change here can't silently rewrite it wrong.
            existing = path.read_text(encoding="utf-8").splitlines()
            line = existing[0] if existing else (header or "")
            path.write_text(line + "\n", encoding="utf-8")
        elif key == "portfolio_json":
            state = PortfolioState.new(capital)
            state.save(path)
        else:
            path.write_text("{}\n", encoding="utf-8")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    fresh = json.loads(cfg.path("portfolio_json").read_text())
    print(f"\nportfolio.json: cash ${fresh['cash']:,.2f}, "
          f"{len(fresh['positions'])} positions, "
          f"{len(fresh['pending_orders'])} pending orders")
    print("kept: fundamentals, universe history, decision history, "
          "weights history, run log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
