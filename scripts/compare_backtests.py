"""Put several backtest summaries side by side as one markdown table.

Reads every summary.json under a directory and prints a comparison sorted
by Sharpe. Written for the variant sweep, where each leg changes exactly
one thing, so the interesting question is how the legs differ from the
baseline rather than what any single number is.

    python scripts/compare_backtests.py all-results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _fmt(value, kind: str = "num") -> str:
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if x != x:  # NaN
        return "—"
    if kind == "pct":
        return f"{x * 100:+.1f}%"
    if kind == "pct0":
        return f"{x * 100:.0f}%"
    return f"{x:.2f}"


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"<!-- unreadable {path}: {exc} -->")
            continue
        # artifact dir looks like backtest-<name>-<run>/...
        label = path.parent.name
        if label in ("latest", "full"):
            label = path.parent.parent.name
        strat = data.get("strategy") or {}
        bench = data.get("benchmark") or {}
        rel = data.get("relative_to_benchmark") or {}
        period = data.get("actual_period") or {}
        rows.append({
            "name": label.replace("backtest-", ""),
            "sessions": period.get("trading_sessions"),
            "total_return": strat.get("total_return"),
            "bench_return": bench.get("total_return"),
            "excess": rel.get("excess_return"),
            "cagr": strat.get("cagr"),
            "sharpe": strat.get("sharpe"),
            "sortino": strat.get("sortino"),
            "max_dd": strat.get("max_drawdown"),
            "vol": strat.get("annualized_volatility"),
            "trades": strat.get("closed_trades"),
            "win_rate": strat.get("win_rate"),
            "avg_positions": strat.get("average_open_positions"),
            "avg_exposure": strat.get("average_exposure"),
        })
    return rows


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "backtest-results")
    rows = load(root)
    if not rows:
        print("No backtest summaries found.")
        return 0

    rows.sort(key=lambda r: (r["sharpe"] is None, -(r["sharpe"] or 0)))

    print("### Backtest variant comparison\n")
    sessions = {r["sessions"] for r in rows if r["sessions"]}
    if sessions:
        print(f"_{min(sessions)}–{max(sessions)} trading sessions per leg._\n")
    print("| variant | return | SPY | excess | CAGR | Sharpe | Sortino | max DD "
          "| vol | trades | win | avg pos | avg expo |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        print(
            f"| `{r['name']}` "
            f"| {_fmt(r['total_return'], 'pct')} "
            f"| {_fmt(r['bench_return'], 'pct')} "
            f"| {_fmt(r['excess'], 'pct')} "
            f"| {_fmt(r['cagr'], 'pct')} "
            f"| {_fmt(r['sharpe'])} "
            f"| {_fmt(r['sortino'])} "
            f"| {_fmt(r['max_dd'], 'pct')} "
            f"| {_fmt(r['vol'], 'pct0')} "
            f"| {r['trades'] if r['trades'] is not None else '—'} "
            f"| {_fmt(r['win_rate'], 'pct0')} "
            f"| {_fmt(r['avg_positions'])} "
            f"| {_fmt(r['avg_exposure'], 'pct0')} |"
        )

    print(
        "\n> Each leg changes one setting against `baseline-cap25`, so a gap "
        "is attributable to that setting. Sharpe and drawdown matter more "
        "than raw return here: the concentrated legs are expected to post "
        "bigger numbers in both directions, and the question is whether the "
        "extra return is paid for by the extra risk.\n"
        "\n> This sweep runs WITHOUT fundamental blocks, so it measures the "
        "momentum/risk/technical/macro core only — not the live strategy.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
