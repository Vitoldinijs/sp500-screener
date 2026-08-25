"""Paper-trading portfolio engine — the "ladder".

Mechanics
---------
* ``n_slots`` (21) concurrent positions, one new entry per trading day,
  each held ``hold_days`` (21) trading days. In steady state the ladder is
  full and one position rolls off each day => ~250 round trips per year.
* **Next-open execution.** A signal computed from day T's close becomes a
  *pending order* filled at day T+1's open. Nothing is ever transacted at a
  price that was used to generate the signal, so the equity curve carries no
  lookahead bias.
* Position size = equity / n_slots at entry, so gains compound and slots stay
  roughly equal-weighted.
* Slippage is charged in basis points on both legs; commission is configurable
  (US brokers are mostly $0).

All state lives in one JSON file so a GitHub Actions run is fully resumable
and every change is visible in the git diff.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

STATE_VERSION = 2


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
@dataclass
class PortfolioState:
    cash: float
    start_capital: float
    bar_count: int = 0
    last_run_date: str | None = None
    positions: list = field(default_factory=list)
    pending_orders: list = field(default_factory=list)
    next_trade_id: int = 1
    version: int = STATE_VERSION

    # -- persistence -------------------------------------------------------
    @classmethod
    def new(cls, start_capital: float) -> "PortfolioState":
        return cls(cash=float(start_capital), start_capital=float(start_capital))

    @classmethod
    def load(cls, path: Path, start_capital: float) -> "PortfolioState":
        if not path.exists():
            return cls.new(start_capital)
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=False, default=str),
            encoding="utf-8",
        )

    # -- derived -----------------------------------------------------------
    @property
    def held_tickers(self) -> set[str]:
        return {p["ticker"] for p in self.positions} | {
            o["ticker"] for o in self.pending_orders if o["side"] == "BUY"
        }

    @property
    def sector_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.positions:
            counts[p.get("sector", "Unknown")] = counts.get(p.get("sector", "Unknown"), 0) + 1
        for o in self.pending_orders:
            if o["side"] == "BUY":
                counts[o.get("sector", "Unknown")] = counts.get(o.get("sector", "Unknown"), 0) + 1
        return counts

    @property
    def reserved_cash(self) -> float:
        return sum(float(o["notional"]) for o in self.pending_orders if o["side"] == "BUY")

    @property
    def available_cash(self) -> float:
        return max(0.0, self.cash - self.reserved_cash)

    def positions_value(self) -> float:
        return float(sum(float(p.get("last_value", p.get("cost_basis", 0.0))) for p in self.positions))

    def equity(self) -> float:
        return float(self.cash + self.positions_value())

    def open_slots(self, n_slots: int) -> int:
        used = len(self.positions) + sum(1 for o in self.pending_orders if o["side"] == "BUY")
        return max(0, n_slots - used)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Ladder:
    def __init__(self, cfg, state: PortfolioState):
        self.cfg = cfg
        self.s = state
        st = cfg.strategy
        self.n_slots = int(st["n_slots"])
        self.hold_days = int(st["hold_days"])
        self.slip = float(st.get("slippage_bps", 0.0)) / 10_000.0
        self.commission = float(st.get("commission_per_trade", 0.0))
        self.fractional = bool(st.get("fractional_shares", True))
        self.trades: list[dict] = []          # trades closed during this run

    # -- helpers -----------------------------------------------------------
    def _buy_price(self, px: float) -> float:
        return px * (1.0 + self.slip)

    def _sell_price(self, px: float) -> float:
        return px * (1.0 - self.slip)

    def _shares(self, notional: float, price: float) -> float:
        if price <= 0:
            return 0.0
        raw = notional / price
        return float(raw) if self.fractional else float(np.floor(raw))

    # -- step 1: fill orders queued yesterday ------------------------------
    def fill_pending(self, bar: pd.DataFrame, trade_date: date) -> list[dict]:
        """Execute pending orders at today's OPEN. `bar` is indexed by ticker
        with columns open/close."""
        events = []
        still_pending = []

        for order in self.s.pending_orders:
            t = order["ticker"]
            if t not in bar.index:
                order["retries"] = int(order.get("retries", 0)) + 1
                if order["retries"] <= 3:
                    still_pending.append(order)         # no data today, retry
                    events.append({"event": "order_deferred", "ticker": t})
                else:
                    events.append({"event": "order_cancelled", "ticker": t,
                                   "reason": "no price data for 3 sessions"})
                continue

            row = bar.loc[t]
            px_open = float(row.get("open") if not pd.isna(row.get("open")) else row["close"])
            if px_open <= 0:
                events.append({"event": "order_cancelled", "ticker": t,
                               "reason": "invalid open price"})
                continue

            if order["side"] == "BUY":
                events.append(self._execute_buy(order, px_open, trade_date))
            else:
                events.append(self._execute_sell(order, px_open, trade_date))

        self.s.pending_orders = still_pending
        return [e for e in events if e]

    def _execute_buy(self, order: dict, px_open: float, trade_date: date) -> dict:
        fill = self._buy_price(px_open)
        notional = min(float(order["notional"]), self.s.cash - self.commission)
        shares = self._shares(notional, fill)
        if shares <= 0 or notional <= 1.0:
            return {"event": "order_cancelled", "ticker": order["ticker"],
                    "reason": "insufficient cash"}

        cost = shares * fill + self.commission
        self.s.cash -= cost
        pos = {
            "trade_id": self.s.next_trade_id,
            "ticker": order["ticker"],
            "sector": order.get("sector", "Unknown"),
            "name": order.get("name", ""),
            "shares": round(shares, 6),
            "entry_date": trade_date.isoformat(),
            "entry_price": round(fill, 4),
            "cost_basis": round(cost, 2),
            "entry_bar": self.s.bar_count,
            "entry_score": order.get("entry_score"),
            "entry_blocks": order.get("blocks"),
            "entry_rank": order.get("rank"),
            "av_verdict": order.get("av_verdict", "unverified"),
            "last_price": round(fill, 4),
            "last_value": round(shares * fill, 2),
            "bars_held": 0,
        }
        self.s.next_trade_id += 1
        self.s.positions.append(pos)
        return {"event": "bought", "ticker": pos["ticker"], "shares": shares,
                "price": fill, "cost": cost, "trade_id": pos["trade_id"]}

    def _execute_sell(self, order: dict, px_open: float, trade_date: date) -> dict:
        tid = order.get("trade_id")
        pos = next((p for p in self.s.positions if p["trade_id"] == tid), None)
        if pos is None:
            return {"event": "order_stale", "ticker": order["ticker"]}

        fill = self._sell_price(px_open)
        proceeds = pos["shares"] * fill - self.commission
        self.s.cash += proceeds
        pnl = proceeds - pos["cost_basis"]
        ret = pnl / pos["cost_basis"] if pos["cost_basis"] else 0.0

        self.trades.append({
            "trade_id": tid,
            "ticker": pos["ticker"],
            "name": pos.get("name", ""),
            "sector": pos.get("sector", "Unknown"),
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "exit_date": trade_date.isoformat(),
            "exit_price": round(fill, 4),
            "shares": pos["shares"],
            "cost_basis": round(pos["cost_basis"], 2),
            "proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(ret, 6),
            "bars_held": self.s.bar_count - pos["entry_bar"],
            "entry_score": pos.get("entry_score"),
            "exit_reason": order.get("reason", "hold period elapsed"),
            "av_verdict": pos.get("av_verdict", "unverified"),
        })
        self.s.positions = [p for p in self.s.positions if p["trade_id"] != tid]
        return {"event": "sold", "ticker": pos["ticker"], "price": fill,
                "pnl": pnl, "trade_id": tid}

    # -- step 2: mark to market -------------------------------------------
    def mark_to_market(self, bar: pd.DataFrame) -> None:
        for p in self.s.positions:
            t = p["ticker"]
            if t in bar.index and not pd.isna(bar.loc[t, "close"]):
                px = float(bar.loc[t, "close"])
                p["last_price"] = round(px, 4)
                p["last_value"] = round(p["shares"] * px, 2)
                p["stale_days"] = 0
            else:
                p["stale_days"] = int(p.get("stale_days", 0)) + 1
            p["bars_held"] = self.s.bar_count - p["entry_bar"]
            if p["cost_basis"]:
                p["unrealised_pct"] = round(
                    p["last_value"] / p["cost_basis"] - 1.0, 6
                )

    # -- step 3: queue exits ----------------------------------------------
    def schedule_exits(self) -> list[dict]:
        queued = []
        pending_sell_ids = {o.get("trade_id") for o in self.s.pending_orders
                            if o["side"] == "SELL"}
        for p in self.s.positions:
            if p["trade_id"] in pending_sell_ids:
                continue
            reason = None
            if p["bars_held"] >= self.hold_days:
                reason = "hold period elapsed"
            elif int(p.get("stale_days", 0)) >= 5:
                reason = "price data lost for 5 sessions"
            if reason:
                order = {"side": "SELL", "ticker": p["ticker"],
                         "trade_id": p["trade_id"], "reason": reason,
                         "notional": 0.0}
                self.s.pending_orders.append(order)
                queued.append(order)
        return queued

    # -- step 4: queue the entry ------------------------------------------
    def schedule_entry(self, shortlist: list[dict], name_map: dict | None = None) -> dict | None:
        """Queue one BUY for the top shortlisted candidate, if a slot is free."""
        if self.s.open_slots(self.n_slots) <= 0 or not shortlist:
            return None

        target = self.s.equity() / self.n_slots
        budget = min(target, self.s.available_cash)
        if budget < 25.0:                    # not worth a position
            return None

        pick = shortlist[0]
        order = {
            "side": "BUY",
            "ticker": pick["ticker"],
            "sector": pick.get("sector", "Unknown"),
            "name": (name_map or {}).get(pick["ticker"], ""),
            "notional": round(budget, 2),
            "entry_score": pick.get("composite"),
            "rank": pick.get("rank"),
            "blocks": pick.get("blocks"),
            "signal_close": pick.get("last_close"),
            "av_verdict": pick.get("av_verdict", "unverified"),
            "av_reason": pick.get("av_reason", ""),
            "reason": "top composite score",
        }
        self.s.pending_orders.append(order)
        return order

    # -- full daily cycle --------------------------------------------------
    def run_day(
        self,
        bar: pd.DataFrame,
        trade_date: date,
        shortlist: list[dict] | None = None,
        name_map: dict | None = None,
    ) -> dict:
        """Process one trading day. `bar` = today's open/close indexed by ticker."""
        self.s.bar_count += 1
        events = self.fill_pending(bar, trade_date)
        self.mark_to_market(bar)
        exits = self.schedule_exits()
        entry = self.schedule_entry(shortlist or [], name_map)
        self.s.last_run_date = trade_date.isoformat()
        return {
            "date": trade_date.isoformat(),
            "events": events,
            "exits_queued": len(exits),
            "entry_queued": entry["ticker"] if entry else None,
            "cash": round(self.s.cash, 2),
            "positions": len(self.s.positions),
            "equity": round(self.s.equity(), 2),
        }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def append_trades(trades: list[dict], path: Path) -> None:
    if not trades:
        return
    df = pd.DataFrame(trades)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_equity(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def performance_stats(equity: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Headline statistics for the dashboard."""
    out = {
        "equity": np.nan, "total_return": np.nan, "cagr": np.nan,
        "max_drawdown": np.nan, "sharpe": np.nan, "volatility": np.nan,
        "n_trades": 0, "win_rate": np.nan, "avg_win": np.nan,
        "avg_loss": np.nan, "profit_factor": np.nan, "avg_hold": np.nan,
        "best_trade": np.nan, "worst_trade": np.nan, "days_live": 0,
        "benchmark_return": np.nan,
    }
    if equity is not None and not equity.empty and "equity" in equity:
        eq = pd.to_numeric(equity["equity"], errors="coerce").dropna()
        if len(eq) >= 1:
            start, last = float(eq.iloc[0]), float(eq.iloc[-1])
            out["equity"] = last
            out["days_live"] = int(len(eq))
            if start > 0:
                out["total_return"] = last / start - 1.0
                years = len(eq) / 252.0
                if years > 0.08:
                    out["cagr"] = (last / start) ** (1.0 / years) - 1.0
            rets = eq.pct_change().dropna()
            if len(rets) > 5:
                sd = rets.std(ddof=0)
                out["volatility"] = float(sd * np.sqrt(252))
                if sd > 0:
                    out["sharpe"] = float(rets.mean() / sd * np.sqrt(252))
            peak = eq.cummax()
            out["max_drawdown"] = float((eq / peak - 1.0).min())
        if "benchmark_equity" in equity:
            b = pd.to_numeric(equity["benchmark_equity"], errors="coerce").dropna()
            if len(b) > 1 and float(b.iloc[0]) > 0:
                out["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1.0)

    if trades is not None and not trades.empty and "return_pct" in trades:
        r = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
        out["n_trades"] = int(len(r))
        if len(r):
            wins, losses = r[r > 0], r[r <= 0]
            out["win_rate"] = float(len(wins) / len(r))
            out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
            out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
            out["best_trade"] = float(r.max())
            out["worst_trade"] = float(r.min())
            gp = float(pd.to_numeric(trades["pnl"], errors="coerce").clip(lower=0).sum())
            gl = float(-pd.to_numeric(trades["pnl"], errors="coerce").clip(upper=0).sum())
            out["profit_factor"] = (gp / gl) if gl > 0 else np.nan
        if "bars_held" in trades:
            out["avg_hold"] = float(pd.to_numeric(trades["bars_held"], errors="coerce").mean())
    return out
