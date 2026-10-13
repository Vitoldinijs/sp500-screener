"""Paper-trading portfolio engine — the "ladder".

Mechanics
---------
* ``n_slots`` is a maximum concurrent-position limit, not a holding target.
  Positions are re-scored daily and exit only when signal/data-risk rules fire.
* **Next-open execution.** A signal computed from day T's close becomes a
  *pending order* filled at day T+1's open. Nothing is ever transacted at a
  price that was used to generate the signal, so the equity curve carries no
  lookahead bias.
* Position targets are rank-weighted and periodically rebalanced. A configurable
  cash reserve and slot normalisation prevent the strongest ranks from taking
  more than the portfolio can fund.
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

from .allocation import allocation_multiplier

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
        return sum(
            float(o.get("notional", 0.0))
            for o in self.pending_orders
            if o["side"] == "BUY"
            or (o["side"] == "REBALANCE" and o.get("action") == "increase")
        )

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
        self.exit_rank_floor = float(st.get("exit_rank_percentile_below", 0.35))
        self.exit_confirmation = max(1, int(st.get("exit_confirmation_observations", 3)))
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
    @staticmethod
    def _order_priority(order: dict) -> int:
        # Free cash first, then the next new entry, then increase existing
        # allocations with whatever cash remains.
        if order.get("side") == "SELL":
            return 0
        if order.get("side") == "REBALANCE" and order.get("action") == "reduce":
            return 1
        if order.get("side") == "BUY":
            return 2
        if order.get("side") == "REBALANCE":
            return 3
        return 4

    def fill_pending(self, bar: pd.DataFrame, trade_date: date) -> list[dict]:
        """Execute pending orders at today's OPEN. `bar` is indexed by ticker
        with columns open/close."""
        events = []
        still_pending = []

        for order in sorted(self.s.pending_orders, key=self._order_priority):
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
            elif order["side"] == "SELL":
                events.append(self._execute_sell(order, px_open, trade_date))
            elif order["side"] == "REBALANCE":
                events.append(self._execute_rebalance(order, px_open, trade_date))
            else:
                events.append({"event": "order_stale", "ticker": t,
                               "reason": "unknown order side"})

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
            "cumulative_cost_basis": float(cost),
            "cumulative_exit_proceeds": 0.0,
            "cumulative_shares_sold": 0.0,
            "total_shares_bought": float(shares),
            "allocation_weight": float(order.get("allocation_weight", 1.0)),
            "entry_bar": self.s.bar_count,
            "entry_score": order.get("entry_score"),
            "signal_date": order.get("signal_date"),
            "signal_id": order.get("signal_id"),
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
        proceeds_now = pos["shares"] * fill - self.commission
        self.s.cash += proceeds_now
        proceeds = float(pos.get("cumulative_exit_proceeds", 0.0)) + proceeds_now
        total_cost_basis = float(pos.get("cumulative_cost_basis", pos["cost_basis"]))
        shares_sold = float(pos.get("cumulative_shares_sold", 0.0)) + float(pos["shares"])
        total_shares_bought = float(pos.get("total_shares_bought", shares_sold))
        average_entry_price = total_cost_basis / total_shares_bought if total_shares_bought else pos["entry_price"]
        pnl = proceeds - total_cost_basis
        ret = pnl / total_cost_basis if total_cost_basis else 0.0

        self.trades.append({
            "trade_id": tid,
            "ticker": pos["ticker"],
            "name": pos.get("name", ""),
            "sector": pos.get("sector", "Unknown"),
            "entry_date": pos["entry_date"],
            "entry_price": round(average_entry_price, 4),
            "exit_date": trade_date.isoformat(),
            "exit_price": round(fill, 4),
            "shares": round(shares_sold, 6),
            "cost_basis": round(total_cost_basis, 2),
            "proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(ret, 6),
            "bars_held": self.s.bar_count - pos["entry_bar"],
            "entry_score": pos.get("entry_score"),
            "signal_date": pos.get("signal_date"),
            "signal_id": pos.get("signal_id"),
            "exit_reason": order.get("reason", "signal weakened"),
            "av_verdict": pos.get("av_verdict", "unverified"),
        })
        self.s.positions = [p for p in self.s.positions if p["trade_id"] != tid]
        return {"event": "sold", "ticker": pos["ticker"], "price": fill,
                "pnl": pnl, "trade_id": tid, "reason": order.get("reason", "signal weakened")}


    def _execute_rebalance(self, order: dict, px_open: float, trade_date: date) -> dict:
        """Move a position toward its target dollar value at the next open."""
        tid = order.get("trade_id")
        pos = next((p for p in self.s.positions if p["trade_id"] == tid), None)
        if pos is None:
            return {"event": "order_stale", "ticker": order.get("ticker")}

        old_shares = float(pos.get("shares", 0.0))
        if old_shares <= 0:
            return {"event": "rebalance_skipped", "ticker": pos["ticker"],
                    "reason": "position has no shares"}

        target_value = max(0.0, float(order.get("target_value", 0.0)))
        desired_shares = target_value / px_open if px_open > 0 else old_shares
        delta = desired_shares - old_shares

        if delta < 0:
            shares_to_sell = min(old_shares, -delta)
            fill = self._sell_price(px_open)
            proceeds = shares_to_sell * fill - self.commission
            cost_basis = float(pos.get("cost_basis", 0.0))
            cost_removed = cost_basis * shares_to_sell / old_shares
            self.s.cash += proceeds
            pos["shares"] = round(old_shares - shares_to_sell, 6)
            pos["cost_basis"] = max(0.0, round(cost_basis - cost_removed, 2))
            pos["cumulative_exit_proceeds"] = (
                float(pos.get("cumulative_exit_proceeds", 0.0)) + proceeds
            )
            pos["cumulative_shares_sold"] = (
                float(pos.get("cumulative_shares_sold", 0.0)) + shares_to_sell
            )
            event = {
                "event": "rebalanced_down", "ticker": pos["ticker"],
                "shares_sold": shares_to_sell, "price": fill, "proceeds": proceeds,
            }
        elif delta > 0:
            fill = self._buy_price(px_open)
            cash_available = max(0.0, self.s.cash - self.commission)
            shares_to_buy = min(delta, self._shares(cash_available, fill))
            if shares_to_buy <= 0:
                return {"event": "rebalance_skipped", "ticker": pos["ticker"],
                        "reason": "insufficient cash"}
            cost = shares_to_buy * fill + self.commission
            self.s.cash -= cost
            pos["shares"] = round(old_shares + shares_to_buy, 6)
            pos["cost_basis"] = round(float(pos.get("cost_basis", 0.0)) + cost, 2)
            pos["cumulative_cost_basis"] = (
                float(pos.get("cumulative_cost_basis", pos.get("cost_basis", 0.0))) + cost
            )
            pos["total_shares_bought"] = (
                float(pos.get("total_shares_bought", old_shares)) + shares_to_buy
            )
            total_bought = float(pos["total_shares_bought"])
            pos["average_entry_price"] = (
                float(pos["cumulative_cost_basis"]) / total_bought if total_bought else fill
            )
            event = {
                "event": "rebalanced_up", "ticker": pos["ticker"],
                "shares_bought": shares_to_buy, "price": fill, "cost": cost,
            }
        else:
            return {"event": "rebalance_skipped", "ticker": pos["ticker"],
                    "reason": "target already met"}

        pos["last_price"] = round(px_open, 4)
        pos["last_value"] = round(float(pos["shares"]) * px_open, 2)
        pos["last_rebalance_bar"] = self.s.bar_count
        pos["allocation_weight"] = float(order.get("allocation_weight", pos.get("allocation_weight", 1.0)))
        return event

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
    def schedule_exits(self, scored: pd.DataFrame | None = None) -> list[dict]:
        """Queue exits when the stock's cross-sectional signal stays weak.

        No maximum holding-period timer is used. Weakness is measured by
        rank percentile within the currently eligible universe and confirmed
        across consecutive daily reviews to avoid a one-day ranking fluctuation.
        """
        queued = []
        pending_sell_ids = {o.get("trade_id") for o in self.s.pending_orders
                            if o["side"] == "SELL"}
        if scored is not None and not scored.empty:
            eligible_mask = (
                scored.get("eligible", pd.Series(False, index=scored.index))
                .fillna(False).astype(bool)
                & pd.to_numeric(
                    scored.get("composite", pd.Series(np.nan, index=scored.index)),
                    errors="coerce",
                ).notna()
            )
            eligible_count = int(eligible_mask.sum())
        else:
            eligible_count = 0

        for p in self.s.positions:
            if p["trade_id"] in pending_sell_ids:
                continue

            reason = None
            ticker = p["ticker"]
            if int(p.get("stale_days", 0)) >= 5:
                reason = "price data lost for 5 sessions"
            elif scored is not None and ticker in scored.index:
                row = scored.loc[ticker]
                eligible = bool(row.get("eligible", False)) if pd.notna(row.get("eligible", False)) else False
                rank = pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").iloc[0]
                if not eligible:
                    weak = True
                    review_detail = str(row.get("exclude_reason") or "screen eligibility lost")
                    rank_pct = None
                elif pd.notna(rank) and eligible_count > 1:
                    rank_pct = float((eligible_count - float(rank)) / (eligible_count - 1))
                    weak = rank_pct < self.exit_rank_floor
                    review_detail = (
                        f"rank percentile {rank_pct:.0%} below threshold {self.exit_rank_floor:.0%}"
                        if weak else f"rank percentile {rank_pct:.0%} remains acceptable"
                    )
                elif eligible and pd.notna(rank):
                    rank_pct = 1.0
                    weak = False
                    review_detail = "only one eligible stock; signal retained"
                else:
                    rank_pct = None
                    weak = False
                    review_detail = "rank unavailable; no exit decision"
                p["last_composite"] = (
                    float(row["composite"]) if pd.notna(row.get("composite")) else None
                )
                p["last_rank_percentile"] = rank_pct
                if weak:
                    p["weak_signal_runs"] = int(p.get("weak_signal_runs", 0)) + 1
                else:
                    p["weak_signal_runs"] = 0
                p["last_signal_review"] = review_detail
                if weak and p["weak_signal_runs"] >= self.exit_confirmation:
                    reason = (
                        f"signal weakened for {p['weak_signal_runs']} consecutive reviews: "
                        f"{review_detail}"
                    )
            if reason:
                order = {"side": "SELL", "ticker": ticker,
                         "trade_id": p["trade_id"], "reason": reason,
                         "notional": 0.0}
                self.s.pending_orders.append(order)
                queued.append(order)
        return queued


    def _cash_reserve_pct(self) -> float:
        value = float(self.cfg.strategy.get("cash_reserve_pct", 0.10))
        return min(0.50, max(0.0, value))

    def _eligible_count(self, scored: pd.DataFrame | None) -> int:
        if scored is None or scored.empty:
            return 0
        eligible = scored.get("eligible", pd.Series(False, index=scored.index)).fillna(False).astype(bool)
        composite = pd.to_numeric(
            scored.get("composite", pd.Series(np.nan, index=scored.index)),
            errors="coerce",
        )
        return int((eligible & composite.notna()).sum())

    def _weight_for_position(
        self, position: dict, scored: pd.DataFrame | None, eligible_count: int,
        pending_sell_ids: set,
    ) -> float:
        # A queued exit occupies a slot until its next-open fill, but should
        # not distort the allocation normalisation for the remaining holdings.
        if position.get("trade_id") in pending_sell_ids:
            return 1.0
        ticker = position.get("ticker")
        if scored is not None and not scored.empty and ticker in scored.index:
            row = scored.loc[ticker]
            eligible = bool(row.get("eligible", False)) if pd.notna(row.get("eligible", False)) else False
            rank = pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").iloc[0]
            if eligible and pd.notna(rank) and eligible_count > 1:
                pct = float((eligible_count - float(rank)) / (eligible_count - 1))
                return allocation_multiplier(pct, self.cfg.strategy)
            if eligible and pd.notna(rank) and eligible_count == 1:
                return allocation_multiplier(1.0, self.cfg.strategy)
            return allocation_multiplier(None, self.cfg.strategy)
        try:
            weight = float(position.get("allocation_weight", 1.0))
            return weight if np.isfinite(weight) and weight > 0 else 1.0
        except (TypeError, ValueError):
            return 1.0

    def _average_slot_weight(
        self, scored: pd.DataFrame | None, candidate_weight: float | None = None,
    ) -> float:
        eligible_count = self._eligible_count(scored)
        pending_sell_ids = {
            o.get("trade_id") for o in self.s.pending_orders if o.get("side") == "SELL"
        }
        weights = [
            self._weight_for_position(p, scored, eligible_count, pending_sell_ids)
            for p in self.s.positions
        ]
        weights.extend(
            max(0.1, float(o.get("allocation_weight", 1.0)))
            for o in self.s.pending_orders if o.get("side") == "BUY"
        )
        if candidate_weight is not None:
            weights.append(max(0.1, float(candidate_weight)))
        remaining_slots = max(0, self.n_slots - len(weights))
        weights.extend([1.0] * remaining_slots)
        return max(sum(weights) / max(1, self.n_slots), 1e-9)

    # -- step 4: queue the entry ------------------------------------------
    def schedule_entry(
        self, shortlist: list[dict], name_map: dict | None = None,
        signal_date: date | None = None, scored: pd.DataFrame | None = None,
    ) -> dict | None:
        """Queue one rank-weighted BUY if a portfolio slot is free."""
        if self.s.open_slots(self.n_slots) <= 0 or not shortlist:
            return None

        pick = shortlist[0]
        try:
            weight = float(pick.get("allocation_weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not np.isfinite(weight) or weight <= 0:
            weight = 1.0
        average_weight = self._average_slot_weight(scored, candidate_weight=weight)
        base_slot_value = self.s.equity() * (1.0 - self._cash_reserve_pct()) / self.n_slots
        target = base_slot_value * weight / average_weight
        budget = min(target, self.s.available_cash)
        if budget < 25.0:                    # not worth a position
            return None

        order = {
            "side": "BUY",
            "ticker": pick["ticker"],
            "sector": pick.get("sector", "Unknown"),
            "name": (name_map or {}).get(pick["ticker"], ""),
            "notional": round(budget, 2),
            "allocation_weight": weight,
            "rank_percentile": pick.get("rank_percentile"),
            "entry_score": pick.get("composite"),
            "rank": pick.get("rank"),
            "blocks": pick.get("blocks"),
            "signal_close": pick.get("last_close"),
            "signal_date": signal_date.isoformat() if signal_date else None,
            "signal_id": f"{signal_date.isoformat()}:{pick['ticker']}" if signal_date else None,
            "av_verdict": pick.get("av_verdict", "unverified"),
            "av_reason": pick.get("av_reason", ""),
            "reason": "rank-weighted composite score",
        }
        self.s.pending_orders.append(order)
        return order

    def schedule_rebalances(self, scored: pd.DataFrame | None = None) -> list[dict]:
        """Queue material target-weight adjustments for the next session open."""
        if scored is None or scored.empty:
            return []

        eligible_count = self._eligible_count(scored)
        pending_sell_ids = {
            o.get("trade_id") for o in self.s.pending_orders if o.get("side") == "SELL"
        }
        pending_rebalance_ids = {
            o.get("trade_id") for o in self.s.pending_orders if o.get("side") == "REBALANCE"
        }
        average_weight = self._average_slot_weight(scored)
        base_slot_value = self.s.equity() * (1.0 - self._cash_reserve_pct()) / self.n_slots
        min_change_pct = max(0.0, float(self.cfg.strategy.get("rebalance_threshold_pct", 0.15)))
        min_notional = max(0.0, float(self.cfg.strategy.get("rebalance_min_notional", 50.0)))
        interval = max(1, int(self.cfg.strategy.get("rebalance_interval_sessions", 5)))
        reductions, increases = [], []

        for position in self.s.positions:
            tid = position.get("trade_id")
            if tid in pending_sell_ids or tid in pending_rebalance_ids:
                continue
            # Do not resize a holding while its exit signal is being confirmed.
            if int(position.get("weak_signal_runs", 0)) > 0:
                continue
            last_rebalance = int(position.get("last_rebalance_bar", position.get("entry_bar", 0)))
            if self.s.bar_count - last_rebalance < interval:
                continue

            weight = self._weight_for_position(position, scored, eligible_count, pending_sell_ids)
            target_value = base_slot_value * weight / average_weight
            current_value = float(position.get("last_value", position.get("cost_basis", 0.0)))
            difference = target_value - current_value
            if current_value <= 0 or abs(difference) < min_notional:
                continue
            if abs(difference) / max(current_value, 1.0) < min_change_pct:
                continue

            order = {
                "side": "REBALANCE",
                "action": "increase" if difference > 0 else "reduce",
                "ticker": position["ticker"],
                "trade_id": tid,
                "target_value": round(target_value, 2),
                "current_value": round(current_value, 2),
                "notional": round(max(0.0, difference), 2),
                "allocation_weight": weight,
                "reason": "rank-weighted target allocation",
            }
            (increases if difference > 0 else reductions).append(order)

        queued = reductions + increases
        self.s.pending_orders.extend(queued)
        return queued

    # -- full daily cycle --------------------------------------------------
    def run_day(
        self,
        bar: pd.DataFrame,
        trade_date: date,
        shortlist: list[dict] | None = None,
        name_map: dict | None = None,
        scored: pd.DataFrame | None = None,
    ) -> dict:
        """Process one trading day. `bar` = today's open/close indexed by ticker."""
        self.s.bar_count += 1
        events = self.fill_pending(bar, trade_date)
        self.mark_to_market(bar)
        exits = self.schedule_exits(scored)
        entry = self.schedule_entry(
            shortlist or [], name_map, signal_date=trade_date, scored=scored
        )
        rebalances = self.schedule_rebalances(scored)
        self.s.last_run_date = trade_date.isoformat()
        return {
            "date": trade_date.isoformat(),
            "events": events,
            "exits_queued": len(exits),
            "entry_queued": entry["ticker"] if entry else None,
            "rebalances_queued": len(rebalances),
            "cash": round(self.s.cash, 2),
            "positions": len(self.s.positions),
            "equity": round(self.s.equity(), 2),
        }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def append_trades(trades: list[dict], path: Path) -> None:
    """Persist closed trades while safely upgrading older CSV schemas.

    New trade records carry lifecycle IDs that older CSV columns may not
    contain. Rewriting the union of old/new columns prevents the header from
    becoming misaligned. Deduplication by trade_id also makes a retried daily
    run idempotent if it wrote a trade before state was saved.
    """
    if not trades:
        return

    incoming = pd.DataFrame(trades)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming

    if "trade_id" in combined.columns:
        has_id = combined["trade_id"].notna()
        identified = combined.loc[has_id].drop_duplicates(
            subset=["trade_id"], keep="last"
        )
        unidentified = combined.loc[~has_id]
        combined = pd.concat([unidentified, identified], ignore_index=True, sort=False)

    combined.to_csv(path, index=False)


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
