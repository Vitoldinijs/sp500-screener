"""Link completed paper trades to the exact screen decision that initiated them.

This is event-driven outcome data: the target is the realised result when the
position actually closes. There is no fixed 21/63/126-session forward-return
label and no trading/learning decision is made in this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRADE_FIELDS = (
    "trade_id", "ticker", "name", "sector", "signal_id", "signal_date",
    "entry_date", "entry_price", "exit_date", "exit_price", "shares",
    "cost_basis", "proceeds", "pnl", "return_pct", "bars_held",
    "exit_reason", "av_verdict",
)
SIGNAL_FEATURE_FIELDS = (
    "last_close", "eligible", "rank", "rank_percentile", "composite",
    "blocks_used", "factor_agreement_positive_share", "fundamental_coverage",
    "is_chosen_candidate", "shortlisted", "portfolio_action", "action_reason",
    "price_provider", "fundamentals_asof",
) + tuple(f"block_{name}" for name in (
    "value", "quality", "growth", "momentum", "earnings", "risk", "technical"
))
OUTCOME_FIELDS = TRADE_FIELDS + (
    "closed_outcome", "feature_link_status", "learning_eligible",
) + tuple(f"signal_{name}" for name in SIGNAL_FEATURE_FIELDS)


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _truthy(value) -> bool:
    if _missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _text(value) -> str | None:
    if _missing(value):
        return None
    result = str(value).strip()
    return result or None


def build_signal_outcomes(decisions: pd.DataFrame,
                          trades: pd.DataFrame) -> pd.DataFrame:
    """Build one row per completed trade, joined to its entry-day signal.

    Historical trades predating signal IDs are retained for audit but marked
    ineligible for learning. A date+ticker ID is reconstructed only when the
    trade has a signal date and the matching decision snapshot actually exists.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=list(OUTCOME_FIELDS))

    decisions = decisions.copy() if decisions is not None else pd.DataFrame()
    if not decisions.empty and "signal_id" in decisions.columns:
        decisions["signal_id"] = decisions["signal_id"].map(_text)
        decisions = decisions[decisions["signal_id"].notna()]
        # A signal ID should be unique. In a damaged/hand-edited ledger, choose
        # the latest stored row deterministically and never multiply a trade.
        decisions = decisions.drop_duplicates("signal_id", keep="last")
        by_signal = decisions.set_index("signal_id", drop=False).to_dict("index")
    else:
        by_signal = {}

    trade_rows = trades.copy()
    if "trade_id" in trade_rows.columns:
        trade_rows = trade_rows.drop_duplicates("trade_id", keep="last")

    records = []
    for _, trade in trade_rows.iterrows():
        row = {field: trade.get(field) for field in TRADE_FIELDS}
        ticker = _text(trade.get("ticker"))
        original_id = _text(trade.get("signal_id"))
        entry_signal_date = _text(trade.get("signal_date"))
        signal_id = original_id
        reconstructed = False
        if signal_id is None and entry_signal_date and ticker:
            signal_id = f"{entry_signal_date}:{ticker}"
            reconstructed = True

        match = by_signal.get(signal_id) if signal_id else None
        status = "missing_signal_id"
        if original_id and match is None:
            status = "signal_snapshot_missing"
        elif signal_id and match is None:
            status = "signal_snapshot_missing"
        elif match is not None:
            match_ticker = _text(match.get("ticker"))
            match_date = _text(match.get("data_date"))
            if ticker and match_ticker != ticker:
                status = "ticker_mismatch"
                match = None
            elif entry_signal_date and match_date and entry_signal_date != match_date:
                status = "signal_date_mismatch"
                match = None
            else:
                status = "linked_reconstructed" if reconstructed else "linked"

        # Keep a reconstructed identifier only when it found an exact row;
        # otherwise preserve the missing original ID as a visible data issue.
        row["signal_id"] = signal_id if (match is not None or original_id) else None
        if _missing(row.get("signal_date")) and match is not None:
            row["signal_date"] = match.get("data_date")

        pnl = pd.to_numeric(pd.Series([trade.get("pnl")]), errors="coerce").iloc[0]
        if pd.isna(pnl):
            closed_outcome = None
        elif pnl > 0:
            closed_outcome = "win"
        elif pnl < 0:
            closed_outcome = "loss"
        else:
            closed_outcome = "flat"
        row["closed_outcome"] = closed_outcome
        row["feature_link_status"] = status

        selected = _truthy(match.get("is_chosen_candidate")) if match is not None else False
        eligible = _truthy(match.get("eligible")) if match is not None else False
        composite = pd.to_numeric(
            pd.Series([match.get("composite") if match is not None else np.nan]),
            errors="coerce",
        ).iloc[0]
        realised_return = pd.to_numeric(
            pd.Series([trade.get("return_pct")]), errors="coerce"
        ).iloc[0]
        row["learning_eligible"] = bool(
            match is not None and status in {"linked", "linked_reconstructed"}
            and selected and eligible and pd.notna(composite)
            and pd.notna(realised_return) and closed_outcome is not None
        )

        for field in SIGNAL_FEATURE_FIELDS:
            row[f"signal_{field}"] = match.get(field) if match is not None else None
        records.append(row)

    result = pd.DataFrame(records, columns=list(OUTCOME_FIELDS))
    if "trade_id" in result.columns:
        result = result.drop_duplicates("trade_id", keep="last")
    if "exit_date" in result.columns:
        result = result.sort_values(["exit_date", "trade_id"], kind="stable",
                                    na_position="last")
    return result.reset_index(drop=True)


def reconcile_signal_outcomes(decision_history_path: Path,
                              trades_path: Path,
                              output_path: Path) -> dict:
    """Rebuild the derived outcome ledger idempotently from source CSVs."""
    def read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            return pd.DataFrame()

    decisions = read(decision_history_path)
    trades = read(trades_path)
    outcomes = build_signal_outcomes(decisions, trades)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output_path, index=False)

    linked = outcomes["feature_link_status"].isin(
        ["linked", "linked_reconstructed"]
    ) if not outcomes.empty else pd.Series(dtype=bool)
    eligible = outcomes["learning_eligible"].fillna(False).astype(bool) \
        if not outcomes.empty else pd.Series(dtype=bool)
    return {
        "closed_trades": int(len(outcomes)),
        "linked_trades": int(linked.sum()),
        "unlinked_trades": int(len(outcomes) - linked.sum()),
        "learning_eligible_trades": int(eligible.sum()),
        "output_path": str(output_path),
    }
