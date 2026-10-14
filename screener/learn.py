"""Lifecycle-driven factor-weight learning with strict validation gates.

How it learns
-------------
Once a month the job reconstructs the screener's own scoring table *as it
would have looked* on a grid of past dates, using only data available then:

  * price factors   <- the cached OHLCV history, truncated at that date
  * fundamentals    <- the newest weekly snapshot dated on/before that date

The former implementation used a fixed forward-return target. It is now
disabled and removed; the replacement will evaluate complete position
lifecycles from actual signal and execution records. No fixed target horizon
is used for learning.

Guardrails, because factor timing is notoriously easy to overfit:
  * weights move at most ``max_step_per_month`` (0.05) per run
  * every weight stays inside [weight_min, weight_max] and they sum to 1
  * nothing changes until ``min_trades_for_update`` linked real trades have closed
  * training trades must have exited before the chronological holdout begins
  * proposals must improve score/return rank correlation on the holdout
  * weights stay bounded and move by no more than the configured monthly step
  * sparse factor coverage shrinks adjustments
  * the same outcome sample is never tuned repeatedly without new results.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# Must stay in sync with config.yml's `metric_blocks` keys — this list drives
# both the walk-forward IC evaluation below AND what `propose_weights` writes
# back into config.yml's `factor_weights` (see Config.save_weights, which
# REPLACES the whole block). A block present in metric_blocks but missing
# here would silently vanish from config.yml the first time learning runs.
BLOCKS = ("value", "quality", "growth", "momentum", "earnings", "risk", "technical")


# ---------------------------------------------------------------------------
# statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def spearman(a: pd.Series, b: pd.Series, minimum_n: int = 8) -> float:
    """Rank correlation; NaN when fewer than the requested pairs overlap."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < max(3, int(minimum_n)):
        return np.nan
    ra = df.iloc[:, 0].rank()
    rb = df.iloc[:, 1].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return np.nan
    return float(ra.corr(rb))


# ---------------------------------------------------------------------------
# walk-forward panel
# ---------------------------------------------------------------------------
def build_panel(
    prices_long: pd.DataFrame,
    universe: pd.DataFrame,
    fundamentals_path: Path,
    cfg,
    grid_step: int = 5,
    max_dates: int = 200,
) -> tuple[pd.DataFrame, list[dict]]:
    """Retired fixed-horizon panel builder; lifecycle outcomes drive learning."""
    return pd.DataFrame(), []


# ---------------------------------------------------------------------------
# weight proposal
# ---------------------------------------------------------------------------
def _renorm_with_caps(vals: dict, lo: dict, hi: dict, target_sum: float = 1.0,
                       max_iter: int = 20) -> dict:
    """Scale `vals` to sum to `target_sum`, water-filling within [lo, hi].

    A uniform rescale (`v / sum(vals) * target_sum`) ignores the per-block
    band and can push a block's *effective* step well past what it looks
    like it moved before the rescale. This instead only ever pushes the
    remaining budget onto blocks that still have room, iterating until
    either the budget is placed or every block is pinned to a bound.
    """
    vals = dict(vals)
    for _ in range(max_iter):
        diff = target_sum - sum(vals.values())
        if abs(diff) < 1e-9:
            break
        free = [k for k, v in vals.items()
                if not ((diff > 0 and v >= hi[k] - 1e-9) or
                        (diff < 0 and v <= lo[k] + 1e-9))]
        if not free:
            break
        share = diff / len(free)
        for k in free:
            vals[k] = min(hi[k], max(lo[k], vals[k] + share))
    return vals


def propose_weights(current: dict, ic_means: dict, cfg, confidence: dict | None = None) -> dict:
    """Blend current weights toward IC-implied weights, respecting bounds.

    Two constraints must hold on the weights this RETURNS, not just on some
    intermediate step: each stays within [weight_min, weight_max], and none
    moves more than `max_step_per_month` from `current`. Those per-block
    step caps are enforced as a band, and only the leftover budget needed to
    reach sum=1.0 gets water-filled across blocks that still have room —
    see `_renorm_with_caps`.
    """
    L = cfg.learning
    wmin, wmax = float(L["weight_min"]), float(L["weight_max"])
    step = float(L["max_step_per_month"])
    confidence = confidence or {}

    # Target: proportional to positive IC. A block with no/negative signal
    # falls back to the floor rather than going to zero.
    pos = {b: max(0.0, float(ic_means.get(b, 0.0) or 0.0)) for b in BLOCKS}
    total = sum(pos.values())
    if total <= 0:
        target = {b: 1.0 / len(BLOCKS) for b in BLOCKS}   # no signal: equal weight
    else:
        target = {b: pos[b] / total for b in BLOCKS}

    cur = {b: float(current.get(b, 0.0)) for b in BLOCKS}
    lo = {b: max(wmin, cur[b] - step) for b in BLOCKS}
    hi = {b: min(wmax, cur[b] + step) for b in BLOCKS}

    proposed = {}
    for b in BLOCKS:
        conf = float(confidence.get(b, 1.0))
        delta = (target[b] - cur[b]) * conf
        delta = max(-step, min(step, delta))
        proposed[b] = min(hi[b], max(lo[b], cur[b] + delta))

    proposed = _renorm_with_caps(proposed, lo, hi)
    s = sum(proposed.values())
    if s <= 0:
        return proposed
    return {b: round(v / s, 4) for b, v in proposed.items()}


def _coverage_confidence(ic_frame: pd.DataFrame) -> dict:
    """Shrink adjustments for blocks whose historical coverage is thin.

    Fundamental snapshots only start accumulating when the bot goes live, so
    early on the value/quality/growth/earnings ICs are measured on stale or
    partly missing data. Momentum, risk and technical are computed purely
    from prices and are always fully covered, so they get full confidence.
    """
    conf = {}
    for b in BLOCKS:
        col = f"cov_{b}"
        cov = float(ic_frame[col].mean()) if col in ic_frame else 0.0
        conf[b] = float(np.clip(cov, 0.0, 1.0))
    return conf


# ---------------------------------------------------------------------------
# lifecycle learner
# ---------------------------------------------------------------------------
def _truthy(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _load_eligible_outcomes(path: Path, lookback_months: int) -> pd.DataFrame:
    """Load only exact-linked, completed trades, never forward-return labels."""
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()
    if data.empty or "learning_eligible" not in data:
        return pd.DataFrame()

    data = data.loc[data["learning_eligible"].map(_truthy)].copy()
    if data.empty:
        return data
    data["return_pct"] = pd.to_numeric(data.get("return_pct"), errors="coerce")
    data["signal_date_parsed"] = pd.to_datetime(
        data.get("signal_date"), errors="coerce", utc=True
    )
    data["exit_date_parsed"] = pd.to_datetime(
        data.get("exit_date"), errors="coerce", utc=True
    )
    valid = (
        data["return_pct"].notna()
        & np.isfinite(data["return_pct"])
        & data["signal_date_parsed"].notna()
        & data["exit_date_parsed"].notna()
        & (data["exit_date_parsed"] >= data["signal_date_parsed"])
    )
    data = data.loc[valid].copy()
    if data.empty:
        return data

    if "trade_id" in data:
        data = data.drop_duplicates("trade_id", keep="last")
    data = data.sort_values(["signal_date_parsed", "exit_date_parsed"], kind="stable")
    latest_signal = data["signal_date_parsed"].max()
    cutoff = latest_signal - pd.DateOffset(months=max(1, int(lookback_months)))
    return data.loc[data["signal_date_parsed"] >= cutoff].reset_index(drop=True)


def _weighted_composite(rows: pd.DataFrame, weights: dict) -> pd.Series:
    """Rebuild a row-wise weighted composite, renormalising missing blocks."""
    numerator = pd.Series(0.0, index=rows.index, dtype=float)
    denominator = pd.Series(0.0, index=rows.index, dtype=float)
    for block in BLOCKS:
        weight = float(weights.get(block, 0.0))
        if weight <= 0:
            continue
        column = f"signal_block_{block}"
        values = pd.to_numeric(
            rows[column] if column in rows else pd.Series(np.nan, index=rows.index),
            errors="coerce",
        )
        present = values.notna() & np.isfinite(values)
        numerator.loc[present] += values.loc[present] * weight
        denominator.loc[present] += weight
    return numerator.div(denominator.where(denominator > 0))


def _last_learning_record(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        history = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return None, None
    if history.empty:
        return None, None
    row = history.iloc[-1]
    sample_key, decision = row.get("sample_key"), row.get("decision")
    sample_key = None if pd.isna(sample_key) else str(sample_key)
    decision = None if pd.isna(decision) else str(decision)
    return sample_key, decision


def _log_learning_decision(cfg, decision: str, reason: str, before: dict,
                           after: dict, **details) -> dict:
    rec = {
        "run_date": date.today().isoformat(),
        "mode": "lifecycle",
        "decision": decision,
        "reason": reason,
        "weights_before": json.dumps(before, sort_keys=True),
        "weights_after": json.dumps(after, sort_keys=True),
        **details,
    }
    _append_log(rec, cfg.path("weights_history"))
    return rec


def run_learning(cfg, prices_long: pd.DataFrame, universe: pd.DataFrame,
                 dry_run: bool = False) -> dict:
    """Update weights only when a held-out, completed-trade check improves."""
    current_weights = {b: float(cfg.weights.get(b, 0.0)) for b in BLOCKS}
    if not bool(cfg.learning.get("enabled", False)):
        return _log_learning_decision(
            cfg, decision="skipped", reason="automatic lifecycle learning is disabled",
            before=current_weights, after=current_weights, sample_key="disabled",
            training_n=0, validation_n=0,
        )

    data = _load_eligible_outcomes(
        cfg.path("signal_outcomes_csv"),
        int(cfg.learning.get("lookback_months", 18)),
    )
    minimum_total = max(1, int(cfg.learning.get("min_trades_for_update", 40)))
    minimum_train = max(1, int(cfg.learning.get("min_training_trades", 20)))
    minimum_validation = max(1, int(cfg.learning.get("min_validation_trades", 10)))
    minimum_factor_n = max(3, int(cfg.learning.get("min_factor_observations", 8)))
    validation_fraction = float(np.clip(
        cfg.learning.get("validation_fraction", 0.30), 0.15, 0.50
    ))

    if data.empty:
        sample_key = "0|none|none|none"
        cutoff = None
        training = validation = data.copy()
        latest_exit = None
    else:
        latest_signal = data["signal_date_parsed"].max()
        latest_exit = data["exit_date_parsed"].max()
        unique_dates = sorted(pd.Timestamp(x) for x in data["signal_date_parsed"].unique())
        cut_index = int(np.floor(len(unique_dates) * (1.0 - validation_fraction)))
        cut_index = max(1, min(len(unique_dates) - 1, cut_index)) if len(unique_dates) >= 2 else 0
        cutoff = unique_dates[cut_index] if len(unique_dates) >= 2 else None
        if cutoff is None:
            sample_key = f"{len(data)}|{latest_signal:%Y-%m-%d}|{latest_exit:%Y-%m-%d}|none"
            training = validation = data.iloc[0:0].copy()
        else:
            sample_key = (
                f"{len(data)}|{latest_signal:%Y-%m-%d}|{latest_exit:%Y-%m-%d}|"
                f"{cutoff:%Y-%m-%d}"
            )
            training = data.loc[
                (data["signal_date_parsed"] < cutoff)
                & (data["exit_date_parsed"] < cutoff)
            ].copy()
            validation = data.loc[data["signal_date_parsed"] >= cutoff].copy()

    shared = {
        "sample_key": sample_key,
        "eligible_trades": int(len(data)),
        "training_n": int(len(training)),
        "validation_n": int(len(validation)),
        "validation_cutoff": cutoff.strftime("%Y-%m-%d") if cutoff is not None else None,
        "latest_exit_date": latest_exit.strftime("%Y-%m-%d") if latest_exit is not None else None,
    }
    last_key, last_decision = _last_learning_record(cfg.path("weights_history"))
    if last_key == sample_key and last_decision not in {
        "dry_run_accept", "dry_run_reject", "dry_run_skipped"
    }:
        return _log_learning_decision(
            cfg, decision="skipped",
            reason="no new eligible closed trades since the previous monthly review",
            before=current_weights, after=current_weights, **shared,
        )

    if len(data) < minimum_total:
        return _log_learning_decision(
            cfg, decision="skipped",
            reason=f"only {len(data)} eligible linked closed trades; need at least {minimum_total}",
            before=current_weights, after=current_weights, **shared,
        )
    if len(training) < minimum_train or len(validation) < minimum_validation:
        return _log_learning_decision(
            cfg, decision="skipped",
            reason=(f"temporal split too small: {len(training)} training and "
                    f"{len(validation)} validation trades; need {minimum_train} and "
                    f"{minimum_validation}"),
            before=current_weights, after=current_weights, **shared,
        )

    train_ic: dict[str, float] = {}
    confidence: dict[str, float] = {}
    train_returns = pd.to_numeric(training["return_pct"], errors="coerce")
    for block in BLOCKS:
        column = f"signal_block_{block}"
        factor = pd.to_numeric(
            training[column] if column in training else pd.Series(np.nan, index=training.index),
            errors="coerce",
        )
        pair = pd.concat([factor.rename("factor"), train_returns.rename("ret")], axis=1).dropna()
        ic = spearman(pair["factor"], pair["ret"], minimum_n=minimum_factor_n)
        train_ic[block] = float(ic) if np.isfinite(ic) else 0.0
        coverage = float(len(pair) / len(training)) if len(training) else 0.0
        support = min(1.0, len(pair) / max(30.0, float(minimum_total)))
        confidence[block] = coverage * support

    if not any(value > 0 for value in train_ic.values()):
        return _log_learning_decision(
            cfg, decision="rejected",
            reason="no factor block has a positive training rank correlation",
            before=current_weights, after=current_weights,
            train_factor_ic=json.dumps(train_ic, sort_keys=True), **shared,
        )

    proposal = propose_weights(current_weights, train_ic, cfg, confidence=confidence)
    largest_change = max(abs(proposal[b] - current_weights[b]) for b in BLOCKS)
    if largest_change < 0.0001:
        return _log_learning_decision(
            cfg, decision="skipped", reason="factor evidence does not justify a weight change",
            before=current_weights, after=current_weights,
            train_factor_ic=json.dumps(train_ic, sort_keys=True), **shared,
        )

    validation_returns = pd.to_numeric(validation["return_pct"], errors="coerce")
    baseline_ic = spearman(
        _weighted_composite(validation, current_weights), validation_returns,
        minimum_n=minimum_factor_n,
    )
    proposed_ic = spearman(
        _weighted_composite(validation, proposal), validation_returns,
        minimum_n=minimum_factor_n,
    )
    baseline_train_ic = spearman(
        _weighted_composite(training, current_weights), train_returns,
        minimum_n=minimum_factor_n,
    )
    proposed_train_ic = spearman(
        _weighted_composite(training, proposal), train_returns,
        minimum_n=minimum_factor_n,
    )
    min_improvement = max(0.0, float(
        cfg.learning.get("min_validation_ic_improvement", 0.01)
    ))
    validation_details = {
        **shared,
        "train_factor_ic": json.dumps(train_ic, sort_keys=True),
        "validation_baseline_ic": float(baseline_ic) if np.isfinite(baseline_ic) else None,
        "validation_proposed_ic": float(proposed_ic) if np.isfinite(proposed_ic) else None,
        "training_baseline_ic": float(baseline_train_ic) if np.isfinite(baseline_train_ic) else None,
        "training_proposed_ic": float(proposed_train_ic) if np.isfinite(proposed_train_ic) else None,
        "min_validation_ic_improvement": min_improvement,
    }
    if not np.isfinite(baseline_ic) or not np.isfinite(proposed_ic):
        return _log_learning_decision(
            cfg, decision="rejected",
            reason="validation score/return correlation is unavailable; weights left unchanged",
            before=current_weights, after=current_weights, **validation_details,
        )
    improvement = float(proposed_ic - baseline_ic)
    if proposed_ic <= 0 or improvement < min_improvement:
        return _log_learning_decision(
            cfg, decision="rejected",
            reason=(f"held-out rank correlation did not improve enough "
                    f"({baseline_ic:.3f} -> {proposed_ic:.3f}; need +{min_improvement:.3f})"),
            before=current_weights, after=current_weights,
            validation_improvement=round(improvement, 6), **validation_details,
        )

    if dry_run:
        return _log_learning_decision(
            cfg, decision="dry_run_accept",
            reason=(f"proposed weights improve held-out rank correlation "
                    f"({baseline_ic:.3f} -> {proposed_ic:.3f}); dry run, not saved"),
            before=current_weights, after=proposal,
            validation_improvement=round(improvement, 6), **validation_details,
        )

    cfg.save_weights(proposal)
    return _log_learning_decision(
        cfg, decision="accepted",
        reason=(f"held-out rank correlation improved {baseline_ic:.3f} -> "
                f"{proposed_ic:.3f} (+{improvement:.3f}); weights updated within monthly bounds"),
        before=current_weights, after=proposal,
        validation_improvement=round(improvement, 6), **validation_details,
    )


def _append_log(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "logged_at": datetime.utcnow().isoformat(timespec="seconds")}
    df = pd.DataFrame([rec])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
