"""Factor-weight constraints and helpers (automatic updates paused).

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
  * nothing changes until ``min_trades_for_update`` real trades have closed
  * automatic tuning remains paused until lifecycle outcomes and validation
    are implemented and reviewed
  * fundamental blocks get a shrink factor while point-in-time coverage is
    still thin, since the snapshot history only starts accumulating now

Weight-proposal helpers remain available for a future validated lifecycle learner.
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
def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation; NaN if fewer than 8 overlapping observations."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 8:
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
    """No fixed-horizon target is generated.

    The previous implementation labelled every stock with an arbitrary
    forward return. That method is intentionally removed. The next learner
    will evaluate completed signal lifecycles using actual entry/exit events.
    """
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
# orchestration
# ---------------------------------------------------------------------------
def run_learning(cfg, prices_long: pd.DataFrame, universe: pd.DataFrame,
                 dry_run: bool = False) -> dict:
    """Learning is paused until completed lifecycle outcomes can be evaluated.

    Do not fall back to forward returns at fixed day counts. Weight changes
    require a separate event-driven evaluation design and validation.
    """
    current = json.dumps(cfg.weights)
    return {
        "run_date": date.today().isoformat(),
        "mode": "off",
        "decision": "skipped",
        "reason": "learning disabled pending lifecycle-based evaluation; fixed-horizon targets are not used",
        "weights_before": current,
        "weights_after": current,
    }


def _append_log(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "logged_at": datetime.utcnow().isoformat(timespec="seconds")}
    df = pd.DataFrame([rec])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
