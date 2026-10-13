"""Conviction-weighted position sizing shared by scoring and portfolio logic."""
from __future__ import annotations

import math


def allocation_multiplier(rank_percentile, strategy_cfg) -> float:
    """Map cross-sectional rank percentile to a bounded allocation weight.

    The multiplier is a relative target weight, not a return forecast. Stronger
    current ranks receive larger target allocations; weak/unavailable signals
    receive the minimum weight. Portfolio logic normalises weights across slots.
    """
    try:
        low = float(strategy_cfg.get("allocation_min_multiplier", 0.60))
        high = float(strategy_cfg.get("allocation_max_multiplier", 1.80))
        power = float(strategy_cfg.get("allocation_rank_power", 1.50))
    except (TypeError, ValueError):
        low, high, power = 0.60, 1.80, 1.50

    if not math.isfinite(low):
        low = 0.60
    if not math.isfinite(high):
        high = 1.80
    if high < low:
        low, high = high, low
    if not math.isfinite(power) or power <= 0:
        power = 1.50

    try:
        pct = float(rank_percentile)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(pct):
        return low
    pct = min(1.0, max(0.0, pct))
    return low + (high - low) * (pct ** power)
