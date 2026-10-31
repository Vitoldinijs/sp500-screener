"""Per-stock sensitivity to macro drivers, turned into one scored metric.

Why betas rather than levels
----------------------------
A macro reading is the same number for every company on a given day, so
dropping "fed funds = 4.0" into the scoring table contributes exactly
nothing: zero cross-sectional variance means a z-score of zero for all 500
names. What *does* differ between companies is how strongly each one
responds to that driver — Chevron's sensitivity to crude is not Exxon's,
and a regional bank's sensitivity to the 2-year yield is not an insurer's.
Those sensitivities are a legitimate cross-sectional signal.

The metric
----------
For each stock we estimate betas to a small set of macro drivers, then
score it by what those betas are worth *given where the drivers are
currently heading*:

    macro_tilt_i = sum_f  beta_(i,f) * trend_f

Both sides are in matching units, so the product reads as an expected
return contribution. Oil trending up gives oil-sensitive names a positive
tilt; the Energy sector therefore gains weight as a *consequence* of its
constituents scoring well, with no separate sector-tilt mechanism to
conflict with the sector-neutral scoring of the fundamental blocks.

Deliberate choices
------------------
* **Weekly, not daily, returns.** Daily co-movement with macro is mostly
  noise; weekly sampling over two years still leaves ~100 observations.
* **Joint regression, not one factor at a time.** Oil, the dollar and
  yields move together. Univariate betas would count the same shared
  movement three times; partial betas from one multivariate fit do not.
* **Trend is capped.** The driver's move is measured in its own standard
  deviations and clipped, so one violent month in crude cannot dominate
  the entire composite.
* **No shrinkage beyond the sample-size gate.** Betas below `min_obs`
  observations are returned as NaN rather than estimated badly, and the
  scorer's winsorization handles the remaining tails.

This block is NOT sector-neutralised. Producing sector tilts is the point.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .data import macro as macro_mod

# label -> how that driver's change is measured.
#   "diff"    : change in level, in the series' own units (yields, spreads)
#   "logret"  : log return (prices: crude, the dollar index)
DRIVERS: dict[str, str] = {
    "treasury_10y": "diff",
    "treasury_2y": "diff",
    "oil_wti": "logret",
    "dollar_index": "logret",
    "high_yield_spread": "diff",
}

TREND_CLIP = 2.0        # driver trend capped at +/- 2 of its own sigma


def _weekly(series: pd.Series) -> pd.Series:
    """Last observation of each ISO week, as a date-indexed series."""
    s = series.dropna()
    if s.empty:
        return s
    s = s.sort_index()
    return s.resample("W-FRI").last()


def driver_changes(macro_frame: pd.DataFrame, asof: date,
                   drivers: dict[str, str] | None = None) -> pd.DataFrame:
    """Weekly changes for each macro driver, published-as-of `asof`."""
    drivers = drivers or DRIVERS
    cols = {}
    for label, mode in drivers.items():
        hist = macro_mod.history(macro_frame, label, asof=asof)
        if hist.empty:
            continue
        weekly = _weekly(hist)
        if len(weekly) < 10:
            continue
        if mode == "logret":
            positive = weekly.where(weekly > 0)
            cols[label] = np.log(positive).diff()
        else:
            cols[label] = weekly.diff()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna(how="all")


def driver_trends(changes: pd.DataFrame, lookback_weeks: int = 13) -> dict[str, float]:
    """Where each driver has been heading, in units of its own volatility.

    Measured as the recent cumulative change divided by the standard
    deviation of weekly changes, then clipped. Scaling by the driver's own
    volatility is what makes a 40bp move in the 10-year and a 15% move in
    crude comparable; without it the regression betas and the trend would be
    multiplied on mismatched scales.
    """
    out: dict[str, float] = {}
    if changes is None or changes.empty:
        return out
    for col in changes.columns:
        s = changes[col].dropna()
        if len(s) < lookback_weeks + 5:
            continue
        sigma = float(s.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        recent = float(s.tail(lookback_weeks).sum())
        # Cumulative move over N weeks scales with sqrt(N) under a random
        # walk, so normalise by that rather than by the single-week sigma.
        z = recent / (sigma * np.sqrt(lookback_weeks))
        out[col] = float(np.clip(z, -TREND_CLIP, TREND_CLIP))
    return out


def estimate_betas(close_wide: pd.DataFrame, changes: pd.DataFrame,
                   window_weeks: int = 104, min_obs: int = 60) -> pd.DataFrame:
    """Multivariate betas of weekly stock returns on weekly driver changes.

    Solved once for the whole cross-section: with a design matrix shared by
    every stock, (X'X)^-1 X'Y gives all 500 name's betas in a single solve.
    """
    if close_wide is None or close_wide.empty or changes is None or changes.empty:
        return pd.DataFrame()

    weekly_px = close_wide.sort_index().resample("W-FRI").last()
    rets = np.log(weekly_px.where(weekly_px > 0)).diff()

    X_all = changes.dropna()
    idx = rets.index.intersection(X_all.index)
    if len(idx) < min_obs:
        return pd.DataFrame()
    idx = idx[-window_weeks:]
    X = X_all.loc[idx]
    Y = rets.loc[idx]

    # Only names with a complete return history over the window get a beta.
    # A partially-observed stock would need its own design matrix, and a
    # beta fitted on a handful of weeks is worse than no beta at all — the
    # scorer renormalises cleanly over a missing block.
    complete = Y.columns[Y.notna().sum() >= min_obs]
    Y = Y[complete]
    keep = Y.dropna(axis=1, how="any")
    if keep.empty:
        return pd.DataFrame()

    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    try:
        coef, *_ = np.linalg.lstsq(design, keep.to_numpy(dtype=float), rcond=None)
    except np.linalg.LinAlgError:
        return pd.DataFrame()

    betas = pd.DataFrame(
        coef[1:].T, index=keep.columns, columns=[f"beta_{c}" for c in X.columns]
    )
    return betas


def macro_tilt(close_wide: pd.DataFrame, macro_frame: pd.DataFrame,
               asof: date, cfg=None) -> pd.DataFrame:
    """One row per ticker: raw betas plus the single scored `macro_tilt`.

    Returns an empty frame when macro data is unavailable, which leaves the
    block absent and lets the composite renormalise over the blocks that do
    exist — the same path a name with missing fundamentals already takes.
    """
    settings = dict((cfg or {}).get("macro_factors") or {}) if cfg is not None else {}
    window_weeks = int(settings.get("window_weeks", 104))
    min_obs = int(settings.get("min_observations", 60))
    lookback = int(settings.get("trend_lookback_weeks", 13))

    changes = driver_changes(macro_frame, asof)
    if changes.empty:
        return pd.DataFrame()
    trends = driver_trends(changes, lookback_weeks=lookback)
    if not trends:
        return pd.DataFrame()

    betas = estimate_betas(close_wide, changes[list(trends)],
                           window_weeks=window_weeks, min_obs=min_obs)
    if betas.empty:
        return pd.DataFrame()

    tilt = pd.Series(0.0, index=betas.index)
    for driver, trend in trends.items():
        col = f"beta_{driver}"
        if col in betas.columns:
            tilt = tilt + betas[col].fillna(0.0) * float(trend)

    out = betas.copy()
    out["macro_tilt"] = tilt
    out.index.name = "ticker"
    return out
