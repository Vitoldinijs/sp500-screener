"""Cross-sectional scoring: filters -> winsorize -> z-score -> composite.

Scoring is deliberately *relative*: on any given day a stock is judged against
the other members of the index, not against absolute thresholds. That keeps the
screener functional in any market regime — there is always a best-ranked name.

Pipeline per metric:
  1. drop non-sensical values (negative P/E means no earnings, not "cheap")
  2. winsorize the tails (default 2%) so one outlier can't dominate the z-score
  3. z-score across the eligible cross-section
  4. flip the sign for "lower is better" metrics (P/E, debt, ...)

Block score = mean of its available metric z-scores.
Composite    = weighted mean of block scores, weights renormalised over the
               blocks that actually have data for that row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Metrics that are meaningless when non-positive (loss-making / no book value).
POSITIVE_ONLY = {"pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"}


def winsorize(s: pd.Series, pct: float = 0.02) -> pd.Series:
    if s.dropna().empty or pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1.0 - pct)
    if np.isnan(lo) or np.isnan(hi):
        return s
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    v = s.astype(float)
    mu, sd = v.mean(), v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=v.index).where(v.notna())
    return (v - mu) / sd


def _num(value, default: float) -> float:
    """Coerce a config value to float.

    YAML is treacherous here: `5.0e6` without an explicit `+` in the exponent
    parses as the *string* "5.0e6" under YAML 1.1, which would silently turn
    every numeric filter comparison into a TypeError. Coercing at the boundary
    means a hand-edited config can't break the screener.
    """
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        print(f"[scoring] could not read {value!r} as a number; using {default}")
        return default


def apply_filters(metrics: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add `eligible` (bool) and `exclude_reason` (str) columns."""
    f = cfg.filters
    min_price = _num(f.get("min_price"), 5.0)
    min_dvol = _num(f.get("min_dollar_volume"), 5e6)
    min_mcap = _num(f.get("min_market_cap"), 2e9)
    min_hist = int(_num(f.get("min_history_days"), 200))
    max_missing = int(_num(f.get("max_missing_fundamentals"), 3))

    m = metrics.copy()
    reasons = pd.Series("", index=m.index)

    def fail(mask: pd.Series, text: str):
        nonlocal reasons
        mask = mask.fillna(True)  # missing data => fail the check
        reasons = reasons.where(~(mask & (reasons == "")), text)

    def col(name: str) -> pd.Series:
        return pd.to_numeric(m[name], errors="coerce")

    fail(col("last_close") < min_price, f"price < ${min_price:g}")
    fail(col("history_days") < min_hist, "insufficient price history")
    if "dollar_volume" in m:
        fail(col("dollar_volume") < min_dvol, "illiquid")
    if "market_cap" in m:
        fail(col("market_cap") < min_mcap, "market cap too small")

    # Fundamental coverage: count how many of the configured fundamental
    # metrics are missing for each name.
    fund_metrics = [
        c for block in ("value", "quality", "growth")
        for c in cfg["metric_blocks"][block]
        if c in m.columns
    ]
    if fund_metrics:
        missing = m[fund_metrics].isna().sum(axis=1)
        m["missing_fundamentals"] = missing
        fail(missing > max_missing, "fundamentals incomplete")
    else:
        m["missing_fundamentals"] = 0

    m["exclude_reason"] = reasons
    m["eligible"] = reasons == ""
    return m


def score(metrics: pd.DataFrame, cfg, weights: dict | None = None) -> pd.DataFrame:
    """Return the scoring table sorted by composite score, best first."""
    if metrics.empty:
        return metrics

    weights = dict(weights or cfg.weights)
    blocks: dict = cfg["metric_blocks"]
    directions: dict = cfg["metric_direction"]
    wpct = float(cfg.get("winsorize_pct", 0.02))

    m = apply_filters(metrics, cfg)
    pool = m[m["eligible"]].copy()
    if pool.empty:
        m["composite"] = np.nan
        return m

    # --- per-metric z-scores (computed on the eligible pool only) ----------
    zcols: dict[str, list[str]] = {b: [] for b in blocks}
    for block, metric_names in blocks.items():
        for name in metric_names:
            if name not in pool.columns:
                continue
            raw = pd.to_numeric(pool[name], errors="coerce")
            if name in POSITIVE_ONLY:
                raw = raw.where(raw > 0)
            if raw.notna().sum() < 5:  # too sparse to rank meaningfully
                continue
            z = zscore(winsorize(raw, wpct)) * float(directions.get(name, 1))
            zname = f"z_{name}"
            pool[zname] = z
            zcols[block].append(zname)

    # --- block scores ------------------------------------------------------
    for block, cols in zcols.items():
        pool[f"block_{block}"] = pool[cols].mean(axis=1) if cols else np.nan

    # --- composite with per-row weight renormalisation ---------------------
    bcols = [f"block_{b}" for b in blocks]
    W = pd.DataFrame(
        {f"block_{b}": float(weights.get(b, 0.0)) for b in blocks},
        index=pool.index,
    )
    present = pool[bcols].notna()
    w_eff = W.where(present, 0.0)
    denom = w_eff.sum(axis=1).replace(0.0, np.nan)
    pool["composite"] = (pool[bcols].fillna(0.0) * w_eff).sum(axis=1) / denom
    pool["blocks_used"] = present.sum(axis=1)

    out = m.join(
        pool[[c for c in pool.columns if c.startswith(("z_", "block_"))]
             + ["composite", "blocks_used"]],
        how="left",
    )
    out = out.sort_values("composite", ascending=False)
    out["rank"] = out["composite"].rank(ascending=False, method="min")
    return out


def pick_candidate(
    scored: pd.DataFrame,
    held_tickers: set[str],
    sector_counts: dict[str, int],
    cfg,
    max_candidates: int = 15,
) -> list[dict]:
    """Ordered shortlist of buyable names, best first.

    Applies the portfolio-level constraints the scorer itself can't know
    about: no duplicate tickers and the per-sector cap.
    """
    strat = cfg.strategy
    cap = int(strat.get("max_positions_per_sector", 99))
    allow_dupes = bool(strat.get("allow_duplicate_tickers", False))

    shortlist = []
    for ticker, row in scored.iterrows():
        if not bool(row.get("eligible", False)) or pd.isna(row.get("composite")):
            continue
        if not allow_dupes and ticker in held_tickers:
            continue
        sector = str(row.get("sector", "Unknown"))
        if sector_counts.get(sector, 0) >= cap:
            continue
        shortlist.append(
            {
                "ticker": ticker,
                "sector": sector,
                "composite": float(row["composite"]),
                "rank": int(row.get("rank", 0) or 0),
                "last_close": float(row["last_close"]),
                "blocks": {
                    b: (None if pd.isna(row.get(f"block_{b}")) else float(row[f"block_{b}"]))
                    for b in cfg["metric_blocks"]
                },
            }
        )
        if len(shortlist) >= max_candidates:
            break
    return shortlist
