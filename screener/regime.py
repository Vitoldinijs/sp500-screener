"""How much of the book should be invested at all.

The screener is purely cross-sectional: it answers "which of these 500 names
is best today", never "is today a good day to own any of them". By
construction it cannot decline. In a falling market the top-ranked stock
still scores +2.5, because it is measured against peers that are falling
too — so no threshold on the composite can ever make the ladder wait. That
decision needs an input of a different kind, and this is it.

The output is a single number: the fraction of equity the ladder is allowed
to deploy. It never reaches zero — `min_exposure` keeps a floor, because a
rule that can take the portfolio fully to cash will eventually do so right
before a rebound, and market-timing rules are the part of any strategy most
prone to looking brilliant in-sample and useless afterwards.

Signals used, all deliberately slow and few:

* **Trend** — benchmark above/below its long moving average. The single most
  robust regime signal there is, and the one with the least to overfit.
* **Volatility** — VIX against its own recent range. High realised fear
  means wider outcomes, which a concentrated book feels more than a broad
  one.
* **Credit** — high-yield spreads widening. Credit markets tend to price
  trouble before equity indices do.
* **Curve** — 10y minus 2y. Slow-moving and weak on its own; carried with a
  small weight as context rather than a trigger.

Each returns a score in [0, 1] (1 = benign). They are averaged with weights,
and the result is mapped onto [min_exposure, 1.0]. Nothing here is a
hard switch: exposure glides rather than flipping, so a single noisy day
cannot liquidate the book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .data import macro as macro_mod


@dataclass
class RegimeAssessment:
    exposure: float                      # fraction of equity deployable
    score: float                         # blended 0..1, 1 = benign
    components: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "exposure": round(float(self.exposure), 4),
            "score": round(float(self.score), 4),
            "components": {k: (None if v is None else round(float(v), 4))
                           for k, v in self.components.items()},
            "notes": list(self.notes),
        }


def _clip01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(min(1.0, max(0.0, x)))


def _trend_score(bench: pd.Series, window: int) -> float:
    """How far the benchmark sits above its long moving average.

    Expressed as a glide, not a switch: 0 at 8% below the average, 1 at 4%
    above. A hard cross would make exposure flip on a single tick when the
    index happens to sit on the line.
    """
    s = pd.to_numeric(bench, errors="coerce").dropna()
    if len(s) < window:
        return float("nan")
    ma = float(s.tail(window).mean())
    last = float(s.iloc[-1])
    if not np.isfinite(ma) or ma <= 0:
        return float("nan")
    gap = last / ma - 1.0
    return _clip01((gap + 0.08) / 0.12)


def _vix_score(vix_hist: pd.Series) -> float:
    """Calm (VIX ~15) scores 1; panic (VIX ~35) scores 0."""
    s = pd.to_numeric(vix_hist, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    last = float(s.iloc[-1])
    return _clip01((35.0 - last) / 20.0)


def _credit_score(spread_hist: pd.Series) -> float:
    """High-yield spread against its own two-year range.

    Absolute spread levels drift with the rate environment, so this is scored
    relative to the series' own recent history rather than a fixed number.
    """
    s = pd.to_numeric(spread_hist, errors="coerce").dropna().tail(500)
    if len(s) < 60:
        return float("nan")
    last = float(s.iloc[-1])
    lo, hi = float(s.quantile(0.10)), float(s.quantile(0.90))
    if not np.isfinite(hi - lo) or hi <= lo:
        return float("nan")
    return _clip01((hi - last) / (hi - lo))


def _curve_score(curve_hist: pd.Series) -> float:
    """10y-2y: inverted (-0.5) scores 0, comfortably positive (+1.0) scores 1."""
    s = pd.to_numeric(curve_hist, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    last = float(s.iloc[-1])
    return _clip01((last + 0.5) / 1.5)


def assess(cfg, benchmark_close: pd.Series | None,
           macro_frame: pd.DataFrame | None, asof: date) -> RegimeAssessment:
    """Blend the available signals into a deployable-capital fraction."""
    rcfg = dict(cfg.get("regime") or {})
    if not bool(rcfg.get("enabled", True)):
        return RegimeAssessment(1.0, 1.0, {}, ["regime gating disabled"])

    min_exposure = float(rcfg.get("min_exposure", 0.30))
    max_exposure = float(rcfg.get("max_exposure", 1.0))
    trend_window = int(rcfg.get("trend_window", 200))
    weights = dict(rcfg.get("weights") or {
        "trend": 0.50, "volatility": 0.20, "credit": 0.20, "curve": 0.10
    })

    components = {
        "trend": _trend_score(benchmark_close, trend_window)
        if benchmark_close is not None else float("nan"),
        "volatility": _vix_score(macro_mod.history(macro_frame, "vix", asof)),
        "credit": _credit_score(
            macro_mod.history(macro_frame, "high_yield_spread", asof)),
        "curve": _curve_score(
            macro_mod.history(macro_frame, "yield_curve_10y_2y", asof)),
    }

    notes = []
    usable = {k: v for k, v in components.items() if np.isfinite(v)}
    missing = [k for k, v in components.items() if not np.isfinite(v)]
    if missing:
        notes.append("no data for: " + ", ".join(sorted(missing)))

    if not usable:
        # Never let an absent data feed decide to sit in cash. With nothing
        # to go on the honest position is full exposure plus a loud note,
        # not a silent de-risking the operator never asked for.
        notes.append("no regime signal available; defaulting to full exposure")
        return RegimeAssessment(max_exposure, 1.0, components, notes)

    # Renormalise across the signals we actually have, so a missing feed
    # dilutes nobody's vote rather than dragging the blend toward zero.
    total_w = sum(float(weights.get(k, 0.0)) for k in usable) or 1.0
    score = sum(float(weights.get(k, 0.0)) * v for k, v in usable.items()) / total_w
    score = _clip01(score)

    exposure = min_exposure + (max_exposure - min_exposure) * score
    return RegimeAssessment(exposure, score, components, notes)
