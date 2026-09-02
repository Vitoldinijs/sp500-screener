"""Synthetic market data, so the whole pipeline can be tested offline.

Generates correlated random-walk prices (a market factor plus idiosyncratic
noise) and plausible fundamentals. Deterministic given a seed, which means the
tests are reproducible and CI never depends on a live data provider.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = [
    "Information Technology", "Health Care", "Financials", "Industrials",
    "Consumer Discretionary", "Consumer Staples", "Energy", "Utilities",
    "Materials", "Real Estate", "Communication Services",
]


def make_universe(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "name": [f"Synthetic Corp {i}" for i in range(n)],
        "sector": [SECTORS[int(rng.integers(0, len(SECTORS)))] for _ in range(n)],
        "industry": "",
    })


def make_prices(
    universe: pd.DataFrame,
    days: int = 520,
    seed: int = 11,
    start_date: str = "2024-01-02",
    include_benchmark: str | None = "SPY",
) -> pd.DataFrame:
    """Long OHLCV frame with a shared market factor and per-name quality tilt.

    A slice of names is given a genuine positive drift correlated with their
    fundamentals, so factor scoring has something real to find — otherwise a
    test of "does the screener pick good stocks" is meaningless on noise.
    """
    rng = np.random.default_rng(seed)
    tickers = universe["ticker"].tolist()
    dates = pd.bdate_range(start=start_date, periods=days)

    market = rng.normal(0.0003, 0.010, size=days)
    frames = []
    alphas = {}
    for i, t in enumerate(tickers):
        beta = float(rng.uniform(0.6, 1.5))
        alpha = float(rng.normal(0.0, 0.0006))
        alphas[t] = alpha
        idio = rng.normal(0, 0.014, size=days)
        rets = alpha + beta * market + idio
        px = 20.0 * float(rng.uniform(1.0, 12.0)) * np.exp(np.cumsum(rets))

        # Build OHLC around the close path.
        close = px
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * (1.0 + rng.normal(0, 0.003, size=days))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.004, days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.004, days)))
        vol = rng.integers(500_000, 12_000_000, size=days)

        frames.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": open_, "high": high,
            "low": low, "close": close, "volume": vol,
        }))

    if include_benchmark:
        bench = 400.0 * np.exp(np.cumsum(0.0002 + market * 0.95))
        prev = np.concatenate([[bench[0]], bench[:-1]])
        frames.append(pd.DataFrame({
            "date": dates, "ticker": include_benchmark,
            "open": prev, "high": bench * 1.002, "low": bench * 0.998,
            "close": bench, "volume": 70_000_000,
        }))

    out = pd.concat(frames, ignore_index=True)
    out.attrs["alphas"] = alphas
    return out


def make_fundamentals(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    asof: str = "2024-01-02",
    seed: int = 13,
    missing_rate: float = 0.06,
) -> pd.DataFrame:
    """Fundamentals partly correlated with each name's hidden alpha.

    Names with positive drift get better ratios on average, so a correctly
    implemented scorer should rank them higher than chance.
    """
    rng = np.random.default_rng(seed)
    alphas = prices.attrs.get("alphas", {})
    last_close = prices.groupby("ticker")["close"].last()

    rows = []
    for t in universe["ticker"]:
        a = float(alphas.get(t, 0.0)) / 0.0006     # normalise to ~N(0,1)
        signal = np.clip(a, -3, 3)
        mc = float(last_close.get(t, 100.0)) * float(rng.uniform(2e7, 6e8))
        rec = {
            "asof": asof,
            "ticker": t,
            "pe_ratio": float(np.clip(26.0 - 4.0 * signal + rng.normal(0, 5), 3, 90)),
            "pb_ratio": float(np.clip(4.0 - 0.7 * signal + rng.normal(0, 1.2), 0.3, 18)),
            "ps_ratio": float(np.clip(3.5 - 0.5 * signal + rng.normal(0, 1.0), 0.2, 20)),
            "ev_ebitda": float(np.clip(15.0 - 2.5 * signal + rng.normal(0, 4), 2, 60)),
            "fcf_yield": float(np.clip(0.045 + 0.012 * signal + rng.normal(0, 0.02), -0.08, 0.20)),
            "roe": float(np.clip(0.16 + 0.05 * signal + rng.normal(0, 0.07), -0.35, 0.75)),
            "gross_margin": float(np.clip(0.42 + 0.04 * signal + rng.normal(0, 0.12), 0.05, 0.92)),
            "net_margin": float(np.clip(0.12 + 0.03 * signal + rng.normal(0, 0.06), -0.25, 0.45)),
            "net_debt_ebitda": float(np.clip(2.0 - 0.4 * signal + rng.normal(0, 1.0), -2.5, 9.0)),
            "revenue_growth": float(np.clip(0.07 + 0.04 * signal + rng.normal(0, 0.09), -0.4, 0.9)),
            "earnings_growth": float(np.clip(0.10 + 0.05 * signal + rng.normal(0, 0.18), -0.8, 1.5)),
            "market_cap": mc,
            # Earnings surprise correlated with the same hidden alpha, dated
            # recently relative to `asof` so the PEAD staleness gate in
            # factors.py doesn't blank it out during tests.
            "last_earnings_date": (
                pd.Timestamp(asof) - pd.Timedelta(days=int(rng.integers(1, 80)))
            ).date().isoformat(),
            "earnings_surprise_pct": float(np.clip(2.0 + 6.0 * signal + rng.normal(0, 8.0), -40, 60)),
        }
        # Punch a few holes, like a real provider would.
        for f in ("pe_ratio", "ev_ebitda", "earnings_growth", "fcf_yield", "earnings_surprise_pct"):
            if rng.random() < missing_rate:
                rec[f] = np.nan
        rows.append(rec)

    from screener.data.fundamentals import SCHEMA
    return pd.DataFrame(rows, columns=SCHEMA)
