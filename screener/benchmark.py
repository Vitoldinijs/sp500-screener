"""Benchmark equity helper — tracks a buy-and-hold SPY line from the same
start capital, so the dashboard can show relative performance."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def benchmark_equity(prices_long: pd.DataFrame, start_capital: float,
                     first_date: str | None, symbol: str = "SPY") -> pd.Series:
    """Value of `start_capital` invested in `symbol` on `first_date`."""
    df = prices_long[prices_long["ticker"] == symbol]
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["close"].sort_index()
    if first_date:
        s = s[s.index >= pd.Timestamp(first_date)]
    if s.empty or s.iloc[0] <= 0:
        return pd.Series(dtype=float)
    return (s / s.iloc[0]) * start_capital
