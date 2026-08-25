"""Factor computation.

Two families:

* **Price factors** — recomputed every day from the batched OHLCV pull.
  Momentum over several horizons, trend position vs the 200-day average,
  RSI, realised volatility, and liquidity.
* **Fundamental factors** — read from the weekly snapshot as-is.

Everything returns one row per ticker, indexed by ticker, so the scorer can
z-score across the cross-section.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Trading-day horizons
H_1M, H_3M, H_6M, H_12M = 21, 63, 126, 252


def rsi(series: pd.Series, window: int = 14) -> float:
    """Wilder's RSI on the last `window` periods. NaN if not enough data."""
    s = series.dropna()
    if len(s) < window + 1:
        return np.nan
    delta = s.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ret(col: pd.Series, lag: int, skip: int = 0) -> float:
    """Return over `lag` bars ending `skip` bars ago."""
    s = col.dropna()
    need = lag + skip + 1
    if len(s) < need:
        return np.nan
    end = s.iloc[-1 - skip]
    start = s.iloc[-1 - skip - lag]
    if start <= 0:
        return np.nan
    return float(end / start - 1.0)


def price_factors(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute per-ticker price factors from wide (date x ticker) frames."""
    if close.empty:
        return pd.DataFrame()

    close = close.sort_index()
    if asof is not None:
        close = close.loc[close.index <= asof]
        if volume is not None and not volume.empty:
            volume = volume.sort_index()
            volume = volume.loc[volume.index <= asof]
    if close.empty:
        return pd.DataFrame()

    rows = {}
    for ticker in close.columns:
        col = close[ticker].dropna()
        if col.empty:
            continue
        last = float(col.iloc[-1])
        ma50 = float(col.tail(50).mean()) if len(col) >= 50 else np.nan
        ma200 = float(col.tail(200).mean()) if len(col) >= 200 else np.nan
        rets = col.pct_change().dropna()
        vol60 = (
            float(rets.tail(60).std() * np.sqrt(252)) if len(rets) >= 60 else np.nan
        )

        dvol = np.nan
        if volume is not None and ticker in getattr(volume, "columns", []):
            v = volume[ticker].dropna()
            if len(v) >= 20:
                aligned = (close[ticker] * volume[ticker]).dropna()
                if len(aligned) >= 20:
                    dvol = float(aligned.tail(20).mean())

        rsi14 = rsi(col, 14)
        rows[ticker] = {
            "last_close": last,
            "history_days": int(len(col)),
            "mom_1m": _ret(col, H_1M),
            "mom_3m": _ret(col, H_3M),
            "mom_6m": _ret(col, H_6M),
            "mom_12m_1m": _ret(col, H_12M - H_1M, skip=H_1M),
            "above_ma50": (last / ma50 - 1.0) if ma50 and not np.isnan(ma50) else np.nan,
            "above_ma200": (last / ma200 - 1.0) if ma200 and not np.isnan(ma200) else np.nan,
            "rsi14": rsi14,
            # Prefers a neutral RSI: penalises overbought AND oversold extremes.
            # Momentum factors already capture trend direction; this keeps us
            # from paying up for euphoria or catching a falling knife.
            "rsi_centered": -abs(rsi14 - 50.0) if not np.isnan(rsi14) else np.nan,
            "volatility": vol60,
            "dollar_volume": dvol,
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    return out


def forward_return(
    close: pd.DataFrame, entry_date: pd.Timestamp, horizon: int = H_1M
) -> pd.Series:
    """Realised forward return per ticker, for factor-performance attribution."""
    if close.empty:
        return pd.Series(dtype=float)
    close = close.sort_index()
    idx = close.index
    try:
        i0 = idx.get_indexer([entry_date], method="ffill")[0]
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)
    i1 = i0 + horizon
    if i0 < 0 or i1 >= len(idx):
        return pd.Series(dtype=float)
    start, end = close.iloc[i0], close.iloc[i1]
    return (end / start - 1.0).replace([np.inf, -np.inf], np.nan)


def build_metrics(
    prices_long: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join price factors + fundamentals + sector into one scoring table."""
    from .data import prices as price_mod

    close = price_mod.to_wide(prices_long, "close")
    volume = price_mod.to_wide(prices_long, "volume")
    pf = price_factors(close, volume, asof=asof)
    if pf.empty:
        return pd.DataFrame()

    fund = fundamentals.copy()
    if not fund.empty:
        fund = fund.set_index("ticker")
        fund = fund.drop(columns=[c for c in ("asof",) if c in fund.columns])
        merged = pf.join(fund, how="left")
    else:
        merged = pf

    uni = universe.set_index("ticker")
    for col in ("sector", "name"):
        if col in uni.columns:
            merged[col] = uni[col].reindex(merged.index)
    merged["sector"] = merged.get("sector", pd.Series(index=merged.index)).fillna("Unknown")
    return merged
