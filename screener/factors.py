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


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """MACD histogram (MACD line minus its own signal line), normalised by price.

    Raw MACD is denominated in price units, so a $900 stock's histogram would
    dwarf a $20 stock's at the exact same % trend strength — dividing by the
    last close makes it comparable across the cross-section, which is what
    the z-score step needs. Requires slow+signal bars for the EMAs to have
    converged past their initial transient.
    """
    s = series.dropna()
    if len(s) < slow + signal:
        return np.nan
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = float((macd_line - signal_line).iloc[-1])
    last = float(s.iloc[-1])
    return hist / last if last else np.nan


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
        # Distance below the 52-week high (George & Hwang, 2004): proximity
        # to the trailing high is a persistent, momentum-adjacent predictor
        # in its own right. Requires near a full year of history, same bar
        # as ma200, so we don't call a 3-month-old listing's own high "the"
        # 52-week high.
        high52 = float(col.tail(H_12M).max()) if len(col) >= 200 else np.nan
        dist_52w_high = (last / high52 - 1.0) if high52 and not np.isnan(high52) else np.nan

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
            "macd_hist": macd_hist(col),
            "volatility": vol60,
            "dollar_volume": dvol,
            "dist_52w_high": dist_52w_high,
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

    # --- PEAD staleness gate -------------------------------------------
    # earnings_surprise_pct is only a meaningful signal in the weeks right
    # after a report; a surprise from two quarters ago is stale news the
    # market has long since priced in. Blank it out once it ages past
    # ~one quarter, same point-in-time discipline as everything else here
    # (days_since_earnings < 0 would mean the "report" is in our future —
    # can't happen with real data, but we guard against it anyway).
    if "last_earnings_date" in merged.columns:
        ref_date = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today()
        led = pd.to_datetime(merged["last_earnings_date"], errors="coerce")
        days_since = (ref_date - led).dt.days
        merged["days_since_earnings"] = days_since
        if "earnings_surprise_pct" in merged.columns:
            stale = days_since.isna() | (days_since > 90) | (days_since < 0)
            merged.loc[stale, "earnings_surprise_pct"] = np.nan

    uni = universe.set_index("ticker")
    for col in ("sector", "name"):
        if col in uni.columns:
            merged[col] = uni[col].reindex(merged.index)
    merged["sector"] = merged.get("sector", pd.Series(index=merged.index)).fillna("Unknown")
    return merged
