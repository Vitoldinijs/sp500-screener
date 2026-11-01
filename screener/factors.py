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

# Observations kept for the bounded factors. The longest of them,
# mom_12m_1m, reads 253 (231 bars plus a 21-bar skip plus one), so 400 is
# comfortably clear of it while sparing a decade-long replay from walking
# thousands of rows per ticker to use the last few hundred.
WINDOW_ROWS = 400


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

    # Every factor here looks back a bounded distance: the longest is
    # mom_12m_1m, which needs 253 observations (231 + a 21-bar skip + 1).
    # A replay hands in a decade, so each per-ticker dropna/tail walks
    # thousands of rows to use the last few hundred. Slicing first makes
    # that work proportional to what is actually read.
    #
    # The slice is on ROWS while the factors consume NON-NaN values, so a
    # column with heavy gaps could lose observations it would otherwise
    # have had. Those columns — and only those — are recomputed from the
    # full history, which keeps the result identical rather than
    # approximately right.
    full_counts = close.notna().sum()

    rows = {}
    for ticker in close.columns:
        col_full = close[ticker].dropna()
        if col_full.empty:
            continue
        # Trimming the DROPPED-NA series (not the raw rows) is exactly
        # equivalent for every bounded factor: tail(400).tail(50) is
        # tail(50). RSI and MACD are deliberately excluded — they are
        # exponentially weighted, so they depend on the entire history and
        # truncating them shifts the value, which would change rankings.
        col = col_full.tail(WINDOW_ROWS) if len(col_full) > WINDOW_ROWS else col_full
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

        rsi14 = rsi(col_full, 14)
        rows[ticker] = {
            "last_close": last,
            # Counted over the whole history, not the slice, so the
            # min_history_days filter sees the same number as before.
            "history_days": int(full_counts.get(ticker, len(col))),
            "mom_1m": _ret(col, H_1M),
            "mom_3m": _ret(col, H_3M),
            "mom_6m": _ret(col, H_6M),
            "mom_12m_1m": _ret(col, H_12M - H_1M, skip=H_1M),
            # Research diagnostics: the 3- and 6-month windows WITHOUT their
            # most recent month. mom_3m and mom_6m above include it, and the
            # last month is the one horizon where returns reliably reverse
            # rather than persist. If that reversal is what is cancelling the
            # momentum block, these two will show a positive IC where the
            # unskipped versions show none. Not in any block, so computing
            # them changes no score.
            "mom_3m_1m": _ret(col, H_3M - H_1M, skip=H_1M),
            "mom_6m_1m": _ret(col, H_6M - H_1M, skip=H_1M),
            "above_ma50": (last / ma50 - 1.0) if ma50 and not np.isnan(ma50) else np.nan,
            "above_ma200": (last / ma200 - 1.0) if ma200 and not np.isnan(ma200) else np.nan,
            "rsi14": rsi14,
            "macd_hist": macd_hist(col_full),
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
    close_wide: pd.DataFrame | None = None,
    volume_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join price factors + fundamentals + sector into one scoring table.

    `close_wide`/`volume_wide` let a caller that has already reshaped the
    price frame hand it over instead of paying for the pivot again. A
    replay reshapes the same decade thousands of times, so this matters
    there; a live run leaves them None and nothing changes. Either way
    price_factors applies `asof` itself, so passing an unclipped frame is
    safe — it never reads past the as-of date.
    """
    from .data import prices as price_mod

    close = close_wide if close_wide is not None else price_mod.to_wide(prices_long, "close")
    volume = volume_wide if volume_wide is not None else price_mod.to_wide(prices_long, "volume")
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
