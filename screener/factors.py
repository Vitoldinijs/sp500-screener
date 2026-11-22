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


def _ewm_last_rsi_macd(close: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """RSI(14) and normalised MACD histogram for many columns at once.

    Bit-for-bit the same as rsi() / macd_hist() on each column's dropna'd
    series, for columns whose observations are CONTIGUOUS (leading and
    trailing NaN only): an adjust=False EWM starts at the first valid value
    and is untouched by NaNs before it, so the wide computation and the
    per-series one run the identical recursion. A column with gaps inside
    its history is not handled here — the caller computes it per series.
    """
    n = close.notna().sum()
    # RSI: diff of the dropna'd series == diff of the column once leading
    # NaNs are past; the first valid row's diff is NaN in both, and dropped.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / 14
    g = gain.ewm(alpha=alpha, adjust=False).mean()
    l = loss.ewm(alpha=alpha, adjust=False).mean()
    last_valid = close.apply(lambda c: c.last_valid_index())
    pos = {t: close.index.get_loc(ix) for t, ix in last_valid.items() if ix is not None}
    def at_last(frame):
        return pd.Series({t: frame[t].iat[i] for t, i in pos.items()}, dtype=float)
    ag, al = at_last(g), at_last(l)
    rs = ag / al
    rsi_v = 100.0 - (100.0 / (1.0 + rs))
    rsi_v = rsi_v.where(al != 0, np.where(ag > 0, 100.0, 50.0))
    rsi_v = rsi_v.where(n.reindex(rsi_v.index) >= 15)

    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=9, adjust=False).mean()
    hist = at_last(line - sig)
    lastpx = at_last(close)
    macd_v = (hist / lastpx).where(lastpx != 0)
    macd_v = macd_v.where(n.reindex(macd_v.index) >= 26 + 9)
    return rsi_v.astype(float), macd_v.astype(float)


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

    # RSI and MACD in one pass for every column without interior gaps —
    # the per-series pandas overhead was a third of the whole function, and
    # a small-cap universe has five times the S&P's columns. Gapped columns
    # fall back to the per-series path below, so every value is unchanged.
    # Contiguous = as many observations as there are rows between the first
    # and the last one. (Counting the observations inside that range instead
    # is always equal to the total, and let gapped columns through.)
    span = close.notna().to_numpy()
    any_obs = span.any(axis=0)
    first_row = span.argmax(axis=0)
    last_row = len(span) - 1 - span[::-1].argmax(axis=0)
    width = last_row - first_row + 1
    counts = span.sum(axis=0)
    contiguous = [t for t, ok in zip(close.columns, any_obs & (counts == width)) if ok]
    fast_rsi, fast_macd = (_ewm_last_rsi_macd(close[contiguous])
                           if contiguous else (pd.Series(dtype=float),) * 2)

    # The per-ticker loop below runs on numpy arrays rather than pandas
    # Series. The arithmetic is the same — sums, means and std use numpy's
    # pairwise reductions in both — so every value is bit-for-bit what the
    # Series version produced (verified in the test suite); what goes is the
    # per-object pandas overhead, which dominated with 2,500 columns.
    C = close.to_numpy(dtype=float)
    VOL = None
    if volume is not None and not getattr(volume, "empty", True):
        VOL = volume.reindex(index=close.index, columns=close.columns).to_numpy(dtype=float)
    sqrt252 = np.sqrt(252)

    def ret(a, lag, skip=0):
        need = lag + skip + 1
        if a.size < need:
            return np.nan
        start, stop = a[-1 - skip - lag], a[-1 - skip]
        if start <= 0:
            return np.nan
        return float(stop / start - 1.0)

    rows = {}
    for j, ticker in enumerate(close.columns):
        colv = C[:, j]
        a_full = colv[~np.isnan(colv)]
        if a_full.size == 0:
            continue
        # Trimming the dropna'd values is exact for every bounded factor:
        # the last 50 of the last 400 are the last 50. RSI and MACD use the
        # full history (they are exponentially weighted).
        a = a_full[-WINDOW_ROWS:] if a_full.size > WINDOW_ROWS else a_full
        last = float(a[-1])
        ma50 = float(a[-50:].mean()) if a.size >= 50 else np.nan
        ma200 = float(a[-200:].mean()) if a.size >= 200 else np.nan
        rets = a[1:] / a[:-1] - 1.0
        rets = rets[~np.isnan(rets)]
        vol60 = float(rets[-60:].std(ddof=1) * sqrt252) if rets.size >= 60 else np.nan
        # Distance below the 52-week high (George & Hwang, 2004). Requires
        # near a full year of history, same bar as ma200.
        high52 = float(a[-H_12M:].max()) if a.size >= 200 else np.nan
        dist_52w_high = (last / high52 - 1.0) if high52 and not np.isnan(high52) else np.nan

        dvol = np.nan
        if VOL is not None:
            prod = colv * VOL[:, j]
            prod = prod[~np.isnan(prod)]
            if prod.size >= 20:
                dvol = float(prod[-20:].mean())

        if ticker in fast_rsi.index:
            rsi14, macd_v = float(fast_rsi[ticker]), float(fast_macd[ticker])
        else:
            series = pd.Series(a_full)
            rsi14, macd_v = rsi(series, 14), macd_hist(series)
        # 12-1 momentum per unit of the same year's volatility. Raw momentum
        # is largest in magnitude for the most volatile names, so averaging
        # its z-score with others pushes volatile stocks to BOTH ends of the
        # ranking — the U-shaped decile profile the 2020-2026 IC study found.
        vol252 = float(rets[-252:].std(ddof=1) * sqrt252) if rets.size >= 200 else np.nan
        mom12 = ret(a, H_12M - H_1M, skip=H_1M)
        mom12_va = (mom12 / vol252 if np.isfinite(mom12) and np.isfinite(vol252)
                    and vol252 > 0 else np.nan)
        rows[ticker] = {
            "last_close": last,
            # Counted over the whole history, not the slice, so the
            # min_history_days filter sees the same number as before.
            "history_days": int(full_counts.get(ticker, a.size)),
            "mom_1m": ret(a, H_1M),
            "mom_3m": ret(a, H_3M),
            "mom_6m": ret(a, H_6M),
            "mom_12m_1m": mom12,
            "mom_12m_1m_va": mom12_va,
            # Research diagnostics: the 3- and 6-month windows WITHOUT their
            # most recent month, the one horizon where returns reverse.
            "mom_3m_1m": ret(a, H_3M - H_1M, skip=H_1M),
            "mom_6m_1m": ret(a, H_6M - H_1M, skip=H_1M),
            "above_ma50": (last / ma50 - 1.0) if ma50 and not np.isnan(ma50) else np.nan,
            "above_ma200": (last / ma200 - 1.0) if ma200 and not np.isnan(ma200) else np.nan,
            "rsi14": rsi14,
            "macd_hist": macd_v,
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
