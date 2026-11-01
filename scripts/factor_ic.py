#!/usr/bin/env python3
"""Measure whether the scoring model actually predicts returns.

Run against the panel a backtest exports:

    python scripts/factor_ic.py backtest-results/latest/factor_panel.csv.gz

What it reports
---------------
**Rank IC** — the cross-sectional Spearman correlation, computed separately
for every session, between a score and the return each name went on to
deliver. An IC of 0.00 means the ranking carries no information; equity
cross-sectional signals that survive publication typically live around
0.02-0.05, and anything much above 0.10 on a long sample is usually a bug
or look-ahead rather than a discovery.

**Decile spread** — average forward return of the top-scored tenth minus the
bottom-scored tenth. This is the same claim as the IC in units a portfolio
cares about, and it also shows whether the relationship is monotone or comes
entirely from one extreme bucket.

Returns are measured in EXCESS of each session's equal-weighted
cross-sectional mean. Without that subtraction, a signal that merely holds
equities during a bull market scores well while predicting nothing about
which equities.

The statistics problem, and the fix
-----------------------------------
Consecutive sessions share almost all of their 21-session forward window, so
the per-session IC series is heavily autocorrelated and the naive t-statistic
is inflated by roughly sqrt(horizon). Both are printed: the naive one for
reference and a Newey-West figure with the lag set to the horizon, which is
the one to read. A result that is significant naively and not under
Newey-West is not a result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ranks-plus-Pearson, already owned by the project. pandas'
# `corr(method="spearman")` imports scipy, which this project has never
# depended on: on pandas 2.x that raises ModuleNotFoundError, and the
# failure surfaces only once a report is being written at the end of an
# hour-long replay.
from screener.learn import spearman  # noqa: E402

HORIZON_DEFAULT = 21
MIN_NAMES = 30          # sessions with a thinner cross-section are skipped
DECILES = 10

# Overlapping windows mean n sessions carry only about n/horizon independent
# observations. Below five of those there is no inference to be had, and
# printing a t-statistic anyway invites exactly the wrong reading: the
# fundamental blocks were scored on 17-26 sessions of a single month and one
# of them came back at t=4.7, sitting in the same column as a figure built
# from 1,695 sessions. The IC is still worth showing; the t-statistic is not.
MIN_INDEPENDENT = 5


def newey_west_tstat(series: pd.Series, lag: int) -> tuple[float, float]:
    """t-statistic for the mean of an autocorrelated series, and its SE.

    Bartlett-weighted long-run variance: the usual standard error assumes
    independent observations, which overlapping forward windows violate by
    construction.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    x = x - x.mean()
    gamma0 = float(x @ x) / n
    total = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        gamma_l = float(x[l:] @ x[:-l]) / n
        total += 2.0 * (1.0 - l / (lag + 1.0)) * gamma_l
    if total <= 0:
        return float("nan"), float("nan")
    se = float(np.sqrt(total / n))
    mean = float(pd.to_numeric(series, errors="coerce").dropna().mean())
    return mean / se if se > 0 else float("nan"), se


def load_panel(path: Path, horizon: int, eligible_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"data_date": str, "ticker": str, "sector": str},
                     low_memory=False)
    target = f"fwd_ret_{horizon}"
    if target not in df.columns:
        have = [c for c in df.columns if c.startswith("fwd_ret_")]
        raise SystemExit(f"Panel has no {target}; available: {have}")
    if eligible_only and "eligible" in df.columns:
        df = df[df["eligible"].astype(str).str.lower().isin({"true", "1"})]
    df = df.dropna(subset=[target])
    # Excess of the session's own cross-section: isolates stock selection
    # from the market move every name shared.
    df["excess"] = df[target] - df.groupby("data_date")[target].transform("mean")
    counts = df.groupby("data_date")["ticker"].transform("size")
    return df[counts >= MIN_NAMES].copy()


def ic_series(df: pd.DataFrame, score: str) -> pd.Series:
    """Per-session Spearman IC, computed without groupby.apply.

    Spearman is Pearson on ranks, and ranking within a session is a single
    grouped transform. Expressing the correlation through grouped sums keeps
    the whole calculation vectorised — a 900k-row panel takes under a second
    instead of minutes — and avoids the apply/include_groups keywords whose
    behaviour differs across pandas 2.x and 3.x.
    """
    sub = df.dropna(subset=[score, "excess"])
    if sub.empty:
        return pd.Series(dtype=float)
    g = sub.groupby("data_date")
    x = g[score].rank()
    y = g["excess"].rank()
    frame = pd.DataFrame({"session": sub["data_date"].to_numpy(), "x": x, "y": y})
    frame["xy"] = frame["x"] * frame["y"]
    frame["xx"] = frame["x"] ** 2
    frame["yy"] = frame["y"] ** 2
    agg = frame.groupby("session").agg(
        n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
        sxy=("xy", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"),
    )
    cov = agg["sxy"] - agg["sx"] * agg["sy"] / agg["n"]
    vx = agg["sxx"] - agg["sx"] ** 2 / agg["n"]
    vy = agg["syy"] - agg["sy"] ** 2 / agg["n"]
    denom = np.sqrt(vx * vy)
    out = (cov / denom.where(denom > 0)).dropna()
    out.index.name = "data_date"
    return out.sort_index()


def decile_profile(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Per score decile: mean excess return, and how dispersed it is.

    Each session is bucketed on its own, so a session with 400 eligible
    names and one with 450 contribute equally rather than by size.

    The dispersion column is what distinguishes a signal from a volatility
    tilt, and it is the reason this returns two numbers rather than one. A
    ranking that predicts direction gives a profile that climbs from D1 to
    D10. A ranking that merely concentrates volatile, high-beta names at
    both ends gives a U — both tails beat the middle — and in a rising
    market both tails look good. Those two stories are indistinguishable
    from the means alone, and tell apart immediately by dispersion: under
    the volatility reading, D1 and D10 have the widest spread of outcomes,
    and the U in the means is just drift times beta.
    """
    sub = df.dropna(subset=[score, "excess"])
    if sub.empty:
        return pd.DataFrame()
    pct = sub.groupby("data_date")[score].rank(pct=True, method="first")
    decile = np.minimum((pct * DECILES).astype(int), DECILES - 1)
    keys = [sub["data_date"].to_numpy(), decile.to_numpy()]
    per_session = sub.groupby(keys)["excess"].agg(["mean", "std"])
    out = per_session.groupby(level=1).mean().sort_index()
    out.columns = ["mean", "dispersion"]
    # Keep the per-session series: the strategy only ever buys the top
    # decile, so whether D10's excess return is distinguishable from zero
    # matters more than the correlation across the whole cross-section.
    out.attrs["per_session"] = per_session["mean"].unstack(level=1)
    return out


def report_score(df: pd.DataFrame, score: str, horizon: int, label: str) -> dict:
    ics = ic_series(df, score)
    if ics.empty:
        print(f"{label:<22} — нет данных")
        return {}
    mean = float(ics.mean())
    naive_t = mean / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std(ddof=1) > 0 else float("nan")
    nw_t, _ = newey_west_tstat(ics, lag=horizon)
    positive = float((ics > 0).mean())
    thin = len(ics) < MIN_INDEPENDENT * horizon
    if thin:
        # Show the IC, withhold the inference, and say so in the row itself.
        print(f"{label:<22}{mean:+8.4f}{'—':>9}{'—':>9}"
              f"{positive * 100:8.0f}%{len(ics):8d}  мало данных")
    else:
        print(f"{label:<22}{mean:+8.4f}{naive_t:9.1f}{nw_t:9.1f}"
              f"{positive * 100:8.0f}%{len(ics):8d}")
    return {"mean_ic": mean, "naive_t": None if thin else naive_t,
            "newey_west_t": None if thin else nw_t,
            "positive_share": positive, "sessions": int(len(ics)),
            "sample_too_thin_for_inference": bool(thin)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel", help="factor_panel.csv.gz from a backtest run")
    ap.add_argument("--horizon", type=int, default=HORIZON_DEFAULT,
                    help="forward-return horizon in sessions (default 21)")
    ap.add_argument("--all-names", action="store_true",
                    help="include names the screen excluded, not just eligible ones")
    ap.add_argument("--by-year", action="store_true",
                    help="also break the composite IC down by calendar year")
    args = ap.parse_args(argv)

    path = Path(args.panel)
    if not path.exists():
        raise SystemExit(f"No such panel: {path}")

    df = load_panel(path, args.horizon, eligible_only=not args.all_names)
    if df.empty:
        raise SystemExit("Panel is empty after filtering.")

    sessions = df["data_date"].nunique()
    print(f"\nпанель: {len(df):,} наблюдений | {sessions} сессий | "
          f"{df['ticker'].nunique()} тикеров | горизонт {args.horizon} сессий")
    print(f"период: {df['data_date'].min()} .. {df['data_date'].max()}")
    print("доходности open-to-open со следующей сессии, в избытке к равновзвешенному срезу\n")

    header = f"{'сигнал':<22}{'ср.IC':>8}{'t наив':>9}{'t NW':>9}{'IC>0':>8}{'сессий':>8}"
    print(header)
    print("-" * len(header))

    results = {}
    if "composite" in df.columns:
        results["composite"] = report_score(df, "composite", args.horizon, "composite")
    if "rank_percentile" in df.columns:
        results["rank_percentile"] = report_score(
            df, "rank_percentile", args.horizon, "rank_percentile"
        )
    print()
    for col in sorted(c for c in df.columns if c.startswith("block_")):
        results[col] = report_score(df, col, args.horizon, col)

    print("\nдецили по composite: средний избыточный доход и разброс, "
          "% за горизонт")
    prof = decile_profile(df, "composite")
    if not prof.empty:
        means, disp = prof["mean"], prof["dispersion"]
        for d in prof.index:
            bar = "#" * int(abs(float(means.loc[d])) * 400)
            print(f"  D{int(d) + 1:<2} {float(means.loc[d]) * 100:+7.2f}%  "
                  f"разброс {float(disp.loc[d]) * 100:6.2f}%  {bar}")
        spread = float(means.iloc[-1] - means.iloc[0])
        rho = spearman(pd.Series(means.to_numpy()),
                       pd.Series(range(len(means)), dtype=float), minimum_n=3)
        print(f"  D10-D1 спред: {spread * 100:+.2f}% | "
              f"монотонность (rho) {rho:+.2f}")

        # Is the profile a U rather than a ramp, and if so does dispersion
        # follow it? That combination is the volatility tilt, not a signal.
        mid = float(means.iloc[2:8].mean())
        tails_beat_middle = (float(means.iloc[0]) > mid
                             and float(means.iloc[-1]) > mid)
        tail_disp = (float(disp.iloc[0]) + float(disp.iloc[-1])) / 2
        mid_disp = float(disp.iloc[2:8].mean())
        print(f"  края vs середина: доход {float(means.iloc[0]) * 100:+.2f}% / "
              f"{mid * 100:+.2f}% / {float(means.iloc[-1]) * 100:+.2f}%  |  "
              f"разброс краёв {tail_disp * 100:.2f}% против "
              f"{mid_disp * 100:.2f}% в середине")
        # The decision-relevant test. A composite IC near zero says the
        # ordering carries no information, but the book only ever holds the
        # top decile — so D10 against zero, and D10 minus D1, are the two
        # numbers that decide whether any of this is tradeable.
        wide = prof.attrs.get("per_session")
        if wide is not None and wide.shape[1] == DECILES:
            top = wide.iloc[:, -1].dropna()
            spread_series = (wide.iloc[:, -1] - wide.iloc[:, 0]).dropna()
            t_top, _ = newey_west_tstat(top, lag=args.horizon)
            t_spread, _ = newey_west_tstat(spread_series, lag=args.horizon)
            print(f"  D10 против нуля: {top.mean() * 100:+.2f}% "
                  f"за горизонт | t NW {t_top:+.1f}")
            print(f"  D10-D1 против нуля: {spread_series.mean() * 100:+.2f}% "
                  f"| t NW {t_spread:+.1f}")
            results["top_decile_t"] = float(t_top)
            results["spread_t"] = float(t_spread)

        if tails_beat_middle and tail_disp > mid_disp * 1.1:
            print("  -> оба края обгоняют середину И разброс на краях шире: "
                  "это наклон в волатильность, а не предсказание направления")
        results["decile_spread"] = spread
        results["decile_monotonicity"] = rho
        results["tails_beat_middle"] = bool(tails_beat_middle)
        results["tail_dispersion_ratio"] = (
            tail_disp / mid_disp if mid_disp > 0 else None
        )

    if args.by_year and "composite" in df.columns:
        print("\ncomposite IC по годам")
        df["year"] = df["data_date"].str.slice(0, 4)
        for year, group in df.groupby("year"):
            ics = ic_series(group, "composite")
            if len(ics) < 20:
                continue
            nw_t, _ = newey_west_tstat(ics, lag=args.horizon)
            print(f"  {year}: IC {ics.mean():+.4f} | t NW {nw_t:+5.1f} "
                  f"| сессий {len(ics)}")

    print("\nчитать так: t NW по модулю ниже ~2 — сигнала не видно на этой выборке.")
    print("перекрывающиеся окна делают t наив завышенным примерно в sqrt(горизонт) раз.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
