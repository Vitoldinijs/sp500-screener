"""Test suite — runs standalone (`python tests/test_screener.py`) or under pytest.

Covers the things that would silently corrupt results if they broke:
accounting identities, the no-lookahead execution rule, filter and scoring
correctness, and the learning module's guardrails.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener import factors, learn, portfolio, scoring  # noqa: E402
from screener.config import Config  # noqa: E402
from screener.data import fundamentals as fund_mod  # noqa: E402
from screener.data import prices as price_mod  # noqa: E402
from tests import synthetic  # noqa: E402

TOL = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cfg() -> Config:
    return Config.load(ROOT / "config.yml")


def _fixture(n=60, days=520):
    uni = synthetic.make_universe(n)
    px = synthetic.make_prices(uni, days=days)
    fund = synthetic.make_fundamentals(uni, px)
    return uni, px, fund


# ---------------------------------------------------------------------------
# factor maths
# ---------------------------------------------------------------------------
def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    flat = pd.Series([100.0] * 60)

    r_up, r_down, r_flat = factors.rsi(up), factors.rsi(down), factors.rsi(flat)
    assert 95 <= r_up <= 100, f"monotonic rise should pin RSI near 100, got {r_up}"
    assert 0 <= r_down <= 5, f"monotonic fall should pin RSI near 0, got {r_down}"
    assert abs(r_flat - 50.0) < TOL, f"flat series should give RSI 50, got {r_flat}"
    assert np.isnan(factors.rsi(pd.Series([1.0, 2.0])))  # too short


def test_return_windows_are_exact():
    # geometric series: each bar +10%
    s = pd.Series([100 * (1.1 ** i) for i in range(30)])
    assert abs(factors._ret(s, 1) - 0.10) < 1e-9
    assert abs(factors._ret(s, 5) - (1.1 ** 5 - 1)) < 1e-9
    # skip=1 must end one bar earlier
    assert abs(factors._ret(s, 5, skip=1) - (1.1 ** 5 - 1)) < 1e-9
    assert np.isnan(factors._ret(s, 100))


def test_price_factors_respect_asof():
    """Factors computed as-of an early date must not see later prices."""
    uni, px, _ = _fixture(n=12, days=400)
    close = price_mod.to_wide(px, "close")
    cut = close.index[300]

    full = factors.price_factors(close, asof=cut)
    truncated = factors.price_factors(close.loc[close.index <= cut])
    pd.testing.assert_frame_equal(
        full.sort_index(), truncated.sort_index(), check_exact=False, rtol=1e-9
    )
    # and the reported last close is the one at the cut, not the newest bar
    for t in full.index[:5]:
        assert abs(full.loc[t, "last_close"] - close.loc[cut, t]) < 1e-9


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_zscore_and_winsorize():
    s = pd.Series([1, 2, 3, 4, 5, 1000.0])
    w = scoring.winsorize(s, 0.02)
    assert w.max() < 1000.0, "winsorize must pull in the outlier"

    z = scoring.zscore(pd.Series([10.0, 20.0, 30.0]))
    assert abs(z.mean()) < TOL and abs(z.std(ddof=0) - 1.0) < TOL
    # constant input must not produce NaN/inf
    zc = scoring.zscore(pd.Series([5.0, 5.0, 5.0]))
    assert zc.notna().all() and (zc == 0).all()


def test_direction_signs_are_applied():
    """Cheaper P/E and higher ROE must both push the score up."""
    cfg = _cfg()
    uni, px, fund = _fixture(n=40)
    metrics = factors.build_metrics(px, fund, uni)
    scored = scoring.score(metrics, cfg)
    pool = scored[scored["eligible"]].dropna(subset=["z_pe_ratio", "pe_ratio"])
    assert len(pool) > 10
    # z_pe_ratio is sign-flipped, so it must correlate NEGATIVELY with raw P/E
    corr = pool["z_pe_ratio"].corr(pool["pe_ratio"])
    assert corr < -0.8, f"P/E direction not inverted (corr={corr:.3f})"
    pool_roe = scored[scored["eligible"]].dropna(subset=["z_roe", "roe"])
    assert pool_roe["z_roe"].corr(pool_roe["roe"]) > 0.8


def test_filters_exclude_and_explain():
    cfg = _cfg()
    uni, px, fund = _fixture(n=30)
    metrics = factors.build_metrics(px, fund, uni)
    metrics.loc[metrics.index[0], "last_close"] = 1.0          # penny stock
    metrics.loc[metrics.index[1], "market_cap"] = 1.0e8        # too small
    metrics.loc[metrics.index[2], "history_days"] = 10         # too short

    out = scoring.apply_filters(metrics, cfg)
    for i, expected in enumerate(["price", "market cap", "history"]):
        row = out.iloc[i]
        assert not row["eligible"], f"row {i} should be filtered out"
        assert expected in row["exclude_reason"], \
            f"row {i}: expected reason about {expected!r}, got {row['exclude_reason']!r}"
    assert (out["eligible"] & (out["exclude_reason"] != "")).sum() == 0


def test_composite_renormalises_over_missing_blocks():
    """A name missing a whole block must not be silently penalised to zero."""
    cfg = _cfg()
    uni, px, fund = _fixture(n=40)
    fund.loc[fund.index[:5], ["revenue_growth", "earnings_growth"]] = np.nan
    metrics = factors.build_metrics(px, fund, uni)
    scored = scoring.score(metrics, cfg)
    pool = scored[scored["eligible"]]
    assert pool["composite"].notna().sum() > 20
    assert np.isfinite(pool["composite"]).all()
    # blocks_used must reflect the missing growth block for the doctored names
    doctored = [t for t in fund["ticker"].iloc[:5] if t in pool.index]
    if doctored:
        assert pool.loc[doctored, "blocks_used"].max() <= len(cfg["metric_blocks"]) - 1


def test_scoring_finds_real_signal():
    """On synthetic data where good fundamentals really do predict drift,
    the top decile must beat the bottom decile."""
    cfg = _cfg()
    uni = synthetic.make_universe(90)
    px = synthetic.make_prices(uni, days=430)
    fund = synthetic.make_fundamentals(uni, px)
    close = price_mod.to_wide(px, "close")
    cut = close.index[-40]

    metrics = factors.build_metrics(px, fund, uni, asof=cut)
    scored = scoring.score(metrics, cfg)
    pool = scored[scored["eligible"] & scored["composite"].notna()]
    fwd = factors.forward_return(close, cut, 21).reindex(pool.index)

    k = max(5, len(pool) // 10)
    top = fwd.loc[pool.nlargest(k, "composite").index].mean()
    bottom = fwd.loc[pool.nsmallest(k, "composite").index].mean()
    assert top > bottom, (
        f"top decile ({top:.4f}) should beat bottom ({bottom:.4f}) on data "
        "constructed to contain signal"
    )


def test_pick_candidate_respects_portfolio_constraints():
    cfg = _cfg()
    uni, px, fund = _fixture(n=50)
    metrics = factors.build_metrics(px, fund, uni)
    scored = scoring.score(metrics, cfg)

    full = scoring.pick_candidate(scored, set(), {}, cfg)
    assert full, "should produce a shortlist"
    top = full[0]["ticker"]

    # holding the winner must remove it
    without = scoring.pick_candidate(scored, {top}, {}, cfg)
    assert all(c["ticker"] != top for c in without)

    # a saturated sector must be skipped
    top_sector = full[0]["sector"]
    cap = int(cfg.strategy["max_positions_per_sector"])
    capped = scoring.pick_candidate(scored, set(), {top_sector: cap}, cfg)
    assert all(c["sector"] != top_sector for c in capped)


# ---------------------------------------------------------------------------
# portfolio engine
# ---------------------------------------------------------------------------
def test_orders_fill_at_next_open_not_signal_close():
    """The core no-lookahead guarantee."""
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)

    day1 = pd.DataFrame({"open": [100.0], "close": [110.0]}, index=["AAA"])
    ladder.run_day(day1, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech",
                               "composite": 1.0, "last_close": 110.0}])
    assert len(state.positions) == 0, "entry must not execute on the signal day"
    assert len(state.pending_orders) == 1

    day2 = pd.DataFrame({"open": [120.0], "close": [130.0]}, index=["AAA"])
    ladder.run_day(day2, date(2025, 1, 3))
    assert len(state.positions) == 1
    pos = state.positions[0]
    slip = 1 + cfg.strategy["slippage_bps"] / 10_000
    assert abs(pos["entry_price"] - 120.0 * slip) < 1e-6, (
        f"must fill at day-2 open (120), got {pos['entry_price']}"
    )
    assert pos["entry_price"] != 110.0, "must never fill at the signal close"


def test_cash_and_equity_accounting_is_exact():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)

    bar = pd.DataFrame({"open": [50.0], "close": [50.0]}, index=["AAA"])
    ladder.run_day(bar, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech",
                               "composite": 1.0, "last_close": 50.0}])
    ladder.run_day(bar, date(2025, 1, 3))

    pos = state.positions[0]
    assert abs(state.cash + pos["cost_basis"] - 10_000.0) < 0.01, \
        "cash spent must equal the position's cost basis"
    assert abs(state.equity() - 10_000.0) < 0.5, \
        "buying at the marking price should leave equity ~flat (minus slippage)"
    assert state.cash >= -TOL, "cash must never go negative"


def test_position_closes_after_hold_period_with_correct_pnl():
    cfg = _cfg()
    hold = int(cfg.strategy["hold_days"])
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)

    bar = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=["AAA"])
    ladder.run_day(bar, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech",
                               "composite": 1.0, "last_close": 100.0}])
    ladder.run_day(bar, date(2025, 1, 3))
    assert len(state.positions) == 1
    entry_cost = state.positions[0]["cost_basis"]
    shares = state.positions[0]["shares"]

    for i in range(hold + 3):
        ladder.run_day(bar, date(2025, 2, 1))
        if not state.positions and not state.pending_orders:
            break
    assert len(state.positions) == 0, "position must close after the hold period"
    assert len(ladder.trades) == 1
    tr = ladder.trades[0]
    slip = cfg.strategy["slippage_bps"] / 10_000
    expected_proceeds = shares * 100.0 * (1 - slip)
    assert abs(tr["proceeds"] - expected_proceeds) < 0.01
    assert abs(tr["pnl"] - (expected_proceeds - entry_cost)) < 0.01
    # round trip at a flat price loses exactly the two-way slippage
    assert tr["return_pct"] < 0, "flat price round trip must lose slippage"


def test_ladder_never_exceeds_slots_and_never_double_buys():
    cfg = _cfg()
    n_slots = int(cfg.strategy["n_slots"])
    uni, px, fund = _fixture(n=45, days=400)
    close = price_mod.to_wide(px, "close")
    open_w = price_mod.to_wide(px, "open")

    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    metrics = factors.build_metrics(px, fund, uni)
    scored = scoring.score(metrics, cfg)

    for d in close.index[-120:]:
        bar = pd.DataFrame({"open": open_w.loc[d], "close": close.loc[d]}).dropna(subset=["close"])
        shortlist = scoring.pick_candidate(scored, state.held_tickers,
                                           state.sector_counts, cfg)
        ladder.run_day(bar, d.date(), shortlist=shortlist)

        assert len(state.positions) <= n_slots, "slot limit breached"
        tickers = [p["ticker"] for p in state.positions]
        assert len(tickers) == len(set(tickers)), "duplicate ticker held"
        assert state.cash >= -0.01, f"negative cash: {state.cash}"
        buys = [o for o in state.pending_orders if o["side"] == "BUY"]
        assert len(buys) <= 1, "more than one entry queued in a day"


def test_state_survives_a_save_load_roundtrip():
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "portfolio.json"
        state = portfolio.PortfolioState.new(10_000.0)
        ladder = portfolio.Ladder(cfg, state)
        bar = pd.DataFrame({"open": [80.0], "close": [80.0]}, index=["AAA"])
        ladder.run_day(bar, date(2025, 1, 2),
                       shortlist=[{"ticker": "AAA", "sector": "Tech",
                                   "composite": 1.0, "last_close": 80.0}])
        ladder.run_day(bar, date(2025, 1, 3))
        state.save(p)

        reloaded = portfolio.PortfolioState.load(p, 10_000.0)
        assert abs(reloaded.cash - state.cash) < TOL
        assert len(reloaded.positions) == len(state.positions)
        assert reloaded.bar_count == state.bar_count
        assert abs(reloaded.equity() - state.equity()) < TOL


def test_missing_price_data_defers_then_cancels():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar_ok = pd.DataFrame({"open": [10.0], "close": [10.0]}, index=["AAA"])
    ladder.run_day(bar_ok, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech",
                               "composite": 1.0, "last_close": 10.0}])
    empty = pd.DataFrame({"open": [], "close": []}, dtype=float)
    for _ in range(3):
        ladder.run_day(empty, date(2025, 1, 3))
        assert state.cash == 10_000.0, "no data must mean no fill"
    ladder.run_day(empty, date(2025, 1, 8))
    assert not state.pending_orders, "order should be cancelled after 3 retries"
    assert state.cash == 10_000.0


# ---------------------------------------------------------------------------
# learning guardrails
# ---------------------------------------------------------------------------
def test_proposed_weights_respect_bounds_and_sum_to_one():
    cfg = _cfg()
    L = cfg.learning
    current = {"value": 0.20, "quality": 0.18, "growth": 0.12, "momentum": 0.32,
               "earnings": 0.12, "risk": 0.06}

    for ic in [
        {"value": 0.9, "quality": 0.0, "growth": 0.0, "momentum": 0.0,
         "earnings": 0.0, "risk": 0.0},   # extreme
        {"value": -0.5, "quality": -0.5, "growth": -0.5, "momentum": -0.5,
         "earnings": -0.5, "risk": -0.5},  # all bad
        {"value": 0.0, "quality": 0.0, "growth": 0.0, "momentum": 0.0,
         "earnings": 0.0, "risk": 0.0},   # no signal
        {"value": 0.02, "quality": 0.03, "growth": 0.01, "momentum": 0.05,
         "earnings": 0.02, "risk": 0.01},  # realistic
    ]:
        w = learn.propose_weights(current, ic, cfg)
        assert abs(sum(w.values()) - 1.0) < 1e-3, f"weights must sum to 1: {w}"
        for k, v in w.items():
            assert L["weight_min"] - 1e-6 <= v <= L["weight_max"] + 1e-6, \
                f"{k}={v} outside [{L['weight_min']}, {L['weight_max']}]"
            step = abs(v - current[k])
            assert step <= L["max_step_per_month"] + 0.05, \
                f"{k} moved {step:.3f}, more than the monthly cap allows"


def test_learning_refuses_to_tune_without_enough_trades():
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(ROOT / "config.yml", Path(td) / "config.yml")
        (Path(td) / "state").mkdir()
        (Path(td) / "screener" / "data").mkdir(parents=True)
        shutil.copy(ROOT / "screener" / "data" / "sp500.csv",
                    Path(td) / "screener" / "data" / "sp500.csv")
        cfg = Config.load(Path(td) / "config.yml")
        cfg.root = Path(td)

        uni, px, _ = _fixture(n=25, days=400)
        rec = learn.run_learning(cfg, px, uni)
        assert rec["decision"] == "skipped"
        assert "closed trades" in rec["reason"]
        assert json.loads(rec["weights_before"]) == json.loads(rec["weights_after"])


def test_spearman_matches_known_cases():
    a = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])
    assert abs(learn.spearman(a, a) - 1.0) < 1e-9
    assert abs(learn.spearman(a, a[::-1].reset_index(drop=True)) + 1.0) < 1e-9
    assert np.isnan(learn.spearman(pd.Series([1.0, 2]), pd.Series([1.0, 2])))
    assert np.isnan(learn.spearman(a, pd.Series([5.0] * 10)))


def test_evaluate_weights_prefers_the_better_block():
    """A weight set pointing at the predictive block must score higher."""
    rng = np.random.default_rng(3)
    snaps = []
    for _ in range(30):
        n = 40
        good = pd.Series(rng.normal(size=n))
        noise = pd.Series(rng.normal(size=n))
        fwd = good * 0.05 + rng.normal(0, 0.01, size=n)   # `good` predicts fwd
        snaps.append({"date": None, "table": pd.DataFrame(
            {"momentum": good, "value": noise, "fwd": fwd})})

    smart = learn.evaluate_weights(snaps, {"momentum": 1.0, "value": 0.0})
    dumb = learn.evaluate_weights(snaps, {"momentum": 0.0, "value": 1.0})
    assert smart["mean"] > dumb["mean"], "weighting the predictive block must pay off"
    assert smart["hit_rate"] > 0.7


# ---------------------------------------------------------------------------
# fundamentals point-in-time behaviour
# ---------------------------------------------------------------------------
def test_latest_asof_never_reads_the_future():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fundamentals.csv"
        uni, px, _ = _fixture(n=6, days=300)
        for asof, pe in [("2024-01-05", 10.0), ("2024-06-05", 20.0),
                         ("2024-12-05", 30.0)]:
            snap = synthetic.make_fundamentals(uni, px, asof=asof)
            snap["pe_ratio"] = pe
            fund_mod.append_snapshot(snap, path)

        mid = fund_mod.latest_asof(path, on_or_before=date(2024, 7, 1))
        assert (mid["pe_ratio"] == 20.0).all(), \
            "must use the June snapshot, not December's"
        early = fund_mod.latest_asof(path, on_or_before=date(2024, 1, 1))
        assert early.empty, "nothing was published yet at that date"
        latest = fund_mod.latest_asof(path)
        assert (latest["pe_ratio"] == 30.0).all()
        assert len(latest) == len(uni), "one row per ticker"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
            failed.append(name)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    print("Running screener test suite\n" + "-" * 60)
    sys.exit(_run_all())
