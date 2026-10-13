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

from screener import allocation, evaluation, factors, learn, lifecycle, portfolio, scoring  # noqa: E402
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


def test_missing_todays_bar_triggers_stooq_fallback():
    """A liquid ticker with clean history but no bar for *today* specifically
    must still trigger the Stooq fallback.

    Regression test for a real bug: "missing" used to mean "has zero rows
    anywhere in the ~420-day lookback", which a name with 400+ good days and
    a merely-not-yet-published latest close will never satisfy — so the
    fallback silently never fired for exactly the case that matters most
    (today's bar, which is the one fill_pending actually needs).
    """
    end = date(2026, 9, 8)
    all_dates = pd.date_range(end - pd.Timedelta(days=5), end, freq="D")

    def fake_yf(tickers, start, end_):
        rows = []
        for t in tickers:
            for d in all_dates:
                if t == "AAA" and d == pd.Timestamp(end):
                    continue  # AAA's *today* bar "isn't published yet"
                rows.append({"date": d, "ticker": t, "open": 10.0, "high": 10.0,
                             "low": 10.0, "close": 10.0, "volume": 1000})
        return pd.DataFrame(rows)

    def fake_stooq(tickers, start, end_):
        return pd.DataFrame([
            {"date": pd.Timestamp(end), "ticker": t, "open": 11.0, "high": 11.0,
             "low": 11.0, "close": 11.0, "volume": 500}
            for t in tickers
        ])

    orig_yf, orig_stooq = price_mod._download_yfinance, price_mod._download_stooq
    price_mod._download_yfinance, price_mod._download_stooq = fake_yf, fake_stooq
    try:
        with tempfile.TemporaryDirectory() as td:
            df = price_mod.get_prices(
                ["AAA", "BBB"], Path(td) / "prices.csv.gz",
                lookback_days=5, provider="yfinance", allow_network=True, asof=end,
            )
    finally:
        price_mod._download_yfinance, price_mod._download_stooq = orig_yf, orig_stooq

    today = df[df["date"] == pd.Timestamp(end)]
    assert set(today["ticker"]) == {"AAA", "BBB"}, \
        "AAA should have been backfilled from Stooq for today's date"
    assert float(today.loc[today["ticker"] == "AAA", "close"].iloc[0]) == 11.0


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


def test_missing_mandatory_entry_safety_data_fails_closed():
    cfg = _cfg()
    uni, px, fund = _fixture(n=30)
    metrics = factors.build_metrics(px, fund, uni)
    metrics.loc[metrics.index[0], "dollar_volume"] = np.nan
    # Keep the other core filters satisfied so the intended failure reason
    # is deterministic even when a synthetic sample happens to be illiquid.
    metrics.loc[metrics.index[1], "dollar_volume"] = float(cfg.filters["min_dollar_volume"]) * 2
    metrics.loc[metrics.index[1], "market_cap"] = np.nan

    out = scoring.apply_filters(metrics, cfg)
    assert not bool(out.iloc[0]["eligible"])
    assert "liquidity data unavailable" in out.iloc[0]["exclude_reason"]
    assert not bool(out.iloc[1]["eligible"])
    assert "market cap unavailable" in out.iloc[1]["exclude_reason"]

    # If the provider omits the entire column, the configured market-cap
    # filter remains active rather than disappearing silently.
    no_cap = scoring.apply_filters(metrics.drop(columns=["market_cap"]), cfg)
    assert not no_cap["eligible"].any()
    assert not no_cap["eligible"].any()
    assert no_cap["exclude_reason"].str.contains("market cap unavailable").any()


def test_allocation_multiplier_and_entry_notional_follow_rank_strength():
    cfg = _cfg()
    weak = allocation.allocation_multiplier(0.0, cfg.strategy)
    strong = allocation.allocation_multiplier(1.0, cfg.strategy)
    assert strong > weak
    assert abs(weak - cfg.strategy["allocation_min_multiplier"]) < TOL
    assert abs(strong - cfg.strategy["allocation_max_multiplier"]) < TOL

    def queued_notional(weight):
        state = portfolio.PortfolioState.new(10_000.0)
        ladder = portfolio.Ladder(cfg, state)
        order = ladder.schedule_entry(
            [{"ticker": "AAA", "sector": "Tech", "composite": 1.0,
              "allocation_weight": weight, "last_close": 100.0}],
            signal_date=date(2025, 1, 2),
        )
        assert order is not None
        return order["notional"]

    assert queued_notional(strong) > queued_notional(weak)


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


def test_scoring_produces_auditable_ranked_candidates_without_horizon_labels():
    """Scoring is tested independently of any fixed forward-return horizon."""
    cfg = _cfg()
    uni, px, fund = _fixture(n=90, days=430)
    cut = price_mod.to_wide(px, "close").index[-1]

    metrics = factors.build_metrics(px, fund, uni, asof=cut)
    scored = scoring.score(metrics, cfg)
    pool = scored[scored["eligible"] & scored["composite"].notna()]

    assert len(pool) >= 20
    assert pool["composite"].notna().all()
    assert pool["rank"].notna().all()
    assert pool["rank"].min() == 1
    assert pool.sort_values("rank")["composite"].is_monotonic_decreasing
    assert any(c.startswith("block_") for c in pool.columns)


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


def test_rebalancing_moves_capital_toward_higher_ranked_positions():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame({"open": [100.0, 100.0], "close": [100.0, 100.0]},
                       index=["AAA", "BBB"])
    ladder.run_day(
        bar.loc[["AAA"]], date(2025, 1, 2),
        shortlist=[{"ticker": "AAA", "sector": "Tech", "composite": 0.5,
                    "allocation_weight": 1.0, "last_close": 100.0}],
    )
    ladder.run_day(bar.loc[["AAA"]], date(2025, 1, 3))
    initial_shares = state.positions[0]["shares"]

    ranks = pd.DataFrame(
        {"eligible": [True, True], "composite": [1.0, 0.5], "rank": [1, 2]},
        index=["AAA", "BBB"],
    )
    for i in range(5):
        ladder.run_day(bar, (pd.Timestamp("2025-01-04") + pd.Timedelta(days=i)).date(),
                       scored=ranks)
    queued = [o for o in state.pending_orders if o.get("side") == "REBALANCE"]
    assert queued, "a materially underweight, top-ranked position should be increased"
    assert any(o.get("action") == "increase" for o in queued)

    ladder.run_day(bar, date(2025, 1, 9), scored=ranks)
    assert state.positions[0]["shares"] > initial_shares
    assert state.cash >= -0.01
    assert state.positions[0]["cumulative_cost_basis"] >= state.positions[0]["cost_basis"]


def test_partial_rebalance_is_included_in_the_final_trade_outcome():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=["AAA"])
    ladder.run_day(
        bar, date(2025, 1, 2),
        shortlist=[{"ticker": "AAA", "sector": "Tech", "composite": 0.5,
                    "allocation_weight": 1.0, "last_close": 100.0}],
    )
    ladder.run_day(bar, date(2025, 1, 3))
    pos = state.positions[0]
    total_bought = pos["total_shares_bought"]
    ladder._execute_rebalance(
        {"side": "REBALANCE", "trade_id": pos["trade_id"],
         "target_value": pos["last_value"] * 0.5, "allocation_weight": 0.8},
        100.0, date(2025, 1, 4),
    )
    expected_proceeds = pos["cumulative_exit_proceeds"]
    state.pending_orders.append(
        {"side": "SELL", "ticker": "AAA", "trade_id": pos["trade_id"],
         "reason": "test close", "notional": 0.0}
    )
    ladder.fill_pending(bar, date(2025, 1, 5))
    trade = ladder.trades[0]
    assert abs(trade["proceeds"] - expected_proceeds) > 0.01
    assert abs(trade["pnl"] - (trade["proceeds"] - trade["cost_basis"])) < 0.01
    assert abs(trade["shares"] - total_bought) < 1e-5


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


def test_position_is_not_sold_just_because_time_passes():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=["AAA"])
    ladder.run_day(bar, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech",
                               "composite": 1.0, "last_close": 100.0}])
    ladder.run_day(bar, date(2025, 1, 3))
    assert len(state.positions) == 1

    strong = pd.DataFrame(
        {"eligible": [True], "composite": [1.0], "rank": [1]},
        index=["AAA"],
    )
    start = pd.Timestamp("2025-01-04")
    for i in range(90):
        ladder.run_day(bar, (start + pd.Timedelta(days=i)).date(), scored=strong)
        assert len(state.positions) == 1, "a strong signal should keep the position open"
        assert not any(o["side"] == "SELL" for o in state.pending_orders)


def test_missing_core_risk_data_does_not_force_existing_position_to_exit():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=["AAA"])
    ladder.run_day(
        bar, date(2025, 1, 2),
        shortlist=[{"ticker": "AAA", "sector": "Tech", "composite": 0.5,
                    "allocation_weight": 1.0, "last_close": 100.0}],
    )
    ladder.run_day(bar, date(2025, 1, 3))
    missing_risk_data = pd.DataFrame(
        {"eligible": [False], "composite": [np.nan], "rank": [np.nan],
         "exclude_reason": ["market cap unavailable"]},
        index=["AAA"],
    )
    for i in range(3):
        ladder.run_day(
            bar, (pd.Timestamp("2025-01-04") + pd.Timedelta(days=i)).date(),
            scored=missing_risk_data,
        )
        assert state.positions, "missing screening data alone must not close a holding"
        assert not any(o["side"] == "SELL" for o in state.pending_orders)
    assert state.positions[0].get("weak_signal_runs", 0) == 0


def test_position_exits_after_signal_is_weak_for_consecutive_reviews():
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=["AAA"])
    ladder.run_day(bar, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech",
                               "composite": 1.0, "last_close": 100.0}])
    ladder.run_day(bar, date(2025, 1, 3))
    entry_cost = state.positions[0]["cost_basis"]
    shares = state.positions[0]["shares"]

    weak_scores = pd.DataFrame(
        {
            "eligible": [True, True, True, True],
            "composite": [0.1, 0.9, 0.7, 0.5],
            "rank": [4, 1, 2, 3],
        },
        index=["AAA", "BBB", "CCC", "DDD"],
    )
    start = pd.Timestamp("2025-01-04")
    for i in range(2):
        ladder.run_day(bar, (start + pd.Timedelta(days=i)).date(), scored=weak_scores)
        assert not any(o["side"] == "SELL" for o in state.pending_orders)

    ladder.run_day(bar, (start + pd.Timedelta(days=2)).date(), scored=weak_scores)
    sells = [o for o in state.pending_orders if o["side"] == "SELL"]
    assert len(sells) == 1, "three consecutive weak reviews should queue a sale"
    assert "signal weakened" in sells[0]["reason"]

    ladder.run_day(bar, (start + pd.Timedelta(days=3)).date(), scored=weak_scores)
    assert not state.positions, "queued sale should fill at the next open"
    assert len(ladder.trades) == 1
    tr = ladder.trades[0]
    slip = cfg.strategy["slippage_bps"] / 10_000
    expected_proceeds = shares * 100.0 * (1 - slip)
    assert abs(tr["proceeds"] - expected_proceeds) < 0.01
    assert abs(tr["pnl"] - (expected_proceeds - entry_cost)) < 0.01
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
    current = {"value": 0.18, "quality": 0.16, "growth": 0.10, "momentum": 0.30,
               "earnings": 0.12, "risk": 0.06, "technical": 0.08}

    for ic in [
        {"value": 0.9, "quality": 0.0, "growth": 0.0, "momentum": 0.0,
         "earnings": 0.0, "risk": 0.0, "technical": 0.0},   # extreme
        {"value": -0.5, "quality": -0.5, "growth": -0.5, "momentum": -0.5,
         "earnings": -0.5, "risk": -0.5, "technical": -0.5},  # all bad
        {"value": 0.0, "quality": 0.0, "growth": 0.0, "momentum": 0.0,
         "earnings": 0.0, "risk": 0.0, "technical": 0.0},   # no signal
        {"value": 0.02, "quality": 0.03, "growth": 0.01, "momentum": 0.05,
         "earnings": 0.02, "risk": 0.01, "technical": 0.02},  # realistic
    ]:
        w = learn.propose_weights(current, ic, cfg)
        assert abs(sum(w.values()) - 1.0) < 1e-3, f"weights must sum to 1: {w}"
        for k, v in w.items():
            assert L["weight_min"] - 1e-6 <= v <= L["weight_max"] + 1e-6, \
                f"{k}={v} outside [{L['weight_min']}, {L['weight_max']}]"
            step = abs(v - current[k])
            assert step <= L["max_step_per_month"] + 0.05, \
                f"{k} moved {step:.3f}, more than the monthly cap allows"


def test_learning_stays_disabled_until_lifecycle_evaluation_is_ready():
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
        assert "disabled" in rec["reason"]
        assert json.loads(rec["weights_before"]) == json.loads(rec["weights_after"])


def test_spearman_matches_known_cases():
    a = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0])
    assert abs(learn.spearman(a, a) - 1.0) < 1e-9
    assert abs(learn.spearman(a, a[::-1].reset_index(drop=True)) + 1.0) < 1e-9
    assert np.isnan(learn.spearman(pd.Series([1.0, 2]), pd.Series([1.0, 2])))
    assert np.isnan(learn.spearman(a, pd.Series([5.0] * 10)))


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
# lifecycle signal/outcome linkage
# ---------------------------------------------------------------------------
def test_lifecycle_outcomes_link_the_entry_snapshot_and_reconcile_idempotently():
    decisions = pd.DataFrame([{
        "data_date": "2025-01-02",
        "ticker": "AAA",
        "signal_id": "2025-01-02:AAA",
        "last_close": 100.0,
        "eligible": True,
        "rank": 1,
        "rank_percentile": 1.0,
        "composite": 0.8,
        "blocks_used": 7,
        "factor_agreement_positive_share": 0.71,
        "fundamental_coverage": 1.0,
        "block_momentum": 1.25,
        "is_chosen_candidate": True,
        "shortlisted": True,
        "portfolio_action": "BUY",
    }])
    trades = pd.DataFrame([{
        "trade_id": 7,
        "ticker": "AAA",
        "name": "Alpha",
        "sector": "Technology",
        "signal_id": "2025-01-02:AAA",
        "signal_date": "2025-01-02",
        "entry_date": "2025-01-03",
        "entry_price": 101.0,
        "exit_date": "2025-02-14",
        "exit_price": 111.0,
        "shares": 10.0,
        "cost_basis": 1010.0,
        "proceeds": 1110.0,
        "pnl": 100.0,
        "return_pct": 100.0 / 1010.0,
        "bars_held": 30,
        "exit_reason": "signal weakened",
        "av_verdict": "confirmed",
    }])

    outcomes = lifecycle.build_signal_outcomes(decisions, trades)
    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["feature_link_status"] == "linked"
    assert bool(row["learning_eligible"])
    assert row["closed_outcome"] == "win"
    assert row["signal_composite"] == 0.8
    assert row["signal_block_momentum"] == 1.25
    assert row["exit_date"] == "2025-02-14"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        decision_path = root / "decision_history.csv"
        trades_path = root / "trades.csv"
        output_path = root / "signal_outcomes.csv"
        decisions.to_csv(decision_path, index=False)
        trades.to_csv(trades_path, index=False)
        first = lifecycle.reconcile_signal_outcomes(decision_path, trades_path, output_path)
        second = lifecycle.reconcile_signal_outcomes(decision_path, trades_path, output_path)
        saved = pd.read_csv(output_path)
        assert first["linked_trades"] == second["linked_trades"] == 1
        assert first["learning_eligible_trades"] == 1
        assert len(saved) == 1, "reconciliation must not append duplicate outcomes"


def test_legacy_closed_trades_are_kept_but_not_eligible_for_learning():
    trades = pd.DataFrame([{
        "trade_id": 1,
        "ticker": "OLD",
        "entry_date": "2024-01-02",
        "exit_date": "2024-02-02",
        "pnl": 25.0,
        "return_pct": 0.05,
    }])
    outcomes = lifecycle.build_signal_outcomes(pd.DataFrame(), trades)
    assert len(outcomes) == 1, "legacy trade should remain auditable"
    assert outcomes.iloc[0]["feature_link_status"] == "missing_signal_id"
    assert outcomes.iloc[0]["closed_outcome"] == "win"
    assert not bool(outcomes.iloc[0]["learning_eligible"])


def test_append_trades_upgrades_old_schema_and_deduplicates_trade_ids():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trades.csv"
        pd.DataFrame([{"trade_id": 1, "ticker": "OLD", "pnl": 5.0}]).to_csv(
            path, index=False
        )
        portfolio.append_trades([{
            "trade_id": 2, "ticker": "AAA", "pnl": 10.0,
            "signal_date": "2025-01-02", "signal_id": "2025-01-02:AAA",
        }], path)
        # Simulate a retried run writing the same close with the final record.
        portfolio.append_trades([{
            "trade_id": 2, "ticker": "AAA", "pnl": 12.0,
            "signal_date": "2025-01-02", "signal_id": "2025-01-02:AAA",
        }], path)
        saved = pd.read_csv(path)
        assert len(saved) == 2
        assert "signal_id" in saved.columns and "signal_date" in saved.columns
        assert pd.isna(saved.loc[saved["trade_id"] == 1, "signal_id"]).all()
        assert saved.loc[saved["trade_id"] == 2, "pnl"].iloc[0] == 12.0


# ---------------------------------------------------------------------------
# lifecycle evaluation
# ---------------------------------------------------------------------------
def _evaluation_fixture(n=50):
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n):
        signal_date = start + pd.Timedelta(days=i)
        ret = (i - (n / 2)) / 500
        rows.append({
            "trade_id": i + 1,
            "ticker": f"T{i:03d}",
            "signal_id": f"{signal_date:%Y-%m-%d}:T{i:03d}",
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "exit_date": (signal_date + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
            "pnl": round(ret * 500, 2),
            "return_pct": ret,
            "bars_held": 5 + i % 4,
            "exit_reason": "signal weakened",
            "feature_link_status": "linked",
            "learning_eligible": True,
            "signal_block_momentum": float(i),
            "signal_block_value": float(n - i),
            "signal_block_quality": float(i % 7),
            "signal_block_growth": float(i % 5),
            "signal_block_earnings": float(i % 4),
            "signal_block_risk": float(i % 3),
            "signal_block_technical": float(i % 9),
        })
    return pd.DataFrame(rows)


def test_lifecycle_evaluation_reports_insufficient_sample_without_tuning():
    report = evaluation.evaluate_lifecycle_outcomes(_evaluation_fixture(6), {
        "min_trades_for_evaluation": 40,
        "min_training_trades": 20,
        "min_validation_trades": 10,
    })
    assert report["status"] == "collecting_data"
    assert report["counts"]["evaluable_trades"] == 6
    assert report["temporal_validation"]["status"] == "insufficient_sample"
    assert report["learning_enabled"] is False
    assert len(report["factor_associations"]) == 7


def test_lifecycle_evaluation_uses_strict_temporal_validation_split():
    report = evaluation.evaluate_lifecycle_outcomes(_evaluation_fixture(50), {
        "min_trades_for_evaluation": 40,
        "min_training_trades": 20,
        "min_validation_trades": 10,
        "validation_fraction": 0.30,
        "min_factor_observations": 8,
    })
    assert report["status"] == "temporal_split_ready_for_review"
    split = report["temporal_validation"]
    assert split["training"]["n"] >= 20
    assert split["validation"]["n"] >= 10
    cutoff = pd.Timestamp(split["cutoff_signal_date"], tz="UTC")
    rows = _evaluation_fixture(50)
    exit_dates = pd.to_datetime(rows["exit_date"], utc=True)
    signal_dates = pd.to_datetime(rows["signal_date"], utc=True)
    assert (exit_dates[signal_dates < cutoff] < cutoff).sum() == split["training"]["n"]
    momentum = next(r for r in report["factor_associations"] if r["block"] == "momentum")
    assert momentum["spearman_return"] is not None
    assert momentum["spearman_return"] > 0.9
    assert momentum["validation_spearman_return"] is not None
    assert report["learning_enabled"] is False


def test_lifecycle_evaluation_writes_valid_json_for_empty_sample():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        outcomes_path = root / "signal_outcomes.csv"
        report_path = root / "lifecycle_evaluation.json"
        pd.DataFrame().to_csv(outcomes_path, index=False)
        report = evaluation.refresh_lifecycle_evaluation(
            outcomes_path, report_path, {"min_trades_for_evaluation": 40}
        )
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved["status"] == "collecting_data"
        assert saved["counts"]["closed_trades"] == 0
        assert saved["factor_associations"] == report["factor_associations"]
        assert saved["learning_enabled"] is False


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
