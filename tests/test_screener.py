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

from screener import allocation, evaluation, factors, learn, lifecycle, portfolio, scoring, universe  # noqa: E402
from screener.config import Config  # noqa: E402
from screener.data import fundamentals as fund_mod  # noqa: E402
from screener.data import prices as price_mod  # noqa: E402
from scripts import backtest as backtest_mod  # noqa: E402
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


def test_new_entries_obey_single_name_and_sector_exposure_caps():
    cfg = _cfg()
    cfg.strategy["max_concurrent_positions"] = 3
    cfg.strategy["max_position_pct_equity"] = 0.08
    cfg.strategy["max_sector_pct_equity"] = 0.25

    def make_state():
        state = portfolio.PortfolioState.new(10_000.0)
        state.cash = 8_000.0
        state.positions = [
            {"trade_id": 1, "ticker": "AAA", "sector": "Tech",
             "shares": 10.0, "entry_price": 100.0, "cost_basis": 1_000.0,
             "last_value": 1_000.0, "allocation_weight": 0.6, "entry_bar": 0},
            {"trade_id": 2, "ticker": "BBB", "sector": "Tech",
             "shares": 10.0, "entry_price": 100.0, "cost_basis": 1_000.0,
             "last_value": 1_000.0, "allocation_weight": 0.6, "entry_bar": 0},
        ]
        state.next_trade_id = 3
        return state

    top_pick = {"ticker": "CCC", "sector": "Health Care", "composite": 2.0,
                "allocation_weight": 1.8, "last_close": 100.0}
    other_sector = portfolio.Ladder(cfg, make_state()).schedule_entry([top_pick])
    assert other_sector is not None
    assert other_sector["notional"] <= 800.0 + 0.01

    same_sector = portfolio.Ladder(cfg, make_state()).schedule_entry(
        [{**top_pick, "sector": "Tech"}]
    )
    assert same_sector is not None
    assert same_sector["notional"] <= 500.0 + 0.01, (
        "the $2,000 held in Tech leaves only $500 under a 25% sector cap"
    )


def test_rebalance_targets_obey_single_name_and_sector_caps():
    cfg = _cfg()
    cfg.strategy["max_concurrent_positions"] = 4
    cfg.strategy["max_position_pct_equity"] = 0.20
    cfg.strategy["max_sector_pct_equity"] = 0.25
    state = portfolio.PortfolioState.new(10_000.0)
    state.cash = 4_000.0
    state.bar_count = 20
    state.positions = [
        {"trade_id": 1, "ticker": "AAA", "sector": "Tech", "shares": 15.0,
         "entry_price": 100.0, "cost_basis": 1_500.0, "last_value": 1_500.0,
         "allocation_weight": 1.0, "entry_bar": 0},
        {"trade_id": 2, "ticker": "BBB", "sector": "Tech", "shares": 15.0,
         "entry_price": 100.0, "cost_basis": 1_500.0, "last_value": 1_500.0,
         "allocation_weight": 1.0, "entry_bar": 0},
        {"trade_id": 3, "ticker": "CCC", "sector": "Health Care", "shares": 15.0,
         "entry_price": 100.0, "cost_basis": 1_500.0, "last_value": 1_500.0,
         "allocation_weight": 1.0, "entry_bar": 0},
        {"trade_id": 4, "ticker": "DDD", "sector": "Energy", "shares": 15.0,
         "entry_price": 100.0, "cost_basis": 1_500.0, "last_value": 1_500.0,
         "allocation_weight": 1.0, "entry_bar": 0},
    ]
    scored = pd.DataFrame(
        {"eligible": [True] * 4, "composite": [4.0, 3.0, 2.0, 1.0],
         "rank": [1, 2, 3, 4], "exclude_reason": [""] * 4},
        index=["AAA", "BBB", "CCC", "DDD"],
    )
    ladder = portfolio.Ladder(cfg, state)
    targets = ladder._position_target_values(scored, pending_sell_ids=set())
    tech_target = targets[1] + targets[2]
    assert tech_target <= 2_500.0 + 0.01
    assert all(value <= 2_000.0 + 0.01 for value in targets.values())


def test_single_stock_cap_bounds_the_strongest_conviction():
    """The per-name ceiling is what sets the minimum number of holdings.

    A maximum-conviction pick (weight 1.8 against an empty book, so 1.8x the
    base target) must still be clipped to max_position_pct_equity. At 25%
    that is what guarantees at least four holdings.
    """
    cfg = _cfg()
    cap = float(cfg.strategy["max_position_pct_equity"])
    assert abs(cap - 0.25) < TOL, "config changed; update this expectation"
    state = portfolio.PortfolioState.new(10_000.0)
    order = portfolio.Ladder(cfg, state).schedule_entry(
        [{"ticker": "AAA", "sector": "Tech", "composite": 2.0,
          "allocation_weight": 1.8, "last_close": 100.0}]
    )
    assert order is not None
    assert abs(order["notional"] - 10_000.0 * cap) < 0.02
    assert round(1.0 / cap) == 4, "a 25% ceiling implies a 4-position minimum"


def test_drawdown_breaker_cancels_pending_buys_without_liquidating_positions():
    cfg = _cfg()
    cfg.strategy["max_concurrent_positions"] = 3
    cfg.strategy["max_portfolio_drawdown_pct"] = 0.20
    cfg.strategy["drawdown_resume_below_pct"] = 0.12
    state = portfolio.PortfolioState.new(10_000.0)
    state.cash = 2_000.0
    state.peak_equity = 10_000.0
    state.positions = [{
        "trade_id": 1, "ticker": "AAA", "sector": "Tech", "shares": 80.0,
        "entry_date": "2025-01-01", "entry_price": 100.0, "cost_basis": 8_000.0,
        "last_value": 8_000.0, "last_price": 100.0, "entry_bar": 0,
        "allocation_weight": 1.0, "bars_held": 0,
    }]
    state.pending_orders = [{
        "side": "BUY", "ticker": "BBB", "sector": "Health Care",
        "notional": 1_000.0, "retries": 0,
    }]
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame(
        {"open": [75.0, 50.0], "close": [75.0, 50.0]},
        index=["AAA", "BBB"],
    )
    result = ladder.run_day(
        bar, date(2025, 1, 2),
        shortlist=[{"ticker": "CCC", "sector": "Energy", "composite": 1.0}],
    )
    assert state.drawdown_paused
    assert abs(state.current_drawdown_pct - 0.20) < TOL
    assert [p["ticker"] for p in state.positions] == ["AAA"]
    assert not any(o.get("side") == "BUY" for o in state.pending_orders)
    assert any(e.get("event") == "order_cancelled" for e in result["events"])
    assert any(e.get("event") == "drawdown_protection_activated" for e in result["events"])


def test_drawdown_breaker_resumes_only_after_recovery_below_hysteresis_level():
    cfg = _cfg()
    cfg.strategy["max_concurrent_positions"] = 3
    cfg.strategy["max_position_pct_equity"] = 0.15
    cfg.strategy["max_portfolio_drawdown_pct"] = 0.20
    cfg.strategy["drawdown_resume_below_pct"] = 0.12
    state = portfolio.PortfolioState.new(10_000.0)
    state.cash = 2_000.0
    state.peak_equity = 10_000.0
    state.current_drawdown_pct = 0.15
    state.drawdown_paused = True
    state.drawdown_pause_date = "2025-01-01"
    state.positions = [{
        "trade_id": 1, "ticker": "AAA", "sector": "Tech", "shares": 70.0,
        "entry_date": "2025-01-01", "entry_price": 100.0, "cost_basis": 6_500.0,
        "last_value": 6_500.0, "last_price": 92.8571, "entry_bar": 0,
        "allocation_weight": 1.0, "bars_held": 0,
    }]
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame(
        {"open": [6900.0 / 70.0, 100.0], "close": [6900.0 / 70.0, 100.0]},
        index=["AAA", "BBB"],
    )
    result = ladder.run_day(
        bar, date(2025, 1, 2),
        shortlist=[{"ticker": "BBB", "sector": "Health Care",
                    "composite": 2.0, "allocation_weight": 1.8}],
    )
    assert not state.drawdown_paused
    assert state.current_drawdown_pct <= 0.12
    assert any(o.get("side") == "BUY" and o.get("ticker") == "BBB"
               for o in state.pending_orders)
    assert any(e.get("event") == "drawdown_protection_resumed" for e in result["events"])


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
    # Equity dips by exactly the slippage paid, which scales with the size
    # of the position — not by a fixed amount.
    expected_slip = pos["cost_basis"] * float(cfg.strategy["slippage_bps"]) / 10_000.0
    assert abs(state.equity() - (10_000.0 - expected_slip)) < 0.02, \
        "buying at the marking price should cost exactly the slippage"
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


def test_every_data_gap_reason_pauses_review_rather_than_forcing_an_exit():
    """No data-availability gap may masquerade as a weakened signal.

    Regression test for a real exit: COF was sold on 2026-10-23 with reason
    "signal weakened for 3 consecutive reviews: fundamentals incomplete" —
    the ladder had no fundamentals that week, not a deteriorating stock.
    The exemption list in review_positions was hand-written and covered only
    two of the five data-gap reasons scoring can emit, so this walks the
    whole list from scoring rather than spot-checking one string.
    """
    for reason in scoring.DATA_GAP_REASONS:
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
        assert state.positions, f"setup failed for {reason!r}"

        gap = pd.DataFrame(
            {"eligible": [False], "composite": [np.nan], "rank": [np.nan],
             "exclude_reason": [reason]},
            index=["AAA"],
        )
        # Well past exit_confirmation_observations: a gap must never age into
        # an exit, however long the provider stays silent.
        for i in range(8):
            ladder.run_day(
                bar, (pd.Timestamp("2025-01-06") + pd.Timedelta(days=i)).date(),
                scored=gap,
            )
            assert state.positions, f"{reason!r} closed a holding"
            assert not any(o["side"] == "SELL" for o in state.pending_orders), \
                f"{reason!r} queued a sell"
        assert state.positions[0].get("weak_signal_runs", 0) == 0, \
            f"{reason!r} counted as a weak-signal review"


def test_genuine_deterioration_still_exits_while_data_gaps_do_not():
    """The fix must not turn the exit rule off for real deterioration."""
    cfg = _cfg()
    state = portfolio.PortfolioState.new(10_000.0)
    ladder = portfolio.Ladder(cfg, state)
    bar = pd.DataFrame({"open": [100.0], "close": [100.0]}, index=["AAA"])
    ladder.run_day(bar, date(2025, 1, 2),
                   shortlist=[{"ticker": "AAA", "sector": "Tech", "composite": 0.5,
                               "allocation_weight": 1.0, "last_close": 100.0}])
    ladder.run_day(bar, date(2025, 1, 3))

    # "illiquid" means we DID see the volume and it was too low — a real
    # change in the name, so the usual confirmation-and-exit path applies.
    bad = pd.DataFrame(
        {"eligible": [False], "composite": [np.nan], "rank": [np.nan],
         "exclude_reason": ["illiquid"]},
        index=["AAA"],
    )
    sold = False
    for i in range(6):
        ladder.run_day(bar, (pd.Timestamp("2025-01-06") + pd.Timedelta(days=i)).date(),
                       scored=bad)
        if not state.positions:
            sold = True
            break
    assert sold, "a genuine eligibility failure must still close the position"


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
    n_slots = int(cfg.strategy["max_concurrent_positions"])
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



def test_universe_history_uses_latest_snapshot_on_or_before_asof():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "screener" / "data").mkdir(parents=True)
        (root / "state").mkdir()
        shutil.copy(ROOT / "config.yml", root / "config.yml")
        cfg = Config.load(root / "config.yml")
        cfg.root = root
        pd.DataFrame([{
            "ticker": "LIVE", "name": "Current member",
            "sector": "Technology", "industry": "Software",
        }]).to_csv(root / "screener" / "data" / "sp500.csv", index=False)

        history_path = root / "state" / "universe_history.csv"
        universe.append_snapshot(pd.DataFrame([{
            "ticker": "AAA", "name": "Former member",
            "sector": "Industrials", "industry": "Machinery",
        }]), history_path, "2025-01-01")
        universe.append_snapshot(pd.DataFrame([{
            "ticker": "BBB", "name": "New member",
            "sector": "Technology", "industry": "Software",
        }]), history_path, "2025-02-01")

        jan = universe.load_universe(cfg, allow_network=False, asof=date(2025, 1, 15))
        feb = universe.load_universe(cfg, allow_network=False, asof=date(2025, 2, 15))
        before_archive = universe.load_universe(cfg, allow_network=False, asof=date(2024, 12, 31))
        assert jan["ticker"].tolist() == ["AAA"]
        assert feb["ticker"].tolist() == ["BBB"]
        assert before_archive["ticker"].tolist() == ["LIVE"]


def test_historical_backtest_replays_production_pipeline_and_writes_benchmark_report():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "screener" / "data").mkdir(parents=True)
        (root / "state").mkdir()
        shutil.copy(ROOT / "config.yml", root / "config.yml")

        uni = synthetic.make_universe(26, seed=23)
        px = synthetic.make_prices(uni, days=360, seed=24)
        first_date = pd.Timestamp(px["date"].min())
        first_day_prices = px[px["date"] <= first_date].copy()
        first_day_prices.attrs["alphas"] = px.attrs.get("alphas", {})
        fund = synthetic.make_fundamentals(
            uni, first_day_prices, asof=first_date.date().isoformat(), seed=25
        )
        uni.to_csv(root / "screener" / "data" / "sp500.csv", index=False)
        universe.append_snapshot(uni, root / "state" / "universe_history.csv", first_date.date())
        fund_mod.append_snapshot(fund, root / "state" / "fundamentals.csv")

        market_days = sorted(pd.to_datetime(
            px.loc[px["ticker"] == "SPY", "date"].unique()
        ))
        start = market_days[305].date()
        end = market_days[318].date()
        output = root / "backtest-output"
        report = backtest_mod.run_backtest(
            root / "config.yml", px, start, end, output,
            warmup_sessions=252, verbose=False,
        )

        curve = pd.read_csv(output / "equity_curve.csv")
        assert report["actual_period"]["start"] == start.isoformat()
        assert report["actual_period"]["end"] == end.isoformat()
        assert report["actual_period"]["trading_sessions"] == 14
        assert report["initial_capital"] == 10_000.0
        assert len(curve) == 14
        assert abs(float(curve["benchmark_equity"].iloc[0]) - 10_000.0) < 0.01
        assert report["benchmark"]["symbol"] == "SPY"
        assert report["benchmark"]["total_return"] is not None
        assert "excess_total_return" in report["relative_to_benchmark"]
        assert (output / "summary.json").exists()
        assert (output / "trades.csv").exists()
        assert report["universe_membership"]["point_in_time_membership_used_for_entire_period"] is True
        assert not any("survivorship bias remains" in x for x in report["limitations"])


def test_historical_backtest_refuses_fundamentals_lookahead():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "screener" / "data").mkdir(parents=True)
        (root / "state").mkdir()
        shutil.copy(ROOT / "config.yml", root / "config.yml")
        uni, px, fund = _fixture(n=8, days=320)
        uni.to_csv(root / "screener" / "data" / "sp500.csv", index=False)
        fund["asof"] = "2024-06-01"
        fund_mod.append_snapshot(fund, root / "state" / "fundamentals.csv")
        start = pd.Timestamp(px["date"].min()).date()
        end = pd.Timestamp(px["date"].max()).date()
        try:
            backtest_mod.run_backtest(
                root / "config.yml", px, start, end, root / "output",
                warmup_sessions=252, verbose=False,
            )
        except ValueError as exc:
            assert "introduce look-ahead bias" in str(exc)
        else:
            raise AssertionError("backtest must not use later fundamental values in earlier dates")


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


def test_lifecycle_learning_waits_for_enough_linked_completed_trades():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shutil.copy(ROOT / "config.yml", root / "config.yml")
        (root / "state").mkdir()
        cfg = Config.load(root / "config.yml")
        cfg.root = root

        rec = learn.run_learning(cfg, pd.DataFrame(), pd.DataFrame())
        assert rec["decision"] == "skipped"
        assert "eligible linked closed trades" in rec["reason"]
        assert json.loads(rec["weights_before"]) == json.loads(rec["weights_after"])
        assert rec["mode"] == "lifecycle"
        assert (root / "state" / "weights_history.csv").exists()


def test_lifecycle_learner_accepts_only_a_validation_improving_proposal():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shutil.copy(ROOT / "config.yml", root / "config.yml")
        (root / "state").mkdir()
        cfg = Config.load(root / "config.yml")
        cfg.root = root
        cfg.learning["enabled"] = True
        cfg.learning["min_trades_for_update"] = 40
        cfg.learning["min_training_trades"] = 20
        cfg.learning["min_validation_trades"] = 10
        cfg.learning["min_factor_observations"] = 8
        cfg.learning["validation_fraction"] = 0.30
        cfg.learning["min_validation_ic_improvement"] = 0.0

        rng = np.random.default_rng(17)
        n = 100
        blocks = {block: rng.normal(size=n) for block in learn.BLOCKS}
        returns = blocks["momentum"] + rng.normal(scale=0.05, size=n)
        start = pd.Timestamp("2025-01-01")
        rows = []
        for i in range(n):
            signal_date = start + pd.Timedelta(days=i)
            row = {
                "trade_id": i + 1,
                "ticker": f"T{i:03d}",
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "exit_date": (signal_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                "return_pct": float(returns[i]),
                "learning_eligible": True,
            }
            row.update({f"signal_block_{b}": float(blocks[b][i]) for b in learn.BLOCKS})
            rows.append(row)
        pd.DataFrame(rows).to_csv(cfg.path("signal_outcomes_csv"), index=False)

        before = cfg.weights.copy()
        rec = learn.run_learning(cfg, pd.DataFrame(), pd.DataFrame())
        assert rec["decision"] == "accepted", rec["reason"]
        after = Config.load(root / "config.yml").weights
        assert abs(sum(after.values()) - 1.0) < 1e-3
        assert after["momentum"] > before["momentum"]
        assert rec["validation_proposed_ic"] > rec["validation_baseline_ic"]
        assert rec["validation_improvement"] > 0
        assert max(abs(after[b] - before[b]) for b in learn.BLOCKS) <= (
            cfg.learning["max_step_per_month"] + 0.001
        )
        assert len(pd.read_csv(cfg.path("weights_history"))) == 1


def test_lifecycle_learner_never_tunes_same_sample_twice():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shutil.copy(ROOT / "config.yml", root / "config.yml")
        (root / "state").mkdir()
        cfg = Config.load(root / "config.yml")
        cfg.root = root
        pd.DataFrame([{
            "sample_key": "0|none|none|none",
            "decision": "skipped",
            "reason": "insufficient data",
        }]).to_csv(cfg.path("weights_history"), index=False)

        rec = learn.run_learning(cfg, pd.DataFrame(), pd.DataFrame())
        assert rec["decision"] == "skipped"
        assert "no new eligible closed trades" in rec["reason"]


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


def test_lifecycle_evaluation_arms_automatic_gate_only_when_enabled_and_ready():
    report = evaluation.evaluate_lifecycle_outcomes(_evaluation_fixture(50), {
        "enabled": True,
        "min_trades_for_evaluation": 40,
        "min_training_trades": 20,
        "min_validation_trades": 10,
        "validation_fraction": 0.30,
        "min_factor_observations": 8,
    })
    assert report["status"] == "ready_for_automatic_gate"
    assert report["learning_enabled"] is True


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


def test_sector_neutral_scoring_removes_structural_sector_bias():
    """Value must rank a name against its peers, not against the index.

    Built so the answer is unambiguous: every Energy name is given a lower
    P/E than every Tech name (the structural pattern seen in real data), but
    the two sectors are otherwise identical and each contains the same spread
    of relative cheapness. Market-wide, every Energy name outranks every Tech
    name on the value block purely because of the industry it is in. Scored
    within sector, the two sectors must come out even, and the ranking must
    instead reflect cheapness relative to peers.
    """
    rows, sectors = [], []
    for i in range(10):
        # Energy P/E 6..15, Tech P/E 26..35 — disjoint ranges, same spread.
        rows.append({"ticker": f"E{i}", "pe_ratio": 6.0 + i, "sector": "Energy"})
        rows.append({"ticker": f"T{i}", "pe_ratio": 26.0 + i, "sector": "Tech"})
    frame = pd.DataFrame(rows).set_index("ticker")
    pe = frame["pe_ratio"]
    sectors = frame["sector"]

    market = scoring.zscore(pe) * -1          # direction: lower P/E is better
    neutral = scoring.zscore_within_sector(pe, sectors, min_names=5) * -1

    energy = [t for t in frame.index if t.startswith("E")]
    tech = [t for t in frame.index if t.startswith("T")]

    # Market-wide: the sectors are cleanly separated — the bias we're removing.
    assert market.loc[energy].min() > market.loc[tech].max(), \
        "fixture is wrong: market-wide scoring should separate the sectors"

    # Sector-neutral: neither sector is advantaged as a group.
    assert abs(neutral.loc[energy].mean()) < 1e-9
    assert abs(neutral.loc[tech].mean()) < 1e-9
    # ...and the cheapest name in each sector scores identically well.
    assert abs(neutral.loc["E0"] - neutral.loc["T0"]) < 1e-9
    # Within a sector, cheaper still beats dearer.
    assert neutral.loc["E0"] > neutral.loc["E9"]


def test_thin_sectors_fall_back_to_market_wide_scoring():
    """A 2-name sector must not get extreme z-scores from its own spread."""
    frame = pd.DataFrame(
        [{"ticker": f"B{i}", "v": 10.0 + i, "sector": "Big"} for i in range(10)]
        + [{"ticker": "S0", "v": 10.2, "sector": "Small"},
           {"ticker": "S1", "v": 10.3, "sector": "Small"}]
    ).set_index("ticker")

    z = scoring.zscore_within_sector(frame["v"], frame["sector"], min_names=5)

    # Scored inside their own 2-name group, S0 and S1 would be forced to
    # exactly -1 and +1 — a full 2 sd apart on a 0.1 difference in the raw
    # metric. The fallback must leave them as close as their values are.
    in_group = scoring.zscore(frame.loc[["S0", "S1"], "v"])
    assert abs(in_group.loc["S1"] - in_group.loc["S0"]) > 1.9, \
        "fixture is wrong: in-group scoring should blow these apart"
    assert abs(z.loc["S1"] - z.loc["S0"]) < 0.2, \
        "thin sector was scored against itself instead of the market"

    # The fallback is the ordinary market-wide z-score, nothing bespoke.
    market = scoring.zscore(frame["v"])
    assert abs(z.loc["S0"] - market.loc["S0"]) < 1e-9


if __name__ == "__main__":
    print("Running screener test suite\n" + "-" * 60)
    sys.exit(_run_all())
