"""Point-in-time fundamentals reconstructed from SEC XBRL filings.

Why this exists
---------------
The live system snapshots yfinance ratios weekly, which only began in
October 2026. That leaves the four fundamental blocks — half the composite's
weight — untestable: a replay cannot use today's ratios for 2021 without
look-ahead. SEC's `companyfacts` API returns every value a company has ever
reported in a 10-K or 10-Q together with the date the filing was made, so a
value is usable from its filing date onward and not a day before. That makes
the history point-in-time by construction.

What is and is not reproduced
-----------------------------
Every ratio in the live schema except `earnings_surprise_pct`, which compares
reported earnings with analyst consensus. Consensus is not an SEC disclosure,
so it cannot be reconstructed here; a replay on these fundamentals drops the
earnings block rather than counting a structurally absent value as missing.

The ratios are rebuilt from statements, so their definitions are close to
but not identical with yfinance's: trailing-twelve-month sums from filed
periods rather than a vendor's own TTM, and debt from a short list of XBRL
tags rather than a curated balance-sheet line. That is fine for ranking a
cross-section consistently; it is NOT fine to splice these with the live
yfinance snapshots in one replay, which is why a replay on this source uses
it for the whole window.

The traps handled below
-----------------------
* 10-Qs report cash-flow items only year-to-date. There is no discrete
  quarter to sum, so trailing twelve months are built as last fiscal year
  plus current YTD minus prior-year YTD.
* 52/53-week fiscal years (Apple, many retailers) put period ends on
  different calendar days each year; matching uses windows, not equality.
* Restatements: the same period appears in several filings. For a given
  cut-off date the latest filing made BEFORE it wins — the value the market
  could actually see on that day, not today's corrected one.
* Tag drift: companies move between equivalent tags (SalesRevenueNet to
  RevenueFromContractWithCustomer... in 2018). Tags are merged per period by
  priority so the switch does not break a trailing sum.
* Splits are NOT handled here; they need prices, and are applied where the
  market capitalisation is built (scripts/build_edgar_fundamentals.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# field -> tags, best first. Merged per period, so a company that switched
# tags keeps a continuous series.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ),
    "cost_of_revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": (
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ),
    "dna": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ),
    "ocf": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    # LongTermDebt includes current maturities by definition; the split
    # noncurrent/current pair is the fallback for filers that only tag those.
    "debt_total": ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"),
    "debt_noncurrent": ("LongTermDebtNoncurrent",),
    "debt_current": ("LongTermDebtCurrent", "DebtCurrent"),
    "short_borrowings": ("ShortTermBorrowings", "CommercialPaper"),
    "cash": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ),
    # Diluted weighted average covers every share class, which the cover-page
    # dei count does not for multi-class filers (companyfacts drops facts
    # that carry a class dimension).
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "shares_outstanding": (
        "dei:EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ),
}

DURATION_FIELDS = {
    "revenue", "cost_of_revenue", "gross_profit", "operating_income",
    "net_income", "dna", "ocf", "capex", "shares_diluted",
}
INSTANT_FIELDS = {
    "equity", "debt_total", "debt_noncurrent", "debt_current",
    "short_borrowings", "cash", "shares_outstanding",
}

FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "20-F", "20-F/A", "40-F"}
SHARE_UNITS = {"shares"}
MONEY_UNITS = {"USD"}

# Period lengths in days (end - start + 1), with room for 52/53-week years.
QUARTER = (80, 100)
ANNUAL = (350, 380)
YTD = (80, 300)
YEAR_BACK = (355, 375)        # how far back "the same period last year" sits
QUARTER_BACK = (84, 98)       # how far back the previous quarter end sits

# A company whose newest filed period is older than this is treated as not
# reporting: its last numbers are not presented as current.
STALE_DAYS = 200


@dataclass(frozen=True)
class _Tagged:
    field: str
    priority: int


_TAG_INDEX: dict[str, _Tagged] = {
    tag: _Tagged(field, rank)
    for field, tags in CONCEPTS.items()
    for rank, tag in enumerate(tags)
}


def parse_companyfacts(payload: dict) -> pd.DataFrame:
    """Flatten a companyfacts payload to the facts this module uses.

    One row per reported value: field, priority, start (NaT for balance-sheet
    instants), end, value, filed, form. Facts from other forms (8-K, S-1...)
    and other units are dropped.
    """
    rows = []
    facts = (payload or {}).get("facts") or {}
    for taxonomy, concepts in facts.items():
        for concept, body in (concepts or {}).items():
            key = concept if taxonomy == "us-gaap" else f"{taxonomy}:{concept}"
            tagged = _TAG_INDEX.get(key)
            if tagged is None:
                continue
            units = (body or {}).get("units") or {}
            allowed = SHARE_UNITS if tagged.field.startswith("shares") else MONEY_UNITS
            for unit, items in units.items():
                if unit not in allowed:
                    continue
                for item in items or []:
                    if item.get("form") not in FORMS:
                        continue
                    rows.append((
                        tagged.field, tagged.priority, item.get("start"),
                        item.get("end"), item.get("val"), item.get("filed"),
                        item.get("form"),
                    ))
    if not rows:
        return _empty_facts()
    df = pd.DataFrame(rows, columns=["field", "priority", "start", "end",
                                     "value", "filed", "form"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["end", "filed", "value"])
    df["days"] = (df["end"] - df["start"]).dt.days + 1
    return df.reset_index(drop=True)


def _empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=["field", "priority", "start", "end", "value",
                                 "filed", "form", "days"])


def visible(facts: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Facts the market could see on `cutoff`, one value per field and period.

    Filed strictly before the cut-off: an EDGAR acceptance can land after the
    close, so a filing is usable from the following day. Within a field and
    period, the highest-priority tag wins, then the most recent filing —
    which is how a restatement replaces the original only from its own
    filing date onward.
    """
    v = facts[facts["filed"] < cutoff]
    if v.empty:
        return v
    v = v.sort_values(["field", "start", "end", "priority", "filed"],
                      ascending=[True, True, True, True, False],
                      na_position="first")
    return v.drop_duplicates(subset=["field", "start", "end"], keep="first")


def _durations(v: pd.DataFrame, field: str) -> pd.DataFrame:
    d = v[(v["field"] == field) & v["start"].notna()]
    return d[["start", "end", "value", "days", "filed"]]


def _between(series: pd.Series, lo, hi) -> pd.Series:
    return (series >= lo) & (series <= hi)


def _annual_ending(d: pd.DataFrame, end: pd.Timestamp):
    hit = d[(d["end"] == end) & _between(d["days"], *ANNUAL)]
    return float(hit["value"].iloc[0]) if not hit.empty else None


def ttm(d: pd.DataFrame, end: pd.Timestamp) -> float:
    """Trailing twelve months ending at `end`, from whatever was filed.

    1. A full fiscal year ending there.
    2. Last fiscal year + current year-to-date - same YTD a year earlier.
       This is the only route for cash-flow items, which 10-Qs report
       year-to-date only.
    3. Four discrete quarters.
    """
    if d.empty:
        return np.nan
    annual = _annual_ending(d, end)
    if annual is not None:
        return annual

    current = d[(d["end"] == end) & _between(d["days"], *YTD)]
    for _, cur in current.sort_values("days", ascending=False).iterrows():
        start = cur["start"]
        prior_fy = d[_between(start - d["end"], pd.Timedelta(days=1), pd.Timedelta(days=6))
                     & _between(d["days"], *ANNUAL)]
        prior_ytd = d[
            _between(end - d["end"], pd.Timedelta(days=YEAR_BACK[0]), pd.Timedelta(days=YEAR_BACK[1]))
            & _between(start - d["start"], pd.Timedelta(days=YEAR_BACK[0]), pd.Timedelta(days=YEAR_BACK[1]))
        ]
        if not prior_fy.empty and not prior_ytd.empty:
            return float(prior_fy["value"].iloc[0] + cur["value"]
                         - prior_ytd["value"].iloc[0])

    total, q_end = 0.0, end
    for _ in range(4):
        q = quarter(d, q_end)
        if not np.isfinite(q):
            return np.nan
        total += q
        prev = d[_between(q_end - d["end"], pd.Timedelta(days=QUARTER_BACK[0]),
                          pd.Timedelta(days=QUARTER_BACK[1]))]["end"]
        if prev.empty:
            return np.nan
        q_end = prev.max()
    return total


def quarter(d: pd.DataFrame, end: pd.Timestamp) -> float:
    """The discrete fiscal quarter ending at `end`."""
    if d.empty:
        return np.nan
    hit = d[(d["end"] == end) & _between(d["days"], *QUARTER)]
    if not hit.empty:
        return float(hit["value"].iloc[0])

    # Q4 is never reported on its own: fiscal year minus nine months. More
    # generally, a YTD minus the YTD one quarter earlier from the same start.
    longer = d[(d["end"] == end) & (d["days"] > QUARTER[1])]
    for _, cur in longer.iterrows():
        earlier = d[(d["start"] == cur["start"])
                    & _between(end - d["end"], pd.Timedelta(days=QUARTER_BACK[0]),
                               pd.Timedelta(days=QUARTER_BACK[1]))]
        if not earlier.empty:
            return float(cur["value"] - earlier["value"].iloc[0])
    return np.nan


def _same_quarter_last_year(d: pd.DataFrame, end: pd.Timestamp) -> float:
    ends = d[_between(end - d["end"], pd.Timedelta(days=YEAR_BACK[0]),
                      pd.Timedelta(days=YEAR_BACK[1]))]["end"]
    if ends.empty:
        return np.nan
    return quarter(d, ends.max())


def _instant(v: pd.DataFrame, field: str, end: pd.Timestamp) -> float:
    """Balance-sheet value at `end`, else the latest within a quarter before."""
    d = v[(v["field"] == field) & v["start"].isna()]
    if d.empty:
        return np.nan
    exact = d[d["end"] == end]
    if not exact.empty:
        return float(exact["value"].iloc[0])
    near = d[(d["end"] <= end) & (end - d["end"] <= pd.Timedelta(days=100))]
    return float(near.sort_values("end")["value"].iloc[-1]) if not near.empty else np.nan


def _shares(v: pd.DataFrame, end: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    """Share count for the period, and the date it was filed.

    The filing date travels with the count because splits are applied from
    it: a filing restates shares for every split made before it was filed,
    so only splits AFTER that date still need applying.
    """
    d = _durations(v, "shares_diluted")
    hit = d[d["end"] == end].sort_values("days")
    if not hit.empty and hit["value"].iloc[0] > 0:
        return float(hit["value"].iloc[0]), hit["filed"].iloc[0]
    inst = v[(v["field"] == "shares_outstanding") & (v["value"] > 0)]
    if not inst.empty:
        latest = inst.sort_values(["end", "filed"]).iloc[-1]
        return float(latest["value"]), latest["filed"]
    return np.nan, None


def state_at(facts: pd.DataFrame, cutoff: pd.Timestamp) -> dict | None:
    """Everything needed for the ratios, as visible on `cutoff`.

    Prices are deliberately absent: the statement side changes only when a
    filing arrives, so it is computed once per filing and joined to weekly
    prices afterwards.
    """
    v = visible(facts, cutoff)
    if v.empty:
        return None
    flows = v[v["start"].notna() & v["field"].isin(DURATION_FIELDS)]
    if flows.empty:
        return None
    end = flows["end"].max()
    if (cutoff - end).days > STALE_DAYS:
        return None

    d = {f: _durations(v, f) for f in DURATION_FIELDS}
    revenue = ttm(d["revenue"], end)
    gross = ttm(d["gross_profit"], end)
    if not np.isfinite(gross):
        cost = ttm(d["cost_of_revenue"], end)
        gross = revenue - cost if np.isfinite(revenue) and np.isfinite(cost) else np.nan
    capex = ttm(d["capex"], end)

    debt = _instant(v, "debt_total", end)
    if not np.isfinite(debt):
        parts = [_instant(v, "debt_noncurrent", end), _instant(v, "debt_current", end)]
        debt = sum(p for p in parts if np.isfinite(p)) if any(np.isfinite(p) for p in parts) else np.nan
    short = _instant(v, "short_borrowings", end)
    if np.isfinite(short):
        debt = (debt if np.isfinite(debt) else 0.0) + short

    shares, shares_filed = _shares(v, end)
    return {
        "period_end": end,
        "filed": v["filed"].max(),
        "revenue_ttm": revenue,
        "gross_profit_ttm": gross,
        "operating_income_ttm": ttm(d["operating_income"], end),
        "net_income_ttm": ttm(d["net_income"], end),
        "dna_ttm": ttm(d["dna"], end),
        "ocf_ttm": ttm(d["ocf"], end),
        # A filer that reports no capex tag is treated as zero capex rather
        # than as unknown cash flow; banks are the usual case.
        "capex_ttm": capex if np.isfinite(capex) else 0.0,
        "revenue_q": quarter(d["revenue"], end),
        "revenue_q_prior": _same_quarter_last_year(d["revenue"], end),
        "net_income_q": quarter(d["net_income"], end),
        "net_income_q_prior": _same_quarter_last_year(d["net_income"], end),
        "equity": _instant(v, "equity", end),
        "debt": debt,
        "cash": _instant(v, "cash", end),
        "shares": shares,
        "shares_filed": shares_filed,
    }


def filing_states(facts: pd.DataFrame) -> pd.DataFrame:
    """One statement state per filing date, ready to join to prices.

    `available_from` is the day after the filing — the first session a
    replay may use it.
    """
    if facts.empty:
        return pd.DataFrame()
    rows = []
    for filed in sorted(facts["filed"].dropna().unique()):
        cutoff = pd.Timestamp(filed) + pd.Timedelta(days=1)
        state = state_at(facts, cutoff)
        if state is not None:
            state["available_from"] = cutoff
            rows.append(state)
    return pd.DataFrame(rows)


def _safe_div(a, b, positive_denominator: bool = True):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    ok = (b > 0) if positive_denominator else (b != 0)
    return (a / b.where(ok)).astype(float)


def ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Live-schema ratios from statement state plus `market_cap`.

    Conventions follow what a vendor would publish: a P/E needs positive
    earnings, P/B needs positive equity, EV/EBITDA needs positive EBITDA.
    Undefined is NaN, never a negative multiple that would rank as cheap.
    """
    f = frame
    mcap = pd.to_numeric(f["market_cap"], errors="coerce")
    ebitda = f["operating_income_ttm"] + f["dna_ttm"]
    ev = mcap + f["debt"].fillna(0.0) - f["cash"].fillna(0.0)
    fcf = f["ocf_ttm"] - f["capex_ttm"].fillna(0.0)
    prior_ni = f["net_income_q_prior"]
    out = pd.DataFrame({
        "pe_ratio": _safe_div(mcap, f["net_income_ttm"]),
        "pb_ratio": _safe_div(mcap, f["equity"]),
        "ps_ratio": _safe_div(mcap, f["revenue_ttm"]),
        "ev_ebitda": _safe_div(ev, ebitda),
        "fcf_yield": _safe_div(fcf, mcap),
        "roe": _safe_div(f["net_income_ttm"], f["equity"]),
        "gross_margin": _safe_div(f["gross_profit_ttm"], f["revenue_ttm"]),
        "net_margin": _safe_div(f["net_income_ttm"], f["revenue_ttm"]),
        "net_debt_ebitda": _safe_div(f["debt"].fillna(0.0) - f["cash"].fillna(0.0), ebitda),
        "revenue_growth": _safe_div(f["revenue_q"], f["revenue_q_prior"]) - 1.0,
        "earnings_growth": _safe_div(f["net_income_q"] - prior_ni, prior_ni.abs()),
        "market_cap": mcap,
    }, index=f.index)
    return out.replace([np.inf, -np.inf], np.nan)


def split_factor(splits: pd.Series, since: pd.Timestamp | None) -> float:
    """Product of split ratios dated after `since`.

    Prices from the cache are adjusted to today's share basis. A share count
    filed on `since` already reflects every split before that date, so only
    later splits stand between it and today's basis. NVDA's 4:1 (2021) and
    10:1 (2024) are why this matters: without it a 2022 market cap built from
    filed shares and adjusted prices is off by a factor of ten.
    """
    if splits is None or splits.empty or since is None or pd.isna(since):
        return 1.0
    later = splits[(splits.index > pd.Timestamp(since)) & (splits > 0)]
    return float(later.prod()) if not later.empty else 1.0
