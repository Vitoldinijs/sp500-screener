"""Synthetic companyfacts payloads that behave like a real filer.

A real 10-Q does not hand over a tidy list of quarters. It reports the
current quarter AND the year-to-date for income items, ONLY the
year-to-date for cash-flow items, and repeats last year's figures as
comparatives. A 10-K reports the full year and never the fourth quarter on
its own. These fixtures reproduce that, so the reconstruction is tested
against the shape SEC actually serves rather than a convenient one.
"""
from __future__ import annotations

from datetime import date, timedelta


def fiscal_quarters(first_fy_end: date, years: int, week_based: bool = False):
    """[(fy, q, start, end)] for consecutive fiscal years.

    `week_based` mimics a 52/53-week year: quarters of 13 weeks ending on the
    same weekday, with a 14-week fourth quarter every few years, so period
    ends drift across calendar days the way Apple's and most retailers' do.
    """
    out = []
    prev_end = None
    for y in range(years):
        if week_based:
            weeks = [13, 13, 13, 14 if y % 5 == 4 else 13]
            # Each year starts the day after the last one ended; only the
            # first year is anchored to the given end date.
            start = (first_fy_end - timedelta(days=7 * sum(weeks) - 1)
                     if prev_end is None else prev_end + timedelta(days=1))
            ends, cursor = [], start - timedelta(days=1)
            for w in weeks:
                cursor = cursor + timedelta(days=7 * w)
                ends.append(cursor)
        else:
            start = (date(first_fy_end.year - 1, first_fy_end.month,
                          first_fy_end.day) + timedelta(days=1)
                     if prev_end is None else prev_end + timedelta(days=1))
            ends = []
            for k in (3, 6, 9, 12):
                m = (start.month - 1 + k) % 12 + 1
                yy = start.year + (start.month - 1 + k) // 12
                ends.append(date(yy, m, 1) - timedelta(days=1))
        q_start = start
        for q, q_end in enumerate(ends, start=1):
            out.append((y, q, start, q_start, q_end))
            q_start = q_end + timedelta(days=1)
        prev_end = ends[-1]
    return out


def _fact(start, end, val, filed, form):
    item = {"end": end.isoformat(), "val": val, "filed": filed.isoformat(), "form": form}
    if start is not None:
        item["start"] = start.isoformat()
    return item


def build_payload(quarters, values: dict, *, tags: dict | None = None,
                  instants: dict | None = None, filing_lag: int = 35,
                  annual_lag: int = 60, tag_switch_year: int | None = None,
                  restate: dict | None = None) -> dict:
    """Assemble a companyfacts payload.

    values:   field -> {(fy, q): discrete quarter value}
    tags:     field -> us-gaap tag (defaults below)
    instants: field -> {(fy, q): balance at quarter end}
    tag_switch_year: from this fiscal year on, revenue is tagged with the
              post-2018 contract-revenue tag instead of SalesRevenueNet.
    restate:  {(field, fy, q): (new_value, filed_date)} — a later filing that
              revises one quarter, as an amended or comparative filing would.
    """
    tags = {"revenue": "SalesRevenueNet", "net_income": "NetIncomeLoss",
            "ocf": "NetCashProvidedByUsedInOperatingActivities",
            "shares_diluted": "WeightedAverageNumberOfDilutedSharesOutstanding",
            **(tags or {})}
    cash_flow = {"ocf"}
    gaap: dict[str, dict] = {}

    def add(tag, unit, item):
        gaap.setdefault(tag, {"units": {}})["units"].setdefault(unit, []).append(item)

    by_year: dict[int, list] = {}
    for fy, q, fy_start, qs, qe in quarters:
        by_year.setdefault(fy, []).append((q, fy_start, qs, qe))

    for fy, qs_list in by_year.items():
        for q, fy_start, qs, qe in qs_list:
            is_annual = q == 4
            form = "10-K" if is_annual else "10-Q"
            filed = qe + timedelta(days=annual_lag if is_annual else filing_lag)
            # Each filing reports the current period AND last year's
            # comparative for the same period.
            for year in (fy, fy - 1):
                if year not in by_year:
                    continue
                row = dict((qq, (fs, s, e)) for qq, fs, s, e in by_year[year])
                fs, s, e = row[q]
                for field, series in values.items():
                    tag = tags[field]
                    if field == "revenue" and tag_switch_year is not None and fy >= tag_switch_year:
                        tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
                    unit = "shares" if field.startswith("shares") else "USD"
                    ytd = sum(series[(year, k)] for k in range(1, q + 1))
                    if field.startswith("shares"):
                        ytd = series[(year, q)]       # an average, not a sum
                    if is_annual:
                        add(tag, unit, _fact(fs, e, ytd, filed, form))
                        continue
                    if field not in cash_flow:
                        add(tag, unit, _fact(s, e, series[(year, q)], filed, form))
                    if q > 1 or field in cash_flow:
                        add(tag, unit, _fact(fs, e, ytd, filed, form))
            for field, series in (instants or {}).items():
                tag = {"equity": "StockholdersEquity", "cash": "CashAndCashEquivalentsAtCarryingValue",
                       "debt_total": "LongTermDebt"}[field]
                add(tag, "USD", _fact(None, qe, series[(fy, q)], filed, form))

    for (field, fy, q), (new_value, filed_date) in (restate or {}).items():
        _, fy_start, qs, qe = next(x for x in by_year[fy] if x[0] == q)
        add(tags[field], "USD", _fact(qs, qe, new_value, filed_date, "10-Q/A"))
    return {"cik": 1, "entityName": "Fixture Corp", "facts": {"us-gaap": gaap}}
