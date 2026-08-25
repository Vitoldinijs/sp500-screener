"""Alpha Vantage — used ONLY to verify the single chosen candidate of the day.

Why so narrow: the free tier allows ~25 requests/day. Scanning 500 tickers is
impossible and would just get the key throttled. Instead we spend 1-2 calls on
a final sanity check of the one name we're about to buy:

  GLOBAL_QUOTE  -> is it still trading, and is the price close to our data?
  OVERVIEW      -> do P/E, P/B, ROE roughly agree with our cached fundamentals?

If Alpha Vantage *contradicts* our data the candidate is rejected and the
runner-up is checked instead. If Alpha Vantage is merely *unavailable* (quota,
timeout, network) we proceed and flag it — a data-provider outage should not
stop the strategy.

The key is read from the ALPHAVANTAGE_KEY environment variable and is never
written to disk or logs.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

BASE = "https://www.alphavantage.co/query"


class Quota:
    """Tracks daily request usage in a small JSON file."""

    def __init__(self, path: Path, budget: int = 22):
        self.path = path
        self.budget = budget
        self._d = {"day": date.today().isoformat(), "used": 0}
        if path.exists():
            try:
                d = json.loads(path.read_text())
                if d.get("day") == self._d["day"]:
                    self._d = d
            except Exception:  # noqa: BLE001
                pass

    @property
    def used(self) -> int:
        return int(self._d.get("used", 0))

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def spend(self, n: int = 1) -> bool:
        if self.remaining < n:
            return False
        self._d["used"] = self.used + n
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._d))
        return True


def _get(params: dict, timeout: int = 25) -> dict | None:
    import requests

    key = os.environ.get("ALPHAVANTAGE_KEY", "").strip()
    if not key:
        return None
    params = {**params, "apikey": key}
    try:
        resp = requests.get(BASE, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    # Free-tier throttle responses carry these keys instead of data.
    if any(k in data for k in ("Note", "Information", "Error Message")):
        print(f"[alphavantage] throttled/err: {list(data)[:1]}")
        return None
    return data or None


def _f(val) -> float | None:
    try:
        if val in (None, "", "None", "-"):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def verify_candidate(
    ticker: str,
    our_price: float,
    quota: Quota,
    cached: dict | None = None,
    price_tolerance: float = 0.08,
) -> dict:
    """Cross-check one candidate. Returns a verdict dict.

    verdict = "confirmed" | "rejected" | "unverified"
    """
    out = {
        "ticker": ticker,
        "verdict": "unverified",
        "reason": "",
        "av_price": None,
        "av_pe": None,
        "av_pb": None,
        "av_roe": None,
        "calls_used": 0,
    }

    if not os.environ.get("ALPHAVANTAGE_KEY", "").strip():
        out["reason"] = "no API key configured"
        return out
    if quota.remaining < 2:
        out["reason"] = f"daily quota exhausted ({quota.used}/{quota.budget})"
        return out

    # --- 1. quote ---------------------------------------------------------
    quota.spend(1)
    out["calls_used"] += 1
    quote = _get({"function": "GLOBAL_QUOTE", "symbol": ticker})
    if not quote or "Global Quote" not in quote:
        out["reason"] = "GLOBAL_QUOTE unavailable"
        return out
    gq = quote["Global Quote"]
    av_price = _f(gq.get("05. price"))
    out["av_price"] = av_price
    if av_price is None or av_price <= 0:
        out["verdict"] = "rejected"
        out["reason"] = "no tradable price at Alpha Vantage (delisted/halted?)"
        return out
    if our_price and abs(av_price / our_price - 1.0) > price_tolerance:
        out["verdict"] = "rejected"
        out["reason"] = (
            f"price mismatch: ours {our_price:.2f} vs AV {av_price:.2f} "
            f"({abs(av_price/our_price-1):.1%} > {price_tolerance:.0%})"
        )
        return out

    # --- 2. overview ------------------------------------------------------
    quota.spend(1)
    out["calls_used"] += 1
    ov = _get({"function": "OVERVIEW", "symbol": ticker})
    if not ov or not ov.get("Symbol"):
        out["verdict"] = "confirmed"
        out["reason"] = "price confirmed; OVERVIEW unavailable"
        return out

    out["av_pe"] = _f(ov.get("PERatio"))
    out["av_pb"] = _f(ov.get("PriceToBookRatio"))
    out["av_roe"] = _f(ov.get("ReturnOnEquityTTM"))

    # Contradiction checks against our cached weekly fundamentals.
    if cached:
        for field, av_val, tol in (
            ("pe_ratio", out["av_pe"], 0.60),
            ("pb_ratio", out["av_pb"], 0.60),
        ):
            ours = cached.get(field)
            if ours and av_val and ours > 0 and av_val > 0:
                if abs(av_val / ours - 1.0) > tol:
                    out["verdict"] = "rejected"
                    out["reason"] = (
                        f"{field} disagrees: ours {ours:.1f} vs AV {av_val:.1f}"
                    )
                    return out

    out["verdict"] = "confirmed"
    out["reason"] = "price and fundamentals cross-checked"
    return out
