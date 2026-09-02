"""Self-learning: walk-forward factor-weight tuning with rollback.

How it learns
-------------
Once a month the job reconstructs the screener's own scoring table *as it
would have looked* on a grid of past dates, using only data available then:

  * price factors   <- the cached OHLCV history, truncated at that date
  * fundamentals    <- the newest weekly snapshot dated on/before that date

For each past date it then measures the **information coefficient** (Spearman
rank correlation) between each factor block's score and the realised forward
21-day return. Blocks that ranked stocks well get more weight, blocks that
didn't get less.

Guardrails, because factor timing is notoriously easy to overfit:
  * weights move at most ``max_step_per_month`` (0.05) per run
  * every weight stays inside [weight_min, weight_max] and they sum to 1
  * nothing changes until ``min_trades_for_update`` real trades have closed
  * the proposal is scored on a held-out window and **rolled back** unless it
    beats the incumbent ("champion") weights
  * fundamental blocks get a shrink factor while point-in-time coverage is
    still thin, since the snapshot history only starts accumulating now

Every decision, accepted or rejected, is appended to weights_history.csv.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import factors as fmod
from . import scoring as smod
from .data import fundamentals as fund_mod

FWD_HORIZON = 21          # trading days, matches the holding period
GRID_STEP = 5             # evaluate every 5th trading day
# Must stay in sync with config.yml's `metric_blocks` keys — this list drives
# both the walk-forward IC evaluation below AND what `propose_weights` writes
# back into config.yml's `factor_weights` (see Config.save_weights, which
# REPLACES the whole block). A block present in metric_blocks but missing
# here would silently vanish from config.yml the first time learning runs.
BLOCKS = ("value", "quality", "growth", "momentum", "earnings", "risk", "technical")


# ---------------------------------------------------------------------------
# statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation; NaN if fewer than 8 overlapping observations."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 8:
        return np.nan
    ra = df.iloc[:, 0].rank()
    rb = df.iloc[:, 1].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return np.nan
    return float(ra.corr(rb))


# ---------------------------------------------------------------------------
# walk-forward panel
# ---------------------------------------------------------------------------
def build_panel(
    prices_long: pd.DataFrame,
    universe: pd.DataFrame,
    fundamentals_path: Path,
    cfg,
    grid_step: int = GRID_STEP,
    max_dates: int = 200,
) -> tuple[pd.DataFrame, list[dict]]:
    """Reconstruct historical block scores + forward returns.

    Returns (ic_frame, snapshots) where ic_frame is one row per evaluation
    date with the IC of each block, and snapshots holds the per-date scoring
    tables needed to compare candidate weight sets.
    """
    from .data import prices as pmod

    close = pmod.to_wide(prices_long, "close")
    volume = pmod.to_wide(prices_long, "volume")
    if close.empty or len(close) < 260 + FWD_HORIZON:
        return pd.DataFrame(), []

    dates = list(close.index)
    # need >=252 bars of history behind and FWD_HORIZON ahead
    usable = dates[252 : len(dates) - FWD_HORIZON]
    grid = usable[::grid_step][-max_dates:]

    ic_rows, snapshots = [], []
    for d in grid:
        fund = fund_mod.latest_asof(fundamentals_path, on_or_before=d.date())
        metrics = fmod.build_metrics(prices_long, fund, universe, asof=d)
        if metrics.empty:
            continue
        scored = smod.score(metrics, cfg)
        pool = scored[scored["eligible"] & scored["composite"].notna()]
        if len(pool) < 20:
            continue

        fwd = fmod.forward_return(close, d, FWD_HORIZON)
        if fwd.empty:
            continue

        row = {"date": d.date().isoformat(), "n": int(len(pool)),
               "fund_asof": (fund["asof"].iloc[0] if not fund.empty and "asof" in fund else None)}
        block_cols = {}
        for b in BLOCKS:
            col = f"block_{b}"
            if col in pool.columns:
                row[f"ic_{b}"] = spearman(pool[col], fwd.reindex(pool.index))
                row[f"cov_{b}"] = float(pool[col].notna().mean())
                block_cols[b] = pool[col]
            else:
                row[f"ic_{b}"] = np.nan
                row[f"cov_{b}"] = 0.0
        ic_rows.append(row)

        snap = pd.DataFrame(block_cols)
        snap["fwd"] = fwd.reindex(snap.index)
        snapshots.append({"date": d, "table": snap})

    return pd.DataFrame(ic_rows), snapshots


def evaluate_weights(snapshots: list[dict], weights: dict, top_n: int = 1) -> dict:
    """Mean forward return of the top-`top_n` picks under a weight set."""
    picks = []
    for snap in snapshots:
        tbl = snap["table"]
        cols = [b for b in BLOCKS if b in tbl.columns]
        if not cols:
            continue
        w = np.array([float(weights.get(b, 0.0)) for b in cols])
        present = tbl[cols].notna().to_numpy()
        wm = np.where(present, w, 0.0)
        denom = wm.sum(axis=1)
        vals = np.nan_to_num(tbl[cols].to_numpy(), nan=0.0)
        comp = np.where(denom > 0, (vals * wm).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)
        comp = pd.Series(comp, index=tbl.index)
        chosen = comp.dropna().sort_values(ascending=False).head(top_n).index
        r = tbl.loc[chosen, "fwd"].dropna()
        if len(r):
            picks.append(float(r.mean()))

    if not picks:
        return {"n": 0, "mean": np.nan, "hit_rate": np.nan, "sharpe": np.nan}
    arr = np.array(picks, dtype=float)
    sd = arr.std(ddof=0)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "hit_rate": float((arr > 0).mean()),
        # per-pick Sharpe-like ratio (not annualised; used only for comparison)
        "sharpe": float(arr.mean() / sd) if sd > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# weight proposal
# ---------------------------------------------------------------------------
def _renorm_with_caps(vals: dict, lo: dict, hi: dict, target_sum: float = 1.0,
                       max_iter: int = 20) -> dict:
    """Scale `vals` to sum to `target_sum`, water-filling within [lo, hi].

    A uniform rescale (`v / sum(vals) * target_sum`) ignores the per-block
    band and can push a block's *effective* step well past what it looks
    like it moved before the rescale. This instead only ever pushes the
    remaining budget onto blocks that still have room, iterating until
    either the budget is placed or every block is pinned to a bound.
    """
    vals = dict(vals)
    for _ in range(max_iter):
        diff = target_sum - sum(vals.values())
        if abs(diff) < 1e-9:
            break
        free = [k for k, v in vals.items()
                if not ((diff > 0 and v >= hi[k] - 1e-9) or
                        (diff < 0 and v <= lo[k] + 1e-9))]
        if not free:
            break
        share = diff / len(free)
        for k in free:
            vals[k] = min(hi[k], max(lo[k], vals[k] + share))
    return vals


def propose_weights(current: dict, ic_means: dict, cfg, confidence: dict | None = None) -> dict:
    """Blend current weights toward IC-implied weights, respecting bounds.

    Two constraints must hold on the weights this RETURNS, not just on some
    intermediate step: each stays within [weight_min, weight_max], and none
    moves more than `max_step_per_month` from `current`. Those per-block
    step caps are enforced as a band, and only the leftover budget needed to
    reach sum=1.0 gets water-filled across blocks that still have room —
    see `_renorm_with_caps`.
    """
    L = cfg.learning
    wmin, wmax = float(L["weight_min"]), float(L["weight_max"])
    step = float(L["max_step_per_month"])
    confidence = confidence or {}

    # Target: proportional to positive IC. A block with no/negative signal
    # falls back to the floor rather than going to zero.
    pos = {b: max(0.0, float(ic_means.get(b, 0.0) or 0.0)) for b in BLOCKS}
    total = sum(pos.values())
    if total <= 0:
        target = {b: 1.0 / len(BLOCKS) for b in BLOCKS}   # no signal: equal weight
    else:
        target = {b: pos[b] / total for b in BLOCKS}

    cur = {b: float(current.get(b, 0.0)) for b in BLOCKS}
    lo = {b: max(wmin, cur[b] - step) for b in BLOCKS}
    hi = {b: min(wmax, cur[b] + step) for b in BLOCKS}

    proposed = {}
    for b in BLOCKS:
        conf = float(confidence.get(b, 1.0))
        delta = (target[b] - cur[b]) * conf
        delta = max(-step, min(step, delta))
        proposed[b] = min(hi[b], max(lo[b], cur[b] + delta))

    proposed = _renorm_with_caps(proposed, lo, hi)
    s = sum(proposed.values())
    if s <= 0:
        return proposed
    return {b: round(v / s, 4) for b, v in proposed.items()}


def _coverage_confidence(ic_frame: pd.DataFrame) -> dict:
    """Shrink adjustments for blocks whose historical coverage is thin.

    Fundamental snapshots only start accumulating when the bot goes live, so
    early on the value/quality/growth/earnings ICs are measured on stale or
    partly missing data. Momentum, risk and technical are computed purely
    from prices and are always fully covered, so they get full confidence.
    """
    conf = {}
    for b in BLOCKS:
        col = f"cov_{b}"
        cov = float(ic_frame[col].mean()) if col in ic_frame else 0.0
        conf[b] = float(np.clip(cov, 0.0, 1.0))
    return conf


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_learning(cfg, prices_long: pd.DataFrame, universe: pd.DataFrame,
                 dry_run: bool = False) -> dict:
    """Full monthly learning cycle. Returns a decision record."""
    L = cfg.learning
    rec: dict = {
        "run_date": date.today().isoformat(),
        "mode": L.get("mode", "auto_bounded"),
        "decision": "skipped",
        "reason": "",
        "weights_before": json.dumps(cfg.weights),
        "weights_after": json.dumps(cfg.weights),
    }
    if not L.get("enabled", False) or L.get("mode") == "off":
        rec["reason"] = "learning disabled in config"
        return rec

    trades_path = cfg.path("trades_csv")
    n_trades = 0
    if trades_path.exists():
        try:
            n_trades = int(len(pd.read_csv(trades_path)))
        except Exception:  # noqa: BLE001
            n_trades = 0
    if n_trades < int(L["min_trades_for_update"]):
        rec["reason"] = (f"only {n_trades} closed trades, need "
                         f"{L['min_trades_for_update']} before tuning")
        rec["n_trades"] = n_trades
        return rec

    ic_frame, snapshots = build_panel(
        prices_long, universe, cfg.path("fundamentals_csv"), cfg
    )
    if ic_frame.empty or len(snapshots) < 12:
        rec["reason"] = "not enough reconstructable history yet"
        return rec

    # train / validation split by date
    n_val = max(3, int(len(snapshots) * 0.25))
    train_ic = ic_frame.iloc[:-n_val] if len(ic_frame) > n_val else ic_frame
    train_snaps, val_snaps = snapshots[:-n_val], snapshots[-n_val:]

    ic_means = {b: float(train_ic[f"ic_{b}"].mean()) if f"ic_{b}" in train_ic else np.nan
                for b in BLOCKS}
    confidence = _coverage_confidence(train_ic)

    champion = cfg.weights
    challenger = propose_weights(champion, ic_means, cfg, confidence)

    champ_val = evaluate_weights(val_snaps, champion)
    chall_val = evaluate_weights(val_snaps, challenger)

    ic_cols = {
        f"ic_{b}": (round(ic_means[b], 5) if ic_means.get(b) == ic_means.get(b) else "")
        for b in BLOCKS
    }
    rec.update({
        "n_trades": n_trades,
        "eval_dates": len(snapshots),
        **ic_cols,
        "champion_val_mean": champ_val["mean"],
        "challenger_val_mean": chall_val["mean"],
        "champion_val_sharpe": champ_val["sharpe"],
        "challenger_val_sharpe": chall_val["sharpe"],
        "weights_after": json.dumps(challenger),
    })

    better = (
        chall_val["n"] > 0
        and champ_val["n"] > 0
        and np.nan_to_num(chall_val["sharpe"], nan=-9) > np.nan_to_num(champ_val["sharpe"], nan=-9)
        and np.nan_to_num(chall_val["mean"], nan=-9) >= np.nan_to_num(champ_val["mean"], nan=-9)
    )

    if L.get("mode") == "propose_pr":
        rec["decision"] = "proposed"
        rec["reason"] = "propose_pr mode: weights left unchanged, see PR"
    elif not better and L.get("rollback_if_worse", True):
        rec["decision"] = "rolled_back"
        rec["reason"] = "challenger did not beat champion out of sample"
        rec["weights_after"] = json.dumps(champion)
    elif dry_run:
        rec["decision"] = "dry_run"
        rec["reason"] = "would have accepted challenger"
    else:
        rec["decision"] = "accepted"
        rec["reason"] = "challenger beat champion out of sample"
        cfg.save_weights(challenger)

    _append_log(rec, cfg.path("weights_history"))
    return rec


def _append_log(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "logged_at": datetime.utcnow().isoformat(timespec="seconds")}
    df = pd.DataFrame([rec])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
