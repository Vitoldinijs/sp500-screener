"""Diagnostics for completed signal lifecycles; never changes factor weights."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BLOCKS = ("value", "quality", "growth", "momentum", "earnings", "risk", "technical")
LINKED_STATUSES = {"linked", "linked_reconstructed"}


def _is_true(value) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _stats(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "n": 0, "win_rate": None, "mean_return": None,
            "median_return": None, "total_pnl": 0.0,
            "mean_bars_held": None, "profit_factor": None,
        }
    returns = pd.to_numeric(rows.get("return_pct", pd.Series(index=rows.index, dtype=float)),
                            errors="coerce").dropna()
    pnl = pd.to_numeric(rows.get("pnl", pd.Series(index=rows.index, dtype=float)),
                        errors="coerce").dropna()
    bars = pd.to_numeric(rows.get("bars_held", pd.Series(index=rows.index, dtype=float)),
                         errors="coerce").dropna()
    gross_profit = float(pnl.clip(lower=0).sum()) if len(pnl) else 0.0
    gross_loss = float(-pnl.clip(upper=0).sum()) if len(pnl) else 0.0
    return {
        "n": int(len(returns)),
        "win_rate": _finite((returns > 0).mean()) if len(returns) else None,
        "mean_return": _finite(returns.mean()) if len(returns) else None,
        "median_return": _finite(returns.median()) if len(returns) else None,
        "total_pnl": _finite(pnl.sum()) if len(pnl) else 0.0,
        "mean_bars_held": _finite(bars.mean()) if len(bars) else None,
        "profit_factor": (
            _finite(gross_profit / gross_loss) if gross_loss > 0
            else (_finite(gross_profit) if gross_profit > 0 else None)
        ),
    }


def _spearman(x: pd.Series, y: pd.Series, minimum_n: int) -> tuple[int, float | None]:
    pair = pd.concat([
        pd.to_numeric(x, errors="coerce").rename("x"),
        pd.to_numeric(y, errors="coerce").rename("y"),
    ], axis=1).dropna()
    n = len(pair)
    if n < minimum_n or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return n, None
    return n, _finite(pair["x"].rank().corr(pair["y"].rank()))


def evaluate_lifecycle_outcomes(outcomes: pd.DataFrame,
                                settings: dict | None = None) -> dict:
    """Create outcome statistics and a strictly chronological split summary."""
    settings = settings or {}
    minimum_total = max(1, int(settings.get(
        "min_trades_for_evaluation", settings.get("min_trades_for_update", 40)
    )))
    minimum_train = max(1, int(settings.get("min_training_trades", 20)))
    minimum_validation = max(1, int(settings.get("min_validation_trades", 10)))
    minimum_factor_n = max(3, int(settings.get("min_factor_observations", 8)))
    validation_fraction = min(0.5, max(0.15, float(settings.get("validation_fraction", 0.30))))

    data = outcomes.copy() if outcomes is not None else pd.DataFrame()
    closed_count = int(len(data))
    if closed_count:
        link_status = data.get("feature_link_status", pd.Series(index=data.index, dtype=str))
        linked_count = int(link_status.isin(LINKED_STATUSES).sum())
        flagged = data.get("learning_eligible", pd.Series(False, index=data.index)).map(_is_true)
        eligible = data.loc[flagged].copy()
    else:
        linked_count = 0
        flagged = pd.Series(dtype=bool)
        eligible = pd.DataFrame()

    if not eligible.empty:
        eligible["return_pct"] = pd.to_numeric(eligible.get("return_pct"), errors="coerce")
        eligible["pnl"] = pd.to_numeric(eligible.get("pnl"), errors="coerce")
        eligible["signal_date_parsed"] = pd.to_datetime(
            eligible.get("signal_date"), errors="coerce", utc=True
        )
        eligible["exit_date_parsed"] = pd.to_datetime(
            eligible.get("exit_date"), errors="coerce", utc=True
        )
        valid = (
            eligible["return_pct"].notna()
            & eligible["signal_date_parsed"].notna()
            & eligible["exit_date_parsed"].notna()
            & (eligible["exit_date_parsed"] >= eligible["signal_date_parsed"])
        )
        usable = eligible.loc[valid].copy()
    else:
        usable = pd.DataFrame(columns=list(eligible.columns))

    evaluable_count = int(len(usable))
    split_cutoff = None
    training = usable.iloc[0:0].copy()
    validation = usable.iloc[0:0].copy()
    split_available = False

    if evaluable_count >= 3:
        unique_dates = sorted(pd.unique(usable["signal_date_parsed"].dropna()))
        if len(unique_dates) >= 3:
            cut_index = int(np.floor(len(unique_dates) * (1.0 - validation_fraction)))
            cut_index = max(1, min(len(unique_dates) - 1, cut_index))
            cutoff = unique_dates[cut_index]
            split_cutoff = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
            # Training labels must have been known before the validation period began.
            training = usable[
                (usable["signal_date_parsed"] < cutoff)
                & (usable["exit_date_parsed"] < cutoff)
            ].copy()
            validation = usable[usable["signal_date_parsed"] >= cutoff].copy()
            split_available = len(training) >= minimum_train and len(validation) >= minimum_validation

    if evaluable_count < minimum_total:
        status = "collecting_data"
        message = (
            f"Недостаточно связанных закрытых сделок: {evaluable_count} из "
            f"{minimum_total}, необходимых для оценки."
        )
    elif split_available:
        status = "temporal_split_ready_for_review"
        message = (
            "Данных достаточно для раздельной временной проверки. Ассоциации "
            "факторов остаются исследовательскими и не меняют веса автоматически."
        )
    else:
        status = "more_data_needed_for_validation"
        message = (
            "Выборки достаточно для первичного анализа, но временная проверочная "
            "часть пока слишком мала."
        )

    returns = usable.get("return_pct", pd.Series(index=usable.index, dtype=float))
    factor_rows = []
    for block in BLOCKS:
        feature = "signal_block_" + block
        values = pd.to_numeric(
            usable[feature] if feature in usable else pd.Series(np.nan, index=usable.index),
            errors="coerce"
        )
        n_all, corr_all = _spearman(values, returns, minimum_factor_n)
        positive = values > 0
        nonpositive = values.notna() & (values <= 0)
        mean_positive = _finite(returns[positive & returns.notna()].mean()) if (positive & returns.notna()).any() else None
        mean_nonpositive = _finite(returns[nonpositive & returns.notna()].mean()) if (nonpositive & returns.notna()).any() else None
        n_train, corr_train = _spearman(
            training.get(feature, pd.Series(index=training.index, dtype=float)),
            training.get("return_pct", pd.Series(index=training.index, dtype=float)),
            minimum_factor_n
        )
        n_validation, corr_validation = _spearman(
            validation.get(feature, pd.Series(index=validation.index, dtype=float)),
            validation.get("return_pct", pd.Series(index=validation.index, dtype=float)),
            minimum_factor_n
        )
        factor_rows.append({
            "block": block, "n": n_all, "spearman_return": corr_all,
            "training_n": n_train, "training_spearman_return": corr_train,
            "validation_n": n_validation, "validation_spearman_return": corr_validation,
            "mean_return_factor_positive": mean_positive,
            "mean_return_factor_nonpositive": mean_nonpositive,
            "positive_minus_nonpositive_mean_return": (
                _finite(mean_positive - mean_nonpositive)
                if mean_positive is not None and mean_nonpositive is not None else None
            ),
            "interpretation": "exploratory" if corr_all is not None else "insufficient_observations",
        })

    exit_reasons = []
    if not usable.empty and "exit_reason" in usable.columns:
        work = usable.copy()
        work["exit_reason"] = work["exit_reason"].fillna("unknown").astype(str)
        for reason, group in work.groupby("exit_reason", sort=False):
            exit_reasons.append({
                "reason": reason, "count": int(len(group)),
                "mean_return": _finite(group["return_pct"].mean())
            })
        exit_reasons.sort(key=lambda item: (-item["count"], item["reason"]))

    flagged_count = int(flagged.sum()) if len(flagged) else 0
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "status_message": message,
        "learning_enabled": False,
        "thresholds": {
            "min_completed_trades": minimum_total,
            "min_training_trades": minimum_train,
            "min_validation_trades": minimum_validation,
            "validation_fraction": validation_fraction,
            "min_factor_observations": minimum_factor_n,
        },
        "counts": {
            "closed_trades": closed_count,
            "linked_trades": linked_count,
            "learning_eligible_flagged": flagged_count,
            "evaluable_trades": evaluable_count,
            "excluded_due_to_missing_outcome_or_dates": max(0, flagged_count - evaluable_count),
        },
        "overall": _stats(usable),
        "temporal_validation": {
            "status": "available_for_review" if split_available else "insufficient_sample",
            "cutoff_signal_date": split_cutoff,
            "training": _stats(training),
            "validation": _stats(validation),
            "training_labels_known_before_cutoff": True,
        },
        "exit_reasons": exit_reasons[:10],
        "factor_associations": factor_rows,
        "caveat": (
            "Descriptive associations only; not causal estimates and not a basis "
            "for changing factor weights before separate validation."
        ),
    }


def refresh_lifecycle_evaluation(outcomes_path: Path, report_path: Path,
                                settings: dict | None = None) -> dict:
    """Read the reconciled ledger and write the current diagnostic report."""
    if outcomes_path.exists():
        try:
            outcomes = pd.read_csv(outcomes_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            outcomes = pd.DataFrame()
    else:
        outcomes = pd.DataFrame()
    report = evaluate_lifecycle_outcomes(outcomes, settings)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, allow_nan=False)
    return report
