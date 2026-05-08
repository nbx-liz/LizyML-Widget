"""Cross-validation helpers for WidgetService.

Pure functions extracted from service.py to keep the service module under
the 800-line ceiling (#137). These helpers operate on plain dicts /
DataFrames; they have no dependency on WidgetService's internal state.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

GROUP_STRATEGIES: frozenset[str] = frozenset(
    {"group_kfold", "stratified_group_kfold", "group_time_series", "blocked_group_kfold"}
)
TIME_STRATEGIES: frozenset[str] = frozenset(
    {"time_series", "purged_time_series", "group_time_series"}
)


def validate_inner_valid(config: dict[str, Any], df_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Check inner_valid method is compatible with the active CV strategy.

    Returns a list of validation-error dicts (empty when compatible).
    """
    training = config.get("training") or {}
    early_stopping = training.get("early_stopping") or {}
    inner_valid = early_stopping.get("inner_valid") or {}
    method = inner_valid.get("method", "holdout") if isinstance(inner_valid, dict) else inner_valid

    cv = df_info.get("cv") or {}
    strategy = cv.get("strategy", "kfold")
    errors: list[dict[str, Any]] = []

    if method == "group_holdout" and strategy not in GROUP_STRATEGIES:
        errors.append(
            {
                "field": "training.early_stopping.inner_valid.method",
                "message": (
                    "group_holdout requires a group-based CV strategy "
                    "(e.g. group_kfold, stratified_group_kfold, group_time_series)."
                ),
                "type": "inner_valid_constraint",
            }
        )
    elif method == "time_holdout" and strategy not in TIME_STRATEGIES:
        errors.append(
            {
                "field": "training.early_stopping.inner_valid.method",
                "message": (
                    "time_holdout requires a time-based CV strategy "
                    "(e.g. time_series, purged_time_series, group_time_series)."
                ),
                "type": "inner_valid_constraint",
            }
        )

    return errors


def default_strategy_for_task(adapter: Any, task: str) -> str:
    """Resolve the default CV strategy for *task* via the adapter contract."""
    try:
        contract = adapter.get_backend_contract()
        defaults = contract.capabilities.get("cv_default_strategy", {})
        if task in defaults:
            return str(defaults[task])
    except (AttributeError, KeyError, TypeError) as exc:
        _log.debug("cv_default_strategy contract read failed: %s", exc)
    return "stratified_kfold" if task in ("binary", "multiclass") else "kfold"


def default_cv_state(adapter: Any, *, strategy: str, n_splits: int) -> dict[str, Any]:
    """Build a default CV state dict, reading defaults from the adapter contract."""
    cv_defaults: dict[str, Any] = {}
    try:
        contract = adapter.get_backend_contract()
        cv_defaults = contract.capabilities.get("cv_defaults", {})
    except (AttributeError, KeyError, TypeError) as exc:
        _log.debug("cv_defaults contract read failed: %s", exc)

    return {
        "strategy": strategy,
        "n_splits": n_splits,
        "group_column": None,
        "time_column": None,
        "random_state": cv_defaults.get("random_state", 42),
        "shuffle": cv_defaults.get("shuffle", True),
        "gap": cv_defaults.get("gap", 0),
        "purge_gap": 0,
        "embargo": 0,
        "train_size_max": None,
        "test_size_max": None,
    }


def compute_preview_splits(df: pd.DataFrame, df_info: dict[str, Any]) -> dict[str, Any]:
    """Estimate fold structure for the ``blocked_group_kfold`` CV strategy.

    Reads ``df_info["cv"]`` for blocks/groups configuration and computes the
    expected fold layout, including per-period row counts.

    Raises
    ------
    ValueError
        If the active CV strategy is not ``blocked_group_kfold``.
    """
    cv = df_info.get("cv", {})
    if cv.get("strategy") != "blocked_group_kfold":
        msg = "preview_splits only supports strategy='blocked_group_kfold'"
        raise ValueError(msg)

    blocks_cfg: dict[str, Any] = cv.get("blocks") or {}
    groups_cfg: dict[str, Any] = cv.get("groups") or {}

    blocks_col: str = blocks_cfg.get("col", "")
    mode: str = blocks_cfg.get("mode", "expanding")
    train_window: int | None = blocks_cfg.get("train_window")
    group_folds: int = int(groups_cfg.get("n_splits", cv.get("n_splits", 2)))

    if blocks_col and blocks_col in df.columns:
        periods: list[str] = sorted(df[blocks_col].dropna().unique().tolist(), key=str)
    else:
        periods = []

    num_periods = len(periods)
    time_folds = max(0, num_periods - 1)
    total_folds = time_folds * group_folds

    period_counts: dict[str, int] = {}
    if blocks_col and blocks_col in df.columns:
        vc = df[blocks_col].value_counts()
        period_counts = {str(k): int(v) for k, v in vc.items()}

    cutoffs: list[Any] = blocks_cfg.get("cutoffs", [])
    if cutoffs and blocks_col and blocks_col in df.columns:
        sorted_values = sorted(df[blocks_col].dropna().unique().tolist(), key=str)
        grouped_periods: list[list[Any]] = []
        current_group: list[Any] = []
        cutoff_idx = 0
        for v in sorted_values:
            if cutoff_idx < len(cutoffs) and str(v) >= str(cutoffs[cutoff_idx]):
                if current_group:
                    grouped_periods.append(current_group)
                current_group = []
                cutoff_idx += 1
            current_group.append(v)
        if current_group:
            grouped_periods.append(current_group)
        periods = [str(gp[0]) if len(gp) == 1 else f"{gp[0]}..{gp[-1]}" for gp in grouped_periods]
        grouped_counts: dict[str, int] = {}
        for gp, label in zip(grouped_periods, periods, strict=True):
            grouped_counts[label] = sum(period_counts.get(str(v), 0) for v in gp)
        period_counts = grouped_counts
        num_periods = len(periods)
        time_folds = max(0, num_periods - 1)
        total_folds = time_folds * group_folds

    folds: list[dict[str, Any]] = []
    fold_index = 0
    for t in range(time_folds):
        valid_period = periods[t + 1]
        if mode == "sliding" and train_window is not None:
            start = max(0, t + 1 - train_window)
            train_periods_list = periods[start : t + 1]
        else:
            train_periods_list = periods[: t + 1]

        train_rows = sum(period_counts.get(str(p), 0) for p in train_periods_list)
        valid_rows = period_counts.get(str(valid_period), 0)

        period_label = " + ".join(str(p) for p in train_periods_list) + " -> " + str(valid_period)

        for group_idx in range(group_folds):
            folds.append(
                {
                    "fold": fold_index,
                    "period_label": period_label,
                    "group_label": f"G{group_idx}",
                    "train_size": train_rows,
                    "valid_size": valid_rows,
                    "train_periods": list(train_periods_list),
                    "valid_period": valid_period,
                }
            )
            fold_index += 1

    return {
        "total_folds": total_folds,
        "time_folds": time_folds,
        "group_folds": group_folds,
        "periods": periods,
        "folds": folds,
    }
