"""Column auto-detection helpers for WidgetService.

Pure functions extracted from service.py (#137) so the service module
stays under the 800-line ceiling. The helpers operate on plain dicts /
DataFrames and have no dependency on WidgetService internal state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_task(col: pd.Series[Any], n_rows: int, n_unique: int) -> str:
    """Auto-detect task type from a target column."""
    if n_unique == 2:
        return "binary"
    if str(col.dtype) in ("object", "str", "string", "category"):
        return "multiclass"
    if pd.api.types.is_numeric_dtype(col):
        threshold = max(20, int(n_rows * 0.05))
        if n_unique <= threshold:
            return "multiclass"
        return "regression"
    return "multiclass"


def auto_configure_column(
    col_info: dict[str, Any], n_rows: int, df: pd.DataFrame | None
) -> dict[str, Any]:
    """Auto-configure a single column (exclusion + type)."""
    name = col_info["name"]
    dtype = col_info["dtype"]
    unique = col_info["unique_count"]

    excluded = False
    exclude_reason: str | None = None
    col_type = "numeric"

    is_float = dtype.startswith("float") or dtype.startswith("Float")
    if unique == n_rows and not is_float:
        excluded = True
        exclude_reason = "id"
    elif unique == 1:
        excluded = True
        exclude_reason = "constant"

    if dtype in ("object", "str", "string", "category", "bool"):
        col_type = "categorical"
    elif df is not None and pd.api.types.is_numeric_dtype(df[name]):
        threshold = max(20, int(n_rows * 0.05))
        if unique <= threshold:
            col_type = "categorical"

    return {
        **col_info,
        "suggested_type": col_type,
        "suggested_excluded": excluded,
        "exclude_reason": exclude_reason,
        "excluded": excluded,
        "col_type": col_type,
    }


def calc_feature_summary(columns: list[dict[str, Any]]) -> dict[str, int]:
    """Calculate feature summary counts (active vs excluded breakdown)."""
    active = [c for c in columns if not c.get("excluded", False)]
    excluded = [c for c in columns if c.get("excluded", False)]
    return {
        "total": len(active),
        "numeric": sum(1 for c in active if c.get("col_type") == "numeric"),
        "categorical": sum(1 for c in active if c.get("col_type") == "categorical"),
        "excluded": len(excluded),
        "excluded_id": sum(1 for c in excluded if c.get("exclude_reason") == "id"),
        "excluded_const": sum(1 for c in excluded if c.get("exclude_reason") == "constant"),
        "excluded_manual": sum(1 for c in excluded if c.get("exclude_reason") is None),
    }


def merge_best_params_into_config(
    base: dict[str, Any],
    *,
    model_p: dict[str, Any],
    smart_p: dict[str, Any],
    training_p: dict[str, Any],
) -> dict[str, Any]:
    """Merge classified best_params into a base config (immutable update).

    Used by ``WidgetService.apply_best_params`` after classification.
    """
    base = {k: v for k, v in base.items() if k != "tuning"}

    model_section = dict(base.get("model", {}))
    model_params = {**dict(model_section.get("params", {})), **model_p}
    model_section = {**model_section, "params": model_params}

    base = {**base, "model": {**model_section, **smart_p}}

    if training_p:
        training = dict(base.get("training", {}))
        es = dict(training.get("early_stopping", {}))
        es_updates: dict[str, Any] = {}
        if "early_stopping_rounds" in training_p:
            es_updates["rounds"] = training_p["early_stopping_rounds"]
        if "validation_ratio" in training_p:
            existing_iv = es.get("inner_valid")
            if isinstance(existing_iv, dict):
                es_updates["inner_valid"] = {
                    **existing_iv,
                    "ratio": training_p["validation_ratio"],
                }
            else:
                es_updates["validation_ratio"] = training_p["validation_ratio"]
                es_updates["inner_valid"] = None
        new_es = {**es, **es_updates}
        base = {**base, "training": {**training, "early_stopping": new_es}}

    return base
