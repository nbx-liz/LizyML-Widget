"""Metric mapping, param classification, and LightGBM defaults catalogue.

Extracted from ``adapter.py`` to keep the main adapter module under the
800-line project limit.  All items here are pure data or pure functions
with no dependency on Widget / traitlets / service.
"""

from __future__ import annotations

import logging
import threading
import types as _types
from collections.abc import Mapping
from typing import Any

_log = logging.getLogger(__name__)

# ── Model metric → eval metric mapping ──────────────────────
# LightGBM model_metric names (used in training) differ from LizyML
# evaluation metric names. This mapping bridges the two namespaces.
MODEL_METRIC_TO_EVAL: Mapping[str, str] = _types.MappingProxyType(
    {
        # Multiclass-specific
        "auc_mu": "auc",
        "multi_logloss": "logloss",
        "multi_error": "accuracy",
        # Binary-specific
        "binary_logloss": "logloss",
        "binary_error": "accuracy",
        "average_precision": "auc_pr",
        "cross_entropy": "logloss",
        "cross_entropy_lambda": "logloss",
        "kullback_leibler": "logloss",
        # Identity mappings (names that match between model_metric and eval_metric)
        "auc": "auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "brier": "brier",
        "ece": "ece",
        "precision_at_k": "precision_at_k",
        # Legacy/translated names — kept for backward compat with saved configs
        "auc_pr": "auc_pr",
        "logloss": "logloss",
        "mae": "mae",
        # Regression
        "l1": "mae",
        "l2": "rmse",
        "huber": "huber",
        "mape": "mape",
        "rmse": "rmse",
        "r2": "r2",
        "rmsle": "rmsle",
        # Regression native metrics without LizyML eval equivalent — map to
        # nearest eval metric for tune direction resolution (all minimize)
        "fair": "mae",
        "poisson": "mae",
        "gamma": "mae",
        "gamma_deviance": "mae",
        "tweedie": "mae",
        "quantile": "mae",
    }
)

# ── Param category classification for best_params routing ────
SMART_PARAMS: frozenset[str] = frozenset(
    {
        "auto_num_leaves",
        "num_leaves_ratio",
        "min_data_in_leaf_ratio",
        "min_data_in_bin_ratio",
        "feature_weights",
        "balanced",
    }
)
TRAINING_PARAMS: frozenset[str] = frozenset(
    {
        "early_stopping_rounds",
        "validation_ratio",
    }
)


def classify_best_params(
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split best_params into (model, smart, training) category dicts.

    Also normalizes model params that require format conversion:
    - ``metric``: Optuna returns a single string but LightGBM and the UI
      expect a list, so a bare string is wrapped in ``[value]``.
    """
    model_p: dict[str, Any] = {}
    smart_p: dict[str, Any] = {}
    training_p: dict[str, Any] = {}
    for k, v in params.items():
        if k in SMART_PARAMS:
            smart_p[k] = v
        elif k in TRAINING_PARAMS:
            training_p[k] = v
        else:
            model_p[k] = v
    # Normalize metric: single string → list (LightGBM expects list)
    if "metric" in model_p and isinstance(model_p["metric"], str):
        model_p = {**model_p, "metric": [model_p["metric"]]}
    return model_p, smart_p, training_p


def resolve_direction(eval_metric_name: str) -> str:
    """Return 'maximize' or 'minimize' based on the metric's greater_is_better."""
    try:
        from lizyml.core.exceptions import LizyMLError as _LizyMLError
    except ImportError:
        _LizyMLError = Exception  # type: ignore[assignment, misc]
    try:
        from lizyml.metrics.registry import get_metric

        metric = get_metric(eval_metric_name)
        return "maximize" if metric.greater_is_better else "minimize"
    except (ImportError, KeyError, ValueError, _LizyMLError) as exc:
        _log.warning(
            "Could not resolve direction for metric %r (%s); defaulting to 'minimize'",
            eval_metric_name,
            exc,
        )
        return "minimize"


# ── LightGBM default params ─────────────────────────────────
LGBM_PARAMS_TASK_INDEPENDENT: Mapping[str, Any] = _types.MappingProxyType(
    {
        "n_estimators": 1500,
        "learning_rate": 0.001,
        "max_depth": 5,
        "max_bin": 511,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 10,
        "lambda_l1": 0.0,
        "lambda_l2": 0.000001,
        "first_metric_only": False,
        "verbose": -1,
        "num_threads": 0,
    }
)

LGBM_PARAMS_BY_TASK: Mapping[str, Mapping[str, Any]] = _types.MappingProxyType(
    {
        "regression": _types.MappingProxyType(
            {"objective": "huber", "metric": ["huber", "mae", "mape"]}
        ),
        "binary": _types.MappingProxyType(
            {"objective": "binary", "metric": ["auc", "binary_logloss"]}
        ),
        "multiclass": _types.MappingProxyType(
            {"objective": "multiclass", "metric": ["auc_mu", "multi_logloss"]}
        ),
    }
)


# ── Eval metrics catalogue (cached, thread-safe) ────────────
_eval_metrics_cache: dict[str, list[str]] | None = None
_eval_metrics_lock: threading.Lock = threading.Lock()


_PREFERRED_METRIC: dict[str, str] = {
    "binary": "auc",
    "regression": "rmse",
    "multiclass": "auc",
}


def _sort_with_preferred(metrics: list[str], task: str) -> list[str]:
    """Sort metrics alphabetically but place the preferred metric first."""
    preferred = _PREFERRED_METRIC.get(task)
    ordered = sorted(metrics)
    if preferred and preferred in ordered:
        ordered = [preferred, *(m for m in ordered if m != preferred)]
    return ordered


def get_eval_metrics_by_task() -> dict[str, list[str]]:
    """Query LizyML's metric registry for available evaluation metrics per task."""
    global _eval_metrics_cache  # noqa: PLW0603

    if _eval_metrics_cache is not None:
        return _eval_metrics_cache
    with _eval_metrics_lock:
        # Double-checked locking
        if _eval_metrics_cache is not None:
            return _eval_metrics_cache
        metrics: dict[str, list[str]]
        try:
            from lizyml.metrics.registry import _TASK_METRICS

            metrics = {
                task: _sort_with_preferred(list(ms), task) for task, ms in _TASK_METRICS.items()
            }
        except (ImportError, AttributeError, TypeError):
            # Fallback for older LizyML versions without _TASK_METRICS
            metrics = {
                "regression": _sort_with_preferred(
                    ["mae", "mape", "rmse", "huber", "r2", "rmsle"], "regression"
                ),
                "binary": _sort_with_preferred(
                    [
                        "auc",
                        "logloss",
                        "auc_pr",
                        "f1",
                        "accuracy",
                        "brier",
                        "ece",
                        "precision_at_k",
                    ],
                    "binary",
                ),
                "multiclass": _sort_with_preferred(
                    ["auc", "logloss", "auc_pr", "f1", "accuracy", "brier"], "multiclass"
                ),
            }
        _eval_metrics_cache = metrics
        return metrics
