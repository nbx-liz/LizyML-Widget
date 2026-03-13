"""UI schema and capabilities for the LizyML backend contract.

Extracted from LizyMLAdapter.get_backend_contract() to keep adapter.py
under 800 lines. Contains only static data structures — no ML logic.
"""

from __future__ import annotations

from typing import Any


def build_ui_schema(all_metrics_by_task: dict[str, list[str]]) -> dict[str, Any]:
    """Build the full UI schema for the backend contract.

    Parameters
    ----------
    all_metrics_by_task:
        Mapping of task name → list of valid metric names.
        Passed through to ``option_sets.metric``.
    """
    return {
        "sections": [
            {"key": "model", "title": "Model"},
            {"key": "training", "title": "Training"},
            {"key": "evaluation", "title": "Evaluation"},
            {"key": "calibration", "title": "Calibration"},
        ],
        "option_sets": {
            "objective": {
                "regression": ["huber", "mse", "mae", "quantile", "mape", "cross_entropy"],
                "binary": ["binary", "cross_entropy", "cross_entropy_lambda"],
                "multiclass": ["multiclass", "softmax", "multiclassova"],
            },
            "metric": dict(all_metrics_by_task),
        },
        "parameter_hints": [
            {"key": "objective", "label": "Objective", "kind": "objective"},
            {"key": "metric", "label": "Metric", "kind": "metric"},
            {"key": "n_estimators", "label": "N Estimators", "kind": "integer", "step": 100},
            {"key": "learning_rate", "label": "Learning Rate", "kind": "number", "step": 0.001},
            {"key": "max_depth", "label": "Max Depth", "kind": "integer", "step": 1},
            {"key": "max_bin", "label": "Max Bin", "kind": "integer", "step": 1},
            {
                "key": "feature_fraction",
                "label": "Feature Fraction",
                "kind": "number",
                "step": 0.05,
            },
            {
                "key": "bagging_fraction",
                "label": "Bagging Fraction",
                "kind": "number",
                "step": 0.05,
            },
            {"key": "bagging_freq", "label": "Bagging Freq", "kind": "integer", "step": 1},
            {"key": "lambda_l1", "label": "Lambda L1", "kind": "number", "step": 0.0001},
            {"key": "lambda_l2", "label": "Lambda L2", "kind": "number", "step": 0.0001},
            {
                "key": "first_metric_only",
                "label": "First Metric Only",
                "kind": "boolean",
            },
        ],
        "search_space_catalog": [
            {
                "key": "objective",
                "title": "Objective",
                "paramType": "string",
                "modes": ["fixed", "choice"],
            },
            {
                "key": "metric",
                "title": "Metric",
                "paramType": "string",
                "modes": ["fixed", "choice"],
            },
            {
                "key": "n_estimators",
                "title": "N Estimators",
                "paramType": "integer",
                "modes": ["fixed", "range"],
            },
            {
                "key": "learning_rate",
                "title": "Learning Rate",
                "paramType": "number",
                "modes": ["fixed", "range"],
            },
            {
                "key": "max_depth",
                "title": "Max Depth",
                "paramType": "integer",
                "modes": ["fixed", "range"],
            },
            {
                "key": "max_bin",
                "title": "Max Bin",
                "paramType": "integer",
                "modes": ["fixed", "range"],
            },
            {
                "key": "feature_fraction",
                "title": "Feature Fraction",
                "paramType": "number",
                "modes": ["fixed", "range"],
            },
            {
                "key": "bagging_fraction",
                "title": "Bagging Fraction",
                "paramType": "number",
                "modes": ["fixed", "range"],
            },
            {
                "key": "bagging_freq",
                "title": "Bagging Freq",
                "paramType": "integer",
                "modes": ["fixed", "range"],
            },
            {
                "key": "lambda_l1",
                "title": "Lambda L1",
                "paramType": "number",
                "modes": ["fixed", "range"],
            },
            {
                "key": "lambda_l2",
                "title": "Lambda L2",
                "paramType": "number",
                "modes": ["fixed", "range"],
            },
            {
                "key": "first_metric_only",
                "title": "First Metric Only",
                "paramType": "boolean",
                "modes": ["fixed", "choice"],
            },
        ],
        "step_map": {
            "n_estimators": 100,
            "learning_rate": 0.001,
            "max_depth": 1,
            "max_bin": 1,
            "feature_fraction": 0.05,
            "bagging_fraction": 0.05,
            "bagging_freq": 1,
            "lambda_l1": 0.0001,
            "lambda_l2": 0.0001,
            "num_leaves_ratio": 0.05,
            "num_leaves": 1,
        },
        "conditional_visibility": {
            "calibration": {"task": ["binary"]},
            "num_leaves_ratio": {"auto_num_leaves": True},
            "num_leaves": {"auto_num_leaves": False},
        },
        "defaults": {
            "calibration": {"method": "platt", "n_splits": 5, "params": {}},
        },
        "inner_valid_options": ["holdout", "group_holdout", "time_holdout"],
    }


def build_capabilities() -> dict[str, Any]:
    """Return the capabilities dict for the backend contract."""
    return {
        "tune": {"allow_empty_space": True},
    }
