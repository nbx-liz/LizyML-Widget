"""UI schema and capabilities for the LizyML backend contract.

Extracted from LizyMLAdapter.get_backend_contract() to keep adapter.py
under 800 lines. Contains only static data structures — no ML logic.
"""

from __future__ import annotations

from typing import Any

# Keys already covered by parameter_hints or search_space_catalog
_KNOWN_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "objective",
        "metric",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "max_bin",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "first_metric_only",
        "verbose",
        "num_threads",
        "num_leaves",
        # Smart params (in search_space_catalog)
        "auto_num_leaves",
        "num_leaves_ratio",
        "min_data_in_leaf_ratio",
        "min_data_in_bin_ratio",
        "feature_weights",
        "balanced",
    }
)

# LightGBM parameters available as additional params (beyond hints/catalog).
# Subset of commonly used LightGBM parameters.
_LGBM_ADDITIONAL_PARAMS: list[str] = sorted(
    [
        "min_child_weight",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "max_cat_threshold",
        "cat_smooth",
        "cat_l2",
        "extra_trees",
        "path_smooth",
        "min_gain_to_split",
        "min_data_in_leaf",
        "min_data_in_bin",
        "max_cat_to_onehot",
        "top_k",
        "min_sum_hessian_in_leaf",
        "linear_tree",
        "feature_pre_filter",
        "force_col_wise",
        "force_row_wise",
        "histogram_pool_size",
        "is_unbalance",
        "scale_pos_weight",
        "sigmoid",
        "boost_from_average",
        "bin_construct_sample_cnt",
        "data_sample_strategy",
        "interaction_constraints",
    ]
)


def _build_additional_params() -> list[str]:
    """Return LightGBM params not in parameter_hints or search_space_catalog."""
    return [p for p in _LGBM_ADDITIONAL_PARAMS if p not in _KNOWN_PARAM_KEYS]


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
            {"key": "calibration", "title": "Calibration"},
            {"key": "evaluation", "title": "Evaluation"},
        ],
        "option_sets": {
            "objective": {
                "regression": ["huber", "mse", "mae", "quantile", "mape", "cross_entropy"],
                "binary": ["binary", "cross_entropy", "cross_entropy_lambda"],
                "multiclass": ["multiclass", "softmax", "multiclassova"],
            },
            "metric": dict(all_metrics_by_task),
            "model_metric": {
                "regression": [
                    "l1",
                    "l2",
                    "rmse",
                    "quantile",
                    "mape",
                    "huber",
                    "fair",
                    "poisson",
                    "gamma",
                    "gamma_deviance",
                    "tweedie",
                    "r2",
                    "rmsle",
                ],
                "binary": [
                    "auc",
                    "binary_logloss",
                    "binary_error",
                    "average_precision",
                    "cross_entropy",
                    "cross_entropy_lambda",
                    "kullback_leibler",
                    "f1",
                    "accuracy",
                    "brier",
                    "ece",
                    "precision_at_k",
                ],
                "multiclass": [
                    "multi_logloss",
                    "multi_error",
                    "auc_mu",
                    "f1",
                    "accuracy",
                    "brier",
                ],
            },
        },
        "parameter_hints": [
            {"key": "objective", "label": "Objective", "kind": "objective"},
            {"key": "metric", "label": "Metric", "kind": "model_metric"},
            {
                "key": "first_metric_only",
                "label": "First Metric Only",
                "kind": "boolean",
            },
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
        ],
        "search_space_catalog": [
            # ── Smart Params group (Fit tab order: Smart Params first) ──
            {
                "key": "auto_num_leaves",
                "title": "Auto Num Leaves",
                "paramType": "boolean",
                "modes": ["fixed", "choice"],
                "group": "smart_params",
                "default": True,
            },
            {
                "key": "num_leaves_ratio",
                "title": "Num Leaves Ratio",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "smart_params",
                "default": 1.0,
            },
            {
                "key": "num_leaves",
                "title": "Num Leaves",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "smart_params",
                "default": 256,
            },
            {
                "key": "min_data_in_leaf_ratio",
                "title": "Min Data In Leaf Ratio",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "smart_params",
                "default": 0.01,
            },
            {
                "key": "min_data_in_bin_ratio",
                "title": "Min Data In Bin Ratio",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "smart_params",
                "default": 0.01,
            },
            {
                "key": "feature_weights",
                "title": "Feature Weights",
                "paramType": "object",
                "modes": ["fixed"],
                "group": "smart_params",
                "default": None,
            },
            {
                "key": "balanced",
                "title": "Balanced",
                "paramType": "boolean",
                "modes": ["fixed", "choice"],
                "group": "smart_params",
                "default": False,
            },
            # ── Model Params group ──
            {
                "key": "objective",
                "title": "Objective",
                "paramType": "string",
                "modes": ["fixed", "choice"],
                "group": "model_params",
            },
            {
                "key": "metric",
                "title": "Metric",
                "paramType": "string",
                "modes": ["fixed", "choice"],
                "group": "model_params",
            },
            {
                "key": "first_metric_only",
                "title": "First Metric Only",
                "paramType": "boolean",
                "modes": ["fixed", "choice"],
                "group": "model_params",
            },
            {
                "key": "n_estimators",
                "title": "N Estimators",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "learning_rate",
                "title": "Learning Rate",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "max_depth",
                "title": "Max Depth",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "max_bin",
                "title": "Max Bin",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "feature_fraction",
                "title": "Feature Fraction",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "bagging_fraction",
                "title": "Bagging Fraction",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "bagging_freq",
                "title": "Bagging Freq",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "lambda_l1",
                "title": "Lambda L1",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "lambda_l2",
                "title": "Lambda L2",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "model_params",
            },
            {
                "key": "verbose",
                "title": "Log Output",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "model_params",
                "default": -1,
            },
            # ── Training group ──
            {
                "key": "seed",
                "title": "Seed",
                "paramType": "integer",
                "modes": ["fixed"],
                "group": "training",
                "default": 42,
            },
            {
                "key": "early_stopping.enabled",
                "title": "Early Stopping",
                "paramType": "boolean",
                "modes": ["fixed"],
                "group": "training",
                "default": True,
            },
            {
                "key": "early_stopping.rounds",
                "title": "Early Stopping Rounds",
                "paramType": "integer",
                "modes": ["fixed", "range"],
                "group": "training",
                "default": 150,
            },
            {
                "key": "validation_ratio",
                "title": "Validation Ratio",
                "paramType": "number",
                "modes": ["fixed", "range"],
                "group": "training",
                "default": 0.1,
            },
            {
                "key": "inner_valid",
                "title": "Inner Validation",
                "paramType": "string",
                "modes": ["fixed"],
                "group": "training",
                "default": "holdout",
            },
        ],
        "additional_params": _build_additional_params(),
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
            "min_data_in_leaf_ratio": 0.01,
            "min_data_in_bin_ratio": 0.01,
            "early_stopping.rounds": 50,
            "validation_ratio": 0.05,
            "seed": 1,
        },
        "conditional_visibility": {
            "calibration": {"task": ["binary"]},
            "num_leaves_ratio": {"auto_num_leaves": True},
            "num_leaves": {"auto_num_leaves": False},
            "early_stopping.rounds": {"early_stopping.enabled": True},
            "validation_ratio": {"early_stopping.enabled": True},
            "inner_valid": {"early_stopping.enabled": True},
        },
        "defaults": {
            "calibration": {"method": "platt", "n_splits": 5, "params": {}},
        },
        "calibration_methods": ["platt", "isotonic", "beta"],
        "calibration_params": {
            "platt": [],
            "isotonic": [
                "objective",
                "metric",
                "num_boost_round",
                "validation_ratio",
                "seed",
                "num_leaves",
                "max_depth",
                "min_data_in_leaf_ratio",
                "learning_rate",
                "lambda_l2",
                "min_gain_to_split",
                "feature_fraction",
                "bagging_fraction",
                "bagging_freq",
            ],
            "beta": [],
        },
        "inner_valid_options": ["holdout", "group_holdout", "time_holdout"],
        "special_search_space_fields": {
            "objective": "objective",
            "metric": "model_metric",
            "inner_valid": "inner_valid_picker",
        },
    }


def build_capabilities() -> dict[str, Any]:
    """Return the capabilities dict for the backend contract."""
    return {
        "tune": {"allow_empty_space": True},
        "cv_strategies": [
            "kfold",
            "stratified_kfold",
            "time_series",
            "group_time_series",
            "purged_time_series",
            "group_kfold",
            "stratified_group_kfold",
            "blocked_group_kfold",
        ],
        "cv_strategy_fields": {
            "kfold": ["n_splits", "shuffle", "random_state"],
            "stratified_kfold": ["n_splits", "shuffle", "random_state"],
            "group_kfold": ["n_splits", "group_col"],
            "stratified_group_kfold": ["n_splits", "group_col"],
            "time_series": ["n_splits", "time_col", "gap", "max_train_size", "max_test_size"],
            "purged_time_series": [
                "n_splits",
                "time_col",
                "purge_gap",
                "embargo",
            ],
            "group_time_series": [
                "n_splits",
                "group_col",
                "time_col",
                "gap",
                "max_train_size",
                "max_test_size",
            ],
            "blocked_group_kfold": [
                "blocks_col",
                "groups_col",
                "mode",
                "train_window",
                "min_train_rows",
                "min_valid_rows",
            ],
        },
        "cv_defaults": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "gap": 0,
        },
        "cv_default_strategy": {
            "binary": "stratified_kfold",
            "multiclass": "stratified_kfold",
            "regression": "kfold",
        },
    }
