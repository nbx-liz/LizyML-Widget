"""Tests for model_metric option set completeness and validity.

Ensures model_metric option set contains ALL usable metrics per task:
- LightGBM native metrics (primary names, no aliases)
- LizyML feval metrics (f1, accuracy, brier, etc.)
- NO translated names (logloss, auc_pr — those belong in evaluation.metrics)
- Every entry has a MODEL_METRIC_TO_EVAL mapping
"""

from __future__ import annotations

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.adapter_params import MODEL_METRIC_TO_EVAL

# ── Expected metric sets (翻訳 excluded) ────────────────────

EXPECTED_REGRESSION = {
    # LightGBM native
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
    # LizyML feval
    "r2",
    "rmsle",
}

EXPECTED_BINARY = {
    # LightGBM native
    "auc",
    "binary_logloss",
    "binary_error",
    "average_precision",
    "cross_entropy",
    "cross_entropy_lambda",
    "kullback_leibler",
    # LizyML feval
    "f1",
    "accuracy",
    "brier",
    "ece",
    "precision_at_k",
}

EXPECTED_MULTICLASS = {
    # LightGBM native
    "multi_logloss",
    "multi_error",
    "auc_mu",
    # LizyML feval
    "f1",
    "accuracy",
    "brier",
}

# Translated names must NOT appear in model_metric
TRANSLATED_NAMES = {"logloss", "auc_pr"}


def _get_model_metrics() -> dict[str, set[str]]:
    adapter = LizyMLAdapter()
    contract = adapter.get_backend_contract()
    mm = contract.ui_schema["option_sets"]["model_metric"]
    return {task: set(metrics) for task, metrics in mm.items()}


class TestModelMetricCompleteness:
    """model_metric must include ALL usable metrics per task."""

    def test_regression_has_all_metrics(self) -> None:
        actual = _get_model_metrics()["regression"]
        missing = EXPECTED_REGRESSION - actual
        assert not missing, f"regression model_metric missing: {sorted(missing)}"

    def test_binary_has_all_metrics(self) -> None:
        actual = _get_model_metrics()["binary"]
        missing = EXPECTED_BINARY - actual
        assert not missing, f"binary model_metric missing: {sorted(missing)}"

    def test_multiclass_has_all_metrics(self) -> None:
        actual = _get_model_metrics()["multiclass"]
        missing = EXPECTED_MULTICLASS - actual
        assert not missing, f"multiclass model_metric missing: {sorted(missing)}"


class TestModelMetricNoTranslatedNames:
    """Translated names (logloss, auc_pr) must NOT be in model_metric."""

    def test_no_translated_names_in_any_task(self) -> None:
        all_metrics = _get_model_metrics()
        for task, metrics in all_metrics.items():
            found = TRANSLATED_NAMES & metrics
            assert not found, (
                f"{task} model_metric contains translated names: {sorted(found)}. "
                f"Use native names instead."
            )


class TestModelMetricNoAliases:
    """model_metric should use primary names, not aliases."""

    def test_regression_uses_primary_names(self) -> None:
        actual = _get_model_metrics()["regression"]
        aliases = {
            "mae",
            "mse",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "l2_root",
            "regression",
            "regression_l1",
            "regression_l2",
            "mean_absolute_percentage_error",
        }
        found = aliases & actual
        assert not found, f"regression model_metric uses aliases: {sorted(found)}"

    def test_binary_uses_primary_names(self) -> None:
        actual = _get_model_metrics()["binary"]
        aliases = {"binary", "xentropy", "xentlambda", "kldiv"}
        found = aliases & actual
        assert not found, f"binary model_metric uses aliases: {sorted(found)}"

    def test_multiclass_uses_primary_names(self) -> None:
        actual = _get_model_metrics()["multiclass"]
        aliases = {"multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"}
        found = aliases & actual
        assert not found, f"multiclass model_metric uses aliases: {sorted(found)}"


class TestModelMetricToEvalMapping:
    """Every model_metric must have a MODEL_METRIC_TO_EVAL mapping."""

    def test_all_model_metrics_have_eval_mapping(self) -> None:
        all_metrics = _get_model_metrics()
        unmapped: list[str] = []
        for task, metrics in all_metrics.items():
            for m in metrics:
                if m not in MODEL_METRIC_TO_EVAL:
                    unmapped.append(f"{task}/{m}")
        assert not unmapped, f"MODEL_METRIC_TO_EVAL missing: {unmapped}"


class TestModelMetricNoUnexpectedEntries:
    """model_metric must not contain unknown or unexpected entries."""

    def test_regression_has_no_extra(self) -> None:
        actual = _get_model_metrics()["regression"]
        extra = actual - EXPECTED_REGRESSION
        assert not extra, f"regression model_metric has unexpected: {sorted(extra)}"

    def test_binary_has_no_extra(self) -> None:
        actual = _get_model_metrics()["binary"]
        extra = actual - EXPECTED_BINARY
        assert not extra, f"binary model_metric has unexpected: {sorted(extra)}"

    def test_multiclass_has_no_extra(self) -> None:
        actual = _get_model_metrics()["multiclass"]
        extra = actual - EXPECTED_MULTICLASS
        assert not extra, f"multiclass model_metric has unexpected: {sorted(extra)}"
