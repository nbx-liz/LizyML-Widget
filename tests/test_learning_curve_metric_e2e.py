"""E2E tests for Learning Curve metric flow (P-026).

Uses the real LizyML backend (not mocked) to verify:
1. model_metric option set contains only LightGBM native names
2. User-specified model.params.metric flows through to fit
3. Learning curve plot filter works with each metric individually
4. Widget._handle_request_plot passes metrics option to adapter.plot
5. Default metric (no user override) still works correctly

These tests require lizyml[plots] to be installed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo

# ── Fixtures ─────────────────────────────────────────────────

_BINARY_DF: pd.DataFrame | None = None
_MULTICLASS_DF: pd.DataFrame | None = None


def _get_binary_df() -> pd.DataFrame:
    global _BINARY_DF  # noqa: PLW0603
    if _BINARY_DF is None:
        data = load_breast_cancer(as_frame=True)
        _BINARY_DF = data.frame
    return _BINARY_DF


def _get_multiclass_df() -> pd.DataFrame:
    global _MULTICLASS_DF  # noqa: PLW0603
    if _MULTICLASS_DF is None:
        data = load_iris(as_frame=True)
        _MULTICLASS_DF = data.frame
    return _MULTICLASS_DF


def _make_widget() -> Any:
    """Create a LizyWidget with the real LizyML adapter."""
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="lizyml", version="test")
        adapter.get_config_schema.return_value = {"type": "object"}
        adapter.validate_config.return_value = []
        adapter.initialize_config.side_effect = real_adapter.initialize_config
        adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
        adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
        adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
        adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
        adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
        adapter.classify_best_params.side_effect = real_adapter.classify_best_params

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


# ── 1. model_metric option set validity ─────────────────────


class TestModelMetricOptionSet:
    """model_metric must not contain translated names (logloss, auc_pr)."""

    # Translated names must NOT appear — use native/feval names instead
    TRANSLATED_NAMES = {"logloss", "auc_pr"}

    def test_binary_no_translated_names(self) -> None:
        adapter = LizyMLAdapter()
        contract = adapter.get_backend_contract()
        binary = set(contract.ui_schema["option_sets"]["model_metric"]["binary"])
        found = self.TRANSLATED_NAMES & binary
        assert not found, f"binary model_metric has translated names: {found}"

    def test_regression_no_translated_names(self) -> None:
        adapter = LizyMLAdapter()
        contract = adapter.get_backend_contract()
        reg = set(contract.ui_schema["option_sets"]["model_metric"]["regression"])
        found = self.TRANSLATED_NAMES & reg
        assert not found, f"regression model_metric has translated names: {found}"


# ── 2. Fit with custom metric → params_table shows it ───────


class TestFitMetricPropagation:
    """User-specified model.params.metric must appear in fit results."""

    @pytest.fixture()
    def fitted_adapter(self) -> tuple[LizyMLAdapter, Any]:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 10, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 50
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["auc", "binary_logloss", "binary_error"]

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model = adapter.create_model(run_config, _get_binary_df())
        adapter.fit(model)
        return adapter, model

    def test_params_table_contains_metric(self, fitted_adapter: tuple[LizyMLAdapter, Any]) -> None:
        _, model = fitted_adapter
        params_df = model.params_table()
        assert "metric" in params_df.index, "params_table must include 'metric' row"
        metric_val = params_df.loc["metric", "value"]
        assert isinstance(metric_val, list)
        assert "auc" in metric_val
        assert "binary_error" in metric_val

    def test_history_contains_all_specified_metrics(
        self, fitted_adapter: tuple[LizyMLAdapter, Any]
    ) -> None:
        _, model = fitted_adapter
        history = model.fit_result.history[0].get("eval_history", {})
        all_metrics: set[str] = set()
        for ds_metrics in history.values():
            all_metrics.update(ds_metrics.keys())
        assert "auc" in all_metrics
        assert "binary_logloss" in all_metrics
        assert "binary_error" in all_metrics


# ── 3. Learning curve filter works per-metric ────────────────


class TestLearningCurveFilter:
    """plot_learning_curve(metrics=[...]) returns only the requested metric."""

    @pytest.fixture()
    def fitted_model(self) -> Any:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 10, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 50
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["auc", "binary_logloss", "binary_error"]

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model = adapter.create_model(run_config, _get_binary_df())
        adapter.fit(model)
        return model

    @pytest.mark.parametrize("metric", ["auc", "binary_logloss", "binary_error"])
    def test_filter_single_metric(self, fitted_model: Any, metric: str) -> None:
        fig = fitted_model.plot_learning_curve(metrics=[metric])
        titles = [a.text for a in fig.layout.annotations] if fig.layout.annotations else []
        # Each title is "dataset/metric" e.g. "valid_0/auc"
        metric_names = [t.split("/")[-1] for t in titles]
        assert metric_names == [metric], f"Expected [{metric}], got {metric_names}"

    def test_no_filter_returns_all(self, fitted_model: Any) -> None:
        fig = fitted_model.plot_learning_curve()
        titles = [a.text for a in fig.layout.annotations] if fig.layout.annotations else []
        metric_names = sorted({t.split("/")[-1] for t in titles})
        assert len(metric_names) == 3, f"Expected 3 metrics, got {metric_names}"


# ── 4. Adapter.plot() passes metrics kwarg correctly ─────────


class TestAdapterPlotMetricsE2E:
    """adapter.plot(model, 'learning-curve', metrics=[...]) filters correctly."""

    @pytest.fixture()
    def fitted(self) -> tuple[LizyMLAdapter, Any]:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 10, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 50
        config["model"]["params"]["verbose"] = -1
        config["model"]["params"]["metric"] = ["auc", "binary_logloss"]

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model = adapter.create_model(run_config, _get_binary_df())
        adapter.fit(model)
        return adapter, model

    def test_adapter_plot_with_metrics_filter(self, fitted: tuple[LizyMLAdapter, Any]) -> None:
        adapter, model = fitted
        result = adapter.plot(model, "learning-curve", metrics=["auc"])
        spec = json.loads(result.plotly_json)
        titles = [a.get("text", "") for a in spec.get("layout", {}).get("annotations", [])]
        assert len(titles) == 1
        assert "auc" in titles[0]

    def test_adapter_plot_without_filter(self, fitted: tuple[LizyMLAdapter, Any]) -> None:
        adapter, model = fitted
        result = adapter.plot(model, "learning-curve")
        spec = json.loads(result.plotly_json)
        titles = [a.get("text", "") for a in spec.get("layout", {}).get("annotations", [])]
        assert len(titles) == 2  # auc + binary_logloss


# ── 5. Widget request_plot with options flows to adapter ─────


class TestWidgetPlotOptionsFlow:
    """Widget._handle_request_plot extracts options and passes to service."""

    def test_options_metrics_reach_adapter(self) -> None:
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        from lizyml_widget.types import PlotData

        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot(
            {
                "plot_type": "learning-curve",
                "options": {"metrics": ["auc"]},
            }
        )

        w._service.get_plot.assert_called_once_with("learning-curve", metrics=["auc"])

    def test_invalid_options_ignored(self) -> None:
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        from lizyml_widget.types import PlotData

        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        # Non-list metrics should be rejected
        w._handle_request_plot(
            {
                "plot_type": "learning-curve",
                "options": {"metrics": "not_a_list", "evil": True},
            }
        )

        w._service.get_plot.assert_called_once_with("learning-curve")


# ── 6. Default metric (no user override) ────────────────────


class TestDefaultMetric:
    """When user doesn't specify metric, task defaults are used."""

    def test_binary_default_metrics_in_history(self) -> None:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "target"}
        config["training"] = {
            "seed": 42,
            "early_stopping": {"enabled": True, "rounds": 10, "validation_ratio": 0.2},
        }
        config["model"]["params"]["n_estimators"] = 50
        config["model"]["params"]["verbose"] = -1
        # Do NOT set model.params.metric — use defaults

        run_config = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model = adapter.create_model(run_config, _get_binary_df())
        adapter.fit(model)

        history = model.fit_result.history[0].get("eval_history", {})
        all_metrics: set[str] = set()
        for ds_metrics in history.values():
            all_metrics.update(ds_metrics.keys())
        # Default binary metrics: auc, binary_logloss
        assert "auc" in all_metrics
        assert "binary_logloss" in all_metrics
