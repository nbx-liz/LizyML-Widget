"""Widget-level E2E tests for Learning Curve metric (P-026).

Exercises the full Widget flow: load → set_config → fit → request_plot
via msg:custom → verify plot response contains the correct metric.

Uses the real LizyML backend. No mocks except for Widget.send().
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from lizyml_widget import LizyWidget


@pytest.fixture(scope="module")
def binary_df() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame


@pytest.fixture()
def widget(binary_df: pd.DataFrame) -> LizyWidget:
    """Create and load a widget with breast cancer data."""
    w = LizyWidget()
    w.load(binary_df, target="target")
    return w


class TestWidgetMetricChipReflectsLgbmNative:
    """Model Params Metric chips should show only LightGBM native names."""

    def test_binary_metric_chips_have_no_translated_names(self, widget: LizyWidget) -> None:
        contract = widget._service._adapter.get_backend_contract()
        binary_model_metrics = contract.ui_schema["option_sets"]["model_metric"]["binary"]

        # Translated names must not appear — use native/feval names instead
        translated = {"logloss", "auc_pr"}
        found = translated & set(binary_model_metrics)
        assert not found, f"Translated metric names in model_metric: {found}"


class TestWidgetFitWithCustomMetric:
    """set_config with custom model.params.metric → fit → verify results."""

    @pytest.fixture()
    def fitted_widget(self, widget: LizyWidget) -> LizyWidget:
        widget.set_config(
            {
                "model": {
                    "name": "lgbm",
                    "params": {
                        "n_estimators": 50,
                        "learning_rate": 0.1,
                        "metric": ["auc", "binary_logloss", "binary_error"],
                    },
                },
                "training": {
                    "seed": 42,
                    "early_stopping": {"enabled": True, "rounds": 10},
                },
            }
        )
        widget.fit()
        return widget

    def test_fit_summary_params_contain_metric(self, fitted_widget: LizyWidget) -> None:
        summary = fitted_widget.get_fit_summary()
        assert summary is not None
        metric_row = next((r for r in summary.params if r.get("parameter") == "metric"), None)
        assert metric_row is not None, "params should contain 'metric' row"
        assert set(metric_row["value"]) == {"auc", "binary_logloss", "binary_error"}

    def test_fit_summary_metrics_has_scores(self, fitted_widget: LizyWidget) -> None:
        summary = fitted_widget.get_fit_summary()
        assert summary is not None
        assert "raw" in summary.metrics or len(summary.metrics) > 0


class TestWidgetLearningCurvePlotRequest:
    """Simulate JS requesting learning-curve plot with metric filter."""

    @pytest.fixture()
    def fitted_widget(self, widget: LizyWidget) -> LizyWidget:
        widget.set_config(
            {
                "model": {
                    "name": "lgbm",
                    "params": {
                        "n_estimators": 50,
                        "learning_rate": 0.1,
                        "metric": ["auc", "binary_logloss", "binary_error"],
                    },
                },
                "training": {
                    "seed": 42,
                    "early_stopping": {"enabled": True, "rounds": 10},
                },
            }
        )
        widget.fit()
        return widget

    def _request_plot(
        self, widget: LizyWidget, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Simulate JS request_plot and capture the response."""
        sent: list[Any] = []
        widget.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        payload: dict[str, Any] = {"plot_type": "learning-curve"}
        if options:
            payload["options"] = options

        widget._handle_request_plot(payload)

        assert len(sent) == 1, f"Expected 1 response, got {len(sent)}"
        return sent[0]

    @pytest.mark.parametrize("metric", ["auc", "binary_logloss", "binary_error"])
    def test_filtered_plot_shows_single_metric(
        self, fitted_widget: LizyWidget, metric: str
    ) -> None:
        resp = self._request_plot(fitted_widget, {"metrics": [metric]})

        assert resp["type"] == "plot_data"
        assert resp["plot_type"] == "learning-curve"

        # Decode plotly JSON (may be inline or binary)
        plotly_json = resp.get("plotly_json", "")
        spec = json.loads(plotly_json)

        # Check subplot titles contain only the requested metric
        annotations = spec.get("layout", {}).get("annotations", [])
        titles = [a.get("text", "") for a in annotations]
        metric_names = [t.split("/")[-1] for t in titles if "/" in t]
        assert metric_names == [metric], f"Expected [{metric}], got {metric_names}"

    def test_unfiltered_plot_shows_all_metrics(self, fitted_widget: LizyWidget) -> None:
        resp = self._request_plot(fitted_widget)

        assert resp["type"] == "plot_data"
        spec = json.loads(resp.get("plotly_json", ""))
        annotations = spec.get("layout", {}).get("annotations", [])
        metric_names = sorted(
            {t.get("text", "").split("/")[-1] for t in annotations if "/" in t.get("text", "")}
        )
        assert len(metric_names) == 3, f"Expected 3 metrics, got {metric_names}"

    def test_invalid_metric_filter_returns_error(self, fitted_widget: LizyWidget) -> None:
        resp = self._request_plot(fitted_widget, {"metrics": ["nonexistent"]})
        # LizyML raises CONFIG_INVALID for unknown metrics
        assert resp["type"] == "plot_error"

    def test_switching_metrics_returns_different_plots(self, fitted_widget: LizyWidget) -> None:
        """Simulates user clicking metric chips in sequence."""
        resp_auc = self._request_plot(fitted_widget, {"metrics": ["auc"]})
        resp_err = self._request_plot(fitted_widget, {"metrics": ["binary_error"]})

        spec_auc = json.loads(resp_auc["plotly_json"])
        spec_err = json.loads(resp_err["plotly_json"])

        # Plot data should be different
        auc_y = spec_auc["data"][0]["y"] if spec_auc.get("data") else []
        err_y = spec_err["data"][0]["y"] if spec_err.get("data") else []
        assert auc_y != err_y, "Different metrics should produce different plot data"


class TestWidgetDefaultMetricLearningCurve:
    """Default metric (user doesn't change model.params.metric) should work."""

    @pytest.fixture()
    def default_fitted_widget(self, widget: LizyWidget) -> LizyWidget:
        widget.set_config(
            {
                "model": {
                    "name": "lgbm",
                    "params": {"n_estimators": 50, "learning_rate": 0.1},
                },
                "training": {
                    "seed": 42,
                    "early_stopping": {"enabled": True, "rounds": 10},
                },
            }
        )
        widget.fit()
        return widget

    def test_default_binary_has_auc_and_binary_logloss(
        self, default_fitted_widget: LizyWidget
    ) -> None:
        sent: list[Any] = []
        default_fitted_widget.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        default_fitted_widget._handle_request_plot({"plot_type": "learning-curve"})

        assert len(sent) == 1
        spec = json.loads(sent[0].get("plotly_json", ""))
        annotations = spec.get("layout", {}).get("annotations", [])
        metric_names = sorted(
            {t.get("text", "").split("/")[-1] for t in annotations if "/" in t.get("text", "")}
        )
        assert "auc" in metric_names
        assert "binary_logloss" in metric_names
