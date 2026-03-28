"""Tests for learning curve metrics filter (P-026).

Verifies that plot() passes metrics kwarg to plot_learning_curve,
service.get_plot() transparently forwards kwargs, and
widget._handle_request_plot() extracts options from payload.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo, PlotData

# ── Helpers ──────────────────────────────────────────────────


def _make_widget() -> Any:
    """Create a LizyWidget with mocked LizyML backend."""
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
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


# ── Adapter.plot() tests ─────────────────────────────────────


class TestAdapterPlotMetricsKwarg:
    """LizyMLAdapter.plot() should pass metrics kwarg to learning curve."""

    def test_plot_learning_curve_receives_metrics(self) -> None:
        """When metrics kwarg is provided, it reaches plot_learning_curve."""
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": []}'
        mock_model.plot_learning_curve.return_value = mock_fig

        adapter.plot(mock_model, "learning-curve", metrics=["auc"])

        mock_model.plot_learning_curve.assert_called_once_with(metrics=["auc"])

    def test_plot_learning_curve_no_metrics_kwarg(self) -> None:
        """Without metrics kwarg, plot_learning_curve is called without it."""
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": []}'
        mock_model.plot_learning_curve.return_value = mock_fig

        adapter.plot(mock_model, "learning-curve")

        mock_model.plot_learning_curve.assert_called_once_with()

    def test_plot_non_learning_curve_ignores_kwargs(self) -> None:
        """Non-learning-curve plots should not receive extra kwargs."""
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": []}'
        mock_model.plot_oof_distribution.return_value = mock_fig

        result = adapter.plot(mock_model, "oof-distribution", metrics=["auc"])

        mock_model.plot_oof_distribution.assert_called_once_with()
        assert result.plotly_json == '{"data": []}'


# ── Service.get_plot() tests ─────────────────────────────────


class TestServiceGetPlotKwargs:
    """WidgetService.get_plot() should forward kwargs to adapter.plot()."""

    def test_get_plot_forwards_metrics(self) -> None:
        """kwargs passed to get_plot reach adapter.plot."""
        from lizyml_widget.service import WidgetService

        mock_adapter = MagicMock()
        mock_adapter.plot.return_value = PlotData(plotly_json='{"data": []}')
        service = WidgetService(mock_adapter)
        # Set a fake model so get_plot doesn't raise
        service._model = MagicMock()

        service.get_plot("learning-curve", metrics=["auc"])

        mock_adapter.plot.assert_called_once_with(service._model, "learning-curve", metrics=["auc"])

    def test_get_plot_no_kwargs(self) -> None:
        """get_plot without kwargs calls adapter.plot without kwargs."""
        from lizyml_widget.service import WidgetService

        mock_adapter = MagicMock()
        mock_adapter.plot.return_value = PlotData(plotly_json='{"data": []}')
        service = WidgetService(mock_adapter)
        service._model = MagicMock()

        service.get_plot("learning-curve")

        mock_adapter.plot.assert_called_once_with(service._model, "learning-curve")


# ── Widget._handle_request_plot() tests ──────────────────────


class TestWidgetRequestPlotOptions:
    """Widget should extract options from payload and pass to service."""

    def test_options_metrics_forwarded(self) -> None:
        """options.metrics in payload should reach service.get_plot."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot(
            {
                "plot_type": "learning-curve",
                "options": {"metrics": ["auc", "binary_logloss"]},
            }
        )

        w._service.get_plot.assert_called_once_with(
            "learning-curve", metrics=["auc", "binary_logloss"]
        )

    def test_no_options_backward_compat(self) -> None:
        """Without options, get_plot is called without kwargs."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot({"plot_type": "learning-curve"})

        w._service.get_plot.assert_called_once_with("learning-curve")

    def test_empty_options_ignored(self) -> None:
        """Empty options dict should not pass kwargs."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot({"plot_type": "learning-curve", "options": {}})

        w._service.get_plot.assert_called_once_with("learning-curve")

    def test_non_learning_curve_options_still_forwarded(self) -> None:
        """Options should be forwarded for any plot type (adapter decides)."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot(
            {
                "plot_type": "oof-distribution",
                "options": {"metrics": ["auc"]},
            }
        )

        w._service.get_plot.assert_called_once_with("oof-distribution", metrics=["auc"])

    def test_unknown_options_filtered_by_allowlist(self) -> None:
        """Unknown option keys are stripped by widget allowlist."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot(
            {
                "plot_type": "learning-curve",
                "options": {"metrics": ["auc"], "__proto__": "x", "evil_key": 42},
            }
        )

        # Only "metrics" should pass through the allowlist
        w._service.get_plot.assert_called_once_with("learning-curve", metrics=["auc"])

    def test_non_dict_options_ignored(self) -> None:
        """Non-dict options value is safely ignored."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot({"plot_type": "learning-curve", "options": "not_a_dict"})

        w._service.get_plot.assert_called_once_with("learning-curve")

    def test_invalid_metrics_type_ignored(self) -> None:
        """Non-list or non-string metrics values are rejected."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))
        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        # metrics with non-string items
        w._handle_request_plot(
            {"plot_type": "learning-curve", "options": {"metrics": [1, None, "auc"]}}
        )
        w._service.get_plot.assert_called_once_with("learning-curve")

        w._service.get_plot.reset_mock()

        # metrics as string instead of list
        w._handle_request_plot({"plot_type": "learning-curve", "options": {"metrics": "auc"}})
        w._service.get_plot.assert_called_once_with("learning-curve")
