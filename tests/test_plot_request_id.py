"""Tests for plot request_id echo-back (B-1).

Verifies that _handle_request_plot echoes back request_id in the response,
allowing JS to discard stale out-of-order responses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo, PlotData


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


class TestPlotRequestIdEchoBack:
    """B-1: request_id should be echoed in plot_data response."""

    def test_request_id_echoed_in_plot_data(self) -> None:
        """When request_id is included in payload, response includes it."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        # Mock service.get_plot to return dummy data
        w._service.get_plot = MagicMock(
            return_value=PlotData(plotly_json='{"data": [], "layout": {}}')
        )

        w._handle_request_plot({"plot_type": "confusion_matrix", "request_id": "req-42"})

        assert len(sent) == 1
        resp = sent[0]
        assert resp["type"] == "plot_data"
        assert resp["plot_type"] == "confusion_matrix"
        assert resp["request_id"] == "req-42"

    def test_request_id_echoed_in_plot_error(self) -> None:
        """When plot generation fails, error response also includes request_id."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        w._service.get_plot = MagicMock(side_effect=ValueError("No model"))

        w._handle_request_plot({"plot_type": "roc_curve", "request_id": "req-99"})

        assert len(sent) == 1
        resp = sent[0]
        assert resp["type"] == "plot_error"
        assert resp["request_id"] == "req-99"

    def test_no_request_id_still_works(self) -> None:
        """Backward compat: response works without request_id in payload."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_plot({"plot_type": "feature_importance"})

        assert len(sent) == 1
        resp = sent[0]
        assert resp["type"] == "plot_data"
        # request_id should not be present or should be None
        assert resp.get("request_id") is None

    def test_inference_plot_echoes_request_id(self) -> None:
        """request_inference_plot also echoes request_id."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg, **kw: sent.append(msg))

        w._service.get_plot = MagicMock(return_value=PlotData(plotly_json='{"data": []}'))

        w._handle_request_inference_plot(
            {"plot_type": "prediction_distribution", "request_id": "req-inf-1"}
        )

        assert len(sent) == 1
        resp = sent[0]
        assert resp.get("request_id") == "req-inf-1"
