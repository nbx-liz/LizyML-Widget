"""Tests for D-1: binary buffer plot response when plotly_json exceeds threshold."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

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


class TestPlotBinaryBuffer:
    """D-1: _send_plot_response sends binary buffer for large payloads."""

    def test_small_plot_sends_inline_json(self) -> None:
        w = _make_widget()
        w.send = MagicMock()  # type: ignore[method-assign]

        small_json = '{"data": [{"x": [1, 2, 3]}]}'
        w._send_plot_response("roc-curve", small_json, "req-1")

        w.send.assert_called_once()
        call_args = w.send.call_args
        msg = call_args[0][0]
        assert msg["type"] == "plot_data"
        assert msg["plot_type"] == "roc-curve"
        assert msg["plotly_json"] == small_json
        assert msg["request_id"] == "req-1"
        assert "binary" not in msg
        # No buffers kwarg
        assert "buffers" not in (call_args[1] if len(call_args) > 1 else {})

    def test_large_plot_sends_binary_buffer(self) -> None:
        w = _make_widget()
        w.send = MagicMock()  # type: ignore[method-assign]

        # Create a JSON string exceeding the 800KB threshold
        large_json = '{"data": "' + "x" * 900_000 + '"}'
        w._send_plot_response("shap-summary", large_json, "req-2")

        w.send.assert_called_once()
        call_args = w.send.call_args
        msg = call_args[0][0]
        assert msg["type"] == "plot_data"
        assert msg["plot_type"] == "shap-summary"
        assert msg["request_id"] == "req-2"
        assert msg["binary"] is True
        assert "plotly_json" not in msg
        # Binary buffer should be passed
        buffers = call_args[1]["buffers"]
        assert len(buffers) == 1
        assert buffers[0] == large_json.encode("utf-8")

    def test_binary_threshold_boundary(self) -> None:
        """Exactly at threshold should use inline; one byte over uses binary."""
        w = _make_widget()

        # Exactly at threshold — inline
        w.send = MagicMock()  # type: ignore[method-assign]
        at_threshold = "x" * 800_000
        w._send_plot_response("t1", at_threshold, None)
        msg = w.send.call_args[0][0]
        assert "plotly_json" in msg
        assert "binary" not in msg

        # One byte over — binary
        w.send = MagicMock()  # type: ignore[method-assign]
        over_threshold = "x" * 800_001
        w._send_plot_response("t2", over_threshold, None)
        msg = w.send.call_args[0][0]
        assert msg["binary"] is True
        assert "plotly_json" not in msg

    def test_no_request_id_omitted_from_message(self) -> None:
        w = _make_widget()
        w.send = MagicMock()  # type: ignore[method-assign]

        w._send_plot_response("roc-curve", '{"data": []}', None)

        msg = w.send.call_args[0][0]
        assert "request_id" not in msg

    def test_handle_request_plot_uses_binary_for_large_response(self) -> None:
        """Integration: _handle_request_plot delegates to _send_plot_response."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        large_json = '{"data": "' + "x" * 900_000 + '"}'
        mock_plot = PlotData(plotly_json=large_json)

        w.send = MagicMock()  # type: ignore[method-assign]
        with patch.object(w._service, "get_plot", return_value=mock_plot):
            w._handle_request_plot({"plot_type": "roc-curve", "request_id": "req-99"})

        call_args = w.send.call_args
        msg = call_args[0][0]
        assert msg["binary"] is True
        assert msg["request_id"] == "req-99"
        assert call_args[1]["buffers"][0] == large_json.encode("utf-8")
