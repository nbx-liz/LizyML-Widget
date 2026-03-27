"""Tests for load_model() and model_info property."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo


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


class TestLoadModel:
    def test_load_model_sets_status_completed(self) -> None:
        w = _make_widget()
        # Mock service methods
        w._service.load_model_from_path = MagicMock()
        w._service.get_available_plots = MagicMock(return_value=["learning-curve"])
        w.load_model("/tmp/fake_model.pkl")
        assert w.status == "completed"

    def test_load_model_populates_available_plots(self) -> None:
        w = _make_widget()
        expected_plots = ["learning-curve", "feature-importance-split"]
        w._service.load_model_from_path = MagicMock()
        w._service.get_available_plots = MagicMock(return_value=expected_plots)
        w.load_model("/tmp/fake_model.pkl")
        assert list(w.available_plots) == expected_plots

    def test_load_model_returns_self(self) -> None:
        w = _make_widget()
        w._service.load_model_from_path = MagicMock()
        w._service.get_available_plots = MagicMock(return_value=[])
        result = w.load_model("/tmp/fake_model.pkl")
        assert result is w


class TestModelInfo:
    def test_model_info_returns_none_without_model(self) -> None:
        w = _make_widget()
        assert w.model_info is None

    def test_model_info_returns_dict_with_model(self) -> None:
        w = _make_widget()
        mock_model = MagicMock()
        w._service._model = mock_model
        w._service._adapter.model_info = MagicMock(return_value={"loaded": True, "task": "binary"})
        info = w.model_info
        assert info is not None
        assert info["loaded"] is True

    def test_model_info_falls_back_on_adapter_error(self) -> None:
        w = _make_widget()
        mock_model = MagicMock()
        w._service._model = mock_model
        w._service._adapter.model_info = MagicMock(side_effect=RuntimeError("boom"))
        info = w.model_info
        assert info == {"loaded": True}
