"""Tests for LizyWidget error handling, security, and edge cases."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
        # Delegate config lifecycle to real adapter
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


class TestErrorTracebackNotSynced:
    """Traceback should not be synced to frontend via error traitlet."""

    def test_job_error_has_no_traceback_key(self) -> None:
        """When a job fails, error traitlet must NOT contain 'traceback'."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        # Make fit raise an error
        w._service.fit = MagicMock(side_effect=RuntimeError("test backend error"))
        w._service.validate_config = MagicMock(return_value=[])

        w._run_job("fit")
        # Wait for background thread to complete
        if w._job_thread:
            w._job_thread.join(timeout=5)

        assert w.status == "failed"
        assert w.error.get("code") in ("BACKEND_ERROR", "INTERNAL_ERROR")
        assert "traceback" not in w.error, (
            "traceback must not be synced to frontend via error traitlet"
        )

    def test_job_error_still_has_message(self) -> None:
        """Error traitlet should still contain the error message."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.fit = MagicMock(side_effect=RuntimeError("specific error msg"))
        w._service.validate_config = MagicMock(return_value=[])

        w._run_job("fit")
        if w._job_thread:
            w._job_thread.join(timeout=5)

        assert "specific error msg" in w.error.get("message", "")


class TestLazyBackendInfo:
    """Widget must not crash when backend import fails."""

    def test_widget_init_without_backend(self) -> None:
        """LizyWidget() should succeed even when adapter.info raises."""
        adapter = MagicMock()
        type(adapter).info = property(
            lambda _: (_ for _ in ()).throw(ModuleNotFoundError("no lizyml"))
        )

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget(adapter=adapter)
        assert w.backend_info == {}

    def test_widget_init_with_backend(self) -> None:
        """When adapter works, backend_info is populated."""
        w = _make_widget()
        assert w.backend_info["name"] == "mock"


class TestClassifyBestParamsProtocol:
    """classify_best_params delegates directly to adapter (Protocol per P-013)."""

    def test_delegates_to_adapter(self) -> None:
        """Service.classify_best_params should call adapter directly."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        result = w._service.classify_best_params({"learning_rate": 0.1, "num_leaves_ratio": 0.5})
        assert len(result) == 3
        model_p, smart_p, _training_p = result
        assert "learning_rate" in model_p
        assert "num_leaves_ratio" in smart_p

    def test_adapter_error_propagates(self) -> None:
        """If adapter.classify_best_params raises, it should propagate."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service._adapter.classify_best_params = MagicMock(side_effect=ValueError("bad params"))

        with pytest.raises(ValueError, match="bad params"):
            w._service.classify_best_params({"lr": 0.1})


class TestErrorCodes:
    """Test BLUEPRINT error codes (NO_DATA, NO_TARGET)."""

    def test_fit_no_data_returns_no_data_error(self) -> None:
        w = _make_widget()
        # Trigger fit without loading data
        w.action = {"type": "fit", "payload": {}}
        assert w.error.get("code") == "NO_DATA"
        assert w.status == "failed"

    def test_fit_no_target_returns_no_target_error(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)  # no target
        w.action = {"type": "fit", "payload": {}}
        assert w.error.get("code") == "NO_TARGET"
        assert w.status == "failed"

    def test_tune_no_data_returns_no_data_error(self) -> None:
        w = _make_widget()
        w.action = {"type": "tune", "payload": {}}
        assert w.error.get("code") == "NO_DATA"
        assert w.status == "failed"
