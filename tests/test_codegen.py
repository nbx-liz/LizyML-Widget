"""Tests for export_code path handling (Issue 2).

TDD: tests written FIRST, implementation follows.
Covers:
- _handle_export_code uses payload path when provided
- _handle_export_code uses tmpdir when path absent
- w.export_code(path) Python API
- w.export_code() Python API default (tmpdir)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_widget_with_model() -> Any:
    """Create a LizyWidget with a mocked adapter that has a trained model."""
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

        # Set up a mock model and export_code on service
        mock_model = MagicMock()
        adapter.export_code.return_value = Path("/tmp/fake_export")

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()

    # Inject a model into the service so export_code doesn't raise "No trained model"
    w._service._model = mock_model  # type: ignore[attr-defined]
    w._service._adapter = adapter  # type: ignore[attr-defined]
    return w, adapter


# ---------------------------------------------------------------------------
# Issue 2a: _handle_export_code uses payload path
# ---------------------------------------------------------------------------


class TestHandleExportCodeWithCustomPath:
    """_handle_export_code must pass payload['path'] to service.export_code."""

    def test_export_code_with_relative_path(self, tmp_path: Path) -> None:
        """Handler accepts relative path from payload."""
        w, adapter = _make_widget_with_model()
        rel_path = "my_export"
        adapter.export_code.return_value = Path(tmp_path / rel_path)

        sent_messages: list[dict[str, Any]] = []
        w.send = lambda msg: sent_messages.append(msg)  # type: ignore[method-assign]

        w.action = {"type": "export_code", "payload": {"path": rel_path}}

        call_args = adapter.export_code.call_args
        assert call_args is not None, "adapter.export_code was not called"
        called_path = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("path")
        assert called_path == rel_path

    def test_export_code_rejects_absolute_path(self) -> None:
        """Handler rejects absolute paths from JS payload."""
        w, _adapter = _make_widget_with_model()
        w.action = {"type": "export_code", "payload": {"path": "/etc/evil"}}
        assert w.error.get("code") == "EXPORT_CODE_ERROR"
        assert "Absolute paths" in w.error.get("message", "")

    def test_export_code_rejects_traversal(self) -> None:
        """Handler rejects path traversal attempts."""
        w, _adapter = _make_widget_with_model()
        w.action = {"type": "export_code", "payload": {"path": "../../etc/evil"}}
        assert w.error.get("code") == "EXPORT_CODE_ERROR"
        assert "traversal" in w.error.get("message", "")


class TestHandleExportCodeWithoutPath:
    """_handle_export_code must use tmpdir when payload has no 'path'."""

    def test_export_code_without_path_uses_tmpdir(self) -> None:
        """Handler uses a temporary directory when payload path is absent."""
        w, adapter = _make_widget_with_model()

        # Capture the path that service.export_code is called with
        captured_paths: list[str | None] = []

        def spy_export_code(path: str | None = None) -> Any:
            captured_paths.append(path)
            # Return a fake path; create a real tmpdir to avoid zip errors
            td = tempfile.mkdtemp()
            return Path(td)

        w._service.export_code = spy_export_code  # type: ignore[method-assign]

        sent_messages: list[dict[str, Any]] = []
        w.send = lambda msg: sent_messages.append(msg)  # type: ignore[method-assign]

        # Invoke without 'path' in payload
        w.action = {"type": "export_code", "payload": {}}

        # service.export_code should have been called with path=None
        assert len(captured_paths) == 1, "export_code should be called exactly once"
        assert captured_paths[0] is None, (
            f"Expected path=None when payload has no path, got {captured_paths[0]!r}"
        )

    def test_export_code_empty_path_treats_as_none(self) -> None:
        """Handler treats empty string path as absent (uses tmpdir)."""
        w, adapter = _make_widget_with_model()

        captured_paths: list[str | None] = []

        def spy_export_code(path: str | None = None) -> Any:
            captured_paths.append(path)
            td = tempfile.mkdtemp()
            return Path(td)

        w._service.export_code = spy_export_code  # type: ignore[method-assign]

        sent_messages: list[dict[str, Any]] = []
        w.send = lambda msg: sent_messages.append(msg)  # type: ignore[method-assign]

        w.action = {"type": "export_code", "payload": {"path": ""}}

        assert len(captured_paths) == 1
        assert captured_paths[0] is None, (
            f"Expected path=None for empty string path, got {captured_paths[0]!r}"
        )


# ---------------------------------------------------------------------------
# Issue 2b: Python API w.export_code(path)
# ---------------------------------------------------------------------------


class TestExportCodePythonAPI:
    """w.export_code(path) and w.export_code() public Python API."""

    def test_export_code_python_api_with_path(self, tmp_path: Path) -> None:
        """w.export_code(path) delegates to service.export_code(path)."""
        w, adapter = _make_widget_with_model()
        custom_path = str(tmp_path / "code_out")
        adapter.export_code.return_value = Path(custom_path)

        w.export_code(custom_path)

        # Adapter's export_code should be called with the custom path
        call_args = adapter.export_code.call_args
        assert call_args is not None
        called_path = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("path")
        assert called_path == custom_path

    def test_export_code_python_api_default_uses_tmpdir(self) -> None:
        """w.export_code() with no path argument uses a temporary directory."""
        w, adapter = _make_widget_with_model()

        captured_paths: list[str | None] = []
        original_export_code = w._service.export_code

        def spy_export_code(path: str | None = None) -> Any:
            captured_paths.append(path)
            return original_export_code(path)

        w._service.export_code = spy_export_code  # type: ignore[method-assign]

        w.export_code()

        assert len(captured_paths) == 1
        assert captured_paths[0] is None, (
            f"Expected path=None when w.export_code() called without args, "
            f"got {captured_paths[0]!r}"
        )

    def test_export_code_python_api_raises_without_model(self) -> None:
        """w.export_code() raises RuntimeError or ValueError when no model is trained."""
        real_adapter = LizyMLAdapter()
        with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
            adapter = MockAdapter.return_value
            adapter.info = BackendInfo(name="mock", version="0.0.0")
            adapter.get_config_schema.return_value = {"type": "object"}
            adapter.validate_config.return_value = []
            adapter.initialize_config.side_effect = real_adapter.initialize_config
            adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract

            from lizyml_widget.widget import LizyWidget

            w = LizyWidget()

        # No model loaded — should raise
        with pytest.raises((ValueError, RuntimeError)):
            w.export_code()

    def test_export_code_python_api_returns_result(self, tmp_path: Path) -> None:
        """w.export_code(path) returns what the service returns."""
        w, adapter = _make_widget_with_model()
        custom_path = str(tmp_path / "code_out")
        expected = Path(custom_path)
        adapter.export_code.return_value = expected

        result = w.export_code(custom_path)

        assert result == expected
