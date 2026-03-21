"""Tests for export_code handler and Python API.

TDD: tests written FIRST, implementation follows.
Covers:
- _handle_export_code sends zip as binary buffer (browser-download flow)
- _handle_export_code error handling
- w.export_code(path) Python API (unchanged)
- w.export_code() Python API default (tmpdir)
"""

from __future__ import annotations

import io
import tempfile
import zipfile
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


def _make_real_export_dir(tmp_path: Path) -> Path:
    """Create a real directory with a sample file for zipping."""
    export_dir = tmp_path / "export_out"
    export_dir.mkdir()
    (export_dir / "train.py").write_text("# generated code\n")
    return export_dir


# ---------------------------------------------------------------------------
# New: _handle_export_code sends binary buffer (browser-download)
# ---------------------------------------------------------------------------


class TestHandleExportCodeSendsBuffer:
    """_handle_export_code must send zip contents as binary buffer."""

    def test_handle_export_code_sends_buffer(self, tmp_path: Path) -> None:
        """Handler sends a non-empty bytes buffer containing a valid zip."""
        w, adapter = _make_widget_with_model()
        export_dir = _make_real_export_dir(tmp_path)
        adapter.export_code.return_value = export_dir

        sent_msgs: list[Any] = []
        sent_buffers: list[Any] = []

        def capture_send(msg: Any, buffers: list[Any] | None = None) -> None:
            sent_msgs.append(msg)
            sent_buffers.append(buffers or [])

        w.send = capture_send  # type: ignore[method-assign]
        w.action = {"type": "export_code", "payload": {}}

        assert len(sent_msgs) == 1, "send() must be called exactly once"
        bufs = sent_buffers[0]
        assert len(bufs) == 1, "send() must pass exactly one buffer"
        zip_bytes = bufs[0]
        assert isinstance(zip_bytes, (bytes, bytearray, memoryview)), (
            f"buffer must be bytes-like, got {type(zip_bytes)}"
        )
        assert len(zip_bytes) > 0, "zip buffer must not be empty"
        # Verify it is a valid zip
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        assert zf.namelist(), "zip must contain at least one file"

    def test_handle_export_code_sends_correct_type(self, tmp_path: Path) -> None:
        """Handler sends msg with type == 'code_export_download'."""
        w, adapter = _make_widget_with_model()
        export_dir = _make_real_export_dir(tmp_path)
        adapter.export_code.return_value = export_dir

        sent_msgs: list[Any] = []

        def capture_send(msg: Any, buffers: list[Any] | None = None) -> None:
            sent_msgs.append(msg)

        w.send = capture_send  # type: ignore[method-assign]
        w.action = {"type": "export_code", "payload": {}}

        assert sent_msgs[0]["type"] == "code_export_download"

    def test_handle_export_code_sends_filename(self, tmp_path: Path) -> None:
        """Handler includes 'filename' field in the message."""
        w, adapter = _make_widget_with_model()
        export_dir = _make_real_export_dir(tmp_path)
        adapter.export_code.return_value = export_dir

        sent_msgs: list[Any] = []

        def capture_send(msg: Any, buffers: list[Any] | None = None) -> None:
            sent_msgs.append(msg)

        w.send = capture_send  # type: ignore[method-assign]
        w.action = {"type": "export_code", "payload": {}}

        msg = sent_msgs[0]
        assert "filename" in msg, "msg must contain 'filename' key"
        assert msg["filename"].endswith(".zip"), (
            f"filename should end with .zip, got {msg['filename']!r}"
        )

    def test_handle_export_code_cleans_temp_files(self, tmp_path: Path) -> None:
        """Handler removes temp zip file and temp zip_dir after sending."""
        import shutil as _shutil

        w, adapter = _make_widget_with_model()
        export_dir = _make_real_export_dir(tmp_path)
        adapter.export_code.return_value = export_dir

        created_zip_paths: list[str] = []
        original_make_archive = _shutil.make_archive

        def spy_make_archive(base_name: str, fmt: str, *args: Any, **kwargs: Any) -> str:
            path = original_make_archive(base_name, fmt, *args, **kwargs)
            created_zip_paths.append(path)
            return path

        w.send = lambda msg, buffers=None: None  # type: ignore[method-assign]

        with patch("shutil.make_archive", side_effect=spy_make_archive):
            w.action = {"type": "export_code", "payload": {}}

        for zip_path in created_zip_paths:
            assert not Path(zip_path).exists(), (
                f"temp zip file should be deleted after send: {zip_path}"
            )

    def test_handle_export_code_error_sets_error(self) -> None:
        """When export_code raises, the error traitlet is set."""
        w, adapter = _make_widget_with_model()
        adapter.export_code.side_effect = RuntimeError("disk full")

        w.action = {"type": "export_code", "payload": {}}

        assert w.error.get("code") == "EXPORT_CODE_ERROR"
        assert "disk full" in w.error.get("message", "")


class TestHandleExportCodeAlwaysUsesTmpdir:
    """_handle_export_code always ignores payload path and uses tmpdir."""

    def test_export_code_always_passes_none_to_service(self) -> None:
        """Handler always calls service.export_code(None) regardless of payload."""
        w, adapter = _make_widget_with_model()

        captured_paths: list[str | None] = []

        def spy_export_code(path: str | None = None) -> Any:
            captured_paths.append(path)
            td = tempfile.mkdtemp()
            (Path(td) / "train.py").write_text("# code\n")
            return Path(td)

        w._service.export_code = spy_export_code  # type: ignore[method-assign]
        w.send = lambda msg, buffers=None: None  # type: ignore[method-assign]

        w.action = {"type": "export_code", "payload": {}}

        assert len(captured_paths) == 1
        assert captured_paths[0] is None


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
