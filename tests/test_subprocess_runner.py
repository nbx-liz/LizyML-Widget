"""Tests for subprocess_runner and _subprocess_entry (TDD — RED phase).

These tests verify the subprocess execution pipeline without requiring
LizyML to be installed. They use mock adapters and controlled data.
"""

from __future__ import annotations

import os
import pickle
import signal
import struct
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget._subprocess_entry import (
    read_input,
    run_job,
    send_message,
)
from lizyml_widget.subprocess_runner import (
    SubprocessJobResult,
    run_job_subprocess,
)

# ---------------------------------------------------------------------------
# Helpers: length-prefixed pickle protocol
# ---------------------------------------------------------------------------


def encode_message(msg: dict[str, Any]) -> bytes:
    """Encode a message as length-prefixed pickle (same as send_message)."""
    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack(">I", len(payload)) + payload


def decode_messages(data: bytes) -> list[dict[str, Any]]:
    """Decode all length-prefixed pickle messages from bytes."""
    messages = []
    offset = 0
    while offset < len(data):
        if offset + 4 > len(data):
            break
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4
        payload = data[offset : offset + length]
        offset += length
        messages.append(pickle.loads(payload))  # noqa: S301
    return messages


# ===========================================================================
# _subprocess_entry: send_message
# ===========================================================================


class TestSendMessage:
    """Test the length-prefixed pickle message encoding."""

    def test_roundtrip(self) -> None:
        """Message can be encoded and decoded back."""
        import io

        buf = io.BytesIO()
        msg = {"type": "progress", "current": 3, "total": 5, "message": "Fold 3/5"}
        send_message(buf, msg)
        raw = buf.getvalue()
        decoded = decode_messages(raw)
        assert len(decoded) == 1
        assert decoded[0] == msg

    def test_multiple_messages(self) -> None:
        """Multiple messages can be written and decoded sequentially."""
        import io

        buf = io.BytesIO()
        msgs = [
            {"type": "progress", "current": 1, "total": 5, "message": "Fold 1/5"},
            {"type": "progress", "current": 2, "total": 5, "message": "Fold 2/5"},
            {"type": "result", "summary": {"fold_count": 5}},
        ]
        for m in msgs:
            send_message(buf, m)
        decoded = decode_messages(buf.getvalue())
        assert decoded == msgs


# ===========================================================================
# _subprocess_entry: read_input
# ===========================================================================


class TestReadInput:
    """Test input deserialization."""

    def test_reads_pickled_input(self) -> None:
        """Reads and unpickles the input dict from a stream."""
        import io

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        input_data = {
            "job_type": "fit",
            "config": {"model": {"name": "lgbm"}},
            "df_bytes": pickle.dumps(df),
            "target": "b",
            "model_out_path": "/tmp/model.txt",
        }
        buf = io.BytesIO(pickle.dumps(input_data))
        result = read_input(buf)
        assert result["job_type"] == "fit"
        assert result["target"] == "b"
        assert isinstance(pickle.loads(result["df_bytes"]), pd.DataFrame)  # noqa: S301


# ===========================================================================
# _subprocess_entry: run_job
# ===========================================================================


class TestRunJob:
    """Test the main job execution function."""

    def test_fit_sends_result(self) -> None:
        """Successful fit sends progress and result messages."""
        import io

        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        output = io.BytesIO()

        mock_summary = MagicMock()
        mock_summary.metrics = {"auc": {"oos": 0.95}}
        mock_summary.fold_count = 2
        mock_summary.params = []

        mock_adapter = MagicMock()
        mock_adapter.fit.return_value = mock_summary
        mock_adapter.evaluate_table.return_value = []
        mock_adapter.split_summary.return_value = []
        mock_adapter.available_plots.return_value = ["learning_curve"]
        mock_adapter.export_model.return_value = "/tmp/model.txt"

        with patch(
            "lizyml_widget._subprocess_entry._create_adapter",
            return_value=mock_adapter,
        ):
            run_job(
                job_type="fit",
                config={"model": {"name": "lgbm"}},
                df=df,
                target="y",
                model_out_path="/tmp/model.txt",
                output=output,
            )

        messages = decode_messages(output.getvalue())
        types = [m["type"] for m in messages]
        assert "result" in types
        result_msg = next(m for m in messages if m["type"] == "result")
        assert "summary" in result_msg
        assert result_msg["available_plots"] == ["learning_curve"]

    def test_tune_sends_result(self) -> None:
        """Successful tune sends result with tuning summary."""
        import io

        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        output = io.BytesIO()

        mock_summary = MagicMock()
        mock_summary.best_params = {"lr": 0.1}
        mock_summary.best_score = 0.98
        mock_summary.trials = []
        mock_summary.metric_name = "auc"
        mock_summary.direction = "maximize"

        mock_adapter = MagicMock()
        mock_adapter.tune.return_value = mock_summary
        mock_adapter.evaluate_table.return_value = []
        mock_adapter.split_summary.return_value = []
        mock_adapter.available_plots.return_value = []
        mock_adapter.export_model.return_value = "/tmp/model.txt"

        with patch(
            "lizyml_widget._subprocess_entry._create_adapter",
            return_value=mock_adapter,
        ):
            run_job(
                job_type="tune",
                config={"model": {"name": "lgbm"}},
                df=df,
                target="y",
                model_out_path="/tmp/model.txt",
                output=output,
            )

        messages = decode_messages(output.getvalue())
        result_msg = next(m for m in messages if m["type"] == "result")
        assert "tune_summary" in result_msg

    def test_error_sends_error_message(self) -> None:
        """Exception during fit sends error message."""
        import io

        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        output = io.BytesIO()

        mock_adapter = MagicMock()
        mock_adapter.fit.side_effect = RuntimeError("LightGBM crashed")

        with patch(
            "lizyml_widget._subprocess_entry._create_adapter",
            return_value=mock_adapter,
        ):
            run_job(
                job_type="fit",
                config={},
                df=df,
                target="y",
                model_out_path=None,
                output=output,
            )

        messages = decode_messages(output.getvalue())
        error_msg = next(m for m in messages if m["type"] == "error")
        assert error_msg["exc_type"] == "RuntimeError"
        assert "crashed" in error_msg["message"]
        assert "traceback" in error_msg

    def test_progress_forwarded(self) -> None:
        """on_progress callback sends progress messages to output."""
        import io

        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        output = io.BytesIO()

        def fake_fit(model: Any, *, on_progress: Any = None) -> MagicMock:
            if on_progress:
                on_progress(1, 5, "Fold 1/5")
                on_progress(2, 5, "Fold 2/5")
            s = MagicMock()
            s.metrics = {}
            s.fold_count = 5
            s.params = []
            return s

        mock_adapter = MagicMock()
        mock_adapter.fit.side_effect = fake_fit
        mock_adapter.evaluate_table.return_value = []
        mock_adapter.split_summary.return_value = []
        mock_adapter.available_plots.return_value = []
        mock_adapter.export_model.return_value = "/tmp/m.txt"

        with patch(
            "lizyml_widget._subprocess_entry._create_adapter",
            return_value=mock_adapter,
        ):
            run_job(
                job_type="fit",
                config={},
                df=df,
                target="y",
                model_out_path="/tmp/m.txt",
                output=output,
            )

        messages = decode_messages(output.getvalue())
        progress_msgs = [m for m in messages if m["type"] == "progress"]
        assert len(progress_msgs) >= 2
        assert progress_msgs[0]["current"] == 1
        assert progress_msgs[1]["current"] == 2


# ===========================================================================
# subprocess_runner: SubprocessJobResult
# ===========================================================================


class TestSubprocessJobResult:
    """Test the result dataclass."""

    def test_creation(self) -> None:
        """SubprocessJobResult can be created with all fields."""
        result = SubprocessJobResult(
            job_type="fit",
            fit_summary={"fold_count": 5, "metrics": {}},
            tune_summary={},
            eval_table=[],
            split_summary=[],
            available_plots=["learning_curve"],
            model_path="/tmp/model.txt",
        )
        assert result.job_type == "fit"
        assert result.available_plots == ["learning_curve"]


# ===========================================================================
# subprocess_runner: run_job_subprocess
# ===========================================================================


class TestRunJobSubprocess:
    """Test the subprocess runner."""

    def test_successful_fit(self) -> None:
        """Fit via subprocess returns correct result."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})

        # Build fake subprocess output
        result_msg = encode_message(
            {
                "type": "result",
                "summary": {"fold_count": 5, "metrics": {"auc": 0.95}},
                "eval_table": [],
                "split_summary": [],
                "available_plots": ["roc_curve"],
                "model_path": None,
            }
        )

        with patch("lizyml_widget.subprocess_runner.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.stdout = MagicMock()
            proc.stdout.read.side_effect = [result_msg[:4], result_msg[4:], b""]
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.pid = 12345
            mock_popen.return_value = proc

            result = run_job_subprocess(
                job_type="fit",
                config={"model": {"name": "lgbm"}},
                df=df,
                target="y",
                libomp_path=None,
                on_progress=None,
                cancel_flag=threading.Event(),
            )

        assert isinstance(result, SubprocessJobResult)
        assert result.job_type == "fit"
        assert result.fit_summary["fold_count"] == 5

    def test_ld_preload_set_when_libomp_provided(self) -> None:
        """When libomp_path is given, LD_PRELOAD is set in subprocess env."""
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        result_msg = encode_message(
            {
                "type": "result",
                "summary": {},
                "eval_table": [],
                "split_summary": [],
                "available_plots": [],
                "model_path": None,
            }
        )

        captured_env: dict[str, str] = {}

        with patch("lizyml_widget.subprocess_runner.subprocess.Popen") as mock_popen:

            def capture_popen(*args: Any, **kwargs: Any) -> MagicMock:
                captured_env.update(kwargs.get("env", {}))
                proc = MagicMock()
                proc.stdout = MagicMock()
                proc.stdout.read.side_effect = [result_msg[:4], result_msg[4:], b""]
                proc.stderr = MagicMock()
                proc.stderr.read.return_value = b""
                proc.wait.return_value = 0
                proc.returncode = 0
                proc.pid = 12345
                return proc

            mock_popen.side_effect = capture_popen

            run_job_subprocess(
                job_type="fit",
                config={},
                df=df,
                target="y",
                libomp_path="/usr/lib/libomp5.so",
                on_progress=None,
                cancel_flag=threading.Event(),
            )

        assert "LD_PRELOAD" in captured_env
        assert "/usr/lib/libomp5.so" in captured_env["LD_PRELOAD"]

    def test_ld_preload_not_set_when_no_libomp(self) -> None:
        """When libomp_path is None, LD_PRELOAD is not modified."""
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        result_msg = encode_message(
            {
                "type": "result",
                "summary": {},
                "eval_table": [],
                "split_summary": [],
                "available_plots": [],
                "model_path": None,
            }
        )

        captured_env: dict[str, str] = {}

        with patch("lizyml_widget.subprocess_runner.subprocess.Popen") as mock_popen:

            def capture_popen(*args: Any, **kwargs: Any) -> MagicMock:
                captured_env.update(kwargs.get("env", {}))
                proc = MagicMock()
                proc.stdout = MagicMock()
                proc.stdout.read.side_effect = [result_msg[:4], result_msg[4:], b""]
                proc.stderr = MagicMock()
                proc.stderr.read.return_value = b""
                proc.wait.return_value = 0
                proc.returncode = 0
                proc.pid = 12345
                return proc

            mock_popen.side_effect = capture_popen

            original_ld = os.environ.get("LD_PRELOAD", "")
            run_job_subprocess(
                job_type="fit",
                config={},
                df=df,
                target="y",
                libomp_path=None,
                on_progress=None,
                cancel_flag=threading.Event(),
            )

        # LD_PRELOAD should be same as original (not modified)
        assert captured_env.get("LD_PRELOAD", "") == original_ld

    def test_progress_callback_invoked(self) -> None:
        """Progress messages from subprocess are forwarded to callback."""
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})

        progress_msg = encode_message(
            {"type": "progress", "current": 3, "total": 5, "message": "Fold 3"}
        )
        result_msg = encode_message(
            {
                "type": "result",
                "summary": {},
                "eval_table": [],
                "split_summary": [],
                "available_plots": [],
                "model_path": None,
            }
        )
        all_output = progress_msg + result_msg

        received: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, message: str) -> None:
            received.append((current, total, message))

        with patch("lizyml_widget.subprocess_runner.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.stdout = MagicMock()
            # Return all output then empty
            proc.stdout.read.side_effect = [
                all_output[:4],
                all_output[4 : len(progress_msg)],  # progress header+body
                all_output[len(progress_msg) : len(progress_msg) + 4],  # result header
                all_output[len(progress_msg) + 4 :],  # result body
                b"",
            ]
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.pid = 12345
            mock_popen.return_value = proc

            run_job_subprocess(
                job_type="fit",
                config={},
                df=df,
                target="y",
                libomp_path=None,
                on_progress=on_progress,
                cancel_flag=threading.Event(),
            )

        assert len(received) == 1
        assert received[0] == (3, 5, "Fold 3")

    def test_subprocess_error_raises(self) -> None:
        """Error message from subprocess raises RuntimeError."""
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        error_msg = encode_message(
            {
                "type": "error",
                "exc_type": "RuntimeError",
                "message": "LightGBM crashed",
                "traceback": "Traceback...",
            }
        )

        with (
            patch("lizyml_widget.subprocess_runner.subprocess.Popen") as mock_popen,
            pytest.raises(RuntimeError, match="LightGBM crashed"),
        ):
            proc = MagicMock()
            proc.stdout = MagicMock()
            proc.stdout.read.side_effect = [error_msg[:4], error_msg[4:], b""]
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b""
            proc.wait.return_value = 1
            proc.returncode = 1
            proc.pid = 12345
            mock_popen.return_value = proc

            run_job_subprocess(
                job_type="fit",
                config={},
                df=df,
                target="y",
                libomp_path=None,
                on_progress=None,
                cancel_flag=threading.Event(),
            )

    def test_cancellation_sends_sigterm(self) -> None:
        """Setting cancel_flag sends SIGTERM to subprocess."""
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        cancel = threading.Event()
        sigterm_sent = threading.Event()

        with patch("lizyml_widget.subprocess_runner.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 99999
            proc.poll.return_value = None  # Process is "running"
            proc.stdout = MagicMock()

            def mock_send_signal(sig: int) -> None:
                sigterm_sent.set()

            proc.send_signal.side_effect = mock_send_signal

            # Simulate a slow subprocess: block on first read until cancelled
            def slow_read(n: int) -> bytes:
                cancel.wait(timeout=5)
                # After cancel, wait for SIGTERM to be sent
                sigterm_sent.wait(timeout=2)
                error_msg = encode_message(
                    {
                        "type": "error",
                        "exc_type": "InterruptedError",
                        "message": "Cancelled",
                        "traceback": "",
                    }
                )
                if n == 4:
                    return error_msg[:4]
                return error_msg[4:]

            call_count = 0

            def counting_read(n: int) -> bytes:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return slow_read(n)
                return b""

            proc.stdout.read.side_effect = counting_read
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b""
            proc.wait.return_value = -signal.SIGTERM
            proc.returncode = -signal.SIGTERM
            mock_popen.return_value = proc

            # Cancel after a short delay
            def cancel_later() -> None:
                time.sleep(0.1)
                cancel.set()

            t = threading.Thread(target=cancel_later, daemon=True)
            t.start()

            with pytest.raises((InterruptedError, RuntimeError)):
                run_job_subprocess(
                    job_type="fit",
                    config={},
                    df=df,
                    target="y",
                    libomp_path=None,
                    on_progress=None,
                    cancel_flag=cancel,
                )

            # Verify SIGTERM was sent
            assert sigterm_sent.is_set()
