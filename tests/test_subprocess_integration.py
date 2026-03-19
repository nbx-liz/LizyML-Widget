"""Tests for subprocess execution strategy integration (Phase 4).

Tests Widget + Service integration with subprocess path.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.subprocess_runner import SubprocessJobResult
from lizyml_widget.types import BackendInfo

# Force subprocess strategy for tests in this module
_force_subprocess = patch.dict(os.environ, {"LZW_FORCE_SUBPROCESS": "1"})


def _make_widget(**kwargs: Any) -> Any:
    """Create a LizyWidget with mocked backend and controllable strategy."""
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

        w = LizyWidget(**kwargs)
    return w


# ===========================================================================
# Service: get_dataframe / load_model_from_path
# ===========================================================================


class TestServiceSubprocessHelpers:
    """Test WidgetService helper methods for subprocess data transfer."""

    def test_get_dataframe_returns_loaded_df(self) -> None:
        """get_dataframe returns the DataFrame after load."""
        from lizyml_widget.adapter import LizyMLAdapter
        from lizyml_widget.service import WidgetService

        svc = WidgetService(adapter=LizyMLAdapter())
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        svc.load_data(df, target="y")
        result = svc.get_dataframe()
        assert result is not None
        assert len(result) == 3

    def test_get_dataframe_raises_when_no_data(self) -> None:
        """get_dataframe raises ValueError when no data loaded."""
        from lizyml_widget.adapter import LizyMLAdapter
        from lizyml_widget.service import WidgetService

        svc = WidgetService(adapter=LizyMLAdapter())
        with pytest.raises(ValueError, match="No data loaded"):
            svc.get_dataframe()

    def test_load_model_from_path(self) -> None:
        """load_model_from_path delegates to adapter.load_model."""
        mock_adapter = MagicMock()
        mock_model = MagicMock()
        mock_adapter.load_model.return_value = mock_model

        from lizyml_widget.service import WidgetService

        svc = WidgetService(adapter=mock_adapter)
        svc.load_model_from_path("/tmp/model.txt")

        mock_adapter.load_model.assert_called_once_with("/tmp/model.txt")


# ===========================================================================
# Widget: execution strategy detection
# ===========================================================================


class TestWidgetExecutionStrategy:
    """Test that Widget detects execution strategy lazily at _run_job time."""

    def test_strategy_none_at_init(self) -> None:
        """Strategy is None at init (lazy detection)."""
        w = _make_widget()
        assert w._execution_strategy is None

    def test_strategy_detected_at_run_job(self) -> None:
        """Strategy is detected when _run_job is called with LZW_FORCE_SUBPROCESS."""
        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", "/usr/lib/libomp5.so"),
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")

            with patch.object(w, "_subprocess_job_worker"):
                w._run_job("fit")
                if w._job_thread:
                    w._job_thread.join(timeout=2)

            assert w._execution_strategy == "subprocess"
            assert w._libomp_path == "/usr/lib/libomp5.so"


# ===========================================================================
# Widget: _run_job branching
# ===========================================================================


class TestRunJobBranching:
    """Test that _run_job delegates to correct worker based on strategy."""

    def test_thread_strategy_uses_job_worker(self) -> None:
        """thread strategy calls _job_worker."""
        with patch(
            "lizyml_widget.widget.get_execution_strategy",
            return_value=("thread", None),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")

            with patch.object(w, "_job_worker") as mock_worker:
                w._run_job("fit")
                # Give thread time to start
                if w._job_thread:
                    w._job_thread.join(timeout=2)
                mock_worker.assert_called_once()

    def test_subprocess_strategy_uses_subprocess_worker(self) -> None:
        """subprocess strategy calls _subprocess_job_worker."""
        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", "/usr/lib/libomp5.so"),
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")

            with patch.object(w, "_subprocess_job_worker") as mock_worker:
                w._run_job("fit")
                if w._job_thread:
                    w._job_thread.join(timeout=2)
                mock_worker.assert_called_once()


# ===========================================================================
# Widget: _subprocess_job_worker
# ===========================================================================


class TestSubprocessJobWorker:
    """Test the subprocess job worker method."""

    def test_fit_updates_traitlets(self) -> None:
        """Subprocess fit updates fit_summary, status, available_plots."""
        mock_result = SubprocessJobResult(
            job_type="fit",
            fit_summary={
                "metrics": {"auc": {"oos": 0.95}},
                "fold_count": 5,
                "params": [],
            },
            tune_summary={},
            eval_table=[{"metric": "auc", "IS": 0.99, "OOS": 0.95, "OOS_std": 0.01}],
            split_summary=[],
            available_plots=["learning_curve", "roc_curve"],
            model_path=None,
        )

        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch(
                "lizyml_widget.widget.run_job_subprocess",
                return_value=mock_result,
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")
            w._run_job("fit")
            if w._job_thread:
                w._job_thread.join(timeout=5)

            assert w.status == "completed"
            assert w.fit_summary["fold_count"] == 5
            assert "learning_curve" in w.available_plots

    def test_tune_updates_traitlets(self) -> None:
        """Subprocess tune updates tune_summary and status."""
        mock_result = SubprocessJobResult(
            job_type="tune",
            fit_summary={},
            tune_summary={
                "best_params": {"lr": 0.1},
                "best_score": 0.98,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
            },
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path=None,
        )

        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch(
                "lizyml_widget.widget.run_job_subprocess",
                return_value=mock_result,
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")
            w._run_job("tune")
            if w._job_thread:
                w._job_thread.join(timeout=5)

            assert w.status == "completed"
            assert w.tune_summary["best_score"] == 0.98

    def test_error_updates_traitlets(self) -> None:
        """Subprocess error sets error traitlet and failed status."""
        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch(
                "lizyml_widget.widget.run_job_subprocess",
                side_effect=RuntimeError("[RuntimeError] LightGBM crashed"),
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")
            w._run_job("fit")
            if w._job_thread:
                w._job_thread.join(timeout=5)

            assert w.status == "failed"
            assert "crashed" in w.error.get("message", "")

    def test_cancellation_updates_traitlets(self) -> None:
        """InterruptedError from subprocess sets cancelled status."""
        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch(
                "lizyml_widget.widget.run_job_subprocess",
                side_effect=InterruptedError("Job cancelled"),
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")
            w._run_job("fit")
            if w._job_thread:
                w._job_thread.join(timeout=5)

            assert w.status == "failed"
            assert w.error["code"] == "CANCELLED"

    def test_model_loaded_from_path(self) -> None:
        """When model_path is returned, Service loads the model."""
        mock_result = SubprocessJobResult(
            job_type="fit",
            fit_summary={"metrics": {}, "fold_count": 1, "params": []},
            tune_summary={},
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path="/tmp/model.txt",
        )

        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch(
                "lizyml_widget.widget.run_job_subprocess",
                return_value=mock_result,
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")

            with patch.object(w._service, "load_model_from_path") as mock_load:
                w._run_job("fit")
                if w._job_thread:
                    w._job_thread.join(timeout=5)
                mock_load.assert_called_once_with("/tmp/model.txt")

    def test_poll_handler_works_with_subprocess(self) -> None:
        """Colab polling handler reads traitlets updated by subprocess worker."""
        mock_result = SubprocessJobResult(
            job_type="fit",
            fit_summary={"metrics": {}, "fold_count": 1, "params": []},
            tune_summary={},
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path=None,
        )

        with (
            _force_subprocess,
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch(
                "lizyml_widget.widget.run_job_subprocess",
                return_value=mock_result,
            ),
        ):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")
            w._run_job("fit")
            if w._job_thread:
                w._job_thread.join(timeout=5)

            # Simulate poll request — should read the updated traitlets
            sent: list[dict[str, Any]] = []
            w.send = lambda data: sent.append(data)  # type: ignore[assignment]
            w._handle_custom_msg({"type": "poll"}, [])

            assert len(sent) == 1
            assert sent[0]["status"] == "completed"
