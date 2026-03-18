"""Tests for LizyWidget job execution, tune, inference, and plots."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo, PlotData, PredictionSummary, TuningSummary


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


class TestTuneDefaults:
    """Regression tests for tuning default complement (P-004 R1)."""

    def test_tune_complements_missing_tuning_config(self) -> None:
        """R1: load() with target auto-populates tuning defaults; tune() uses them."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # Config should have tuning defaults populated after load()
        tuning = w.config.get("tuning")
        assert tuning is not None, "tuning should be populated after load with target"
        space = (tuning.get("optuna") or {}).get("space", {})
        assert len(space) > 0, "search space should be populated"
        # Trigger tune — should not fail with CONFIG_INVALID due to missing tuning
        w.action = {"type": "tune", "payload": {}}
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)
        # Validation may still fail for other reasons (mock adapter returns []),
        # but specifically NOT "No tuning configuration"
        if w.status == "failed":
            msg = w.error.get("message", "").lower()
            assert "tuning" not in msg or "configuration" not in msg


class TestTuneProgress:
    """Tune sets initial progress with n_trials info."""

    def test_tune_sets_initial_progress(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        def mock_tune(config: Any, *, on_progress: Any = None) -> Any:
            # Capture progress calls and return a mock summary
            return TuningSummary(
                best_params={"lr": 0.01},
                best_score=0.9,
                trials=[],
                metric_name="auc",
                direction="maximize",
            )

        w._service.tune = mock_tune  # type: ignore[assignment]

        # Set config with tuning section
        w.config = {
            **dict(w.config),
            "tuning": {
                "optuna": {"params": {"n_trials": 30}, "space": {}},
            },
        }

        # Run tune — check that progress traitlet is set before tune starts
        w.action = {"type": "tune", "payload": {}}

        # Poll for progress instead of fixed sleep (avoids CI flakiness)
        import time

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.progress.get("total"):
                break
            time.sleep(0.05)

        # Progress should contain n_trials info
        assert w.progress["total"] == 30
        assert "30" in w.progress["message"]


class TestInference:
    def test_load_inference(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [1, 2, 3]})
        w.load_inference(df)
        assert w.inference_result["status"] == "ready"
        assert w.inference_result["rows"] == 3


class TestPredictAndSaveModel:
    """Cover predict() and save_model() (widget.py lines 202-208)."""

    def test_predict_delegates_to_service(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        expected = PredictionSummary(
            predictions=pd.DataFrame({"pred": [0, 1]}),
            warnings=[],
        )
        w._service.predict = MagicMock(return_value=expected)
        result = w.predict(df)
        assert result is expected
        w._service.predict.assert_called_once_with(df, return_shap=False)

    def test_predict_with_shap(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        expected = PredictionSummary(
            predictions=pd.DataFrame({"pred": [0, 1]}),
            warnings=[],
        )
        w._service.predict = MagicMock(return_value=expected)
        w.predict(df, return_shap=True)
        w._service.predict.assert_called_once_with(df, return_shap=True)

    def test_save_model_delegates_to_service(self) -> None:
        w = _make_widget()
        w._service.save_model = MagicMock(return_value="/tmp/model.pkl")
        result = w.save_model("/tmp/model.pkl")
        assert result == "/tmp/model.pkl"
        w._service.save_model.assert_called_once_with("/tmp/model.pkl")


class TestRequestInferencePlot:
    """Cover _handle_request_inference_plot (widget.py lines 418-439)."""

    def test_empty_plot_type_ignored(self) -> None:
        """widget.py line 421: empty plot_type returns early."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": ""}}
        assert len(sent) == 0

    def test_inference_plot_with_data_success(self) -> None:
        """widget.py lines 427-436: inference plot with data."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        # Set inference_result with data
        w.inference_result = {
            "status": "completed",
            "rows": 2,
            "data": [{"pred": 0}, {"pred": 1}],
            "warnings": [],
        }

        plot_data = PlotData(plotly_json='{"data": [], "layout": {}}')
        w._service.get_inference_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": "scatter"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_data"
        assert sent[0]["plot_type"] == "scatter"

    def test_inference_plot_fallback_on_error(self) -> None:
        """widget.py lines 437-439: falls back to fit plot on inference plot error."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.inference_result = {
            "status": "completed",
            "rows": 2,
            "data": [{"pred": 0}, {"pred": 1}],
            "warnings": [],
        }

        w._service.get_inference_plot = MagicMock(side_effect=RuntimeError("no plot"))
        # Fallback calls _handle_request_plot which calls get_plot
        plot_data = PlotData(plotly_json='{"data": []}')
        w._service.get_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": "roc"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_data"

    def test_no_inference_data_falls_back_to_fit_plot(self) -> None:
        """widget.py lines 424-426: no inference data delegates to request_plot."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        plot_data = PlotData(plotly_json='{"data": []}')
        w._service.get_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": "roc"}}
        assert len(sent) == 1


class TestRequestPlotSuccess:
    """Cover _handle_request_plot success path (widget.py lines 325-332)."""

    def test_request_plot_sends_plotly_json(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        plot_data = PlotData(plotly_json='{"data": [], "layout": {}}')
        w._service.get_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_plot", "payload": {"plot_type": "feature_importance"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_data"
        assert sent[0]["plot_type"] == "feature_importance"
        assert sent[0]["plotly_json"] == '{"data": [], "layout": {}}'

    def test_request_plot_error_sends_plot_error(self) -> None:
        """widget.py lines 333-340: plot error path."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.get_plot = MagicMock(side_effect=RuntimeError("no model"))

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_plot", "payload": {"plot_type": "roc"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_error"
        assert "no model" in sent[0]["message"]


class TestRunJobGuard:
    """Cover _run_job guard conditions (widget.py lines 461-477)."""

    def test_run_job_already_running_ignored(self) -> None:
        """widget.py line 463: already running returns early."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.status = "running"
        # Trigger fit action — should be ignored
        w.action = {"type": "fit", "payload": {}}
        # Status should remain running, no error set
        assert w.status == "running"
        assert w.error == {}


class TestRunJobConfigError:
    """_run_job must transition to failed if prepare_run_config raises."""

    def test_config_build_error_transitions_to_failed(self) -> None:
        """Status must be 'failed' (not stuck at 'running') on config error."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.prepare_run_config = MagicMock(side_effect=ValueError("No features available"))

        w._run_job("fit")
        assert w.status == "failed"
        assert w.error.get("code") == "CONFIG_ERROR"
        assert "No features" in w.error.get("message", "")

    def test_config_error_allows_retry(self) -> None:
        """After config error, a subsequent job should be able to run."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        # First call fails
        w._service.prepare_run_config = MagicMock(side_effect=ValueError("bad config"))
        w._run_job("fit")
        assert w.status == "failed"

        # Second call should not be blocked by "running" guard
        w._service.prepare_run_config = MagicMock(side_effect=ValueError("still bad"))
        w._run_job("fit")
        assert w.status == "failed"
        assert "still bad" in w.error.get("message", "")


class TestRunBlockingJob:
    """Cover _run_blocking_job status→exception paths (widget.py lines 119-132)."""

    def test_immediate_completion_returns(self) -> None:
        """widget.py:123-124: job completes synchronously before observer fires."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        def _instant_complete(job_type: str) -> None:
            w.status = "completed"

        w._run_job = _instant_complete  # type: ignore[assignment]
        result = w.fit()
        assert result is w

    def test_failed_status_raises_runtime_error(self) -> None:
        """widget.py:129-131: status=='failed' → RuntimeError with message."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        def _instant_fail(job_type: str) -> None:
            w.error = {"code": "BACKEND_ERROR", "message": "boom"}
            w.status = "failed"

        w._run_job = _instant_fail  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="boom"):
            w.fit()

    def test_failed_status_default_message(self) -> None:
        """widget.py:130: fallback message when error dict has no 'message'."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        def _instant_fail(job_type: str) -> None:
            w.error = {"code": "INTERNAL_ERROR"}
            w.status = "failed"

        w._run_job = _instant_fail  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="Fit failed"):
            w.fit()


class TestJobWorkerErrorClassification:
    """Cover BACKEND_ERROR vs INTERNAL_ERROR classification (widget.py:635-639)."""

    def test_lizyml_exception_gets_backend_error(self) -> None:
        """Exceptions from lizyml module → BACKEND_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        class FakeLizyMLError(Exception):
            pass

        FakeLizyMLError.__module__ = "lizyml.core.exceptions"

        w._service.fit = MagicMock(side_effect=FakeLizyMLError("backend crash"))
        w._service.validate_config = MagicMock(return_value=[])

        w._run_job("fit")
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.error["code"] == "BACKEND_ERROR"
        assert "backend crash" in w.error["message"]

    def test_generic_exception_gets_internal_error(self) -> None:
        """Non-lizyml exceptions → INTERNAL_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.fit = MagicMock(side_effect=RuntimeError("oops"))
        w._service.validate_config = MagicMock(return_value=[])

        w._run_job("fit")
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.error["code"] == "INTERNAL_ERROR"
        assert "oops" in w.error["message"]

    def test_exception_with_none_module_gets_internal_error(self) -> None:
        """Exception with __module__=None → INTERNAL_ERROR (the `or ''` fallback)."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        class WeirdError(Exception):
            pass

        WeirdError.__module__ = None  # type: ignore[assignment]

        w._service.fit = MagicMock(side_effect=WeirdError("weird"))
        w._service.validate_config = MagicMock(return_value=[])

        w._run_job("fit")
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.error["code"] == "INTERNAL_ERROR"


class TestBlockingHelperTimeout:
    """fit()/tune() must raise TimeoutError when timeout expires."""

    def test_fit_timeout_raises(self) -> None:
        """fit(timeout=...) raises TimeoutError if job doesn't complete."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        def _stuck_job(job_type: str) -> None:
            w.status = "running"

        w._run_job = _stuck_job  # type: ignore[assignment]

        with pytest.raises(TimeoutError):
            w.fit(timeout=0.1)

    def test_tune_timeout_raises(self) -> None:
        """tune(timeout=...) raises TimeoutError if job doesn't complete."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        def _stuck_job(job_type: str) -> None:
            w.status = "running"

        w._run_job = _stuck_job  # type: ignore[assignment]

        with pytest.raises(TimeoutError):
            w.tune(timeout=0.1)
