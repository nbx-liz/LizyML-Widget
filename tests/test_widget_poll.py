"""Tests for LizyWidget Colab polling handler (_handle_custom_msg)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo, FitSummary, TuningSummary


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


class TestCustomMsgOverride:
    """_handle_custom_msg overrides ipywidgets base method."""

    def test_method_exists(self) -> None:
        """_handle_custom_msg is defined on LizyWidget."""
        w = _make_widget()
        assert hasattr(w, "_handle_custom_msg")
        assert callable(w._handle_custom_msg)


class TestPollHandlerIdle:
    """Poll response in idle / data_loaded states."""

    def test_poll_idle_returns_job_state(self) -> None:
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"type": "poll"}, [])

        assert len(sent) == 1
        state = sent[0]
        assert state["type"] == "job_state"
        assert state["status"] == "idle"
        assert "progress" in state
        assert "elapsed_sec" in state
        assert "error" in state
        # idle: no result payloads
        assert "fit_summary" not in state
        assert "tune_summary" not in state

    def test_poll_data_loaded(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["status"] == "data_loaded"


class TestPollHandlerRunning:
    """Poll response during running state."""

    def test_poll_running_returns_progress(self) -> None:
        w = _make_widget()
        # Simulate running state
        w.status = "running"
        w.job_type = "fit"
        w.job_index = 1
        w.progress = {"current": 3, "total": 5, "message": "Fold 3/5"}
        w.elapsed_sec = 2.5

        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["status"] == "running"
        assert state["job_type"] == "fit"
        assert state["job_index"] == 1
        assert state["progress"]["current"] == 3
        assert state["elapsed_sec"] == 2.5
        # running: no result payloads
        assert "fit_summary" not in state


class TestPollHandlerCompleted:
    """Poll response after job completion."""

    def test_poll_completed_includes_results(self) -> None:
        w = _make_widget()
        w.status = "completed"
        w.job_type = "fit"
        w.job_index = 1
        w.fit_summary = {"metrics": {"auc": {"oos": 0.95}}, "fold_count": 5, "params": []}
        w.tune_summary = {}
        w.available_plots = ["learning_curve", "roc_curve"]

        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["status"] == "completed"
        assert state["fit_summary"]["fold_count"] == 5
        assert "learning_curve" in state["available_plots"]

    def test_poll_failed_includes_error_and_results(self) -> None:
        w = _make_widget()
        w.status = "failed"
        w.error = {"code": "BACKEND_ERROR", "message": "Something went wrong"}
        w.fit_summary = {}
        w.tune_summary = {}
        w.available_plots = []

        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["status"] == "failed"
        assert state["error"]["code"] == "BACKEND_ERROR"
        assert "fit_summary" in state  # included for terminal states


def _make_widget_for_jobs() -> Any:
    """Create a LizyWidget with full mock adapter for running fit/tune jobs."""
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = {"type": "object", "properties": {}}
        adapter.validate_config.return_value = []
        adapter.initialize_config.side_effect = real_adapter.initialize_config
        adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
        adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
        adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
        adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
        adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
        adapter.classify_best_params.side_effect = real_adapter.classify_best_params

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.fit.return_value = FitSummary(
            metrics={"auc": {"is": 0.95, "oos": 0.90}},
            fold_count=5,
            params=[{"index": "n_estimators", "value": 100}],
        )
        adapter.tune.return_value = TuningSummary(
            best_params={"learning_rate": 0.01},
            best_score=0.92,
            trials=[],
            metric_name="auc",
            direction="maximize",
        )
        adapter.evaluate_table.return_value = [{"index": "auc", "if_mean": 0.95, "oof": 0.90}]
        adapter.split_summary.return_value = [{"fold": 0, "n_train": 40, "n_valid": 10}]
        adapter.available_plots.return_value = ["learning-curve"]

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


class TestPollStateTransitions:
    """Poll responses across job lifecycle — regression tests for A-1 and the 'frozen UI' bug."""

    def test_consecutive_fit_increments_job_index_in_poll(self) -> None:
        """After Fit→Fit, poll must return the second job's index."""
        w = _make_widget_for_jobs()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        # First fit (via msg:custom, same as JS path)
        w._handle_custom_msg({"type": "action", "action_type": "fit", "payload": {}}, [])
        if w._job_thread:
            w._job_thread.join(timeout=30)
        assert w.status == "completed"
        assert w.job_index == 1

        # Second fit
        w._handle_custom_msg({"type": "action", "action_type": "fit", "payload": {}}, [])
        if w._job_thread:
            w._job_thread.join(timeout=30)
        assert w.status == "completed"
        assert w.job_index == 2

        # Poll after second fit must reflect job_index=2
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))
        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["job_index"] == 2
        assert state["status"] == "completed"
        assert "fit_summary" in state

    def test_tune_then_fit_poll_returns_fit_results(self) -> None:
        """After Tune→Apply→Fit, poll must return fit (not tune) results."""
        w = _make_widget_for_jobs()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        # Tune (via msg:custom)
        w._handle_custom_msg({"type": "action", "action_type": "tune", "payload": {}}, [])
        if w._job_thread:
            w._job_thread.join(timeout=60)
        assert w.status == "completed"
        tune_best = w.tune_summary.get("best_params", {})

        # Apply best params + Fit
        w._handle_apply_best_params({"params": tune_best})
        w._handle_custom_msg({"type": "action", "action_type": "fit", "payload": {}}, [])
        if w._job_thread:
            w._job_thread.join(timeout=30)
        assert w.status == "completed"

        # Poll: fit_summary should reflect the Fit, not be empty
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))
        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["status"] == "completed"
        assert state["job_index"] == 2
        # fit_summary should have metrics from the Fit
        fit_s = state.get("fit_summary", {})
        assert fit_s.get("fold_count", 0) > 0, "fit_summary should have results from the Fit run"

    def test_poll_during_running_excludes_results(self) -> None:
        """Poll while status=running should NOT include fit_summary/tune_summary."""
        w = _make_widget()
        w.status = "running"
        w.job_type = "fit"
        w.job_index = 1
        # Stale results from a previous run
        w.fit_summary = {"metrics": {"auc": {"oos": 0.9}}, "fold_count": 5, "params": []}

        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))
        w._handle_custom_msg({"type": "poll"}, [])

        state = sent[0]
        assert state["status"] == "running"
        # Running state should NOT include result payloads
        assert "fit_summary" not in state
        assert "tune_summary" not in state


class TestPollHandlerIgnoresOtherTypes:
    """Non-poll messages are ignored."""

    def test_non_poll_message_ignored(self) -> None:
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"type": "other"}, [])
        assert len(sent) == 0

    def test_empty_content_ignored(self) -> None:
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({}, [])
        assert len(sent) == 0

    def test_missing_type_ignored(self) -> None:
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = MagicMock(side_effect=lambda msg: sent.append(msg))

        w._handle_custom_msg({"data": "something"}, [])
        assert len(sent) == 0
