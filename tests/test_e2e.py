"""End-to-end integration tests with mocked backend adapter."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import (
    BackendInfo,
    FitSummary,
    PredictionSummary,
    TuningSummary,
)


def _make_widget_with_adapter() -> tuple[Any, MagicMock]:
    """Create a LizyWidget with a controllable mock adapter."""
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = {"type": "object", "properties": {}}
        adapter.validate_config.return_value = []
        # Delegate config lifecycle to real adapter
        adapter.initialize_config.side_effect = real_adapter.initialize_config
        adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
        adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
        adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
        adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w, adapter


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})


class TestFitE2E:
    """Full fit workflow: load → fit → completed → get_fit_summary."""

    def test_fit_workflow(self) -> None:
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.fit.return_value = FitSummary(
            metrics={"auc": {"is": 0.95, "oos": 0.90}},
            fold_count=5,
            params=[{"index": "n_estimators", "value": 100}],
        )
        adapter.evaluate_table.return_value = [
            {"index": "auc", "if_mean": 0.95, "oof": 0.90, "fold_0": 0.88, "fold_1": 0.92}
        ]
        adapter.split_summary.return_value = [{"fold": 0, "n_train": 40, "n_valid": 10}]
        adapter.available_plots.return_value = ["learning-curve", "roc-curve"]

        w.load(df, target="y")
        assert w.status == "data_loaded"
        assert w.config["model"]["name"] == "lgbm"

        # Trigger fit via action
        w.action = {"type": "fit", "payload": {}}

        # Wait for background thread to complete
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.status == "completed"
        assert w.fit_summary["fold_count"] == 5
        assert len(w.available_plots) > 0


class TestTuneE2E:
    """Full tune workflow: load → tune → completed → get_tune_summary."""

    def test_tune_workflow(self) -> None:
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.tune.return_value = TuningSummary(
            best_params={"learning_rate": 0.01},
            best_score=0.92,
            trials=[],
            metric_name="auc",
            direction="maximize",
        )
        adapter.evaluate_table.return_value = []
        adapter.split_summary.return_value = []
        adapter.available_plots.return_value = ["optimization-history"]

        w.load(df, target="y")
        w.action = {"type": "tune", "payload": {}}

        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.status == "completed"
        assert w.tune_summary["best_score"] == 0.92
        assert w.tune_summary["best_params"]["learning_rate"] == 0.01


class TestInferenceE2E:
    """Inference workflow: load → fit → load_inference → run_inference."""

    def test_inference_workflow(self) -> None:
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.fit.return_value = FitSummary(
            metrics={"auc": {"is": 0.95}}, fold_count=1, params=[]
        )
        adapter.evaluate_table.return_value = []
        adapter.split_summary.return_value = []
        adapter.available_plots.return_value = []
        adapter.predict.return_value = PredictionSummary(
            predictions=pd.DataFrame({"pred": [0, 1, 0, 1]}),
            warnings=[],
        )

        w.load(df, target="y")
        w.action = {"type": "fit", "payload": {}}
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.status == "completed"

        # Load inference data
        test_df = pd.DataFrame({"x": [10, 20, 30, 40]})
        w.load_inference(test_df)

        # Run inference
        w.action = {"type": "run_inference", "payload": {"return_shap": False}}
        assert w.inference_result["status"] == "completed"
        assert w.inference_result["rows"] == 4


class TestErrorFlows:
    """Test error handling paths."""

    def test_no_data_error(self) -> None:
        w, _ = _make_widget_with_adapter()
        w.action = {"type": "fit", "payload": {}}
        assert w.error["code"] == "NO_DATA"
        assert w.status == "failed"

    def test_no_target_error(self) -> None:
        w, _ = _make_widget_with_adapter()
        w.load(_sample_df())  # no target
        w.action = {"type": "fit", "payload": {}}
        assert w.error["code"] == "NO_TARGET"
        assert w.status == "failed"

    def test_validation_error(self) -> None:
        w, adapter = _make_widget_with_adapter()
        adapter.validate_config.return_value = [
            {"field": "model.name", "message": "Missing required field"}
        ]
        w.load(_sample_df(), target="y")
        w.action = {"type": "fit", "payload": {}}

        # Validation happens synchronously before thread spawn
        assert w.error["code"] == "VALIDATION_ERROR"
        assert w.status == "failed"

    def test_backend_error(self) -> None:
        w, adapter = _make_widget_with_adapter()
        adapter.validate_config.return_value = []
        adapter.create_model.side_effect = RuntimeError("Backend crashed")

        w.load(_sample_df(), target="y")
        w.action = {"type": "fit", "payload": {}}

        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.status == "failed"
        assert w.error["code"] in ("BACKEND_ERROR", "INTERNAL_ERROR")
        assert "Backend crashed" in w.error["message"]


class TestTuneOnlyE2E:
    """P-004: Tune-only execution tests (R3/R4)."""

    def test_tune_succeeds_when_evaluate_table_raises(self) -> None:
        """R3: Tune should complete even if evaluate_table raises (MODEL_NOT_FIT)."""
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.tune.return_value = TuningSummary(
            best_params={"learning_rate": 0.01},
            best_score=0.92,
            trials=[],
            metric_name="auc",
            direction="maximize",
        )
        adapter.evaluate_table.side_effect = RuntimeError("Model has not been fitted")
        adapter.available_plots.return_value = ["optimization-history"]

        w.load(df, target="y")
        w.action = {"type": "tune", "payload": {}}

        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.status == "completed"
        assert w.tune_summary["best_score"] == 0.92
        # fit_summary should remain empty since evaluate_table failed
        assert w.fit_summary == {} or w.fit_summary.get("metrics") is None

    def test_tune_with_empty_space_completes(self) -> None:
        """R1: Tune with default (empty) tuning config should complete."""
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.tune.return_value = TuningSummary(
            best_params={"lr": 0.05},
            best_score=0.88,
            trials=[],
            metric_name="auc",
            direction="maximize",
        )
        adapter.evaluate_table.return_value = []
        adapter.split_summary.return_value = []
        adapter.available_plots.return_value = ["optimization-history"]

        w.load(df, target="y")
        # Ensure no tuning config is set
        cfg = w.get_config()
        cfg.pop("tuning", None)
        w.set_config(cfg)

        w.action = {"type": "tune", "payload": {}}

        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)

        assert w.status == "completed"
        assert w.tune_summary["best_score"] == 0.88


class TestContractE2E:
    """Phase 22: Full-flow contract validation tests."""

    def test_fit_with_correct_contract_completes(self) -> None:
        """Normal flow: valid config → fit completes."""
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.fit.return_value = FitSummary(metrics={"auc": {"is": 0.9}}, fold_count=5, params=[])
        adapter.evaluate_table.return_value = []
        adapter.split_summary.return_value = []
        adapter.available_plots.return_value = []

        w.load(df, target="y")
        w.action = {"type": "fit", "payload": {}}
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)
        assert w.status == "completed"

    def test_tune_with_invalid_space_type_fails(self) -> None:
        """Invalid space type value should cause VALIDATION_ERROR."""
        w, adapter = _make_widget_with_adapter()
        df = _sample_df()

        adapter.validate_config.return_value = [
            {
                "field": "tuning.optuna.space.lr",
                "message": "Invalid type 'range'",
                "type": "invalid_space_type",
            }
        ]

        w.load(df, target="y")
        w.set_config(
            {
                **w.get_config(),
                "tuning": {
                    "optuna": {
                        "params": {"n_trials": 50},
                        "space": {"lr": {"type": "range", "low": 0.01, "high": 0.1}},
                    }
                },
            }
        )
        w.action = {"type": "tune", "payload": {}}
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)
        assert w.error.get("code") == "VALIDATION_ERROR"
        assert w.status == "failed"


class TestCancelFlow:
    """Test job cancellation."""

    def test_cancel_during_fit(self) -> None:
        w, adapter = _make_widget_with_adapter()

        # Make fit block until cancelled
        cancel_event = threading.Event()

        def slow_fit(*args: Any, **kwargs: Any) -> FitSummary:
            cancel_event.wait(timeout=5.0)
            # Check if on_progress raises InterruptedError
            on_progress = kwargs.get("on_progress")
            if on_progress:
                on_progress(1, 5, "Fold 1/5")
            return FitSummary(metrics={}, fold_count=1, params=[])

        mock_model = MagicMock()
        adapter.create_model.return_value = mock_model
        adapter.validate_config.return_value = []
        adapter.fit.side_effect = slow_fit

        w.load(_sample_df(), target="y")
        w.action = {"type": "fit", "payload": {}}

        # Cancel immediately
        w.action = {"type": "cancel", "payload": {}}
        cancel_event.set()

        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)
