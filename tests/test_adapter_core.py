"""Tests for LizyMLAdapter — Core adapter operations."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget import adapter_params
from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import (
    FitSummary,
    PlotData,
    PredictionSummary,
    TuningSummary,
)


@pytest.fixture(autouse=True)
def _reset_eval_metrics_cache() -> None:  # type: ignore[misc]
    """Reset the module-level eval metrics cache between tests."""
    adapter_params._eval_metrics_cache = None
    yield  # type: ignore[misc]
    adapter_params._eval_metrics_cache = None


@dataclass
class FakeFitResult:
    metrics: dict[str, Any]
    splits: Any


@dataclass
class FakeOuter:
    outer: list[int]


@dataclass
class FakeTrial:
    number: int
    params: dict[str, Any]
    score: float
    state: str


@dataclass
class FakeTuningResult:
    best_params: dict[str, Any]
    best_score: float
    trials: list[FakeTrial]
    metric_name: str
    direction: str


@dataclass
class FakePredResult:
    pred: list[float]
    proba: list[float] | None
    warnings: list[str]


@contextlib.contextmanager
def _capture_log(logger: logging.Logger, level: int = logging.WARNING):
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler(level)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class TestInfo:
    @patch.dict("sys.modules", {"lizyml": MagicMock(__version__="1.2.3")})
    def test_info(self) -> None:
        adapter = LizyMLAdapter()
        info = adapter.info
        assert info.name == "lizyml"
        assert info.version == "1.2.3"


class TestConfigContractValidation:
    @pytest.mark.parametrize(
        "config,expected_error_type,expected_field_pattern",
        [
            (
                {
                    "tuning": {
                        "optuna": {
                            "space": {
                                "lr": {"mode": "range", "low": 0.01, "high": 0.1},
                            },
                        },
                    },
                },
                "search_space_format",
                "tuning.optuna.space.lr",
            ),
            (
                {
                    "tuning": {
                        "optuna": {
                            "space": {
                                "lr": {"type": "range", "low": 0.01, "high": 0.1},
                            },
                        },
                    },
                },
                "invalid_space_type",
                "tuning.optuna.space.lr",
            ),
            (
                {
                    "tuning": {
                        "optuna": {
                            "space": {
                                "metric": {"type": "choice", "choices": ["auc"]},
                            },
                        },
                    },
                },
                "invalid_space_type",
                "tuning.optuna.space.metric",
            ),
        ],
        ids=["legacy-mode", "invalid-type-range", "invalid-type-choice"],
    )
    def test_contract_violations(
        self,
        config: dict[str, Any],
        expected_error_type: str,
        expected_field_pattern: str,
    ) -> None:
        adapter = LizyMLAdapter()
        errors = adapter.validate_config(config)
        assert len(errors) >= 1
        assert any(e.get("type") == expected_error_type for e in errors)
        assert any(expected_field_pattern in e.get("field", "") for e in errors)

    @pytest.mark.parametrize(
        "space_entry",
        [
            {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            {"type": "int", "low": 1, "high": 10},
            {"type": "categorical", "choices": ["auc", "binary_logloss"]},
        ],
        ids=["float-range", "int-range", "categorical"],
    )
    def test_contract_compliant_space(self, space_entry: dict[str, Any]) -> None:
        mock_load_config = MagicMock()
        mock_loader = MagicMock(load_config=mock_load_config)
        with patch.dict(
            "sys.modules",
            {
                "lizyml": MagicMock(),
                "lizyml.config": MagicMock(),
                "lizyml.config.loader": mock_loader,
            },
        ):
            adapter = LizyMLAdapter()
            config = {
                "tuning": {
                    "optuna": {
                        "params": {"n_trials": 50},
                        "space": {"param": space_entry},
                    },
                },
            }
            errors = adapter.validate_config(config)
            assert errors == []


class TestFit:
    def test_fit_returns_summary(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.fit.return_value = FakeFitResult(
            metrics={"raw": {"oof": {"rmse": 0.5}}},
            splits=FakeOuter(outer=[1, 2, 3]),
        )
        mock_model.params_table.return_value = pd.DataFrame({"param": ["a", "b"], "value": [1, 2]})

        result = adapter.fit(mock_model)
        assert isinstance(result, FitSummary)
        assert result.metrics == {"raw": {"oof": {"rmse": 0.5}}}
        assert result.fold_count == 3
        assert len(result.params) == 2

    def test_fit_with_params(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.fit.return_value = FakeFitResult(metrics={}, splits=FakeOuter(outer=[1]))
        mock_model.params_table.return_value = pd.DataFrame()

        adapter.fit(mock_model, params={"n_estimators": 100})
        mock_model.fit.assert_called_once_with(params={"n_estimators": 100})


class TestTune:
    def test_tune_returns_summary(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.tune.return_value = FakeTuningResult(
            best_params={"lr": 0.01},
            best_score=0.95,
            trials=[FakeTrial(number=1, params={"lr": 0.01}, score=0.95, state="COMPLETE")],
            metric_name="accuracy",
            direction="maximize",
        )

        result = adapter.tune(mock_model)
        assert isinstance(result, TuningSummary)
        assert result.best_score == 0.95
        assert result.best_params == {"lr": 0.01}
        assert len(result.trials) == 1
        assert result.metric_name == "accuracy"

    def test_tune_calls_on_progress_periodically(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()

        # Simulate a tune that takes ~1.5 seconds
        def slow_tune() -> FakeTuningResult:
            time.sleep(1.5)
            return FakeTuningResult(
                best_params={"lr": 0.01},
                best_score=0.95,
                trials=[FakeTrial(number=1, params={"lr": 0.01}, score=0.95, state="COMPLETE")],
                metric_name="accuracy",
                direction="maximize",
            )

        mock_model.tune.side_effect = slow_tune

        progress_calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, message: str) -> None:
            progress_calls.append((current, total, message))

        result = adapter.tune(mock_model, on_progress=on_progress)
        assert isinstance(result, TuningSummary)
        # on_progress should have been called at least once during the 1.5s tune
        assert len(progress_calls) >= 1

    def test_tune_cancellation_via_on_progress(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()

        # Simulate a long tune
        def long_tune() -> FakeTuningResult:
            time.sleep(10.0)
            return FakeTuningResult(
                best_params={},
                best_score=0.0,
                trials=[],
                metric_name="acc",
                direction="maximize",
            )

        mock_model.tune.side_effect = long_tune

        call_count = 0

        def on_progress_cancel(current: int, total: int, message: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise InterruptedError("Cancelled")

        with pytest.raises(InterruptedError, match="Cancelled"):
            adapter.tune(mock_model, on_progress=on_progress_cancel)

    def test_fit_calls_on_progress_periodically(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()

        def slow_fit(params: Any = None) -> FakeFitResult:
            time.sleep(1.5)
            return FakeFitResult(
                metrics={"rmse": 0.5},
                splits=FakeOuter(outer=[1, 2, 3]),
            )

        mock_model.fit.side_effect = slow_fit
        mock_model.params_table.return_value = pd.DataFrame()

        progress_calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, message: str) -> None:
            progress_calls.append((current, total, message))

        result = adapter.fit(mock_model, on_progress=on_progress)
        assert isinstance(result, FitSummary)
        assert len(progress_calls) >= 1

    def test_fit_cancellation_via_on_progress(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()

        def long_fit(params: Any = None) -> FakeFitResult:
            time.sleep(10.0)
            return FakeFitResult(metrics={}, splits=FakeOuter(outer=[1]))

        mock_model.fit.side_effect = long_fit
        mock_model.params_table.return_value = pd.DataFrame()

        call_count = 0

        def on_progress_cancel(current: int, total: int, message: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise InterruptedError("Cancelled")

        with pytest.raises(InterruptedError, match="Cancelled"):
            adapter.fit(mock_model, on_progress=on_progress_cancel)


class TestPredict:
    def test_predict_classification(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.predict.return_value = FakePredResult(
            pred=[0, 1, 1], proba=[0.2, 0.8, 0.9], warnings=[]
        )
        df = pd.DataFrame({"x": [1, 2, 3]})

        result = adapter.predict(mock_model, df)
        assert isinstance(result, PredictionSummary)
        assert list(result.predictions["pred"]) == [0, 1, 1]
        assert list(result.predictions["proba"]) == [0.2, 0.8, 0.9]

    def test_predict_regression_no_proba(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.predict.return_value = FakePredResult(
            pred=[1.0, 2.0], proba=None, warnings=["low confidence"]
        )
        df = pd.DataFrame({"x": [1, 2]})

        result = adapter.predict(mock_model, df)
        assert "proba" not in result.predictions.columns
        assert result.warnings == ["low confidence"]


class TestPlot:
    def test_plot_returns_plotdata(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_model.plot_learning_curve.return_value = mock_fig

        result = adapter.plot(mock_model, "learning-curve")
        assert isinstance(result, PlotData)
        assert result.plotly_json == '{"data": [], "layout": {}}'

    @pytest.mark.parametrize("kind", ["split", "gain", "shap"])
    def test_plot_feature_importance_variants(self, kind: str) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_model.importance_plot.return_value = mock_fig

        result = adapter.plot(mock_model, f"feature-importance-{kind}")
        assert isinstance(result, PlotData)
        mock_model.importance_plot.assert_called_once_with(kind=kind)

    def test_plot_unknown_type_raises(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="Unknown plot type"):
            adapter.plot(mock_model, "nonexistent-plot")


class TestAvailablePlots:
    def test_regression_plots(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._cfg.task = "regression"
        mock_model.fit_result.calibrator = None
        del mock_model._tuning_result

        plots = adapter.available_plots(mock_model)
        assert "residuals" in plots
        assert "roc-curve" not in plots
        assert "learning-curve" in plots
        assert "feature-importance-split" in plots
        assert "feature-importance-gain" in plots
        assert "feature-importance-shap" in plots

    def test_binary_with_calibration(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._cfg.task = "binary"
        mock_model.fit_result.calibrator = MagicMock()
        del mock_model._tuning_result

        plots = adapter.available_plots(mock_model)
        assert "roc-curve" in plots
        assert "calibration" in plots
        assert "probability-histogram" in plots

    def test_binary_without_calibration(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._cfg.task = "binary"
        mock_model.fit_result.calibrator = None
        del mock_model._tuning_result

        plots = adapter.available_plots(mock_model)
        assert "roc-curve" in plots
        assert "calibration" not in plots

    def test_tuning_plot_when_tuned(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._cfg.task = "binary"
        mock_model.fit_result.calibrator = None
        mock_model._tuning_result = MagicMock()

        plots = adapter.available_plots(mock_model)
        assert "optimization-history" in plots

    def test_available_plots_fallback_when_no_cfg(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock(spec=[])
        mock_model._widget_config = {"task": "binary"}
        mock_model.fit_result = MagicMock()
        mock_model.fit_result.calibrator = None

        plots = adapter.available_plots(mock_model)
        assert "roc-curve" in plots
        assert "learning-curve" in plots

    def test_tune_only_no_fit_dependent_plots(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._cfg.task = "binary"
        mock_model.fit_result = None  # Not fitted
        mock_model._tuning_result = MagicMock()

        plots = adapter.available_plots(mock_model)
        assert "optimization-history" in plots
        assert "learning-curve" not in plots
        assert "oof-distribution" not in plots
        assert "feature-importance-split" not in plots
        assert "feature-importance-gain" not in plots
        assert "feature-importance-shap" not in plots
        assert "roc-curve" not in plots

    def test_fitted_and_tuned_model_has_all_plots(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._cfg.task = "binary"
        mock_model.fit_result = MagicMock()
        mock_model.fit_result.calibrator = None
        mock_model._tuning_result = MagicMock()

        plots = adapter.available_plots(mock_model)
        assert "optimization-history" in plots
        assert "learning-curve" in plots
        assert "feature-importance-split" in plots
        assert "feature-importance-gain" in plots
        assert "feature-importance-shap" in plots
        assert "roc-curve" in plots


class TestResultDelegation:
    def test_evaluate_table(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.evaluate_table.return_value = pd.DataFrame(
            {"metric": ["rmse"], "value": [0.5]}, index=[0]
        )
        result = adapter.evaluate_table(mock_model)
        assert len(result) == 1
        assert result[0]["metric"] == "rmse"

    def test_split_summary(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.split_summary.return_value = pd.DataFrame(
            {"fold": [0, 1], "train_size": [80, 80]}
        )
        result = adapter.split_summary(mock_model)
        assert len(result) == 2

    def test_importance(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model.importance.return_value = {"x": 0.5, "y": 0.3}
        result = adapter.importance(mock_model, "split")
        assert result == {"x": 0.5, "y": 0.3}


class TestStubs:
    def test_export_model(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        result = adapter.export_model(mock_model, "/tmp/model.pkl")
        mock_model.save.assert_called_once_with("/tmp/model.pkl")
        assert result == "/tmp/model.pkl"

    def test_load_model_not_implemented(self) -> None:
        adapter = LizyMLAdapter()
        with pytest.raises(NotImplementedError):
            adapter.load_model("/tmp/model.pkl")

    def test_model_info_not_implemented(self) -> None:
        adapter = LizyMLAdapter()
        with pytest.raises(NotImplementedError):
            adapter.model_info(MagicMock())


class TestNumThreadsExplicit:
    @staticmethod
    def _binary_config() -> dict[str, Any]:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        return {
            **config,
            "task": "binary",
            "data": {"target": "y"},
            "features": {"exclude": [], "categorical": []},
            "split": {"method": "kfold", "n_splits": 5},
        }

    def test_task_independent_params_include_num_threads(self) -> None:
        assert "num_threads" in LizyMLAdapter._LGBM_PARAMS_TASK_INDEPENDENT

    def test_num_threads_value_is_zero(self) -> None:
        assert LizyMLAdapter._LGBM_PARAMS_TASK_INDEPENDENT["num_threads"] == 0

    def test_prepare_run_config_preserves_num_threads_for_fit(self) -> None:
        config = self._binary_config()
        adapter = LizyMLAdapter()
        result = adapter.prepare_run_config(config, job_type="fit", task="binary")
        assert result.get("model", {}).get("params", {}).get("num_threads") == 0

    def test_user_num_threads_overrides_default(self) -> None:
        config = self._binary_config()
        model_section = dict(config.get("model", {}))
        params = {**dict(model_section.get("params", {})), "num_threads": 4}
        config = {**config, "model": {**model_section, "params": params}}
        adapter = LizyMLAdapter()
        result = adapter.prepare_run_config(config, job_type="fit", task="binary")
        assert result.get("model", {}).get("params", {}).get("num_threads") == 4


class TestAbandonedThreadTracking:
    def test_warns_when_previous_thread_still_alive(self) -> None:
        adapter = LizyMLAdapter()
        barrier = threading.Event()

        # First call: slow target that blocks until we release it
        def slow_target() -> str:
            barrier.wait(timeout=5.0)
            return "done"

        def cancel_after_start(_c: int, _t: int, _m: str) -> None:
            raise InterruptedError("cancel")

        # Start and cancel (abandons the daemon thread)
        with pytest.raises(InterruptedError):
            adapter._run_with_cancel_polling(
                slow_target,
                cancel_after_start,
                poll_interval=0.05,
            )

        # Second call: should log a warning about abandoned thread
        logger = logging.getLogger("lizyml_widget.adapter")
        with (
            _capture_log(logger, logging.WARNING) as records,
            pytest.raises(InterruptedError),
        ):
            adapter._run_with_cancel_polling(
                slow_target,
                cancel_after_start,
                poll_interval=0.05,
            )

        warned = any("still running" in r.getMessage() for r in records)
        assert warned, "Expected warning about abandoned thread still running"

        # Cleanup
        barrier.set()

    def test_no_warning_when_previous_thread_finished(self) -> None:
        adapter = LizyMLAdapter()

        call_count = 0

        def noop_progress(_c: int, _t: int, _m: str) -> None:
            nonlocal call_count
            call_count += 1

        # First call: fast target that completes via the threaded path
        result = adapter._run_with_cancel_polling(
            lambda: "done",
            noop_progress,
            poll_interval=0.05,
        )
        assert result == "done"
        # Thread should have completed; verify it's tracked
        assert adapter._last_worker_thread is not None
        assert not adapter._last_worker_thread.is_alive()

        # Second call: should NOT warn
        logger = logging.getLogger("lizyml_widget.adapter")
        with _capture_log(logger, logging.WARNING) as records:
            result2 = adapter._run_with_cancel_polling(
                lambda: "done2",
                noop_progress,
                poll_interval=0.05,
            )

        assert result2 == "done2"
        warned = any("still running" in r.getMessage() for r in records)
        assert not warned, "Should not warn when previous thread finished"


class TestMulticlassProbaExpansion:
    def test_binary_proba_single_column(self) -> None:
        import numpy as np

        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.pred = np.array([0, 1, 0])
        mock_result.proba = np.array([0.2, 0.8, 0.3])
        mock_result.warnings = []
        mock_model.predict.return_value = mock_result

        summary = adapter.predict(mock_model, pd.DataFrame({"x": [1, 2, 3]}))
        assert "proba" in summary.predictions.columns
        assert "proba_0" not in summary.predictions.columns

    def test_multiclass_proba_per_class_columns(self) -> None:
        import numpy as np

        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.pred = np.array([0, 1, 2])
        mock_result.proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_result.warnings = []
        mock_model.predict.return_value = mock_result

        summary = adapter.predict(mock_model, pd.DataFrame({"x": [1, 2, 3]}))
        assert "proba_0" in summary.predictions.columns
        assert "proba_1" in summary.predictions.columns
        assert "proba_2" in summary.predictions.columns
        assert "proba" not in summary.predictions.columns


class TestShapOutput:
    def test_predict_with_shap_values(self) -> None:
        import numpy as np

        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.pred = np.array([0, 1])
        mock_result.proba = np.array([0.3, 0.9])
        mock_result.warnings = []
        mock_result.shap_values = np.array([[0.1, -0.2], [0.3, 0.4]])
        mock_model.predict.return_value = mock_result

        data = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4]})
        summary = adapter.predict(mock_model, data, return_shap=True)
        assert "shap_feat_a" in summary.predictions.columns
        assert "shap_feat_b" in summary.predictions.columns

    def test_predict_without_shap_no_columns(self) -> None:
        import numpy as np

        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.pred = np.array([0, 1])
        mock_result.proba = np.array([0.3, 0.9])
        mock_result.warnings = []
        # shap_values not present
        del mock_result.shap_values
        mock_model.predict.return_value = mock_result

        data = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4]})
        summary = adapter.predict(mock_model, data, return_shap=False)
        shap_cols = [c for c in summary.predictions.columns if c.startswith("shap_")]
        assert shap_cols == []


class TestCreateModelWidgetConfig:
    """create_model should attach _widget_config to the model."""

    def test_widget_config_attached(self) -> None:
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config.update(
            {
                "task": "binary",
                "data": {"target": "y"},
                "features": {"categorical": [], "exclude": []},
                "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
            }
        )
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        model = adapter.create_model(config, df)
        assert hasattr(model, "_widget_config")
        assert model._widget_config["task"] == "binary"
        # Should be a deep copy, not the same object
        assert model._widget_config is not config


class TestCancelPollingErrorPropagation:
    """_run_with_cancel_polling should propagate worker thread errors."""

    def test_worker_exception_propagates(self) -> None:
        adapter = LizyMLAdapter()

        def failing_target() -> None:
            raise RuntimeError("worker failed")

        with pytest.raises(RuntimeError, match="worker failed"):
            adapter._run_with_cancel_polling(failing_target, on_progress=None)

    def test_worker_exception_with_progress_propagates(self) -> None:
        adapter = LizyMLAdapter()

        def failing_target() -> None:
            raise ValueError("bad value")

        def on_progress(cur: int, total: int, msg: str) -> None:
            pass

        with pytest.raises(ValueError, match="bad value"):
            adapter._run_with_cancel_polling(failing_target, on_progress)


class TestPlotInference:
    """plot_inference should generate prediction-distribution and shap-summary."""

    def test_prediction_distribution(self) -> None:
        adapter = LizyMLAdapter()
        df = pd.DataFrame({"pred": [0.1, 0.5, 0.9, 0.3, 0.7]})
        result = adapter.plot_inference(df, "prediction-distribution")
        assert result.plotly_json
        import json

        spec = json.loads(result.plotly_json)
        assert "data" in spec
        assert "layout" in spec

    def test_shap_summary(self) -> None:
        adapter = LizyMLAdapter()
        df = pd.DataFrame(
            {"pred": [0.1, 0.5], "shap_feat_a": [0.2, -0.1], "shap_feat_b": [0.3, 0.4]}
        )
        result = adapter.plot_inference(df, "shap-summary")
        assert result.plotly_json

    def test_shap_summary_missing_columns_raises(self) -> None:
        adapter = LizyMLAdapter()
        df = pd.DataFrame({"pred": [0.1, 0.5]})
        with pytest.raises(ValueError, match="No SHAP values"):
            adapter.plot_inference(df, "shap-summary")

    def test_unknown_plot_type_raises(self) -> None:
        adapter = LizyMLAdapter()
        df = pd.DataFrame({"pred": [0.1, 0.5]})
        with pytest.raises(ValueError, match="Unknown inference plot type"):
            adapter.plot_inference(df, "nonexistent")
