"""Tests for LizyMLAdapter using mocked LizyML library."""

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

from lizyml_widget import adapter_params, adapter_schema
from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import (
    BackendContract,
    ConfigPatchOp,
    FitSummary,
    PlotData,
    PredictionSummary,
    TuningSummary,
)

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_eval_metrics_cache() -> None:  # type: ignore[misc]
    """Reset the module-level eval metrics cache between tests."""
    adapter_params._eval_metrics_cache = None
    yield  # type: ignore[misc]
    adapter_params._eval_metrics_cache = None


# ── Helpers ──────────────────────────────────────────────────


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


# ── BackendInfo ──────────────────────────────────────────────


class TestInfo:
    @patch.dict("sys.modules", {"lizyml": MagicMock(__version__="1.2.3")})
    def test_info(self) -> None:
        adapter = LizyMLAdapter()
        info = adapter.info
        assert info.name == "lizyml"
        assert info.version == "1.2.3"


# ── Config schema ────────────────────────────────────────────


class TestConfigSchema:
    def test_get_config_schema(self) -> None:
        mock_config_cls = MagicMock()
        mock_config_cls.model_json_schema.return_value = {"type": "object", "properties": {}}
        with patch.dict(
            "sys.modules",
            {
                "lizyml": MagicMock(),
                "lizyml.config": MagicMock(),
                "lizyml.config.schema": MagicMock(LizyMLConfig=mock_config_cls),
            },
        ):
            adapter = LizyMLAdapter()
            schema = adapter.get_config_schema()
            assert schema == {"type": "object", "properties": {}}

    def test_validate_config_valid(self) -> None:
        mock_load_config = MagicMock()
        mock_loader = MagicMock(load_config=mock_load_config)
        mock_config_cls = MagicMock()
        mock_config_cls.model_json_schema.return_value = {"properties": {}}
        old_cache = adapter_schema._schema_cache
        adapter_schema.reset_schema_cache()
        try:
            with patch.dict(
                "sys.modules",
                {
                    "lizyml": MagicMock(),
                    "lizyml.config": MagicMock(),
                    "lizyml.config.loader": mock_loader,
                    "lizyml.config.schema": MagicMock(LizyMLConfig=mock_config_cls),
                },
            ):
                adapter = LizyMLAdapter()
                errors = adapter.validate_config({"model": {"name": "lgbm"}})
                assert errors == []
        finally:
            adapter_schema._schema_cache = old_cache

    def test_validate_config_invalid(self) -> None:
        mock_load_config = MagicMock(side_effect=ValueError("config_version is required"))
        mock_loader = MagicMock(load_config=mock_load_config)
        mock_config_cls = MagicMock()
        mock_config_cls.model_json_schema.return_value = {"properties": {}}
        old_cache = adapter_schema._schema_cache
        adapter_schema.reset_schema_cache()
        try:
            with patch.dict(
                "sys.modules",
                {
                    "lizyml": MagicMock(),
                    "lizyml.config": MagicMock(),
                    "lizyml.config.loader": mock_loader,
                    "lizyml.config.schema": MagicMock(LizyMLConfig=mock_config_cls),
                },
            ):
                adapter = LizyMLAdapter()
                errors = adapter.validate_config({})
                assert len(errors) == 1
                assert "config_version" in errors[0]["message"]
        finally:
            adapter_schema._schema_cache = old_cache

    def test_validate_config_catches_legacy_mode_format(self) -> None:
        """R2 defense: legacy 'mode' format should be caught before backend validation."""
        adapter = LizyMLAdapter()
        config = {
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 50},
                    "space": {"learning_rate": {"mode": "range", "low": 0.001, "high": 0.1}},
                }
            }
        }
        errors = adapter.validate_config(config)
        assert len(errors) == 1
        assert "Legacy" in errors[0]["message"]
        assert errors[0]["field"] == "tuning.optuna.space.learning_rate"

    def test_validate_config_passes_type_format(self) -> None:
        """Correct type-based format should pass pre-validation."""
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
                        "space": {
                            "learning_rate": {
                                "type": "float",
                                "low": 0.001,
                                "high": 0.1,
                                "log": True,
                            },
                        },
                    }
                }
            }
            errors = adapter.validate_config(config)
            assert errors == []


# ── Config Contract Validation (Phase 22) ────────────────────


class TestConfigContractValidation:
    """Phase 22: Table-driven config contract violation/compliance tests."""

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
        """Valid type-based format should pass pre-validation."""
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


# ── Fit ──────────────────────────────────────────────────────


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


# ── Tune ─────────────────────────────────────────────────────


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
        """on_progress must be called during tune so cancel checks fire."""
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
        """When on_progress raises InterruptedError, tune should stop."""
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
        """on_progress must be called during fit so cancel checks fire."""
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
        """When on_progress raises InterruptedError, fit should stop."""
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


# ── Predict ──────────────────────────────────────────────────


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


# ── Plot ─────────────────────────────────────────────────────


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


# ── Available plots ──────────────────────────────────────────


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
        """Regression: no _cfg attribute should not raise."""
        adapter = LizyMLAdapter()
        mock_model = MagicMock(spec=[])
        mock_model._widget_config = {"task": "binary"}
        mock_model.fit_result = MagicMock()
        mock_model.fit_result.calibrator = None

        plots = adapter.available_plots(mock_model)
        assert "roc-curve" in plots
        assert "learning-curve" in plots

    def test_tune_only_no_fit_dependent_plots(self) -> None:
        """R4: Tune-only model should not include fit-dependent plots."""
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
        """Model that is both fitted and tuned should have all plots."""
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


# ── Evaluate / Split / Importance ────────────────────────────


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


# ── Stubs ────────────────────────────────────────────────────


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


# ── Backend Contract (Phase 25) ──────────────────────────────


def _mock_schema_modules() -> dict[str, Any]:
    """Return sys.modules patch dict for a minimal config schema."""
    mock_config_cls = MagicMock()
    mock_config_cls.model_json_schema.return_value = {
        "type": "object",
        "properties": {
            "config_version": {"type": "integer", "default": 1},
            "model": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": "lgbm"},
                    "auto_num_leaves": {"type": "boolean", "default": True},
                    "num_leaves_ratio": {"type": "number", "default": 1.0},
                    "params": {"type": "object", "additionalProperties": True},
                },
            },
            "training": {"type": "object", "properties": {}},
            "evaluation": {"type": "object", "properties": {}},
            "calibration": {"type": "object", "properties": {}},
            "output_dir": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Output Dir",
            },
        },
    }
    return {
        "lizyml": MagicMock(),
        "lizyml.config": MagicMock(),
        "lizyml.config.schema": MagicMock(LizyMLConfig=mock_config_cls),
    }


class TestGetBackendContract:
    def test_returns_backend_contract(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            assert isinstance(contract, BackendContract)
            assert contract.schema_version == 1

    def test_ui_schema_sections(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            section_keys = [s["key"] for s in contract.ui_schema["sections"]]
            assert section_keys == ["model", "training", "calibration", "evaluation"]

    def test_option_sets_all_tasks(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            option_sets = contract.ui_schema["option_sets"]
            for task in ("regression", "binary", "multiclass"):
                assert task in option_sets["objective"]
                assert task in option_sets["metric"]
                assert len(option_sets["objective"][task]) > 0
                assert len(option_sets["metric"][task]) > 0

    def test_model_metric_option_set_uses_lgbm_names(self) -> None:
        """model_metric option set must contain LightGBM-native metric names, not LizyML eval names."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            option_sets = contract.ui_schema["option_sets"]
            assert "model_metric" in option_sets, "option_sets must include model_metric"
            for task in ("regression", "binary", "multiclass"):
                assert task in option_sets["model_metric"]
                assert len(option_sets["model_metric"][task]) > 0

    def test_multiclass_model_metric_uses_multi_prefix(self) -> None:
        """Multiclass model_metric must use multi_logloss/auc_mu, not logloss/auc."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            mc_metrics = contract.ui_schema["option_sets"]["model_metric"]["multiclass"]
            # Must contain LightGBM multiclass-specific names
            assert "multi_logloss" in mc_metrics
            assert "auc_mu" in mc_metrics
            # Must NOT contain bare names that fail with multiclass objective
            assert "logloss" not in mc_metrics
            assert "auc" not in mc_metrics

    def test_parameter_hints_metric_uses_model_metric_kind(self) -> None:
        """The metric parameter_hint must use kind='model_metric' to pick from model_metric option set."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            hints = contract.ui_schema["parameter_hints"]
            metric_hint = next(h for h in hints if h["key"] == "metric")
            assert metric_hint["kind"] == "model_metric"

    def test_parameter_hints_complete(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            hints = contract.ui_schema["parameter_hints"]
            hint_keys = [h["key"] for h in hints]
            assert "objective" in hint_keys
            assert "metric" in hint_keys
            assert "n_estimators" in hint_keys
            assert "learning_rate" in hint_keys
            # All hints have required fields
            for h in hints:
                assert "key" in h
                assert "label" in h
                assert "kind" in h

    def test_search_space_catalog_complete(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            catalog_keys = [c["key"] for c in catalog]
            assert "objective" in catalog_keys
            assert "learning_rate" in catalog_keys
            # All catalog entries have modes
            for entry in catalog:
                assert "modes" in entry
                assert len(entry["modes"]) > 0

    def test_step_map(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            step_map = contract.ui_schema["step_map"]
            assert step_map["n_estimators"] == 100
            assert step_map["learning_rate"] == 0.001

    def test_conditional_visibility(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            cv = contract.ui_schema["conditional_visibility"]
            assert cv["calibration"]["task"] == ["binary"]
            assert cv["num_leaves_ratio"]["auto_num_leaves"] is True
            assert cv["num_leaves"]["auto_num_leaves"] is False

    def test_defaults(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            defaults = contract.ui_schema["defaults"]
            assert defaults["calibration"]["method"] == "platt"

    def test_capabilities(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            assert contract.capabilities["tune"]["allow_empty_space"] is True


class TestInitializeConfig:
    def test_no_task(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config()
            assert config["config_version"] == 1
            assert config["model"]["name"] == "lgbm"
            params = config["model"]["params"]
            assert params["n_estimators"] == 1500
            assert params["learning_rate"] == 0.001
            # No task-specific params
            assert "objective" not in params

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_with_task(self, task: str) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config(task=task)
            params = config["model"]["params"]
            assert "objective" in params
            assert "metric" in params

    def test_regression_defaults(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config(task="regression")
            params = config["model"]["params"]
            assert params["objective"] == "huber"
            assert "huber" in params["metric"]

    def test_binary_defaults(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config(task="binary")
            params = config["model"]["params"]
            assert params["objective"] == "binary"
            assert "auc" in params["metric"]

    def test_multiclass_defaults(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config(task="multiclass")
            params = config["model"]["params"]
            assert params["objective"] == "multiclass"
            assert "auc_mu" in params["metric"]

    def test_output_dir_default(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config()
            assert config["output_dir"] == "outputs/"


class TestApplyConfigPatch:
    def test_set_operation(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"params": {"learning_rate": 0.001}}}
        ops = [ConfigPatchOp(op="set", path="model.params.learning_rate", value=0.01)]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["params"]["learning_rate"] == 0.01

    def test_unset_operation(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"params": {"learning_rate": 0.001, "max_depth": 5}}}
        ops = [ConfigPatchOp(op="unset", path="model.params.max_depth")]
        result = adapter.apply_config_patch(config, ops)
        assert "max_depth" not in result["model"]["params"]
        assert result["model"]["params"]["learning_rate"] == 0.001

    def test_merge_operation(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"params": {"a": 1}}}
        ops = [ConfigPatchOp(op="merge", path="model.params", value={"b": 2, "c": 3})]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["params"] == {"a": 1, "b": 2, "c": 3}

    def test_set_creates_intermediates(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {}
        ops = [ConfigPatchOp(op="set", path="model.params.lr", value=0.01)]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["params"]["lr"] == 0.01

    def test_auto_num_leaves_true_removes_num_leaves(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {
                "auto_num_leaves": True,
                "params": {"num_leaves": 256, "learning_rate": 0.001},
            }
        }
        ops = [ConfigPatchOp(op="set", path="model.params.learning_rate", value=0.01)]
        result = adapter.apply_config_patch(config, ops)
        assert "num_leaves" not in result["model"]["params"]

    def test_auto_num_leaves_false_adds_num_leaves(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"auto_num_leaves": False, "params": {"learning_rate": 0.001}}}
        ops = [ConfigPatchOp(op="set", path="model.params.learning_rate", value=0.01)]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["params"]["num_leaves"] == 256

    def test_multiple_ops(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"auto_num_leaves": False, "params": {"a": 1, "b": 2}}}
        ops = [
            ConfigPatchOp(op="set", path="model.params.a", value=10),
            ConfigPatchOp(op="unset", path="model.params.b"),
            ConfigPatchOp(op="set", path="model.params.c", value=3),
        ]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["params"]["a"] == 10
        assert "b" not in result["model"]["params"]
        assert result["model"]["params"]["c"] == 3

    def test_does_not_mutate_input(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"auto_num_leaves": False, "params": {"a": 1}}}
        ops = [ConfigPatchOp(op="set", path="model.params.a", value=99)]
        adapter.apply_config_patch(config, ops)
        assert config["model"]["params"]["a"] == 1


class TestApplyTaskDefaults:
    """Phase 26: apply_task_defaults applies task-specific params via adapter."""

    def test_binary_defaults(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {"n_estimators": 100}}}
        result = adapter.apply_task_defaults(config, task="binary")
        assert result["model"]["params"]["objective"] == "binary"
        assert result["model"]["params"]["metric"] == ["auc", "binary_logloss"]
        assert result["model"]["params"]["n_estimators"] == 100

    def test_regression_defaults(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {}}}
        result = adapter.apply_task_defaults(config, task="regression")
        assert result["model"]["params"]["objective"] == "huber"

    def test_multiclass_defaults(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {}}}
        result = adapter.apply_task_defaults(config, task="multiclass")
        assert result["model"]["params"]["objective"] == "multiclass"
        assert "auc_mu" in result["model"]["params"]["metric"]

    def test_unknown_task_returns_copy(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {"lr": 0.1}}}
        result = adapter.apply_task_defaults(config, task="unknown")
        assert result["model"]["params"]["lr"] == 0.1

    def test_does_not_mutate_input(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {"n_estimators": 100}}}
        adapter.apply_task_defaults(config, task="binary")
        assert "objective" not in config["model"]["params"]

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_all_eval_metrics_on_by_default(self, task: str) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
        }
        result = adapter.apply_task_defaults(config, task=task)
        expected = adapter._get_eval_metrics_by_task()[task]
        assert result["evaluation"]["metrics"] == expected

    def test_eval_metrics_are_lizyml_names_not_lgbm(self) -> None:
        """Eval metrics must use LizyML registry names, not LightGBM training names."""
        adapter = LizyMLAdapter()
        eval_metrics = adapter._get_eval_metrics_by_task()
        lgbm_only_names = {
            "binary_logloss",
            "binary_error",
            "multi_logloss",
            "multi_error",
            "auc_mu",
            "mse",
            "quantile",
        }
        for task, metrics in eval_metrics.items():
            overlap = set(metrics) & lgbm_only_names
            assert not overlap, f"Task '{task}' has LightGBM-only metric names: {overlap}"

    def test_eval_metrics_not_overwritten_if_set(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc"]},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        assert result["evaluation"]["metrics"] == ["auc"]

    def test_task_change_filters_stale_metrics(self) -> None:
        """When task changes, stale metrics from old task should be removed."""
        adapter = LizyMLAdapter()
        # Start with regression metrics
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["mae", "rmse", "r2"]},
        }
        # Switch to binary → regression-only metrics are invalid
        result = adapter.apply_task_defaults(config, task="binary")
        binary_valid = set(adapter._get_eval_metrics_by_task()["binary"])
        actual = set(result["evaluation"]["metrics"])
        # All returned metrics must be valid for binary
        assert actual <= binary_valid
        # Since no valid metrics survived, should populate defaults
        assert len(actual) > 0

    def test_task_change_keeps_valid_metrics(self) -> None:
        """When task changes, metrics valid for new task should be kept."""
        adapter = LizyMLAdapter()
        # Config has mix of binary and invalid metrics
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc", "mae", "logloss"]},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        # auc and logloss are valid for binary; mae is not
        assert "auc" in result["evaluation"]["metrics"]
        assert "logloss" in result["evaluation"]["metrics"]
        assert "mae" not in result["evaluation"]["metrics"]


class TestApplyConfigPatchCanonicalInvariant:
    """Phase 26: apply_config_patch re-completes required fields after unset."""

    def test_unset_model_name_re_completed(self) -> None:
        adapter = LizyMLAdapter()
        config = {"config_version": 1, "model": {"name": "lgbm", "params": {}}}
        ops = [ConfigPatchOp(op="unset", path="model.name")]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["name"] == "lgbm"

    def test_unset_config_version_re_completed(self) -> None:
        adapter = LizyMLAdapter()
        config = {"config_version": 1, "model": {"name": "lgbm", "params": {}}}
        ops = [ConfigPatchOp(op="unset", path="config_version")]
        result = adapter.apply_config_patch(config, ops)
        assert result["config_version"] == 1

    def test_unset_entire_model_re_completed(self) -> None:
        adapter = LizyMLAdapter()
        config = {"config_version": 1, "model": {"name": "lgbm", "params": {}}}
        ops = [ConfigPatchOp(op="unset", path="model")]
        result = adapter.apply_config_patch(config, ops)
        assert result["model"]["name"] == "lgbm"
        # Params should be re-initialized from defaults, not silently dropped
        assert "params" in result["model"]
        assert result["model"]["params"].get("n_estimators") is not None


class TestPrepareRunConfig:
    def test_fit_basic(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {}}}
        result = adapter.prepare_run_config(config, job_type="fit")
        assert result["model"]["name"] == "lgbm"
        assert "tuning" not in result

    def test_fit_ensures_model_name(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"params": {}}}
        result = adapter.prepare_run_config(config, job_type="fit")
        assert result["model"]["name"] == "lgbm"

    def test_tune_adds_defaults(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {}}}
        result = adapter.prepare_run_config(config, job_type="tune")
        assert result["tuning"]["optuna"]["params"]["n_trials"] == 50
        assert result["tuning"]["optuna"]["space"] == {}

    def test_tune_preserves_existing(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "tuning": {"optuna": {"params": {"n_trials": 100}, "space": {"lr": {"type": "float"}}}},
        }
        result = adapter.prepare_run_config(config, job_type="tune")
        assert result["tuning"]["optuna"]["params"]["n_trials"] == 100

    def test_auto_num_leaves_exclusivity(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "auto_num_leaves": True, "params": {"num_leaves": 256}}}
        result = adapter.prepare_run_config(config, job_type="fit")
        assert "num_leaves" not in result["model"]["params"]

    def test_does_not_mutate_input(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {"a": 1}}}
        adapter.prepare_run_config(config, job_type="tune")
        assert "tuning" not in config

    def test_evaluation_params_stripped_for_backend(self) -> None:
        """evaluation.params is widget-only and must be stripped before backend submission."""
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc"], "params": {"precision_at_k_k": 20}},
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        eval_section = result.get("evaluation", {})
        assert "params" not in eval_section, "evaluation.params should be stripped"
        assert eval_section["metrics"] == ["auc"]

    def test_evaluation_params_survives_config_patch(self) -> None:
        """evaluation.params should survive config patching (not stripped in patch)."""
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc"], "params": {"precision_at_k_k": 20}},
        }
        ops = [ConfigPatchOp(op="set", path="evaluation.metrics", value=["auc", "logloss"])]
        result = adapter.apply_config_patch(config, ops)
        assert result["evaluation"]["params"] == {"precision_at_k_k": 20}
        assert result["evaluation"]["metrics"] == ["auc", "logloss"]

    def test_inner_valid_normalized_in_prepare(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "training": {"early_stopping": {"inner_valid": "holdout"}},
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "holdout"}


# ── Phase 26: Canonicalize Config ────────────────────────────


class TestCanonicalizeConfig:
    """Phase 26: canonicalize_config unifies partial config into canonical shape."""

    def test_partial_config_gets_defaults(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config({"model": {"params": {"n_estimators": 500}}})
        assert result["model"]["name"] == "lgbm"
        assert result["model"]["params"]["n_estimators"] == 500
        assert result["config_version"] == 1

    def test_empty_config_gets_full_defaults(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config({})
        assert result["model"]["name"] == "lgbm"
        assert "params" in result["model"]
        assert result["config_version"] == 1

    def test_user_values_override_defaults(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config(
            {"model": {"name": "lgbm", "params": {"learning_rate": 0.1}}}
        )
        assert result["model"]["params"]["learning_rate"] == 0.1

    def test_auto_num_leaves_exclusivity(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config(
            {"model": {"auto_num_leaves": True, "params": {"num_leaves": 256}}}
        )
        assert "num_leaves" not in result["model"]["params"]

    def test_auto_num_leaves_false_adds_default(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config({"model": {"auto_num_leaves": False}})
        assert result["model"]["params"]["num_leaves"] == 256

    def test_preserves_config_version(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config({"config_version": 3})
        assert result["config_version"] == 3

    def test_task_specific_defaults(self) -> None:
        adapter = LizyMLAdapter()
        result = adapter.canonicalize_config({}, task="binary")
        assert result["model"]["params"]["objective"] == "binary"


class TestInnerValidNormalization:
    """Phase 26: inner_valid legacy strings are normalized to object/null."""

    def test_holdout_string_to_object(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "holdout"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "holdout"}

    def test_group_holdout_string_to_object(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "group_holdout"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "group_holdout"}

    def test_time_holdout_string_to_object(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "time_holdout"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "time_holdout"}

    def test_null_preserved(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": None}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] is None

    def test_object_preserved(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": {"method": "holdout"}}},
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert isinstance(iv, dict)
        assert iv["method"] == "holdout"

    def test_fold_string_becomes_null(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "fold_0"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] is None

    def test_unrecognized_type_becomes_null(self) -> None:
        """Integer, list, or boolean inner_valid should be discarded as None."""
        adapter = LizyMLAdapter()
        for bad_value in [42, [1, 2], True]:
            config = {
                "training": {"early_stopping": {"inner_valid": bad_value}},
            }
            result = adapter.canonicalize_config(config)
            assert result["training"]["early_stopping"]["inner_valid"] is None, (
                f"inner_valid={bad_value!r} should become None"
            )

    def test_normalization_in_apply_config_patch(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {}}, "training": {"early_stopping": {}}}
        ops = [ConfigPatchOp(op="set", path="training.early_stopping.inner_valid", value="holdout")]
        result = adapter.apply_config_patch(config, ops)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "holdout"}


class TestValidationDiagnostics:
    """Phase 26: validate_config extracts __cause__ chain for field paths."""

    def test_cause_chain_extraction(self) -> None:
        """validate_config should walk __cause__ chain for Pydantic errors."""
        adapter = LizyMLAdapter()

        # Create a mock that simulates LizyMLError wrapping a ValidationError
        class FakeValidationError(Exception):
            def errors(self) -> list[dict[str, Any]]:
                return [
                    {
                        "loc": ("training", "early_stopping", "inner_valid"),
                        "msg": "bad type",
                        "type": "type_error",
                    }
                ]

        class FakeLizyMLError(Exception):
            pass

        outer = FakeLizyMLError("config invalid")
        outer.__cause__ = FakeValidationError("validation failed")

        with patch("lizyml.config.loader.load_config", side_effect=outer):
            errors = adapter.validate_config({"model": {"name": "lgbm"}})

        assert len(errors) >= 1
        assert errors[0]["field"] == "training.early_stopping.inner_valid"
        assert errors[0]["type"] == "type_error"

    def test_fallback_when_no_structured_errors(self) -> None:
        """When no __cause__ has .errors(), return generic error with type."""
        adapter = LizyMLAdapter()

        with patch("lizyml.config.loader.load_config", side_effect=ValueError("bad config")):
            errors = adapter.validate_config({"model": {"name": "lgbm"}})

        assert len(errors) == 1
        assert errors[0]["message"] == "bad config"
        assert errors[0]["type"] == "unknown"


class TestInnerValidFieldStripping:
    """inner_valid normalization must strip fields not allowed by the method."""

    def test_time_holdout_strips_random_state_and_stratify(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "time_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert iv == {"method": "time_holdout", "ratio": 0.1}

    def test_group_holdout_strips_stratify(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "group_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert iv == {"method": "group_holdout", "ratio": 0.1, "random_state": 42}

    def test_holdout_preserves_all_fields(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": True,
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert iv == {
            "method": "holdout",
            "ratio": 0.1,
            "random_state": 42,
            "stratify": True,
        }

    def test_prepare_run_config_strips_inner_valid_extras(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "time_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    }
                }
            },
            "data": {"target": "y"},
            "features": {"categorical": [], "exclude": []},
            "split": {"method": "kfold", "n_splits": 5},
            "task": "regression",
            "config_version": 1,
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert "random_state" not in iv
        assert "stratify" not in iv

    def test_validate_config_normalizes_inner_valid(self) -> None:
        """validate_config should normalize inner_valid before LizyML validation."""
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {"n_estimators": 100}},
            "task": "regression",
            "config_version": 1,
            "data": {"target": "y"},
            "features": {"categorical": [], "exclude": []},
            "split": {"method": "kfold", "n_splits": 5},
            "training": {
                "seed": 42,
                "early_stopping": {
                    "enabled": True,
                    "inner_valid": {
                        "method": "time_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    },
                    "rounds": 150,
                },
            },
        }
        # Raw load_config would reject extra fields
        from lizyml.config.loader import load_config

        with pytest.raises(Exception, match="CONFIG_INVALID"):
            load_config(config)

        # But validate_config normalizes first, so it passes
        errors = adapter.validate_config(config)
        assert errors == []

    def test_unknown_method_nulled_out(self) -> None:
        """Unknown inner_valid method should be set to None."""
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "unknown_method",
                        "ratio": 0.1,
                        "extra_field": "should_be_dropped",
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] is None


class TestGetDefaultSearchSpace:
    """get_default_search_space returns LizyML default space as dict."""

    @pytest.mark.parametrize("task", ["binary", "regression", "multiclass"])
    def test_returns_dict_for_known_tasks(self, task: str) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space(task)
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_binary_contains_expected_keys(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # Should contain range params from LizyML default_space
        assert "n_estimators" in space
        assert "learning_rate" in space
        assert "max_depth" in space
        assert "feature_fraction" in space

    def test_range_param_format(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # n_estimators should be int range
        n_est = space["n_estimators"]
        assert n_est["type"] == "int"
        assert "low" in n_est
        assert "high" in n_est

    def test_categorical_param_format(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # objective should be categorical
        obj = space["objective"]
        assert obj["type"] == "categorical"
        assert "choices" in obj

    def test_log_scale_preserved(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # learning_rate is log-scale in LizyML
        lr = space["learning_rate"]
        assert lr["log"] is True

    def test_apply_task_defaults_without_lgbm_params_still_populates_metrics(self) -> None:
        """H-4 regression: metrics must be populated even when no LGBM param defaults exist."""
        adapter = LizyMLAdapter()
        # Patch _LGBM_PARAMS_BY_TASK to simulate a task with no LGBM defaults
        original = dict(adapter._LGBM_PARAMS_BY_TASK)
        adapter._LGBM_PARAMS_BY_TASK = {}
        try:
            config: dict[str, Any] = {
                "model": {"name": "lgbm", "params": {}},
                "evaluation": {"metrics": []},
            }
            result = adapter.apply_task_defaults(config, task="binary")
            # Metrics should still be populated despite no LGBM param defaults
            assert len(result["evaluation"]["metrics"]) > 0
            assert "auc" in result["evaluation"]["metrics"]
        finally:
            adapter._LGBM_PARAMS_BY_TASK = original

    def test_apply_task_defaults_without_lgbm_params_still_populates_space(self) -> None:
        """H-4 regression: search space must be populated even when no LGBM param defaults exist."""
        adapter = LizyMLAdapter()
        original = dict(adapter._LGBM_PARAMS_BY_TASK)
        adapter._LGBM_PARAMS_BY_TASK = {}
        try:
            config: dict[str, Any] = {
                "model": {"name": "lgbm", "params": {}},
                "evaluation": {"metrics": []},
                "tuning": {"optuna": {"params": {"n_trials": 50}, "space": {}}},
            }
            result = adapter.apply_task_defaults(config, task="binary")
            space = result["tuning"]["optuna"]["space"]
            assert len(space) > 0
            assert "n_estimators" in space
        finally:
            adapter._LGBM_PARAMS_BY_TASK = original

    def test_apply_task_defaults_populates_space(self) -> None:
        """apply_task_defaults fills tuning.optuna.space with defaults."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
            "tuning": {"optuna": {"params": {"n_trials": 50}, "space": {}}},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        space = result["tuning"]["optuna"]["space"]
        assert "n_estimators" in space
        assert space["learning_rate"]["log"] is True

    def test_apply_task_defaults_does_not_overwrite_existing_space(
        self,
    ) -> None:
        """Existing search space is not overwritten."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 50},
                    "space": {"n_estimators": {"type": "int", "low": 10, "high": 20, "log": False}},
                }
            },
        }
        result = adapter.apply_task_defaults(config, task="binary")
        space = result["tuning"]["optuna"]["space"]
        # Should keep existing, not overwrite
        assert space["n_estimators"]["low"] == 10

    def test_apply_task_defaults_populates_space_when_tuning_is_none(self) -> None:
        """When tuning is None (schema default), space should still be populated."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
            "tuning": None,
        }
        result = adapter.apply_task_defaults(config, task="binary")
        # Tuning section should be created with default space
        assert result["tuning"] is not None
        assert isinstance(result["tuning"], dict)
        space = result["tuning"]["optuna"]["space"]
        assert "n_estimators" in space

    def test_apply_task_defaults_populates_space_when_tuning_absent(self) -> None:
        """When tuning key is absent, space should still be populated."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        assert result["tuning"] is not None
        space = result["tuning"]["optuna"]["space"]
        assert "n_estimators" in space

    def test_unknown_task_still_returns_space(self) -> None:
        """LizyML returns a default space for any task string."""
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("unknown_task")
        assert isinstance(space, dict)


class TestStripForBackend:
    """strip_for_backend removes non-schema fields at all nesting levels."""

    def test_strips_extra_top_level(self) -> None:
        LizyMLAdapter()
        config = {"config_version": 1, "task": "binary", "extra": "bad"}
        result = adapter_schema.strip_for_backend(config)
        assert "extra" not in result
        assert result["config_version"] == 1

    def test_strips_extra_in_model(self) -> None:
        LizyMLAdapter()
        config = {
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 100},
                "widget_only_field": True,
            }
        }
        result = adapter_schema.strip_for_backend(config)
        assert "widget_only_field" not in result["model"]
        assert result["model"]["name"] == "lgbm"
        assert result["model"]["params"]["n_estimators"] == 100

    def test_strips_extra_in_training(self) -> None:
        LizyMLAdapter()
        config = {
            "training": {"seed": 42, "ui_state": "running"},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "ui_state" not in result["training"]
        assert result["training"]["seed"] == 42

    def test_strips_extra_in_data(self) -> None:
        LizyMLAdapter()
        config = {
            "data": {"target": "y", "widget_managed": True},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "widget_managed" not in result["data"]
        assert result["data"]["target"] == "y"

    def test_strips_extra_in_features(self) -> None:
        LizyMLAdapter()
        config = {
            "features": {"exclude": [], "categorical": [], "js_extra": True},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "js_extra" not in result["features"]

    def test_preserves_model_params_additionalProperties(self) -> None:
        """model.params has additionalProperties=true, so custom keys survive."""
        LizyMLAdapter()
        config = {
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 100, "custom_lgbm_param": 42},
            }
        }
        result = adapter_schema.strip_for_backend(config)
        assert result["model"]["params"]["custom_lgbm_param"] == 42

    def test_multiple_levels_stripped_at_once(self) -> None:
        LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "top_extra": "x",
            "data": {"target": "y", "data_extra": "x"},
            "model": {"name": "lgbm", "params": {}, "model_extra": "x"},
            "training": {"seed": 42, "training_extra": "x"},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "top_extra" not in result
        assert "data_extra" not in result["data"]
        assert "model_extra" not in result["model"]
        assert "training_extra" not in result["training"]

    def test_prepare_run_config_strips_extra_fields(self) -> None:
        """Integration: prepare_run_config output has no extra fields."""
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "ui_state": "running",
            "data": {"target": "y"},
            "features": {"exclude": [], "categorical": []},
            "split": {"method": "kfold", "n_splits": 5},
            "model": {"name": "lgbm", "params": {}, "js_flag": True},
            "training": {"seed": 42, "widget_extra": True},
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        assert "ui_state" not in result
        assert "js_flag" not in result.get("model", {})
        assert "widget_extra" not in result.get("training", {})

    def test_validate_config_ignores_extra_fields(self) -> None:
        """Integration: validate_config passes even with extra fields."""
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "extra_field": "from_js",
            "data": {"target": "y"},
            "features": {"exclude": [], "categorical": []},
            "split": {"method": "kfold", "n_splits": 5},
            "model": {
                "name": "lgbm",
                "params": {"objective": "binary", "metric": ["auc"]},
                "widget_flag": True,
            },
            "training": {"seed": 42},
        }
        errors = adapter.validate_config(config)
        assert errors == []


# ── Bug 1: Tune metric mapping and direction auto-set ─────


class TestTuneMetricMapping:
    """Tests for model_metric → eval_metric mapping and direction auto-set."""

    def test_model_metric_to_eval_mapping_exists(self) -> None:
        """Adapter should have a mapping from model_metric names to eval metric names."""
        adapter = LizyMLAdapter()
        mapping = adapter._MODEL_METRIC_TO_EVAL
        # Key model_metric names must be present
        assert "auc_mu" in mapping
        assert "multi_logloss" in mapping
        assert "binary_logloss" in mapping
        assert "multi_error" in mapping

    def test_model_metric_to_eval_mapping_values(self) -> None:
        """Mapped values should be valid LizyML evaluation metric names."""
        adapter = LizyMLAdapter()
        mapping = adapter._MODEL_METRIC_TO_EVAL
        assert mapping["auc_mu"] == "auc"
        assert mapping["multi_logloss"] == "logloss"
        assert mapping["binary_logloss"] == "logloss"
        assert mapping["multi_error"] == "accuracy"
        # Identity mappings for names that already match
        assert mapping["auc"] == "auc"
        assert mapping["rmse"] == "rmse"

    def test_resolve_direction_maximize(self) -> None:
        """Metrics with greater_is_better=True should resolve to 'maximize'."""
        assert adapter_params.resolve_direction("auc") == "maximize"
        assert adapter_params.resolve_direction("accuracy") == "maximize"
        assert adapter_params.resolve_direction("f1") == "maximize"

    def test_resolve_direction_minimize(self) -> None:
        """Metrics with greater_is_better=False should resolve to 'minimize'."""
        assert adapter_params.resolve_direction("logloss") == "minimize"
        assert adapter_params.resolve_direction("rmse") == "minimize"
        assert adapter_params.resolve_direction("brier") == "minimize"

    def test_resolve_direction_unknown_fallback(self) -> None:
        """Unknown metrics should fall back to 'minimize'."""
        assert adapter_params.resolve_direction("nonexistent_metric") == "minimize"


# ── Immutability: class-level mapping protection ─────────


class TestModelMetricToEvalImmutability:
    """_MODEL_METRIC_TO_EVAL must be protected from mutation."""

    def test_mutation_raises_type_error(self) -> None:
        """Assigning to _MODEL_METRIC_TO_EVAL should raise TypeError."""
        adapter = LizyMLAdapter()
        with pytest.raises(TypeError):
            adapter._MODEL_METRIC_TO_EVAL["injected"] = "bad"  # type: ignore[index]

    def test_read_access_works(self) -> None:
        """Reading from _MODEL_METRIC_TO_EVAL must still work normally."""
        adapter = LizyMLAdapter()
        assert adapter._MODEL_METRIC_TO_EVAL.get("auc_mu") == "auc"
        assert "auc" in adapter._MODEL_METRIC_TO_EVAL


class TestPrepareRunConfigTuneMetric:
    """Tests for prepare_run_config applying tune metric and direction."""

    @staticmethod
    def _make_tune_config(adapter: LizyMLAdapter, task: str, metric: str) -> dict[str, Any]:
        """Helper: build a config with tuning.optuna.params.metric set."""
        config = adapter.initialize_config(task=task)
        config["task"] = task
        config["data"] = {"target": "y"}
        config["features"] = {"exclude": [], "categorical": []}
        config["split"] = {"method": "kfold", "n_splits": 5}
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 50, "metric": metric}, "space": {}}
        }
        return config

    def test_tune_metric_placed_first_in_evaluation_metrics(self) -> None:
        """When tuning.optuna.params.metric is set, the corresponding eval metric
        should be evaluation.metrics[0]."""
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "multiclass", "auc_mu")
        config["evaluation"] = {"metrics": ["accuracy", "auc", "logloss"]}

        result = adapter.prepare_run_config(config, job_type="tune", task="multiclass")

        eval_metrics = result.get("evaluation", {}).get("metrics", [])
        assert eval_metrics[0] == "auc", (
            f"Expected 'auc' as first eval metric, got {eval_metrics}"
        )

    def test_tune_direction_auto_set_maximize(self) -> None:
        """Direction should be auto-set to 'maximize' for auc_mu (auc)."""
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "multiclass", "auc_mu")

        result = adapter.prepare_run_config(config, job_type="tune", task="multiclass")

        direction = result.get("tuning", {}).get("optuna", {}).get("params", {}).get("direction")
        assert direction == "maximize", f"Expected 'maximize', got {direction}"

    def test_tune_direction_auto_set_minimize(self) -> None:
        """Direction should be auto-set to 'minimize' for multi_logloss (logloss)."""
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "multiclass", "multi_logloss")

        result = adapter.prepare_run_config(config, job_type="tune", task="multiclass")

        direction = result.get("tuning", {}).get("optuna", {}).get("params", {}).get("direction")
        assert direction == "minimize", f"Expected 'minimize', got {direction}"

    def test_tune_metric_stripped_from_optuna_params(self) -> None:
        """The widget-only 'metric' key must be stripped before backend validation."""
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "binary", "auc")

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        optuna_params = result.get("tuning", {}).get("optuna", {}).get("params", {})
        assert "metric" not in optuna_params, (
            f"Widget-only 'metric' key should be stripped, but found in {optuna_params}"
        )

    def test_fit_job_ignores_tune_metric(self) -> None:
        """For fit jobs, tuning.optuna.params.metric should be ignored (no crash)."""
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "binary", "auc")

        # Should not raise
        result = adapter.prepare_run_config(config, job_type="fit", task="binary")
        assert result is not None


# ── Bug 2: Apply best_params routing by category ─────────


class TestClassifyBestParams:
    """Tests for classifying best_params into model/smart/training categories."""

    def test_classify_splits_correctly(self) -> None:
        """best_params should be split into model, smart, and training dicts."""
        adapter = LizyMLAdapter()
        params = {
            "learning_rate": 0.01,
            "max_depth": 7,
            "n_estimators": 1500,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "objective": "binary",
            "num_leaves_ratio": 0.7,
            "min_data_in_leaf_ratio": 0.05,
            "early_stopping_rounds": 80,
            "validation_ratio": 0.2,
        }
        model_p, smart_p, training_p = adapter.classify_best_params(params)

        assert "learning_rate" in model_p
        assert "max_depth" in model_p
        assert "n_estimators" in model_p
        assert "objective" in model_p

        assert "num_leaves_ratio" in smart_p
        assert "min_data_in_leaf_ratio" in smart_p

        assert "early_stopping_rounds" in training_p
        assert "validation_ratio" in training_p

    def test_classify_unknown_defaults_to_model(self) -> None:
        """Unknown param names should default to model category."""
        adapter = LizyMLAdapter()
        params = {"some_custom_param": 42}
        model_p, smart_p, training_p = adapter.classify_best_params(params)
        assert "some_custom_param" in model_p
        assert not smart_p
        assert not training_p

    def test_classify_empty_params(self) -> None:
        """Empty params should return three empty dicts."""
        adapter = LizyMLAdapter()
        model_p, smart_p, training_p = adapter.classify_best_params({})
        assert model_p == {}
        assert smart_p == {}
        assert training_p == {}


# ── normalize_inner_valid: mutual exclusivity ─────────────


class TestNormalizeInnerValidExclusivity:
    """Tests for validation_ratio / inner_valid mutual exclusivity in apply_best_params."""

    def test_validation_ratio_set_inner_valid_none_removes_inner_valid_key(self) -> None:
        """When validation_ratio is set and inner_valid is None,
        inner_valid key must be removed entirely (not kept as None).

        Pydantic's model_fields_set treats explicit None as "set", causing
        the 'Specify either validation_ratio or inner_valid' error.
        """
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                    "validation_ratio": 0.2,
                    "inner_valid": None,
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert "inner_valid" not in es, (
            "inner_valid key must be removed when validation_ratio is set"
        )
        assert es["validation_ratio"] == 0.2

    def test_inner_valid_set_validation_ratio_present_removes_validation_ratio(self) -> None:
        """When inner_valid is explicitly set (non-None), validation_ratio
        must be removed to avoid mutual exclusivity error.
        """
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                    "validation_ratio": 0.2,
                    "inner_valid": {"method": "holdout", "ratio": 0.1},
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert "validation_ratio" not in es, (
            "validation_ratio must be removed when inner_valid is set"
        )
        assert es["inner_valid"] == {"method": "holdout", "ratio": 0.1}

    def test_neither_set_leaves_config_unchanged(self) -> None:
        """When neither validation_ratio nor inner_valid is set, config is unchanged."""
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert "inner_valid" not in es
        assert "validation_ratio" not in es

    def test_validation_ratio_only_no_inner_valid_key_unchanged(self) -> None:
        """When only validation_ratio is present (no inner_valid key at all), unchanged."""
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                    "validation_ratio": 0.2,
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert es["validation_ratio"] == 0.2
        assert "inner_valid" not in es


# ── Immutability: prepare_run_config must not mutate input ─


class TestPrepareRunConfigImmutability:
    """prepare_run_config must not mutate the input config dict."""

    def test_tune_config_input_not_mutated(self) -> None:
        """The original config dict must be unchanged after prepare_run_config."""
        import copy

        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config = {**config, "task": "binary"}
        config = {**config, "data": {"target": "y"}}
        config = {**config, "features": {"exclude": [], "categorical": []}}
        config = {**config, "split": {"method": "kfold", "n_splits": 5}}
        config = {**config, "tuning": {
            "optuna": {"params": {"n_trials": 50, "metric": "auc"}, "space": {}}
        }}

        original = copy.deepcopy(config)
        adapter.prepare_run_config(config, job_type="tune", task="binary")

        assert config == original, (
            "prepare_run_config must not mutate the input config"
        )


# ── Phase 1: num_threads must be explicitly set ──────────────


class TestNumThreadsExplicit:
    """LightGBM num_threads must be explicitly set to avoid OpenMP degradation."""

    def test_task_independent_params_include_num_threads(self) -> None:
        """_LGBM_PARAMS_TASK_INDEPENDENT must contain num_threads."""
        assert "num_threads" in LizyMLAdapter._LGBM_PARAMS_TASK_INDEPENDENT

    def test_num_threads_value_is_minus_one(self) -> None:
        """num_threads should be -1 (use all cores, bypass auto-detect)."""
        assert LizyMLAdapter._LGBM_PARAMS_TASK_INDEPENDENT["num_threads"] == -1

    def test_prepare_run_config_preserves_num_threads_for_fit(self) -> None:
        """num_threads must appear in the config produced by prepare_run_config."""
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config = {**config, "task": "binary"}
        config = {**config, "data": {"target": "y"}}
        config = {**config, "features": {"exclude": [], "categorical": []}}
        config = {**config, "split": {"method": "kfold", "n_splits": 5}}

        result = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model_params = result.get("model", {}).get("params", {})
        assert model_params.get("num_threads") == -1

    def test_user_num_threads_overrides_default(self) -> None:
        """User-specified num_threads must not be overwritten by the default."""
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config = {**config, "task": "binary"}
        config = {**config, "data": {"target": "y"}}
        config = {**config, "features": {"exclude": [], "categorical": []}}
        config = {**config, "split": {"method": "kfold", "n_splits": 5}}
        # User explicitly sets num_threads=4
        model_section = dict(config.get("model", {}))
        params = dict(model_section.get("params", {}))
        params["num_threads"] = 4
        model_section["params"] = params
        config = {**config, "model": model_section}

        result = adapter.prepare_run_config(config, job_type="fit", task="binary")
        model_params = result.get("model", {}).get("params", {})
        assert model_params.get("num_threads") == 4


# ── Phase 2: Abandoned thread tracking ───────────────────────


class TestAbandonedThreadTracking:
    """_run_with_cancel_polling must track and warn about abandoned threads."""

    def test_warns_when_previous_thread_still_alive(self) -> None:
        """Starting a new job while the previous is running should log a warning."""
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
                slow_target, cancel_after_start, poll_interval=0.05,
            )

        # Second call: should log a warning about abandoned thread
        logger = logging.getLogger("lizyml_widget.adapter")
        with (
            _capture_log(logger, logging.WARNING) as records,
            pytest.raises(InterruptedError),
        ):
            adapter._run_with_cancel_polling(
                slow_target, cancel_after_start, poll_interval=0.05,
            )

        warned = any("still running" in r.getMessage() for r in records)
        assert warned, "Expected warning about abandoned thread still running"

        # Cleanup
        barrier.set()

    def test_no_warning_when_previous_thread_finished(self) -> None:
        """No warning when previous thread completed normally (with on_progress)."""
        adapter = LizyMLAdapter()

        call_count = 0

        def noop_progress(_c: int, _t: int, _m: str) -> None:
            nonlocal call_count
            call_count += 1

        # First call: fast target that completes via the threaded path
        result = adapter._run_with_cancel_polling(
            lambda: "done", noop_progress, poll_interval=0.05,
        )
        assert result == "done"
        # Thread should have completed; verify it's tracked
        assert adapter._last_worker_thread is not None
        assert not adapter._last_worker_thread.is_alive()

        # Second call: should NOT warn
        logger = logging.getLogger("lizyml_widget.adapter")
        with _capture_log(logger, logging.WARNING) as records:
            result2 = adapter._run_with_cancel_polling(
                lambda: "done2", noop_progress, poll_interval=0.05,
            )

        assert result2 == "done2"
        warned = any("still running" in r.getMessage() for r in records)
        assert not warned, "Should not warn when previous thread finished"


@contextlib.contextmanager
def _capture_log(logger: logging.Logger, level: int = logging.WARNING):
    """Context manager to capture log records from a specific logger."""
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
