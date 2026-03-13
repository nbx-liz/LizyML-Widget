"""Tests for LizyMLAdapter using mocked LizyML library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import (
    BackendContract,
    ConfigPatchOp,
    FitSummary,
    PlotData,
    PredictionSummary,
    TuningSummary,
)

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
        old_cache = LizyMLAdapter._schema_cache
        LizyMLAdapter._schema_cache = None
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
            LizyMLAdapter._schema_cache = old_cache

    def test_validate_config_invalid(self) -> None:
        mock_load_config = MagicMock(side_effect=ValueError("config_version is required"))
        mock_loader = MagicMock(load_config=mock_load_config)
        mock_config_cls = MagicMock()
        mock_config_cls.model_json_schema.return_value = {"properties": {}}
        old_cache = LizyMLAdapter._schema_cache
        LizyMLAdapter._schema_cache = None
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
            LizyMLAdapter._schema_cache = old_cache

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
        assert "feature-importance" in plots

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
        assert "feature-importance" not in plots
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
        assert "feature-importance" in plots
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
            assert section_keys == ["model", "training", "evaluation", "calibration"]

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


class TestStripForBackend:
    """strip_for_backend removes non-schema fields at all nesting levels."""

    def test_strips_extra_top_level(self) -> None:
        adapter = LizyMLAdapter()
        config = {"config_version": 1, "task": "binary", "extra": "bad"}
        result = adapter.strip_for_backend(config)
        assert "extra" not in result
        assert result["config_version"] == 1

    def test_strips_extra_in_model(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 100},
                "widget_only_field": True,
            }
        }
        result = adapter.strip_for_backend(config)
        assert "widget_only_field" not in result["model"]
        assert result["model"]["name"] == "lgbm"
        assert result["model"]["params"]["n_estimators"] == 100

    def test_strips_extra_in_training(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"seed": 42, "ui_state": "running"},
        }
        result = adapter.strip_for_backend(config)
        assert "ui_state" not in result["training"]
        assert result["training"]["seed"] == 42

    def test_strips_extra_in_data(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "data": {"target": "y", "widget_managed": True},
        }
        result = adapter.strip_for_backend(config)
        assert "widget_managed" not in result["data"]
        assert result["data"]["target"] == "y"

    def test_strips_extra_in_features(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "features": {"exclude": [], "categorical": [], "js_extra": True},
        }
        result = adapter.strip_for_backend(config)
        assert "js_extra" not in result["features"]

    def test_preserves_model_params_additionalProperties(self) -> None:
        """model.params has additionalProperties=true, so custom keys survive."""
        adapter = LizyMLAdapter()
        config = {
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 100, "custom_lgbm_param": 42},
            }
        }
        result = adapter.strip_for_backend(config)
        assert result["model"]["params"]["custom_lgbm_param"] == 42

    def test_multiple_levels_stripped_at_once(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "top_extra": "x",
            "data": {"target": "y", "data_extra": "x"},
            "model": {"name": "lgbm", "params": {}, "model_extra": "x"},
            "training": {"seed": 42, "training_extra": "x"},
        }
        result = adapter.strip_for_backend(config)
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
