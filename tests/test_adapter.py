"""Tests for LizyMLAdapter using mocked LizyML library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import FitSummary, PlotData, PredictionSummary, TuningSummary

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
            assert schema.json_schema == {"type": "object", "properties": {}}

    def test_validate_config_valid(self) -> None:
        mock_config_cls = MagicMock()
        mock_config_cls.return_value = MagicMock()  # no error
        with patch.dict(
            "sys.modules",
            {
                "lizyml": MagicMock(),
                "lizyml.config": MagicMock(),
                "lizyml.config.schema": MagicMock(LizyMLConfig=mock_config_cls),
                "pydantic": MagicMock(),
            },
        ):
            adapter = LizyMLAdapter()
            errors = adapter.validate_config({"model": {"name": "lgbm"}})
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
        mock_model._config.task = "regression"
        mock_model.fit_result.calibrator = None
        del mock_model._tuning_result

        plots = adapter.available_plots(mock_model)
        assert "residuals" in plots
        assert "roc-curve" not in plots
        assert "learning-curve" in plots
        assert "importance" in plots

    def test_binary_with_calibration(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._config.task = "binary"
        mock_model.fit_result.calibrator = MagicMock()
        del mock_model._tuning_result

        plots = adapter.available_plots(mock_model)
        assert "roc-curve" in plots
        assert "calibration" in plots
        assert "probability-histogram" in plots

    def test_binary_without_calibration(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._config.task = "binary"
        mock_model.fit_result.calibrator = None
        del mock_model._tuning_result

        plots = adapter.available_plots(mock_model)
        assert "roc-curve" in plots
        assert "calibration" not in plots

    def test_tuning_plot_when_tuned(self) -> None:
        adapter = LizyMLAdapter()
        mock_model = MagicMock()
        mock_model._config.task = "binary"
        mock_model.fit_result.calibrator = None
        mock_model._tuning_result = MagicMock()

        plots = adapter.available_plots(mock_model)
        assert "tuning" in plots


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
    def test_export_model_not_implemented(self) -> None:
        adapter = LizyMLAdapter()
        with pytest.raises(NotImplementedError):
            adapter.export_model(MagicMock(), "/tmp/model.pkl")

    def test_load_model_not_implemented(self) -> None:
        adapter = LizyMLAdapter()
        with pytest.raises(NotImplementedError):
            adapter.load_model("/tmp/model.pkl")

    def test_model_info_not_implemented(self) -> None:
        adapter = LizyMLAdapter()
        with pytest.raises(NotImplementedError):
            adapter.model_info(MagicMock())
