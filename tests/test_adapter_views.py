"""Tests for adapter_views — typed views over lizyml result objects (#116).

The views are the only place in the widget that read internal attributes
off lizyml result types. Their job is to either return a typed value or
raise ``LizyMLContractError`` when a required field is missing — silent
``None`` fallbacks are forbidden at the boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lizyml_widget.adapter_views import (
    LizyMLContractError,
    view_boundary_report,
    view_fit_result,
    view_prediction_result,
    view_rounds,
    view_tune_progress,
    view_tuning_result,
)


class TestViewRounds:
    def test_empty_input_returns_empty_tuple(self) -> None:
        assert view_rounds(()) == ()
        assert view_rounds(None) == ()

    def test_wraps_round_summary(self) -> None:
        r = SimpleNamespace(
            round=2,
            n_trials=30,
            best_score_before=0.85,
            best_score_after=0.91,
            expanded_dims=["lr", "num_leaves"],
        )
        rounds = view_rounds([r])
        assert len(rounds) == 1
        assert rounds[0].round == 2
        assert rounds[0].n_trials == 30
        assert rounds[0].best_score_before == 0.85
        assert rounds[0].best_score_after == 0.91
        assert rounds[0].expanded_dims == ("lr", "num_leaves")

    def test_missing_required_field_raises_contract_error(self) -> None:
        r = SimpleNamespace(round=1, n_trials=10, best_score_after=0.9)
        # best_score_before is optional; expanded_dims is required.
        with pytest.raises(LizyMLContractError, match="expanded_dims"):
            view_rounds([r])

    def test_missing_round_raises_contract_error(self) -> None:
        r = SimpleNamespace(n_trials=10, best_score_after=0.9, expanded_dims=())
        with pytest.raises(LizyMLContractError, match="round"):
            view_rounds([r])


class TestViewBoundaryReport:
    def test_none_returns_none(self) -> None:
        assert view_boundary_report(None) is None

    def test_empty_dims_returns_empty_tuple(self) -> None:
        report = SimpleNamespace(dims=(), expanded_names=())
        result = view_boundary_report(report)
        assert result is not None
        assert result.dims == ()
        assert result.expanded_names == ()

    def test_wraps_dims(self) -> None:
        d = SimpleNamespace(
            name="lr",
            best_value=0.05,
            low=0.001,
            high=0.1,
            position_pct=0.85,
            edge="upper",
            expanded=True,
            new_low=0.001,
            new_high=0.5,
        )
        report = SimpleNamespace(dims=[d], expanded_names=["lr"])
        result = view_boundary_report(report)
        assert result is not None
        assert len(result.dims) == 1
        dim = result.dims[0]
        assert dim.name == "lr"
        assert dim.expanded is True
        assert dim.edge == "upper"
        assert result.expanded_names == ("lr",)

    def test_missing_dim_name_raises(self) -> None:
        d = SimpleNamespace(best_value=0.05, low=0.001, high=0.1)
        report = SimpleNamespace(dims=[d], expanded_names=())
        with pytest.raises(LizyMLContractError, match="name"):
            view_boundary_report(report)


class TestViewTuneProgress:
    def test_required_fields_only(self) -> None:
        info = SimpleNamespace(
            current_trial=5,
            total_trials=50,
        )
        view = view_tune_progress(info)
        assert view.current_trial == 5
        assert view.total_trials == 50
        # Re-tune fields default to round=1 / cumulative=current_trial.
        assert view.round == 1
        assert view.cumulative_trials == 5
        assert view.expanded_dims == ()
        assert view.latest_score is None
        assert view.best_score is None

    def test_all_fields_present(self) -> None:
        info = SimpleNamespace(
            current_trial=12,
            total_trials=100,
            round=2,
            cumulative_trials=70,
            expanded_dims=("lr",),
            latest_score=0.91,
            latest_state="COMPLETE",
            best_score=0.92,
        )
        view = view_tune_progress(info)
        assert view.round == 2
        assert view.cumulative_trials == 70
        assert view.expanded_dims == ("lr",)
        assert view.latest_score == 0.91
        assert view.latest_state == "COMPLETE"
        assert view.best_score == 0.92

    def test_missing_required_field_raises(self) -> None:
        info = SimpleNamespace(total_trials=50)
        with pytest.raises(LizyMLContractError, match="current_trial"):
            view_tune_progress(info)


class TestViewTuningResult:
    def test_full_result(self) -> None:
        result = SimpleNamespace(
            best_params={"lr": 0.05},
            best_score=0.92,
            metric_name="auc",
            direction="maximize",
            rounds=[],
            boundary_report=None,
            trials=[],
        )
        view = view_tuning_result(result)
        assert view.best_params == {"lr": 0.05}
        assert view.best_score == 0.92
        assert view.metric_name == "auc"
        assert view.direction == "maximize"
        assert view.rounds == ()
        assert view.boundary_report is None

    def test_missing_best_score_raises(self) -> None:
        result = SimpleNamespace(
            best_params={},
            metric_name="auc",
            direction="maximize",
            rounds=[],
            boundary_report=None,
            trials=[],
        )
        with pytest.raises(LizyMLContractError, match="best_score"):
            view_tuning_result(result)


class TestViewFitResult:
    def test_with_splits(self) -> None:
        result = SimpleNamespace(
            metrics={"auc": {"oos": 0.9}},
            splits=SimpleNamespace(outer=[1, 2, 3, 4, 5]),
        )
        view = view_fit_result(result)
        assert view.metrics == {"auc": {"oos": 0.9}}
        assert view.fold_count == 5

    def test_missing_splits_falls_back_to_zero_folds(self) -> None:
        result = SimpleNamespace(metrics={"auc": {}})
        # splits is optional; absent → fold_count=0.
        view = view_fit_result(result)
        assert view.fold_count == 0

    def test_missing_metrics_raises(self) -> None:
        result = SimpleNamespace(splits=SimpleNamespace(outer=[]))
        with pytest.raises(LizyMLContractError, match="metrics"):
            view_fit_result(result)


class TestViewPredictionResult:
    def test_full_result(self) -> None:
        result = SimpleNamespace(
            pred=[1, 0, 1],
            proba=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
            shap_values=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            warnings=["w1"],
        )
        view = view_prediction_result(result)
        assert list(view.pred) == [1, 0, 1]
        assert view.proba is not None
        assert view.shap_values is not None
        assert view.warnings == ["w1"]

    def test_minimal_result(self) -> None:
        result = SimpleNamespace(pred=[0, 1])
        # proba / shap_values / warnings all optional.
        view = view_prediction_result(result)
        assert list(view.pred) == [0, 1]
        assert view.proba is None
        assert view.shap_values is None
        assert view.warnings == []

    def test_missing_pred_raises(self) -> None:
        result = SimpleNamespace(proba=[[0.5, 0.5]])
        with pytest.raises(LizyMLContractError, match="pred"):
            view_prediction_result(result)
