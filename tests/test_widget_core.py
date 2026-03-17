"""Tests for LizyWidget core functionality: load, chaining, summary, properties."""

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


class TestLoadData:
    def test_widget_accepts_injected_adapter(self) -> None:
        from unittest.mock import MagicMock

        adapter = MagicMock()
        adapter.info = BackendInfo(name="custom", version="1.0.0")
        adapter.get_config_schema.return_value = {"type": "object"}

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget(adapter=adapter)
        assert w.backend_info == {"name": "custom", "version": "1.0.0"}

    def test_load_sets_status(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.status == "data_loaded"
        assert w.df_info["target"] == "y"
        assert w.df_info["task"] == "binary"

    def test_load_without_target(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": range(50)})
        w.load(df)
        assert w.status == "data_loaded"
        assert w.df_info["target"] is None

    def test_load_resets_summaries(self) -> None:
        w = _make_widget()
        w.fit_summary = {"some": "data"}
        w.tune_summary = {"some": "data"}
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        assert w.fit_summary == {}
        assert w.tune_summary == {}


class TestChaining:
    def test_load_returns_self(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        result = w.load(df)
        assert result is w

    def test_set_config_returns_self(self) -> None:
        w = _make_widget()
        result = w.set_config({"model": {"name": "lgbm"}})
        assert result is w

    def test_load_inference_returns_self(self) -> None:
        w = _make_widget()
        result = w.load_inference(pd.DataFrame({"x": [1]}))
        assert result is w


class TestSummaryAPI:
    def test_get_fit_summary_empty(self) -> None:
        w = _make_widget()
        assert w.get_fit_summary() is None

    def test_get_fit_summary_returns_type(self) -> None:
        w = _make_widget()
        w.fit_summary = {"metrics": {"rmse": 0.5}, "fold_count": 3, "params": []}
        result = w.get_fit_summary()
        assert isinstance(result, FitSummary)
        assert result.fold_count == 3

    def test_get_tune_summary_empty(self) -> None:
        w = _make_widget()
        assert w.get_tune_summary() is None

    def test_get_tune_summary_returns_type(self) -> None:
        w = _make_widget()
        w.tune_summary = {
            "best_params": {"lr": 0.01},
            "best_score": 0.95,
            "trials": [],
            "metric_name": "auc",
            "direction": "maximize",
        }
        result = w.get_tune_summary()
        assert isinstance(result, TuningSummary)
        assert result.best_score == 0.95

    def test_get_model_none(self) -> None:
        w = _make_widget()
        assert w.get_model() is None


class TestNormalizeMetrics:
    def test_normalize_metrics_basic(self) -> None:
        w = _make_widget()
        records = [
            {"index": "rmse", "if_mean": 0.5, "oof": 0.6, "fold_0": 0.55, "fold_1": 0.65},
            {"index": "mae", "if_mean": 0.3, "oof": 0.4},
        ]
        result = w._normalize_metrics(records)
        assert "rmse" in result
        assert result["rmse"]["is"] == 0.5
        assert result["rmse"]["oos"] == 0.6
        assert "oos_std" in result["rmse"]
        assert "mae" in result
        assert result["mae"]["is"] == 0.3
        assert "oos_std" not in result["mae"]

    def test_normalize_metrics_empty(self) -> None:
        w = _make_widget()
        assert w._normalize_metrics([]) == {}


class TestDfInfoChangeDetection:
    """Verify traitlet change notification fires on every df_info update."""

    def _count_changes(self, w: Any, action: dict[str, Any]) -> int:
        count = [0]

        def on_change(_change: dict[str, Any]) -> None:
            count[0] += 1

        w.observe(on_change, names=["df_info"])
        w.action = action
        w.unobserve(on_change, names=["df_info"])
        return count[0]

    def test_set_target_fires_observe(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        n = self._count_changes(w, {"type": "set_target", "payload": {"target": "y"}})
        assert n >= 1

    def test_set_task_fires_observe(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        n = self._count_changes(w, {"type": "set_task", "payload": {"task": "regression"}})
        assert n >= 1

    def test_update_column_fires_observe(self) -> None:
        w = _make_widget()
        # "feat" has low unique count -> not auto-excluded, toggling excluded triggers a real change
        df = pd.DataFrame({"feat": [0, 1, 2] * 17, "y": [0, 1] * 25 + [0]})
        w.load(df, target="y")
        n = self._count_changes(
            w,
            {
                "type": "update_column",
                "payload": {"name": "feat", "excluded": True, "col_type": "numeric"},
            },
        )
        assert n >= 1

    def test_update_cv_fires_observe(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        n = self._count_changes(
            w,
            {"type": "update_cv", "payload": {"strategy": "kfold", "n_splits": 3}},
        )
        assert n >= 1

    def test_import_yaml_fires_observe(self) -> None:
        import yaml

        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        content = yaml.dump({"data": {"target": "y"}, "split": {"method": "kfold", "n_splits": 3}})
        n = self._count_changes(w, {"type": "import_yaml", "payload": {"content": content}})
        assert n >= 1


class TestModelNameRegression:
    """Regression tests for model.name in initial config (A-2026-03-12)."""

    def test_load_ensures_model_name(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.config.get("model", {}).get("name") == "lgbm"

    def test_load_with_empty_schema_still_has_model_name(self) -> None:
        """Even with a schema that returns empty defaults, model.name is set."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        model = w.config.get("model", {})
        assert model.get("name") == "lgbm"
        assert "params" in model


class TestSetTarget:
    """Cover set_target() public API (widget.py lines 81-86)."""

    def test_set_target_updates_df_info(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        w.set_target("y")
        assert w.df_info["target"] == "y"
        assert w.status == "data_loaded"

    def test_set_target_returns_self(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        result = w.set_target("y")
        assert result is w


class TestProperties:
    """Cover widget properties (widget.py lines 133-156)."""

    def test_task_property(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.task in ("binary", "regression", "multiclass", None)

    def test_task_property_none_without_target(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        assert w.task is None

    def test_cv_method_property(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert isinstance(w.cv_method, str)

    def test_cv_n_splits_property(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert isinstance(w.cv_n_splits, int)
        assert w.cv_n_splits > 0

    def test_df_shape_property(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        shape = w.df_shape
        assert shape == [50, 2]

    def test_df_columns_property(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"a": range(50), "b": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        cols = w.df_columns
        assert isinstance(cols, list)
        assert len(cols) > 0
        names = [c["name"] for c in cols]
        # Target column may be excluded from feature columns list
        assert "a" in names or "b" in names


class TestUpdateColumnError:
    """Cover _handle_update_column error path (widget.py lines 282-283)."""

    def test_update_column_error_sets_error(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # Missing required 'name' key triggers KeyError
        w.action = {"type": "update_column", "payload": {}}
        assert w.error["code"] == "COLUMN_ERROR"

    def test_invalid_col_type_rejected(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {"type": "update_column", "payload": {"name": "x", "col_type": "evil"}}
        assert w.error["code"] == "COLUMN_ERROR"
        assert "Invalid col_type" in w.error["message"]


class TestUpdateCvValidation:
    """Cover _handle_update_cv validation (strategy allowlist, n_splits range)."""

    def test_invalid_strategy_rejected(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {"type": "update_cv", "payload": {"strategy": "../../etc"}}
        assert w.error["code"] == "CV_ERROR"
        assert "Invalid strategy" in w.error["message"]

    def test_n_splits_out_of_range_rejected(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {"type": "update_cv", "payload": {"strategy": "kfold", "n_splits": 0}}
        assert w.error["code"] == "CV_ERROR"
        assert "n_splits" in w.error["message"]


class TestUpdateCvError:
    """Cover _handle_update_cv error path (widget.py lines 301-302)."""

    def test_update_cv_error_sets_error(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # Invalid strategy triggers error in service
        w._service.update_cv = MagicMock(side_effect=ValueError("bad strategy"))
        w.action = {"type": "update_cv", "payload": {"strategy": "bad", "n_splits": 3}}
        assert w.error["code"] == "CV_ERROR"
