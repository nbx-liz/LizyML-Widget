"""Tests for LizyWidget Python API methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from lizyml_widget.types import BackendInfo, ConfigSchema, FitSummary, TuningSummary


def _make_widget() -> Any:
    """Create a LizyWidget with mocked LizyML backend."""
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = ConfigSchema(json_schema={"type": "object"})
        adapter.validate_config.return_value = []

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


class TestLoadData:
    def test_load_sets_status(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.status == "data_loaded"
        assert w.df_info["target"] == "y"
        assert w.df_info["task"] == "binary"

    def test_load_without_target(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        w.load(df)
        assert w.status == "data_loaded"
        assert w.df_info["target"] is None

    def test_load_resets_summaries(self) -> None:
        w = _make_widget()
        w.fit_summary = {"some": "data"}
        w.tune_summary = {"some": "data"}
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
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


class TestConfig:
    def test_set_get_config(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm"}})
        cfg = w.get_config()
        assert cfg["model"] == {"name": "lgbm"}
        # set_config preserves config_version if not explicitly supplied
        assert cfg["config_version"] == 1

    def test_get_config_returns_copy(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm"}})
        cfg = w.get_config()
        cfg["extra"] = True
        assert "extra" not in w.get_config()


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


class TestConfigVersion:
    def test_load_includes_config_version(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.config.get("config_version") == 1

    def test_config_version_preserved_after_update(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_config",
            "payload": {"config": {**w.config, "model": {"name": "lgbm"}}},
        }
        assert w.config.get("config_version") == 1


class TestInference:
    def test_load_inference(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [1, 2, 3]})
        w.load_inference(df)
        assert w.inference_result["status"] == "ready"
        assert w.inference_result["rows"] == 3


class TestActionDispatch:
    def test_set_target_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df)
        w.action = {"type": "set_target", "payload": {"target": "y"}}
        assert w.df_info["target"] == "y"

    def test_update_column_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_column",
            "payload": {"name": "x", "excluded": True, "col_type": "numeric"},
        }
        x_col = next(c for c in w.df_info["columns"] if c["name"] == "x")
        assert x_col["excluded"] is True

    def test_set_task_action_updates_cv_strategy(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(100), "y": range(100)})
        w.load(df, target="y")
        assert w.df_info["cv"]["strategy"] == "kfold"

        w.action = {"type": "set_task", "payload": {"task": "binary"}}
        assert w.df_info["task"] == "binary"
        assert w.df_info["cv"]["strategy"] == "stratified_kfold"

    def test_update_cv_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_cv",
            "payload": {"strategy": "group_kfold", "n_splits": 3, "group_column": "x"},
        }
        assert w.df_info["cv"]["strategy"] == "group_kfold"
        assert w.df_info["cv"]["n_splits"] == 3

    def test_update_config_action(self) -> None:
        w = _make_widget()
        w.action = {
            "type": "update_config",
            "payload": {"config": {"model": {"name": "rf"}}},
        }
        assert w.config == {"model": {"name": "rf"}}

    def test_unknown_action_ignored(self) -> None:
        w = _make_widget()
        # Should not raise
        w.action = {"type": "nonexistent", "payload": {}}

    def test_empty_action_ignored(self) -> None:
        w = _make_widget()
        w.action = {}

    def test_apply_best_params_action(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm", "params": {"lr": 0.1}}})
        w.action = {
            "type": "apply_best_params",
            "payload": {"params": {"lr": 0.01, "depth": 5}},
        }
        assert w.config["model"]["params"]["lr"] == 0.01
        assert w.config["model"]["params"]["depth"] == 5

    def test_request_inference_plot_action(self) -> None:
        w = _make_widget()
        # Should not raise even without a model
        w.action = {
            "type": "request_inference_plot",
            "payload": {"plot_type": "roc-curve"},
        }


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
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df)
        n = self._count_changes(w, {"type": "set_target", "payload": {"target": "y"}})
        assert n >= 1

    def test_set_task_fires_observe(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        n = self._count_changes(w, {"type": "set_task", "payload": {"task": "regression"}})
        assert n >= 1

    def test_update_column_fires_observe(self) -> None:
        w = _make_widget()
        # "feat" has low unique count → not auto-excluded, toggling excluded triggers a real change
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
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        n = self._count_changes(
            w,
            {"type": "update_cv", "payload": {"strategy": "kfold", "n_splits": 3}},
        )
        assert n >= 1

    def test_import_yaml_fires_observe(self) -> None:
        import yaml

        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df)
        content = yaml.dump({"data": {"target": "y"}, "split": {"method": "kfold", "n_splits": 3}})
        n = self._count_changes(w, {"type": "import_yaml", "payload": {"content": content}})
        assert n >= 1


class TestModelNameRegression:
    """Regression tests for model.name in initial config (A-2026-03-12)."""

    def test_load_ensures_model_name(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.config.get("model", {}).get("name") == "lgbm"

    def test_load_with_empty_schema_still_has_model_name(self) -> None:
        """Even with a schema that returns empty defaults, model.name is set."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        model = w.config.get("model", {})
        assert model.get("name") == "lgbm"
        assert "params" in model


class TestErrorCodes:
    """Test BLUEPRINT §6.1 error codes (NO_DATA, NO_TARGET)."""

    def test_fit_no_data_returns_no_data_error(self) -> None:
        w = _make_widget()
        # Trigger fit without loading data
        w.action = {"type": "fit", "payload": {}}
        assert w.error.get("code") == "NO_DATA"
        assert w.status == "failed"

    def test_fit_no_target_returns_no_target_error(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df)  # no target
        w.action = {"type": "fit", "payload": {}}
        assert w.error.get("code") == "NO_TARGET"
        assert w.status == "failed"

    def test_tune_no_data_returns_no_data_error(self) -> None:
        w = _make_widget()
        w.action = {"type": "tune", "payload": {}}
        assert w.error.get("code") == "NO_DATA"
        assert w.status == "failed"
