"""Tests for WidgetService auto-detection and data management logic."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lizyml_widget.service import WidgetService
from lizyml_widget.types import BackendInfo, ConfigSchema


def _mock_adapter() -> Any:
    adapter = MagicMock()
    adapter.info = BackendInfo(name="mock", version="0.0.0")
    adapter.get_config_schema.return_value = ConfigSchema(json_schema={})
    return adapter


# ── Task detection ───────────────────────────────────────────


class TestDetectTask:
    def test_binary_two_unique(self) -> None:
        df = pd.DataFrame({"y": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["task"] == "binary"

    def test_binary_string_two_unique(self) -> None:
        df = pd.DataFrame({"y": ["yes", "no", "yes", "no"], "x": [1, 2, 3, 4]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["task"] == "binary"

    def test_multiclass_object_dtype(self) -> None:
        df = pd.DataFrame({"y": ["a", "b", "c", "a"], "x": [1, 2, 3, 4]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["task"] == "multiclass"

    def test_multiclass_numeric_low_unique(self) -> None:
        n = 100
        df = pd.DataFrame({"y": list(range(5)) * 20, "x": range(n)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        # 5 unique with threshold = max(20, 100*0.05) = 20 → 5 <= 20 → multiclass
        assert info["task"] == "multiclass"

    def test_regression_numeric_high_unique(self) -> None:
        n = 100
        df = pd.DataFrame({"y": range(n), "x": range(n)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        # 100 unique with threshold = max(20, 100*0.05) = 20 → 100 > 20 → regression
        assert info["task"] == "regression"

    def test_regression_float_target(self) -> None:
        n = 200
        df = pd.DataFrame({"y": [i * 0.1 for i in range(n)], "x": range(n)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["task"] == "regression"


# ── Column auto-configuration ────────────────────────────────


class TestAutoConfigureColumns:
    def test_exclude_id_column(self) -> None:
        n = 100
        df = pd.DataFrame({"id": range(n), "x": [1] * n, "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        id_col = next(c for c in info["columns"] if c["name"] == "id")
        assert id_col["excluded"] is True
        assert id_col["exclude_reason"] == "id"

    def test_exclude_constant_column(self) -> None:
        n = 100
        df = pd.DataFrame({"const": [42] * n, "x": range(n), "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        const_col = next(c for c in info["columns"] if c["name"] == "const")
        assert const_col["excluded"] is True
        assert const_col["exclude_reason"] == "constant"

    def test_object_column_is_categorical(self) -> None:
        df = pd.DataFrame({"city": ["a", "b", "c", "a"], "y": [0, 1, 0, 1]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        city_col = next(c for c in info["columns"] if c["name"] == "city")
        assert city_col["col_type"] == "categorical"

    def test_numeric_low_unique_is_categorical(self) -> None:
        df = pd.DataFrame({"cat": [1, 2, 3] * 33 + [1], "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        cat_col = next(c for c in info["columns"] if c["name"] == "cat")
        # 3 unique <= max(20, 100*0.05) = 20 → categorical
        assert cat_col["col_type"] == "categorical"

    def test_numeric_high_unique_is_numeric(self) -> None:
        n = 100
        df = pd.DataFrame({"feat": range(n), "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        feat_col = next(c for c in info["columns"] if c["name"] == "feat")
        # Excluded as ID, but col_type would be numeric if not excluded
        # Actually 100 unique == 100 rows → excluded as ID
        assert feat_col["excluded"] is True

    def test_target_excluded_from_columns(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        names = [c["name"] for c in info["columns"]]
        assert "y" not in names
        assert "x" in names


# ── CV defaults ──────────────────────────────────────────────


class TestCVDefaults:
    def test_binary_uses_stratified(self) -> None:
        df = pd.DataFrame({"y": [0, 1] * 50, "x": range(100)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["cv"]["strategy"] == "stratified_kfold"

    def test_regression_uses_kfold(self) -> None:
        n = 200
        df = pd.DataFrame({"y": range(n), "x": range(n)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["cv"]["strategy"] == "kfold"

    def test_set_target_sets_auto_task_and_strategy(self) -> None:
        df = pd.DataFrame({"y": [0, 1] * 50, "x": range(100)})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df)
        info = svc.set_target("y")
        assert info["task"] == "binary"
        assert info["auto_task"] == "binary"
        assert info["cv"]["strategy"] == "stratified_kfold"

    def test_set_task_updates_strategy_to_stratified_for_classification(self) -> None:
        n = 100
        df = pd.DataFrame({"y": range(n), "x": range(n)})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.set_task("multiclass")
        assert info["task"] == "multiclass"
        assert info["cv"]["strategy"] == "stratified_kfold"

    def test_set_task_resets_cv_fields_to_task_defaults(self) -> None:
        df = pd.DataFrame({"y": [0, 1] * 50, "x": range(100), "g": ["a", "b"] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        svc.update_cv("group_kfold", 3, group_column="g")
        info = svc.set_task("regression")
        assert info["cv"]["strategy"] == "kfold"
        assert info["cv"]["n_splits"] == 3
        assert info["cv"]["group_column"] is None


# ── Feature summary ──────────────────────────────────────────


class TestFeatureSummary:
    def test_counts(self) -> None:
        n = 100
        df = pd.DataFrame(
            {
                "id": range(n),
                "const": [1] * n,
                "num1": range(n),
                "num2": range(n),
                "cat1": ["a", "b"] * 50,
                "y": [0, 1] * 50,
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        fs = info["feature_summary"]
        # id → excluded (ID), const → excluded (constant)
        # num1 → excluded (ID, 100 unique == 100 rows), num2 → same
        # cat1 → active, categorical
        assert fs["excluded_id"] >= 1
        assert fs["excluded_const"] == 1
        assert fs["categorical"] >= 1


# ── Data management ──────────────────────────────────────────


class TestDataManagement:
    def test_update_column(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_column("x", excluded=True, col_type="numeric")
        x_col = next(c for c in info["columns"] if c["name"] == "x")
        assert x_col["excluded"] is True

    def test_update_cv(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_cv("group_kfold", 3, group_column="x")
        assert info["cv"]["strategy"] == "group_kfold"
        assert info["cv"]["n_splits"] == 3
        assert info["cv"]["group_column"] == "x"

    def test_update_cv_time_series_fields(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_cv(
            "purged_time_series",
            5,
            time_column="x",
            purge_gap=3,
            embargo=2,
        )
        assert info["cv"]["strategy"] == "purged_time_series"
        assert info["cv"]["time_column"] == "x"
        assert info["cv"]["purge_gap"] == 3
        assert info["cv"]["embargo"] == 2

    def test_shape_info(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df)
        assert info["shape"] == [50, 2]

    def test_load_data_without_target(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df)
        assert info["target"] is None
        assert info["task"] is None

    def test_set_target_no_data_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        with pytest.raises(ValueError, match="No data loaded"):
            svc.set_target("y")

    def test_build_config(self) -> None:
        df = pd.DataFrame({"num": range(100), "cat": ["a", "b"] * 50, "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        config = svc.build_config({"model": {"name": "lgbm"}})
        assert config["data"]["target"] == "y"
        assert "task" not in config.get("data", {})
        assert config["task"] == "binary"
        assert config["split"]["method"] == "stratified_kfold"
        assert config["split"]["random_state"] == 42
        assert "model" in config

    def test_build_config_group_kfold(self) -> None:
        df = pd.DataFrame({"x": range(50), "g": ["a", "b"] * 25, "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        svc.update_cv("group_kfold", 5, group_column="g")
        config = svc.build_config({})
        assert config["data"]["group_col"] == "g"
        assert config["split"]["method"] == "group_kfold"

    def test_build_config_time_series(self) -> None:
        df = pd.DataFrame({"x": range(50), "t": range(50), "y": range(50)})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        svc.update_cv(
            "time_series",
            5,
            time_column="t",
            gap=2,
            train_size_max=30,
        )
        config = svc.build_config({})
        assert config["data"]["time_col"] == "t"
        assert config["split"]["gap"] == 2
        assert config["split"]["train_size_max"] == 30
        assert config["task"] == "regression"

    def test_build_config_preserves_config_version(self) -> None:
        df = pd.DataFrame({"num": range(100), "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        config = svc.build_config({"config_version": 1, "model": {"name": "lgbm"}})
        assert config["config_version"] == 1
        assert config["task"] == "binary"


class TestReturnValueIndependence:
    """Verify each mutating method returns an independent copy of df_info."""

    def test_load_data_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df)
        info["shape"] = [0, 0]
        assert svc._df_info["shape"] == [50, 2]

    def test_set_target_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df)
        info = svc.set_target("y")
        info["target"] = "tampered"
        assert svc._df_info["target"] == "y"

    def test_set_task_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.set_task("regression")
        info["task"] = "tampered"
        assert svc._df_info["task"] == "regression"

    def test_update_column_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_column("x", excluded=True, col_type="numeric")
        info["columns"].clear()
        assert len(svc._df_info["columns"]) > 0

    def test_update_cv_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_cv("group_kfold", 3, group_column="x")
        info["cv"]["strategy"] = "tampered"
        assert svc._df_info["cv"]["strategy"] == "group_kfold"

    def test_get_df_info_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.get_df_info()
        info["target"] = "tampered"
        assert svc._df_info["target"] == "y"
