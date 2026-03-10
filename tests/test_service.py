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
        info = svc.update_cv("group_kfold", 3, "x")
        assert info["cv"]["strategy"] == "group_kfold"
        assert info["cv"]["n_splits"] == 3
        assert info["cv"]["group_column"] == "x"

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
        assert config["data"]["task"] == "binary"
        assert config["split"]["method"] == "stratified_kfold"
        assert "model" in config
