"""Tests for WidgetService auto-detection and data management logic."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.service import WidgetService
from lizyml_widget.types import BackendInfo, ConfigPatchOp


def _mock_adapter() -> Any:
    """Create a mock adapter that delegates config lifecycle to LizyMLAdapter."""
    real_adapter = LizyMLAdapter()
    adapter = MagicMock()
    adapter.info = BackendInfo(name="mock", version="0.0.0")
    adapter.get_config_schema.return_value = {}

    # Delegate config lifecycle methods to real adapter
    adapter.initialize_config.side_effect = real_adapter.initialize_config
    adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
    adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
    adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
    adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
    adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
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
        df = pd.DataFrame({"y": list(range(5)) * 20, "x": [i % 10 for i in range(n)]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        # 5 unique with threshold = max(20, 100*0.05) = 20 → 5 <= 20 → multiclass
        assert info["task"] == "multiclass"

    def test_regression_numeric_high_unique(self) -> None:
        n = 100
        df = pd.DataFrame({"y": range(n), "x": [i % 10 for i in range(n)]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        # 100 unique with threshold = max(20, 100*0.05) = 20 → 100 > 20 → regression
        assert info["task"] == "regression"

    def test_regression_float_target(self) -> None:
        n = 200
        df = pd.DataFrame({"y": [i * 0.1 for i in range(n)], "x": [i % 10 for i in range(n)]})
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
        df = pd.DataFrame({"const": [42] * n, "x": [i % 10 for i in range(n)], "y": [0, 1] * 50})
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
        df = pd.DataFrame({"y": [0, 1] * 50, "x": [i % 10 for i in range(100)]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["cv"]["strategy"] == "stratified_kfold"

    def test_regression_uses_kfold(self) -> None:
        n = 200
        df = pd.DataFrame({"y": range(n), "x": [i % 10 for i in range(n)]})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        assert info["cv"]["strategy"] == "kfold"

    def test_set_target_sets_auto_task_and_strategy(self) -> None:
        df = pd.DataFrame({"y": [0, 1] * 50, "x": [i % 10 for i in range(100)]})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df)
        info = svc.set_target("y")
        assert info["task"] == "binary"
        assert info["auto_task"] == "binary"
        assert info["cv"]["strategy"] == "stratified_kfold"

    def test_switch_target_preserves_previous_target_in_columns(self) -> None:
        """Switching target should restore the previously excluded target column."""
        df = pd.DataFrame(
            {
                "x": range(100),
                "y": [0, 1] * 50,
                "z": range(100),
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df)

        # Set first target
        info1 = svc.set_target("y")
        cols1 = [c["name"] for c in info1["columns"]]
        assert "y" not in cols1
        assert "x" in cols1
        assert "z" in cols1

        # Switch to second target
        info2 = svc.set_target("z")
        cols2 = [c["name"] for c in info2["columns"]]
        assert "z" not in cols2
        assert "y" in cols2, "Previous target 'y' should be restored in columns"
        assert "x" in cols2

        # Re-select first target (must be possible)
        info3 = svc.set_target("y")
        cols3 = [c["name"] for c in info3["columns"]]
        assert "y" not in cols3
        assert "z" in cols3, "Previous target 'z' should be restored in columns"
        assert "x" in cols3

    def test_switch_target_preserves_column_settings(self) -> None:
        """Manual column settings (excluded, col_type) should survive target switching."""
        df = pd.DataFrame(
            {
                "x": range(100),
                "y": [0, 1] * 50,
                "z": range(100),
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")

        # Manually exclude column x
        svc.update_column("x", excluded=True, col_type="numeric")
        info_before = svc.set_target("z")
        cols_before = {c["name"]: c for c in info_before["columns"]}
        # x should be excluded AND y should be back
        assert cols_before["x"]["excluded"] is True
        assert "y" in cols_before
        assert "z" not in cols_before  # new target

    def test_set_task_updates_strategy_to_stratified_for_classification(self) -> None:
        n = 100
        df = pd.DataFrame({"y": range(n), "x": [i % 10 for i in range(n)]})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.set_task("multiclass")
        assert info["task"] == "multiclass"
        assert info["cv"]["strategy"] == "stratified_kfold"

    def test_set_task_resets_cv_fields_to_task_defaults(self) -> None:
        df = pd.DataFrame(
            {
                "y": [0, 1] * 50,
                "x": [i % 10 for i in range(100)],
                "g": ["a", "b"] * 50,
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        svc.update_cv("group_kfold", 3, group_column="g")
        info = svc.set_task("regression")
        assert info["cv"]["strategy"] == "kfold"
        assert info["cv"]["n_splits"] == 3
        # group_column is preserved across task changes (H3 fix)
        assert info["cv"]["group_column"] == "g"


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
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_column("x", excluded=True, col_type="numeric")
        x_col = next(c for c in info["columns"] if c["name"] == "x")
        assert x_col["excluded"] is True

    def test_update_cv(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_cv("group_kfold", 3, group_column="x")
        assert info["cv"]["strategy"] == "group_kfold"
        assert info["cv"]["n_splits"] == 3
        assert info["cv"]["group_column"] == "x"

    def test_update_cv_time_series_fields(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
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
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": range(50)})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df)
        assert info["shape"] == [50, 2]

    def test_load_data_without_target(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": range(50)})
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
        df = pd.DataFrame(
            {
                "x": [i % 10 for i in range(50)],
                "g": ["a", "b"] * 25,
                "y": [0, 1] * 25,
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        svc.update_cv("group_kfold", 5, group_column="g")
        config = svc.build_config({})
        assert config["data"]["group_col"] == "g"
        assert config["split"]["method"] == "group_kfold"

    def test_build_config_time_series(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3] * 17 + [1], "t": range(52), "y": range(52)})
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
        df = pd.DataFrame({"num": [1, 2, 3] * 34, "y": [0, 1] * 51})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        config = svc.build_config({"config_version": 1, "model": {"name": "lgbm"}})
        assert config["config_version"] == 1
        assert config["task"] == "binary"


class TestReturnValueIndependence:
    """Verify each mutating method returns an independent copy of df_info."""

    def test_load_data_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df)
        info["shape"] = [0, 0]
        assert svc._df_info["shape"] == [50, 2]

    def test_set_target_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df)
        info = svc.set_target("y")
        info["target"] = "tampered"
        assert svc._df_info["target"] == "y"

    def test_set_task_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.set_task("regression")
        info["task"] = "tampered"
        assert svc._df_info["task"] == "regression"

    def test_update_column_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_column("x", excluded=True, col_type="numeric")
        info["columns"].clear()
        assert len(svc._df_info["columns"]) > 0

    def test_update_cv_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.update_cv("group_kfold", 3, group_column="x")
        info["cv"]["strategy"] = "tampered"
        assert svc._df_info["cv"]["strategy"] == "group_kfold"

    def test_get_df_info_returns_independent_copy(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        info = svc.get_df_info()
        info["target"] = "tampered"
        assert svc._df_info["target"] == "y"


class TestModelNameBackfill:
    """Regression tests for model.name missing (A-2026-03-12)."""

    def test_build_config_backfills_model_name_when_missing(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3] * 17 + [1], "y": [0, 1] * 26})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        config = svc.build_config({"model": {"params": {"n_estimators": 100}}})
        assert config["model"]["name"] == "lgbm"

    def test_build_config_preserves_existing_model_name(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3] * 17 + [1], "y": [0, 1] * 26})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        config = svc.build_config({"model": {"name": "lgbm", "params": {}}})
        assert config["model"]["name"] == "lgbm"

    def test_build_config_adds_model_when_absent(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3] * 17 + [1], "y": [0, 1] * 26})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        config = svc.build_config({})
        assert config["model"]["name"] == "lgbm"


class TestServiceOwnedConfigLifecycle:
    """Config initialization and run preparation delegated to Adapter."""

    def test_initialize_config_populates_defaults(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")

        config = svc.initialize_config()

        assert config["config_version"] == 1
        assert config["model"]["name"] == "lgbm"
        assert config["model"]["params"]["objective"] == "binary"

    def test_apply_task_params_returns_updated_copy(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        config = {"model": {"name": "lgbm", "params": {"n_estimators": 100}}}

        updated = svc.apply_task_params(config, "binary")

        assert updated["model"]["params"]["objective"] == "binary"
        assert updated["model"]["params"]["metric"] == ["auc", "binary_logloss"]
        assert "objective" not in config["model"]["params"]

    def test_apply_config_patch_delegates_to_adapter(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        config = {"model": {"params": {"learning_rate": 0.001}}}
        ops = [ConfigPatchOp(op="set", path="model.params.learning_rate", value=0.01)]

        result = svc.apply_config_patch(config, ops)

        assert result["model"]["params"]["learning_rate"] == 0.01

    def test_prepare_run_config_complements_tune_defaults(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3] * 17 + [1], "y": [0, 1] * 26})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")

        config = svc.prepare_run_config({"model": {"name": "lgbm"}}, job_type="tune")

        assert config["tuning"]["optuna"]["params"]["n_trials"] == 50
        assert config["tuning"]["optuna"]["space"] == {}

    def test_apply_loaded_config_updates_service_state(self) -> None:
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df)

        remaining = svc.apply_loaded_config(
            {
                "data": {"target": "y", "time_col": "x"},
                "split": {"method": "time_series", "n_splits": 3, "gap": 2},
                "model": {"name": "lgbm"},
            }
        )

        info = svc.get_df_info()
        assert info["target"] == "y"
        assert info["cv"]["strategy"] == "time_series"
        assert info["cv"]["time_column"] == "x"
        assert info["cv"]["gap"] == 2
        # apply_loaded_config now returns canonicalized config with all defaults
        assert remaining["model"]["name"] == "lgbm"
        assert "config_version" in remaining


class TestCanonicalizeConfigDelegation:
    """Phase 26: Service.canonicalize_config delegates to adapter."""

    def test_canonicalize_config_produces_canonical(self) -> None:
        adapter = _mock_adapter()
        svc = WidgetService(adapter)
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        svc.load_data(df)
        result = svc.canonicalize_config({"model": {"params": {"n_estimators": 500}}})
        assert result["model"]["name"] == "lgbm"
        assert result["model"]["params"]["n_estimators"] == 500

    def test_build_config_no_auto_num_leaves_logic(self) -> None:
        """build_config should not enforce auto_num_leaves (adapter responsibility)."""
        adapter = _mock_adapter()
        svc = WidgetService(adapter)
        df = pd.DataFrame({"x": [1, 2, 3] * 17 + [1], "y": [0, 1] * 26})
        svc.load_data(df)
        svc.set_target("y")
        # build_config just merges df_info; auto_num_leaves is adapter's job
        user_cfg: dict[str, Any] = {
            "model": {"name": "lgbm", "auto_num_leaves": True, "params": {"num_leaves": 256}},
        }
        result = svc.build_config(user_cfg)
        # build_config should pass through as-is (adapter handles exclusivity later)
        assert result["model"]["params"]["num_leaves"] == 256

    def test_apply_task_params_delegates_to_adapter(self) -> None:
        """26-4: apply_task_params delegates to adapter.apply_task_defaults."""
        adapter = _mock_adapter()
        svc = WidgetService(adapter)
        config = {"model": {"name": "lgbm", "params": {"n_estimators": 100}}}
        result = svc.apply_task_params(config, "binary")
        assert result["model"]["params"]["objective"] == "binary"
        assert result["model"]["params"]["n_estimators"] == 100

    def test_service_has_no_backend_specific_constants(self) -> None:
        """26-4: Service source must not contain backend-specific constants."""
        import inspect

        from lizyml_widget import service

        source = inspect.getsource(service)
        # These backend-specific strings should not appear as literals in service
        assert '"lgbm"' not in source
        assert '"objective"' not in source
        assert '"metric"' not in source


class TestFloatIdDetection:
    """Float columns with all-unique values should NOT be excluded as IDs."""

    def test_float_column_not_excluded_as_id(self) -> None:
        import numpy as np

        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "feat_float": np.random.randn(n),
                "y": [0, 1] * 50,
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        feat = next(c for c in info["columns"] if c["name"] == "feat_float")
        assert feat["excluded"] is False
        assert feat["exclude_reason"] is None

    def test_int_column_with_all_unique_still_excluded(self) -> None:
        n = 100
        df = pd.DataFrame({"id": range(n), "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        id_col = next(c for c in info["columns"] if c["name"] == "id")
        assert id_col["excluded"] is True
        assert id_col["exclude_reason"] == "id"

    def test_string_column_with_all_unique_still_excluded(self) -> None:
        n = 100
        df = pd.DataFrame(
            {
                "uuid": [f"id-{i}" for i in range(n)],
                "y": [0, 1] * 50,
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        uuid_col = next(c for c in info["columns"] if c["name"] == "uuid")
        assert uuid_col["excluded"] is True
        assert uuid_col["exclude_reason"] == "id"

    def test_multiple_float_features_not_excluded(self) -> None:
        import numpy as np

        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
                "y": [0, 1] * 100,
            }
        )
        svc = WidgetService(adapter=_mock_adapter())
        info = svc.load_data(df, target="y")
        active = [c for c in info["columns"] if not c["excluded"]]
        assert len(active) == 2


# ── Fix 3: group_col round-trip ───────────────────────────────


class TestGroupColRoundTrip:
    """group_col from data section must survive export/import."""

    def test_apply_loaded_config_reads_group_col_from_data_section(self) -> None:
        """apply_loaded_config must read group_col from data section (export format)."""
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25, "grp": ["a", "b"] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        svc.update_cv("group_kfold", 5, group_column="grp")

        # Export config
        config = svc.build_config(svc.canonicalize_config({}))
        assert config.get("data", {}).get("group_col") == "grp"

        # Reset CV and re-import
        svc.update_cv("kfold", 5)
        assert svc._df_info["cv"]["group_column"] is None

        # Import the exported config
        svc.apply_loaded_config(config)
        assert svc._df_info["cv"]["group_column"] == "grp"


class TestApplyLoadedConfigFeaturesAndTask:
    """apply_loaded_config must restore features and task from imported config."""

    def test_restores_excluded_features(self) -> None:
        """Excluded columns in features section should be reflected in df_info."""
        import numpy as np

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")

        # Verify b is not excluded initially
        col_b = next(c for c in svc._df_info["columns"] if c["name"] == "b")
        assert not col_b["excluded"]

        # Import config with b excluded
        config = {"features": {"exclude": ["b"], "categorical": []}}
        svc.apply_loaded_config(config)

        col_b_after = next(c for c in svc._df_info["columns"] if c["name"] == "b")
        assert col_b_after["excluded"]

    def test_restores_categorical_type(self) -> None:
        """Categorical columns in features section should update col_type."""
        df = pd.DataFrame({"a": range(50), "b": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")

        col_a = next(c for c in svc._df_info["columns"] if c["name"] == "a")
        assert col_a["col_type"] != "categorical"

        config = {"features": {"exclude": [], "categorical": ["a"]}}
        svc.apply_loaded_config(config)

        col_a_after = next(c for c in svc._df_info["columns"] if c["name"] == "a")
        assert col_a_after["col_type"] == "categorical"

    def test_restores_task_override(self) -> None:
        """Explicit task in imported config should override auto-detected task."""
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        assert svc._df_info["task"] == "binary"

        config = {"task": "regression"}
        svc.apply_loaded_config(config)

        assert svc._df_info["task"] == "regression"


class TestZeroFeatureGuard:
    """build_config should raise ValueError when all features are excluded."""

    def test_all_excluded_raises(self) -> None:
        n = 100
        df = pd.DataFrame({"id": range(n), "y": [0, 1] * 50})
        svc = WidgetService(adapter=_mock_adapter())
        svc.load_data(df, target="y")
        # All non-target columns are excluded as IDs
        with pytest.raises(ValueError, match="No features available"):
            svc.build_config({"model": {"name": "lgbm"}})
