"""Tests for WidgetService config routing: apply_best_params and zero-feature guard."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.service import WidgetService
from lizyml_widget.types import BackendInfo


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


class TestApplyBestParams:
    """Tests for Service.apply_best_params() — consolidates config routing logic."""

    def _make_service(self) -> WidgetService:
        adapter = _mock_adapter()
        adapter.classify_best_params = MagicMock(side_effect=LizyMLAdapter().classify_best_params)
        svc = WidgetService(adapter=adapter)
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        svc.load_data(df, target="y")
        return svc

    def test_model_params_merged(self) -> None:
        """Model-category params go to model.params."""
        svc = self._make_service()
        config = svc.initialize_config()
        result = svc.apply_best_params(
            {"learning_rate": 0.05, "max_depth": 7}, config, tune_snapshot=None
        )
        assert result["model"]["params"]["learning_rate"] == 0.05
        assert result["model"]["params"]["max_depth"] == 7

    def test_smart_params_at_model_level(self) -> None:
        """Smart-category params go to model.* (not model.params)."""
        svc = self._make_service()
        config = svc.initialize_config()
        result = svc.apply_best_params(
            {"num_leaves_ratio": 0.7, "min_data_in_leaf_ratio": 0.05},
            config,
            tune_snapshot=None,
        )
        assert result["model"]["num_leaves_ratio"] == 0.7
        assert result["model"]["min_data_in_leaf_ratio"] == 0.05
        # Should NOT be in model.params
        assert "num_leaves_ratio" not in result["model"].get("params", {})

    def test_training_params_routed(self) -> None:
        """Training-category params go to training.early_stopping.*."""
        svc = self._make_service()
        config = svc.initialize_config()
        result = svc.apply_best_params(
            {"early_stopping_rounds": 80, "validation_ratio": 0.2},
            config,
            tune_snapshot=None,
        )
        es = result.get("training", {}).get("early_stopping", {})
        assert es.get("rounds") == 80
        # validation_ratio updates inner_valid.ratio when inner_valid exists
        iv = es.get("inner_valid")
        if iv is not None:
            assert iv["ratio"] == 0.2
        else:
            assert es.get("validation_ratio") == 0.2

    def test_validation_ratio_updates_inner_valid_ratio(self) -> None:
        """When validation_ratio is applied and inner_valid exists, ratio is updated."""
        svc = self._make_service()
        config = svc.initialize_config()
        result = svc.apply_best_params({"validation_ratio": 0.2}, config, tune_snapshot=None)
        es = result.get("training", {}).get("early_stopping", {})
        iv = es.get("inner_valid")
        if iv is not None:
            # inner_valid preserved with updated ratio
            assert iv["ratio"] == 0.2
        else:
            # No inner_valid in base — fallback to validation_ratio
            assert es.get("validation_ratio") == 0.2

    def test_mixed_categories(self) -> None:
        """All 3 categories in one call."""
        svc = self._make_service()
        config = svc.initialize_config()
        result = svc.apply_best_params(
            {
                "learning_rate": 0.05,
                "num_leaves_ratio": 0.7,
                "early_stopping_rounds": 80,
            },
            config,
            tune_snapshot=None,
        )
        assert result["model"]["params"]["learning_rate"] == 0.05
        assert result["model"]["num_leaves_ratio"] == 0.7
        es = result["training"]["early_stopping"]
        assert es["rounds"] == 80

    def test_tune_snapshot_restored(self) -> None:
        """When tune_snapshot is provided, it replaces current config."""
        svc = self._make_service()
        config = svc.initialize_config()
        # Modify config (simulating user changes after tune)
        config["model"]["params"]["n_estimators"] = 9999

        snapshot = copy.deepcopy(config)
        snapshot["model"]["params"]["n_estimators"] = 1500  # original value
        snapshot["evaluation"] = {"metrics": ["auc"]}

        result = svc.apply_best_params({"learning_rate": 0.01}, config, tune_snapshot=snapshot)
        # Should use snapshot, not current config
        assert result["model"]["params"]["n_estimators"] == 1500
        assert result["model"]["params"]["learning_rate"] == 0.01
        assert result.get("evaluation") == {"metrics": ["auc"]}

    def test_tune_snapshot_strips_service_keys(self) -> None:
        """Snapshot should have data/features/split/task stripped."""
        svc = self._make_service()
        config = svc.initialize_config()
        snapshot = copy.deepcopy(config)
        snapshot["data"] = {"target": "y"}
        snapshot["features"] = {"exclude": []}
        snapshot["split"] = {"method": "kfold"}
        snapshot["task"] = "binary"

        result = svc.apply_best_params({"learning_rate": 0.01}, config, tune_snapshot=snapshot)
        assert "data" not in result
        assert "features" not in result
        assert "split" not in result
        assert "task" not in result

    def test_empty_params_returns_canonicalized(self) -> None:
        """Empty params should still return a valid canonicalized config."""
        svc = self._make_service()
        config = svc.initialize_config()
        result = svc.apply_best_params({}, config, tune_snapshot=None)
        assert "model" in result
        assert result["model"]["name"] == "lgbm"

    def test_no_mutation_of_input(self) -> None:
        """Input config must not be mutated."""
        svc = self._make_service()
        config = svc.initialize_config()
        original = copy.deepcopy(config)
        svc.apply_best_params({"learning_rate": 0.05}, config, tune_snapshot=None)
        assert config == original

    def test_wrong_arity_classify_raises(self) -> None:
        """If adapter returns wrong-length tuple, unpacking should fail."""
        svc = self._make_service()
        config = svc.initialize_config()
        svc._adapter.classify_best_params = MagicMock(
            return_value=({"lr": 0.1}, {"ratio": 0.5})  # 2-tuple
        )
        with pytest.raises(ValueError):
            svc.apply_best_params({"lr": 0.1}, config, tune_snapshot=None)


class TestServiceValidationGuards:
    """Service methods must raise ValueError for invalid state."""

    def test_set_target_unknown_column_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        svc.load_data(df, target="y")
        with pytest.raises(ValueError, match="Unknown target column"):
            svc.set_target("nonexistent")

    def test_set_task_invalid_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        svc.load_data(df, target="y")
        with pytest.raises(ValueError, match="Invalid task"):
            svc.set_task("clustering")

    def test_fit_no_data_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        with pytest.raises(ValueError, match="No data loaded"):
            svc.fit({})

    def test_tune_no_data_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        with pytest.raises(ValueError, match="No data loaded"):
            svc.tune({})

    def test_predict_no_model_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        svc.load_data(df, target="y")
        with pytest.raises(ValueError, match="No trained model"):
            svc.predict(df)

    def test_get_plot_no_model_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        with pytest.raises(ValueError, match="No trained model"):
            svc.get_plot("learning-curve")

    def test_save_model_no_model_raises(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        with pytest.raises(ValueError, match="No trained model"):
            svc.save_model("/tmp/model.pkl")

    def test_get_inference_plot_no_adapter_support_raises(self) -> None:
        """When adapter doesn't implement plot_inference, TypeError is raised."""
        adapter = _mock_adapter()
        # Remove plot_inference to simulate missing method
        del adapter.plot_inference
        svc = WidgetService(adapter=adapter)
        with pytest.raises(TypeError, match="not supported"):
            svc.get_inference_plot(pd.DataFrame({"pred": [1, 2]}), "dist")


class TestSetTargetPreservesOverrides:
    """set_target should preserve manual column overrides."""

    def test_manual_exclude_preserved(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"a": range(50), "b": range(50), "y": [0, 1] * 25})
        svc.load_data(df, target="y")
        # Manually exclude column 'a'
        svc.update_column("a", excluded=True, col_type="numeric")
        # Re-set target — manual exclude should be preserved
        info = svc.set_target("y")
        col_a = next(c for c in info["columns"] if c["name"] == "a")
        assert col_a["excluded"] is True

    def test_manual_col_type_preserved(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"a": range(50), "b": range(50), "y": [0, 1] * 25})
        svc.load_data(df, target="y")
        # Manually change col_type
        svc.update_column("b", excluded=False, col_type="categorical")
        info = svc.set_target("y")
        col_b = next(c for c in info["columns"] if c["name"] == "b")
        assert col_b["col_type"] == "categorical"


class TestStratifiedGroupKfold:
    """stratified_group_kfold CV strategy support (LizyML v0.2.0)."""

    def test_build_config_includes_random_state_and_shuffle(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"x": range(50), "g": [0, 1] * 25, "y": [0, 1] * 25})
        svc.load_data(df, target="y")
        svc.update_cv(
            "stratified_group_kfold",
            n_splits=3,
            group_column="g",
            random_state=123,
            shuffle=False,
        )
        config = svc.build_config({"model": {"name": "lgbm"}})
        split = config["split"]
        assert split["method"] == "stratified_group_kfold"
        assert split["n_splits"] == 3
        assert split["random_state"] == 123
        assert split["shuffle"] is False

    def test_build_config_group_col_in_data(self) -> None:
        svc = WidgetService(adapter=_mock_adapter())
        df = pd.DataFrame({"x": range(50), "g": [0, 1] * 25, "y": [0, 1] * 25})
        svc.load_data(df, target="y")
        svc.update_cv("stratified_group_kfold", n_splits=3, group_column="g")
        config = svc.build_config({"model": {"name": "lgbm"}})
        assert config["data"]["group_col"] == "g"

    def test_capabilities_includes_strategy(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        assert "stratified_group_kfold" in caps["cv_strategies"]


class TestBuildConfigCVStrategySplitFields:
    """Cover CV strategy-dependent split field logic (service.py lines 260-274)."""

    def _make_service(self, cv_strategy: str, **cv_kwargs: Any) -> WidgetService:
        adapter = _mock_adapter()
        adapter.classify_best_params = MagicMock(side_effect=LizyMLAdapter().classify_best_params)
        svc = WidgetService(adapter=adapter)
        df = pd.DataFrame({"x": range(50), "t": range(50), "g": [0, 1] * 25, "y": [0, 1] * 25})
        svc.load_data(df, target="y")
        svc.update_cv(cv_strategy, n_splits=3, **cv_kwargs)
        return svc

    def test_purged_time_series_includes_purge_gap_embargo(self) -> None:
        """service.py:267-269: purged_time_series adds purge_gap, embargo."""
        svc = self._make_service(
            "purged_time_series",
            time_column="t",
            purge_gap=10,
            embargo=5,
            train_size_max=100,
            test_size_max=50,
        )
        config = svc.build_config({"model": {"name": "lgbm"}})
        split = config["split"]
        assert split["method"] == "purged_time_series"
        assert split["purge_gap"] == 10
        assert split["embargo"] == 5
        assert split["train_size_max"] == 100
        assert split["test_size_max"] == 50
        assert "gap" not in split
        assert "random_state" not in split

    def test_time_series_includes_gap(self) -> None:
        """service.py:265-266: time_series adds gap."""
        svc = self._make_service("time_series", time_column="t", gap=3)
        config = svc.build_config({"model": {"name": "lgbm"}})
        split = config["split"]
        assert split["method"] == "time_series"
        assert split["gap"] == 3
        assert "purge_gap" not in split
        assert "embargo" not in split

    def test_group_time_series_includes_gap_and_size_limits(self) -> None:
        """service.py:265-266,271-274: group_time_series adds gap + size limits."""
        svc = self._make_service(
            "group_time_series",
            time_column="t",
            group_column="g",
            gap=2,
            train_size_max=200,
        )
        config = svc.build_config({"model": {"name": "lgbm"}})
        split = config["split"]
        assert split["method"] == "group_time_series"
        assert split["gap"] == 2
        assert split["train_size_max"] == 200
        assert config["data"]["group_col"] == "g"
        assert config["data"]["time_col"] == "t"

    def test_kfold_includes_random_state_and_shuffle(self) -> None:
        """service.py:260-264: kfold adds random_state and shuffle."""
        svc = self._make_service("kfold", random_state=99, shuffle=False)
        config = svc.build_config({"model": {"name": "lgbm"}})
        split = config["split"]
        assert split["random_state"] == 99
        assert split["shuffle"] is False
        assert "gap" not in split
        assert "purge_gap" not in split

    def test_stratified_kfold_has_random_state_no_shuffle(self) -> None:
        """service.py:261-264: stratified_kfold adds random_state but not shuffle."""
        svc = self._make_service("stratified_kfold", random_state=42)
        config = svc.build_config({"model": {"name": "lgbm"}})
        split = config["split"]
        assert split["random_state"] == 42
        assert "shuffle" not in split
        assert "gap" not in split


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
