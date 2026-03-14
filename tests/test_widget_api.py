"""Tests for LizyWidget Python API methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo, FitSummary, PlotData, PredictionSummary, TuningSummary


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


class TestConfig:
    def test_set_get_config(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm"}})
        cfg = w.get_config()
        assert cfg["model"]["name"] == "lgbm"
        # set_config canonicalizes and preserves config_version
        assert cfg["config_version"] == 1
        assert "params" in cfg["model"]

    def test_get_config_returns_copy(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm", "params": {}}})
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
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.config.get("config_version") == 1

    def test_config_version_preserved_after_patch(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "set", "path": "model.name", "value": "lgbm"}]},
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
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        w.action = {"type": "set_target", "payload": {"target": "y"}}
        assert w.df_info["target"] == "y"

    def test_update_column_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_column",
            "payload": {"name": "x", "excluded": True, "col_type": "numeric"},
        }
        x_col = next(c for c in w.df_info["columns"] if c["name"] == "x")
        assert x_col["excluded"] is True

    def test_set_task_action_updates_cv_strategy(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(100)], "y": range(100)})
        w.load(df, target="y")
        assert w.df_info["cv"]["strategy"] == "kfold"

        w.action = {"type": "set_task", "payload": {"task": "binary"}}
        assert w.df_info["task"] == "binary"
        assert w.df_info["cv"]["strategy"] == "stratified_kfold"

    def test_update_cv_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_cv",
            "payload": {"strategy": "group_kfold", "n_splits": 3, "group_column": "x"},
        }
        assert w.df_info["cv"]["strategy"] == "group_kfold"
        assert w.df_info["cv"]["n_splits"] == 3

    def test_patch_config_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "patch_config",
            "payload": {
                "ops": [{"op": "set", "path": "model.params.learning_rate", "value": 0.05}],
            },
        }
        assert w.config["model"]["params"]["learning_rate"] == 0.05

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
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)  # no target
        w.action = {"type": "fit", "payload": {}}
        assert w.error.get("code") == "NO_TARGET"
        assert w.status == "failed"

    def test_tune_no_data_returns_no_data_error(self) -> None:
        w = _make_widget()
        w.action = {"type": "tune", "payload": {}}
        assert w.error.get("code") == "NO_DATA"
        assert w.status == "failed"


class TestTuneDefaults:
    """Regression tests for tuning default complement (P-004 R1)."""

    def test_tune_complements_missing_tuning_config(self) -> None:
        """R1: load() with target auto-populates tuning defaults; tune() uses them."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # Config should have tuning defaults populated after load()
        tuning = w.config.get("tuning")
        assert tuning is not None, "tuning should be populated after load with target"
        space = (tuning.get("optuna") or {}).get("space", {})
        assert len(space) > 0, "search space should be populated"
        # Trigger tune — should not fail with CONFIG_INVALID due to missing tuning
        w.action = {"type": "tune", "payload": {}}
        if w._job_thread is not None:
            w._job_thread.join(timeout=5.0)
        # Validation may still fail for other reasons (mock adapter returns []),
        # but specifically NOT "No tuning configuration"
        if w.status == "failed":
            msg = w.error.get("message", "").lower()
            assert "tuning" not in msg or "configuration" not in msg


class TestApplyBestParamsSnapshot:
    """Tests for Apply to Fit config snapshot restoration (P-005)."""

    def test_apply_best_params_restores_tune_snapshot(self) -> None:
        """When tune snapshot exists, Apply to Fit restores it."""
        import copy

        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        # Simulate a tune snapshot with specific training config
        snapshot = {
            "config_version": 1,
            "model": {"name": "lgbm", "params": {"n_estimators": 1500}},
            "training": {"seed": 42, "early_stopping": {"enabled": True, "rounds": 100}},
            "evaluation": {"metrics": ["auc"]},
            "data": {"target": "y"},
            "features": {"include": ["x"]},
            "split": {"method": "kfold"},
            "task": "binary",
        }
        w._tune_config_snapshot = copy.deepcopy(snapshot)

        # Change config after tune
        w.config = {
            "config_version": 1,
            "model": {"name": "lgbm", "params": {"n_estimators": 500}},
            "training": {"seed": 99},
        }

        # Apply best params
        w.action = {
            "type": "apply_best_params",
            "payload": {"params": {"learning_rate": 0.01}},
        }

        # Should restore snapshot's training, not post-tune changes
        assert w.config["training"]["seed"] == 42
        assert w.config["model"]["params"]["learning_rate"] == 0.01
        assert w.config["model"]["params"]["n_estimators"] == 1500
        assert w.config.get("evaluation") == {"metrics": ["auc"]}

    def test_apply_best_params_strips_data_keys(self) -> None:
        """Snapshot restoration should not include data/features/split/task."""
        import copy

        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        snapshot = {
            "model": {"name": "lgbm", "params": {}},
            "data": {"target": "y"},
            "features": {"include": ["x"]},
            "split": {"method": "kfold"},
            "task": "binary",
        }
        w._tune_config_snapshot = copy.deepcopy(snapshot)

        w.action = {
            "type": "apply_best_params",
            "payload": {"params": {"lr": 0.01}},
        }

        assert "data" not in w.config
        assert "features" not in w.config
        assert "split" not in w.config
        assert "task" not in w.config

    def test_apply_best_params_without_snapshot(self) -> None:
        """Without snapshot, falls back to current config (backward compatible)."""
        w = _make_widget()
        w.config = {"model": {"name": "lgbm", "params": {"lr": 0.1}}}

        w.action = {
            "type": "apply_best_params",
            "payload": {"params": {"lr": 0.01}},
        }

        assert w.config["model"]["params"]["lr"] == 0.01

    def test_apply_best_params_without_snapshot_no_mutation(self) -> None:
        """Non-snapshot path must not mutate the original config's nested dicts."""
        w = _make_widget()
        original_params = {"lr": 0.1, "depth": 5}
        original_model = {"name": "lgbm", "params": original_params}
        w.config = {"model": original_model}

        w.action = {
            "type": "apply_best_params",
            "payload": {"params": {"lr": 0.01}},
        }

        # The original nested dict objects should NOT have been mutated
        assert original_params["lr"] == 0.1
        assert original_model["params"]["lr"] == 0.1


class TestBackendContract:
    """Phase 25: backend_contract traitlet and patch_config action."""

    def test_backend_contract_set_on_load(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.backend_contract["schema_version"] == 1
        assert "ui_schema" in w.backend_contract
        assert "capabilities" in w.backend_contract
        assert "config_schema" in w.backend_contract

    def test_backend_contract_ui_schema_has_sections(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        sections = w.backend_contract["ui_schema"]["sections"]
        assert len(sections) == 4
        assert sections[0]["key"] == "model"

    def test_patch_config_set(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "patch_config",
            "payload": {
                "ops": [{"op": "set", "path": "model.params.learning_rate", "value": 0.05}]
            },
        }
        assert w.config["model"]["params"]["learning_rate"] == 0.05

    def test_patch_config_unset(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # First set, then unset
        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "unset", "path": "model.params.max_depth"}]},
        }
        assert "max_depth" not in w.config["model"]["params"]

    def test_patch_config_empty_ops_ignored(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        original = dict(w.config)
        w.action = {
            "type": "patch_config",
            "payload": {"ops": []},
        }
        assert w.config == original


class TestContractViolationPayloads:
    """Phase 22: Contract-violating payloads should be caught."""

    def test_missing_model_name_auto_corrected(self) -> None:
        """build_config should auto-correct missing model.name."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # Unset model.name via patch
        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "unset", "path": "model.name"}]},
        }
        # build_config should add model.name="lgbm"
        full = w._service.build_config(dict(w.config))
        assert full["model"]["name"] == "lgbm"


class TestCanonicalConfigUnification:
    """Phase 26: All config entry points produce canonical config."""

    def test_set_config_canonicalizes(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.set_config({"model": {"params": {"n_estimators": 500}}})
        cfg = w.get_config()
        assert cfg["model"]["name"] == "lgbm"
        assert cfg["model"]["params"]["n_estimators"] == 500
        assert cfg["config_version"] == 1

    def test_set_config_preserves_config_version(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.set_config({"config_version": 5, "model": {"name": "lgbm"}})
        assert w.get_config()["config_version"] == 5

    def test_set_config_auto_num_leaves_exclusivity(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.set_config({"model": {"auto_num_leaves": True, "params": {"num_leaves": 256}}})
        assert "num_leaves" not in w.get_config()["model"]["params"]

    def test_import_yaml_canonicalizes(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        yaml_content = "model:\n  params:\n    n_estimators: 999\n"
        w.action = {"type": "import_yaml", "payload": {"content": yaml_content}}
        cfg = w.get_config()
        assert cfg["model"]["name"] == "lgbm"
        assert cfg["model"]["params"]["n_estimators"] == 999
        assert cfg["config_version"] == 1

    def test_set_config_inner_valid_legacy_normalized(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.set_config(
            {
                "training": {"early_stopping": {"inner_valid": "holdout"}},
            }
        )
        iv = w.get_config()["training"]["early_stopping"]["inner_valid"]
        assert iv == {"method": "holdout"}

    def test_patch_config_unset_model_name_re_completed(self) -> None:
        """26-1: unset model.name should be re-completed to canonical value."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "unset", "path": "model.name"}]},
        }
        assert w.config["model"]["name"] == "lgbm"

    def test_patch_config_unset_config_version_re_completed(self) -> None:
        """26-1: unset config_version should be re-completed."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "unset", "path": "config_version"}]},
        }
        assert w.config["config_version"] == 1

    def test_load_config_path_round_trip(self, tmp_path: Any) -> None:
        """26-5: load_config(path) produces canonical config."""
        import yaml

        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        path = str(tmp_path / "test_config.yaml")
        with open(path, "w") as f:
            yaml.dump(
                {
                    "model": {"params": {"n_estimators": 777}},
                    "data": {"target": "y"},
                    "split": {"method": "kfold", "n_splits": 3},
                },
                f,
            )

        w.load_config(path)
        cfg = w.get_config()
        assert cfg["model"]["name"] == "lgbm"
        assert cfg["model"]["params"]["n_estimators"] == 777
        assert cfg["config_version"] == 1
        # data/split/task should NOT be in widget config (service-managed)
        assert "data" not in cfg
        assert "split" not in cfg

    def test_save_config_canonical_output(self, tmp_path: Any) -> None:
        """26-5: save_config() exports canonical full config."""
        import yaml

        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        path = str(tmp_path / "test_save.yaml")
        w.save_config(path)
        with open(path) as f:
            saved: dict[str, Any] = yaml.safe_load(f)

        assert saved["model"]["name"] == "lgbm"
        assert saved["config_version"] == 1
        assert saved["data"]["target"] == "y"
        assert "task" in saved

    def test_export_yaml_canonical_output(self) -> None:
        """26-5: export_yaml action sends canonical full config."""
        import yaml

        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]

        w.action = {"type": "export_yaml", "payload": {}}

        assert len(sent) == 1
        content = yaml.safe_load(sent[0]["content"])
        assert content["model"]["name"] == "lgbm"
        assert content["config_version"] == 1


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


class TestPredictAndSaveModel:
    """Cover predict() and save_model() (widget.py lines 202-208)."""

    def test_predict_delegates_to_service(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        expected = PredictionSummary(
            predictions=pd.DataFrame({"pred": [0, 1]}),
            warnings=[],
        )
        w._service.predict = MagicMock(return_value=expected)
        result = w.predict(df)
        assert result is expected
        w._service.predict.assert_called_once_with(df, return_shap=False)

    def test_predict_with_shap(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        expected = PredictionSummary(
            predictions=pd.DataFrame({"pred": [0, 1]}),
            warnings=[],
        )
        w._service.predict = MagicMock(return_value=expected)
        w.predict(df, return_shap=True)
        w._service.predict.assert_called_once_with(df, return_shap=True)

    def test_save_model_delegates_to_service(self) -> None:
        w = _make_widget()
        w._service.save_model = MagicMock(return_value="/tmp/model.pkl")
        result = w.save_model("/tmp/model.pkl")
        assert result == "/tmp/model.pkl"
        w._service.save_model.assert_called_once_with("/tmp/model.pkl")


class TestActionHandlerEdgeCases:
    """Cover edge cases in action handlers."""

    def test_set_target_empty_target_ignored(self) -> None:
        """widget.py line 251: empty target returns early."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        w.action = {"type": "set_target", "payload": {"target": ""}}
        assert w.df_info.get("target") is None

    def test_set_task_empty_task_ignored(self) -> None:
        """widget.py line 265: empty task returns early."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        original_task = w.df_info.get("task")
        w.action = {"type": "set_task", "payload": {"task": ""}}
        assert w.df_info.get("task") == original_task

    def test_request_plot_empty_type_ignored(self) -> None:
        """widget.py line 323: empty plot_type returns early."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_plot", "payload": {"plot_type": ""}}
        assert len(sent) == 0

    def test_run_inference_no_data_sets_error(self) -> None:
        """widget.py lines 344-345: no inference data loaded."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {"type": "run_inference", "payload": {}}
        assert w.error["code"] == "INFERENCE_ERROR"

    def test_run_inference_with_data_success(self) -> None:
        """widget.py lines 347-355: inference with loaded data."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.load_inference(df)

        expected = PredictionSummary(
            predictions=pd.DataFrame({"pred": [0] * 50}),
            warnings=["test_warning"],
        )
        w._service.predict = MagicMock(return_value=expected)
        w.action = {"type": "run_inference", "payload": {}}
        assert w.inference_result["status"] == "completed"
        assert w.inference_result["rows"] == 50
        assert w.inference_result["warnings"] == ["test_warning"]

    def test_run_inference_error_sets_failed(self) -> None:
        """widget.py lines 356-360: inference failure."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.load_inference(df)

        w._service.predict = MagicMock(side_effect=RuntimeError("model not fitted"))
        w.action = {"type": "run_inference", "payload": {}}
        assert w.inference_result["status"] == "failed"
        assert "model not fitted" in w.inference_result["message"]

    def test_import_yaml_invalid_type_sets_error(self) -> None:
        """widget.py lines 370-372: YAML content is not a dict."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {"type": "import_yaml", "payload": {"content": "- list_item"}}
        assert w.error["code"] == "IMPORT_ERROR"

    def test_import_yaml_empty_content_ignored(self) -> None:
        """widget.py lines 364-365: empty content returns early."""
        w = _make_widget()
        w.action = {"type": "import_yaml", "payload": {"content": ""}}
        assert w.error == {}

    def test_import_yaml_parse_error(self) -> None:
        """widget.py lines 374-375: invalid YAML raises exception."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {"type": "import_yaml", "payload": {"content": "{{bad: yaml: ::"}}
        assert w.error["code"] == "IMPORT_ERROR"

    def test_export_yaml_error(self) -> None:
        """widget.py lines 384-385: export failure sets error."""
        w = _make_widget()
        w._service.build_config = MagicMock(side_effect=RuntimeError("build error"))
        w.action = {"type": "export_yaml", "payload": {}}
        assert w.error["code"] == "EXPORT_ERROR"

    def test_raw_config_action(self) -> None:
        """widget.py lines 388-393: raw_config sends msg:custom."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "raw_config", "payload": {}}
        assert len(sent) == 1
        assert sent[0]["type"] == "raw_config"
        assert "content" in sent[0]

    def test_raw_config_without_data_sends_user_config(self) -> None:
        """raw_config without data loaded sends current config as-is."""
        w = _make_widget()
        w.config = {"model": {"name": "lgbm", "params": {}}}

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "raw_config", "payload": {}}
        assert len(sent) == 1
        assert sent[0]["type"] == "raw_config"
        assert "model" in sent[0]["content"]

    def test_raw_config_error_sends_msg(self) -> None:
        """raw_config error sends error via custom msg and sets error traitlet."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w._service.build_config = MagicMock(side_effect=RuntimeError("boom"))

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "raw_config", "payload": {}}
        assert len(sent) == 1
        assert sent[0]["type"] == "raw_config_error"
        assert "boom" in sent[0]["message"]
        assert w.error["code"] == "EXPORT_ERROR"

    def test_raw_config_error_with_disconnected_send(self) -> None:
        """H-1: send() raising in error path should not propagate."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w._service.build_config = MagicMock(side_effect=RuntimeError("boom"))

        # Simulate disconnected widget — send() raises
        def broken_send(msg: Any) -> None:
            raise ConnectionError("Widget disconnected")

        w.send = broken_send  # type: ignore[assignment]

        # Should NOT raise even though send() fails in error path
        w.action = {"type": "raw_config", "payload": {}}

        # Error traitlet should still be set
        assert w.error["code"] == "EXPORT_ERROR"
        assert "boom" in w.error["message"]

    def test_apply_best_params_empty_ignored(self) -> None:
        """widget.py line 400: empty params returns early."""
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm", "params": {"lr": 0.1}}})
        original = dict(w.config)
        w.action = {"type": "apply_best_params", "payload": {"params": {}}}
        assert w.config["model"]["params"]["lr"] == original["model"]["params"]["lr"]


class TestRequestInferencePlot:
    """Cover _handle_request_inference_plot (widget.py lines 418-439)."""

    def test_empty_plot_type_ignored(self) -> None:
        """widget.py line 421: empty plot_type returns early."""
        w = _make_widget()
        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": ""}}
        assert len(sent) == 0

    def test_inference_plot_with_data_success(self) -> None:
        """widget.py lines 427-436: inference plot with data."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        # Set inference_result with data
        w.inference_result = {
            "status": "completed",
            "rows": 2,
            "data": [{"pred": 0}, {"pred": 1}],
            "warnings": [],
        }

        plot_data = PlotData(plotly_json='{"data": [], "layout": {}}')
        w._service.get_inference_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": "scatter"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_data"
        assert sent[0]["plot_type"] == "scatter"

    def test_inference_plot_fallback_on_error(self) -> None:
        """widget.py lines 437-439: falls back to fit plot on inference plot error."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.inference_result = {
            "status": "completed",
            "rows": 2,
            "data": [{"pred": 0}, {"pred": 1}],
            "warnings": [],
        }

        w._service.get_inference_plot = MagicMock(side_effect=RuntimeError("no plot"))
        # Fallback calls _handle_request_plot which calls get_plot
        plot_data = PlotData(plotly_json='{"data": []}')
        w._service.get_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": "roc"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_data"

    def test_no_inference_data_falls_back_to_fit_plot(self) -> None:
        """widget.py lines 424-426: no inference data delegates to request_plot."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        plot_data = PlotData(plotly_json='{"data": []}')
        w._service.get_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_inference_plot", "payload": {"plot_type": "roc"}}
        assert len(sent) == 1


class TestRunJobGuard:
    """Cover _run_job guard conditions (widget.py lines 461-477)."""

    def test_run_job_already_running_ignored(self) -> None:
        """widget.py line 463: already running returns early."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w.status = "running"
        # Trigger fit action — should be ignored
        w.action = {"type": "fit", "payload": {}}
        # Status should remain running, no error set
        assert w.status == "running"
        assert w.error == {}


class TestTuneProgress:
    """Tune sets initial progress with n_trials info."""

    def test_tune_sets_initial_progress(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        def mock_tune(config: Any, *, on_progress: Any = None) -> Any:
            # Capture progress calls and return a mock summary
            return TuningSummary(
                best_params={"lr": 0.01},
                best_score=0.9,
                trials=[],
                metric_name="auc",
                direction="maximize",
            )

        w._service.tune = mock_tune  # type: ignore[assignment]

        # Set config with tuning section
        w.config = {
            **dict(w.config),
            "tuning": {
                "optuna": {"params": {"n_trials": 30}, "space": {}},
            },
        }

        # Run tune — check that progress traitlet is set before tune starts
        w.action = {"type": "tune", "payload": {}}

        # Poll for progress instead of fixed sleep (avoids CI flakiness)
        import time

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.progress.get("total"):
                break
            time.sleep(0.05)

        # Progress should contain n_trials info
        assert w.progress["total"] == 30
        assert "30" in w.progress["message"]


class TestRequestPlotSuccess:
    """Cover _handle_request_plot success path (widget.py lines 325-332)."""

    def test_request_plot_sends_plotly_json(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        plot_data = PlotData(plotly_json='{"data": [], "layout": {}}')
        w._service.get_plot = MagicMock(return_value=plot_data)

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_plot", "payload": {"plot_type": "feature_importance"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_data"
        assert sent[0]["plot_type"] == "feature_importance"
        assert sent[0]["plotly_json"] == '{"data": [], "layout": {}}'

    def test_request_plot_error_sends_plot_error(self) -> None:
        """widget.py lines 333-340: plot error path."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.get_plot = MagicMock(side_effect=RuntimeError("no model"))

        sent: list[dict[str, Any]] = []
        w.send = lambda msg: sent.append(msg)  # type: ignore[assignment]
        w.action = {"type": "request_plot", "payload": {"plot_type": "roc"}}
        assert len(sent) == 1
        assert sent[0]["type"] == "plot_error"
        assert "no model" in sent[0]["message"]


class TestUpdateColumnError:
    """Cover _handle_update_column error path (widget.py lines 282-283)."""

    def test_update_column_error_sets_error(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        # Missing required 'name' key triggers KeyError
        w.action = {"type": "update_column", "payload": {}}
        assert w.error["code"] == "COLUMN_ERROR"


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
