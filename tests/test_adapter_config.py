"""Tests for LizyMLAdapter — Config lifecycle operations."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget import adapter_params, adapter_schema
from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import (
    BackendContract,
    ConfigPatchOp,
)


@pytest.fixture(autouse=True)
def _reset_eval_metrics_cache() -> None:  # type: ignore[misc]
    """Reset the module-level eval metrics cache between tests."""
    adapter_params._eval_metrics_cache = None
    yield  # type: ignore[misc]
    adapter_params._eval_metrics_cache = None


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
            "output_dir": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Output Dir",
            },
        },
    }
    return {
        "lizyml": MagicMock(),
        "lizyml.config": MagicMock(),
        "lizyml.config.schema": MagicMock(LizyMLConfig=mock_config_cls),
    }


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
        old_cache = adapter_schema._schema_cache
        adapter_schema.reset_schema_cache()
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
            adapter_schema._schema_cache = old_cache

    def test_validate_config_invalid(self) -> None:
        mock_load_config = MagicMock(side_effect=ValueError("config_version is required"))
        mock_loader = MagicMock(load_config=mock_load_config)
        mock_config_cls = MagicMock()
        mock_config_cls.model_json_schema.return_value = {"properties": {}}
        old_cache = adapter_schema._schema_cache
        adapter_schema.reset_schema_cache()
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
            adapter_schema._schema_cache = old_cache

    def test_validate_config_catches_legacy_mode_format(self) -> None:
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
            assert section_keys == ["model", "training", "calibration", "evaluation"]

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

    def test_model_metric_option_set_uses_lgbm_names(self) -> None:
        """model_metric option set must use LightGBM-native names."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            option_sets = contract.ui_schema["option_sets"]
            assert "model_metric" in option_sets, "option_sets must include model_metric"
            for task in ("regression", "binary", "multiclass"):
                assert task in option_sets["model_metric"]
                assert len(option_sets["model_metric"][task]) > 0

    def test_multiclass_model_metric_uses_multi_prefix(self) -> None:
        """Multiclass model_metric must use multi_logloss/auc_mu, not logloss/auc."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            mc_metrics = contract.ui_schema["option_sets"]["model_metric"]["multiclass"]
            # Must contain LightGBM multiclass-specific names
            assert "multi_logloss" in mc_metrics
            assert "auc_mu" in mc_metrics
            # Must NOT contain bare names that fail with multiclass objective
            assert "logloss" not in mc_metrics
            assert "auc" not in mc_metrics

    def test_parameter_hints_metric_uses_model_metric_kind(self) -> None:
        """metric parameter_hint must use kind='model_metric'."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            hints = contract.ui_schema["parameter_hints"]
            metric_hint = next(h for h in hints if h["key"] == "metric")
            assert metric_hint["kind"] == "model_metric"

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
            assert defaults["calibration"]["method"] == "isotonic"

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

    def test_output_dir_default(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            config = adapter.initialize_config()
            assert config["output_dir"] == "outputs/"


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

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_all_eval_metrics_on_by_default(self, task: str) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
        }
        result = adapter.apply_task_defaults(config, task=task)
        expected = adapter._get_eval_metrics_by_task()[task]
        assert result["evaluation"]["metrics"] == expected

    def test_eval_metrics_are_lizyml_names_not_lgbm(self) -> None:
        """Eval metrics must use LizyML registry names, not LightGBM training names."""
        adapter = LizyMLAdapter()
        eval_metrics = adapter._get_eval_metrics_by_task()
        lgbm_only_names = {
            "binary_logloss",
            "binary_error",
            "multi_logloss",
            "multi_error",
            "auc_mu",
            "mse",
            "quantile",
        }
        for task, metrics in eval_metrics.items():
            overlap = set(metrics) & lgbm_only_names
            assert not overlap, f"Task '{task}' has LightGBM-only metric names: {overlap}"

    def test_eval_metrics_not_overwritten_if_set(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc"]},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        assert result["evaluation"]["metrics"] == ["auc"]

    def test_task_change_filters_stale_metrics(self) -> None:
        """When task changes, stale metrics from old task should be removed."""
        adapter = LizyMLAdapter()
        # Start with regression metrics
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["mae", "rmse", "r2"]},
        }
        # Switch to binary -> regression-only metrics are invalid
        result = adapter.apply_task_defaults(config, task="binary")
        binary_valid = set(adapter._get_eval_metrics_by_task()["binary"])
        actual = set(result["evaluation"]["metrics"])
        # All returned metrics must be valid for binary
        assert actual <= binary_valid
        # Since no valid metrics survived, should populate defaults
        assert len(actual) > 0

    def test_task_change_keeps_valid_metrics(self) -> None:
        """When task changes, metrics valid for new task should be kept."""
        adapter = LizyMLAdapter()
        # Config has mix of binary and invalid metrics
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc", "mae", "logloss"]},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        # auc and logloss are valid for binary; mae is not
        assert "auc" in result["evaluation"]["metrics"]
        assert "logloss" in result["evaluation"]["metrics"]
        assert "mae" not in result["evaluation"]["metrics"]


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
        assert result["tuning"]["optuna"]["params"]["n_trials"] == 10
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

    def test_evaluation_params_stripped_for_backend(self) -> None:
        """evaluation.params is widget-only and must be stripped before backend submission."""
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc"], "params": {"precision_at_k_k": 20}},
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        eval_section = result.get("evaluation", {})
        assert "params" not in eval_section, "evaluation.params should be stripped"
        assert eval_section["metrics"] == ["auc"]

    def test_evaluation_params_survives_config_patch(self) -> None:
        """evaluation.params should survive config patching (not stripped in patch)."""
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": ["auc"], "params": {"precision_at_k_k": 20}},
        }
        ops = [ConfigPatchOp(op="set", path="evaluation.metrics", value=["auc", "logloss"])]
        result = adapter.apply_config_patch(config, ops)
        assert result["evaluation"]["params"] == {"precision_at_k_k": 20}
        assert result["evaluation"]["metrics"] == ["auc", "logloss"]

    def test_inner_valid_normalized_in_prepare(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "training": {"early_stopping": {"inner_valid": "holdout"}},
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "holdout"}


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


class TestEnforceAutoNumLeavesFalse:
    """Cover auto_num_leaves=False path (adapter.py lines 186-187, 306-318)."""

    def test_enforce_false_inserts_default_256(self) -> None:
        """auto_num_leaves=False with no num_leaves → inserts 256."""
        result = LizyMLAdapter._enforce_auto_num_leaves({"auto_num_leaves": False, "params": {}})
        assert result["params"]["num_leaves"] == 256

    def test_enforce_false_preserves_existing_num_leaves(self) -> None:
        """auto_num_leaves=False with existing num_leaves → preserves value."""
        result = LizyMLAdapter._enforce_auto_num_leaves(
            {"auto_num_leaves": False, "params": {"num_leaves": 128}}
        )
        assert result["params"]["num_leaves"] == 128


class TestExtractDefaultsSchemaResolution:
    """Cover allOf + $ref schema resolution (adapter.py lines 420-423)."""

    def test_allof_single_ref_resolved(self) -> None:
        """allOf with a single $ref should be resolved and defaults extracted."""
        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "allOf": [{"$ref": "#/$defs/ModelConfig"}],
                },
            },
            "$defs": {
                "ModelConfig": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "default": "lgbm"},
                        "depth": {"type": "integer", "default": 5},
                    },
                },
            },
        }
        result = LizyMLAdapter._extract_defaults(schema)
        assert result["model"]["name"] == "lgbm"
        assert result["model"]["depth"] == 5

    def test_simple_ref_resolved(self) -> None:
        """Plain $ref should be resolved and defaults extracted."""
        schema = {
            "type": "object",
            "properties": {
                "training": {"$ref": "#/$defs/TrainConfig"},
            },
            "$defs": {
                "TrainConfig": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer", "default": 42},
                    },
                },
            },
        }
        result = LizyMLAdapter._extract_defaults(schema)
        assert result["training"]["seed"] == 42


class TestVersionFallbackPaths:
    """Cover LizyML version fallback paths for import failures."""

    def test_search_space_fallback_returns_empty(self) -> None:
        """adapter_schema.py:250-258: when all imports fail, returns {}."""
        adapter_schema.reset_schema_cache()
        old_cache = adapter_schema._schema_cache
        try:
            with patch.dict(
                "sys.modules",
                {
                    "lizyml.estimators.lgbm.defaults": None,
                    "lizyml.estimators.lgbm": None,
                    "lizyml.estimators": None,
                    "lizyml.tuning": None,
                },
            ):
                result = adapter_schema.get_default_search_space("binary")
                assert result == {}
        finally:
            adapter_schema._schema_cache = old_cache

    def test_resolve_direction_import_error_returns_minimize(self) -> None:
        """adapter_params.py:83-84: when metric registry unavailable, returns minimize."""
        from lizyml_widget.adapter_params import resolve_direction

        with patch.dict(
            "sys.modules",
            {
                "lizyml.metrics.registry": None,
                "lizyml.metrics": None,
                "lizyml.core.exceptions": None,
                "lizyml.core": None,
            },
        ):
            result = resolve_direction("auc")
            assert result == "minimize"

    def test_eval_metrics_fallback_uses_hardcoded(self) -> None:
        """adapter_params.py:170-172: fallback metrics when _TASK_METRICS unavailable."""
        from lizyml_widget.adapter_params import get_eval_metrics_by_task

        adapter_params._eval_metrics_cache = None
        try:
            # Mock _TASK_METRICS to raise AttributeError
            mock_registry = MagicMock(spec=[])  # no _TASK_METRICS attribute
            with patch.dict(
                "sys.modules",
                {
                    "lizyml.metrics.registry": mock_registry,
                    "lizyml.metrics": MagicMock(),
                },
            ):
                result = get_eval_metrics_by_task()
                assert "binary" in result
                assert "regression" in result
                assert "multiclass" in result
                assert len(result["binary"]) > 0
        finally:
            adapter_params._eval_metrics_cache = None


class TestPrepareRunConfigImmutability:
    """prepare_run_config must not mutate the input config dict."""

    def test_tune_config_input_not_mutated(self) -> None:
        """The original config dict must be unchanged after prepare_run_config."""
        adapter = LizyMLAdapter()
        config = adapter.initialize_config(task="binary")
        config = {**config, "task": "binary"}
        config = {**config, "data": {"target": "y"}}
        config = {**config, "features": {"exclude": [], "categorical": []}}
        config = {**config, "split": {"method": "kfold", "n_splits": 5}}
        config = {
            **config,
            "tuning": {"optuna": {"params": {"n_trials": 50, "metric": "auc"}, "space": {}}},
        }

        original = copy.deepcopy(config)
        adapter.prepare_run_config(config, job_type="tune", task="binary")

        assert config == original, "prepare_run_config must not mutate the input config"
