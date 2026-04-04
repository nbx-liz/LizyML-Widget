"""Tests for LizyMLAdapter — Tune and search space operations."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget import adapter_params, adapter_schema
from lizyml_widget.adapter import LizyMLAdapter


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


class TestGetDefaultSearchSpace:
    @pytest.mark.parametrize("task", ["binary", "regression", "multiclass"])
    def test_returns_dict_for_known_tasks(self, task: str) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space(task)
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_binary_contains_expected_keys(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # Should contain range params from LizyML default_space
        assert "n_estimators" in space
        assert "learning_rate" in space
        assert "max_depth" in space
        assert "feature_fraction" in space

    def test_range_param_format(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # n_estimators should be int range
        n_est = space["n_estimators"]
        assert n_est["type"] == "int"
        assert "low" in n_est
        assert "high" in n_est

    def test_categorical_param_format(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # objective should be categorical
        obj = space["objective"]
        assert obj["type"] == "categorical"
        assert "choices" in obj

    def test_log_scale_preserved(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("binary")
        # learning_rate is log-scale in LizyML
        lr = space["learning_rate"]
        assert lr["log"] is True

    def test_apply_task_defaults_without_lgbm_params_still_populates_metrics(self) -> None:
        adapter = LizyMLAdapter()
        # Patch _LGBM_PARAMS_BY_TASK to simulate a task with no LGBM defaults
        original = dict(adapter._LGBM_PARAMS_BY_TASK)
        adapter._LGBM_PARAMS_BY_TASK = {}
        try:
            config: dict[str, Any] = {
                "model": {"name": "lgbm", "params": {}},
                "evaluation": {"metrics": []},
            }
            result = adapter.apply_task_defaults(config, task="binary")
            # Metrics should still be populated despite no LGBM param defaults
            assert len(result["evaluation"]["metrics"]) > 0
            assert "auc" in result["evaluation"]["metrics"]
        finally:
            adapter._LGBM_PARAMS_BY_TASK = original

    def test_apply_task_defaults_without_lgbm_params_still_populates_space(self) -> None:
        adapter = LizyMLAdapter()
        original = dict(adapter._LGBM_PARAMS_BY_TASK)
        adapter._LGBM_PARAMS_BY_TASK = {}
        try:
            config: dict[str, Any] = {
                "model": {"name": "lgbm", "params": {}},
                "evaluation": {"metrics": []},
                "tuning": {"optuna": {"params": {"n_trials": 50}, "space": {}}},
            }
            result = adapter.apply_task_defaults(config, task="binary")
            space = result["tuning"]["optuna"]["space"]
            assert len(space) > 0
            assert "n_estimators" in space
        finally:
            adapter._LGBM_PARAMS_BY_TASK = original

    def test_apply_task_defaults_populates_space(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
            "tuning": {"optuna": {"params": {"n_trials": 50}, "space": {}}},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        space = result["tuning"]["optuna"]["space"]
        assert "n_estimators" in space
        assert space["learning_rate"]["log"] is True

    def test_apply_task_defaults_does_not_overwrite_existing_space(
        self,
    ) -> None:
        """Existing search space is not overwritten."""
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
            "tuning": {
                "optuna": {
                    "params": {"n_trials": 50},
                    "space": {"n_estimators": {"type": "int", "low": 10, "high": 20, "log": False}},
                }
            },
        }
        result = adapter.apply_task_defaults(config, task="binary")
        space = result["tuning"]["optuna"]["space"]
        # Should keep existing, not overwrite
        assert space["n_estimators"]["low"] == 10

    def test_apply_task_defaults_populates_space_when_tuning_is_none(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
            "tuning": None,
        }
        result = adapter.apply_task_defaults(config, task="binary")
        # Tuning section should be created with default space
        assert result["tuning"] is not None
        assert isinstance(result["tuning"], dict)
        space = result["tuning"]["optuna"]["space"]
        assert "n_estimators" in space

    def test_apply_task_defaults_populates_space_when_tuning_absent(self) -> None:
        adapter = LizyMLAdapter()
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "evaluation": {"metrics": []},
        }
        result = adapter.apply_task_defaults(config, task="binary")
        assert result["tuning"] is not None
        space = result["tuning"]["optuna"]["space"]
        assert "n_estimators" in space

    def test_unknown_task_still_returns_space(self) -> None:
        LizyMLAdapter()
        space = adapter_schema.get_default_search_space("unknown_task")
        assert isinstance(space, dict)


class TestTuneMetricMapping:
    def test_model_metric_to_eval_mapping_exists(self) -> None:
        adapter = LizyMLAdapter()
        mapping = adapter._MODEL_METRIC_TO_EVAL
        # Key model_metric names must be present
        assert "auc_mu" in mapping
        assert "multi_logloss" in mapping
        assert "binary_logloss" in mapping
        assert "multi_error" in mapping

    def test_model_metric_to_eval_mapping_values(self) -> None:
        adapter = LizyMLAdapter()
        mapping = adapter._MODEL_METRIC_TO_EVAL
        assert mapping["auc_mu"] == "auc"
        assert mapping["multi_logloss"] == "logloss"
        assert mapping["binary_logloss"] == "logloss"
        assert mapping["multi_error"] == "accuracy"
        # Identity mappings for names that already match
        assert mapping["auc"] == "auc"
        assert mapping["rmse"] == "rmse"

    def test_resolve_direction_maximize(self) -> None:
        assert adapter_params.resolve_direction("auc") == "maximize"
        assert adapter_params.resolve_direction("accuracy") == "maximize"
        assert adapter_params.resolve_direction("f1") == "maximize"

    def test_resolve_direction_minimize(self) -> None:
        assert adapter_params.resolve_direction("logloss") == "minimize"
        assert adapter_params.resolve_direction("rmse") == "minimize"
        assert adapter_params.resolve_direction("brier") == "minimize"

    def test_resolve_direction_unknown_fallback(self) -> None:
        assert adapter_params.resolve_direction("nonexistent_metric") == "minimize"


class TestModelMetricToEvalImmutability:
    def test_mutation_raises_type_error(self) -> None:
        adapter = LizyMLAdapter()
        with pytest.raises(TypeError):
            adapter._MODEL_METRIC_TO_EVAL["injected"] = "bad"  # type: ignore[index]

    def test_read_access_works(self) -> None:
        adapter = LizyMLAdapter()
        assert adapter._MODEL_METRIC_TO_EVAL.get("auc_mu") == "auc"
        assert "auc" in adapter._MODEL_METRIC_TO_EVAL


class TestPrepareRunConfigTuneMetric:
    @staticmethod
    def _make_tune_config(adapter: LizyMLAdapter, task: str, metric: str) -> dict[str, Any]:
        """Helper: build a config with tuning.optuna.params.metric set."""
        config = adapter.initialize_config(task=task)
        config["task"] = task
        config["data"] = {"target": "y"}
        config["features"] = {"exclude": [], "categorical": []}
        config["split"] = {"method": "kfold", "n_splits": 5}
        config["tuning"] = {"optuna": {"params": {"n_trials": 50, "metric": metric}, "space": {}}}
        return config

    def test_tune_metric_placed_first_in_evaluation_metrics(self) -> None:
        """When tuning.optuna.params.metric is set, the corresponding eval metric
        should be evaluation.metrics[0]."""
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "multiclass", "auc_mu")
        config["evaluation"] = {"metrics": ["accuracy", "auc", "logloss"]}

        result = adapter.prepare_run_config(config, job_type="tune", task="multiclass")

        eval_metrics = result.get("evaluation", {}).get("metrics", [])
        assert eval_metrics[0] == "auc", f"Expected 'auc' as first eval metric, got {eval_metrics}"

    def test_tune_direction_auto_set_maximize(self) -> None:
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "multiclass", "auc_mu")

        result = adapter.prepare_run_config(config, job_type="tune", task="multiclass")

        direction = result.get("tuning", {}).get("optuna", {}).get("params", {}).get("direction")
        assert direction == "maximize", f"Expected 'maximize', got {direction}"

    def test_tune_direction_auto_set_minimize(self) -> None:
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "multiclass", "multi_logloss")

        result = adapter.prepare_run_config(config, job_type="tune", task="multiclass")

        direction = result.get("tuning", {}).get("optuna", {}).get("params", {}).get("direction")
        assert direction == "minimize", f"Expected 'minimize', got {direction}"

    def test_tune_metric_stripped_from_optuna_params(self) -> None:
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "binary", "auc")

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        optuna_params = result.get("tuning", {}).get("optuna", {}).get("params", {})
        assert "metric" not in optuna_params, (
            f"Widget-only 'metric' key should be stripped, but found in {optuna_params}"
        )

    def test_fit_job_ignores_tune_metric(self) -> None:
        adapter = LizyMLAdapter()
        config = self._make_tune_config(adapter, "binary", "auc")

        # Should not raise
        result = adapter.prepare_run_config(config, job_type="fit", task="binary")
        assert result is not None


class TestClassifyBestParams:
    def test_classify_splits_correctly(self) -> None:
        adapter = LizyMLAdapter()
        params = {
            "learning_rate": 0.01,
            "max_depth": 7,
            "n_estimators": 1500,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "objective": "binary",
            "num_leaves_ratio": 0.7,
            "min_data_in_leaf_ratio": 0.05,
            "early_stopping_rounds": 80,
            "validation_ratio": 0.2,
        }
        model_p, smart_p, training_p = adapter.classify_best_params(params)

        assert "learning_rate" in model_p
        assert "max_depth" in model_p
        assert "n_estimators" in model_p
        assert "objective" in model_p

        assert "num_leaves_ratio" in smart_p
        assert "min_data_in_leaf_ratio" in smart_p

        assert "early_stopping_rounds" in training_p
        assert "validation_ratio" in training_p

    def test_classify_unknown_defaults_to_model(self) -> None:
        adapter = LizyMLAdapter()
        params = {"some_custom_param": 42}
        model_p, smart_p, training_p = adapter.classify_best_params(params)
        assert "some_custom_param" in model_p
        assert not smart_p
        assert not training_p

    def test_classify_empty_params(self) -> None:
        adapter = LizyMLAdapter()
        model_p, smart_p, training_p = adapter.classify_best_params({})
        assert model_p == {}
        assert smart_p == {}
        assert training_p == {}


class TestP014SearchSpaceCatalogGroups:
    def test_catalog_entries_have_group(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            for entry in catalog:
                assert "group" in entry, f"Entry {entry['key']} missing 'group'"

    def test_catalog_has_model_params_group(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            model_entries = [e for e in catalog if e["group"] == "model_params"]
            model_keys = {e["key"] for e in model_entries}
            assert "objective" in model_keys
            assert "n_estimators" in model_keys
            assert "learning_rate" in model_keys

    def test_catalog_has_smart_params_group(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            smart_entries = [e for e in catalog if e["group"] == "smart_params"]
            smart_keys = {e["key"] for e in smart_entries}
            assert "auto_num_leaves" in smart_keys
            assert "num_leaves_ratio" in smart_keys
            assert "min_data_in_leaf_ratio" in smart_keys
            assert "min_data_in_bin_ratio" in smart_keys
            assert "balanced" in smart_keys

    def test_feature_weights_is_fixed_only(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            fw = next((e for e in catalog if e["key"] == "feature_weights"), None)
            assert fw is not None, "feature_weights missing from catalog"
            assert fw["group"] == "smart_params"
            assert fw["modes"] == ["fixed"]

    def test_catalog_has_training_group(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            training_entries = [e for e in catalog if e["group"] == "training"]
            training_keys = {e["key"] for e in training_entries}
            expected = {
                "seed",
                "early_stopping.enabled",
                "early_stopping.rounds",
                "validation_ratio",
                "inner_valid",
            }
            assert expected.issubset(training_keys), (
                f"Missing training keys: {expected - training_keys}"
            )

    def test_training_fixed_only_constraints(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            catalog_map = {e["key"]: e for e in catalog}
            for key in ("seed", "early_stopping.enabled", "inner_valid"):
                assert catalog_map[key]["modes"] == ["fixed"], (
                    f"{key} must be Fixed-only, got {catalog_map[key]['modes']}"
                )

    def test_training_range_constraints(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            catalog_map = {e["key"]: e for e in catalog}
            for key in ("early_stopping.rounds", "validation_ratio"):
                assert catalog_map[key]["modes"] == ["fixed", "range"], (
                    f"{key} must support Fixed/Range, got {catalog_map[key]['modes']}"
                )


class TestP014AdditionalParams:
    def test_additional_params_exists(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            assert "additional_params" in contract.ui_schema

    def test_additional_params_excludes_hints_and_catalog(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            hint_keys = {h["key"] for h in contract.ui_schema["parameter_hints"]}
            catalog_keys = {e["key"] for e in contract.ui_schema["search_space_catalog"]}
            additional = set(contract.ui_schema["additional_params"])
            assert additional.isdisjoint(hint_keys), (
                f"additional_params overlaps with parameter_hints: {additional & hint_keys}"
            )
            assert additional.isdisjoint(catalog_keys), (
                f"additional_params overlaps with search_space_catalog: {additional & catalog_keys}"
            )

    def test_additional_params_is_nonempty(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            assert len(contract.ui_schema["additional_params"]) > 0


class TestP014StripWidgetOnlyTuningFields:
    def test_strips_tuning_model_params(self) -> None:
        config = {
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "model_params": {"n_estimators": 1000},
            }
        }
        result = adapter_schema.strip_for_backend(config)
        tuning = result.get("tuning", {})
        assert "model_params" not in tuning

    def test_strips_tuning_training(self) -> None:
        config = {
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "training": {"seed": 42},
            }
        }
        result = adapter_schema.strip_for_backend(config)
        tuning = result.get("tuning", {})
        assert "training" not in tuning

    def test_strips_tuning_evaluation(self) -> None:
        config = {
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "evaluation": {"metrics": ["auc"]},
            }
        }
        result = adapter_schema.strip_for_backend(config)
        tuning = result.get("tuning", {})
        assert "evaluation" not in tuning

    def test_preserves_optuna_section(self) -> None:
        config = {
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "model_params": {"n_estimators": 1000},
                "training": {"seed": 42},
                "evaluation": {"metrics": ["auc"]},
            }
        }
        result = adapter_schema.strip_for_backend(config)
        tuning = result.get("tuning", {})
        assert "optuna" in tuning
        assert tuning["optuna"]["params"]["n_trials"] == 50


class TestP014PrepareRunConfigTuneNewFields:
    @staticmethod
    def _base_config(adapter: LizyMLAdapter) -> dict[str, Any]:
        config = adapter.initialize_config(task="binary")
        config["task"] = "binary"
        config["data"] = {"target": "y"}
        config["features"] = {"exclude": [], "categorical": []}
        config["split"] = {"method": "kfold", "n_splits": 5}
        return config

    def test_tuning_model_params_replaces_model_params(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["model"]["params"]["n_estimators"] = 1500  # Fit value
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "model_params": {"n_estimators": 800, "learning_rate": 0.01},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        # Tune's model_params should override Fit's
        assert result["model"]["params"]["n_estimators"] == 800
        assert result["model"]["params"]["learning_rate"] == 0.01

    def test_tuning_model_params_preserves_unrelated_keys(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["model"]["params"]["n_estimators"] = 1500
        config["model"]["params"]["verbose"] = -1
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "model_params": {"n_estimators": 800},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        assert result["model"]["params"]["n_estimators"] == 800
        assert result["model"]["params"]["verbose"] == -1  # preserved

    def test_tuning_training_replaces_training(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["training"] = {"seed": 42, "early_stopping": {"enabled": True, "rounds": 150}}
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "training": {"seed": 99, "early_stopping": {"enabled": True, "rounds": 50}},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        assert result["training"]["seed"] == 99
        assert result["training"]["early_stopping"]["rounds"] == 50

    def test_tuning_evaluation_replaces_evaluation(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["evaluation"] = {"metrics": ["auc", "logloss", "f1"]}
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "evaluation": {"metrics": ["f1", "accuracy"]},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        eval_metrics = result["evaluation"]["metrics"]
        assert eval_metrics[0] == "f1"
        assert "accuracy" in eval_metrics

    def test_direction_from_tuning_evaluation_first_metric_maximize(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "evaluation": {"metrics": ["auc", "logloss"]},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        direction = result["tuning"]["optuna"]["params"].get("direction")
        assert direction == "maximize"

    def test_direction_from_tuning_evaluation_first_metric_minimize(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "evaluation": {"metrics": ["logloss", "auc"]},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        direction = result["tuning"]["optuna"]["params"].get("direction")
        assert direction == "minimize"

    def test_fallback_to_fit_when_tuning_fields_absent(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["model"]["params"]["n_estimators"] = 2000
        config["training"] = {"seed": 42}
        config["evaluation"] = {"metrics": ["auc", "logloss"]}
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        # Fit values should be preserved (fallback)
        assert result["model"]["params"]["n_estimators"] == 2000
        assert result["training"]["seed"] == 42

    def test_smart_params_preserved_for_tune(self) -> None:
        """Smart params must be preserved for tune (LizyML uses them per trial)."""
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["model"]["auto_num_leaves"] = True
        config["model"]["num_leaves_ratio"] = 0.8
        config["model"]["min_data_in_leaf_ratio"] = 0.01
        config["model"]["balanced"] = True
        config["calibration"] = {"method": "platt", "n_splits": 5, "params": {}}
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "model_params": {"n_estimators": 800},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        # Smart params must be preserved in the output model section
        model = result.get("model", {})
        assert model.get("auto_num_leaves") is True
        assert model.get("num_leaves_ratio") == 0.8
        assert model.get("min_data_in_leaf_ratio") == 0.01
        assert model.get("balanced") is True
        # Calibration must still be stripped (LizyML tune does not use it)
        assert "calibration" not in result

    def test_fit_job_not_affected(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["model"]["params"]["n_estimators"] = 1500
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "model_params": {"n_estimators": 800},
        }

        result = adapter.prepare_run_config(config, job_type="fit", task="binary")

        # Fit should use original model.params, not tuning.model_params
        assert result["model"]["params"]["n_estimators"] == 1500

    def test_widget_only_tuning_fields_stripped_from_output(self) -> None:
        adapter = LizyMLAdapter()
        config = self._base_config(adapter)
        config["tuning"] = {
            "optuna": {"params": {"n_trials": 10}, "space": {}},
            "model_params": {"n_estimators": 800},
            "training": {"seed": 99},
            "evaluation": {"metrics": ["auc"]},
        }

        result = adapter.prepare_run_config(config, job_type="tune", task="binary")

        tuning = result.get("tuning", {})
        assert "model_params" not in tuning
        assert "training" not in tuning
        assert "evaluation" not in tuning


class TestAuditCatalogVerboseAndNumLeaves:
    def test_catalog_contains_verbose(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            verbose_entry = next((e for e in catalog if e["key"] == "verbose"), None)
            assert verbose_entry is not None, "verbose missing from catalog"
            assert verbose_entry["group"] == "model_params"
            assert verbose_entry["paramType"] == "integer"
            assert "fixed" in verbose_entry["modes"]
            assert "range" in verbose_entry["modes"]

    def test_catalog_contains_num_leaves(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            entry = next((e for e in catalog if e["key"] == "num_leaves"), None)
            assert entry is not None, "num_leaves missing from catalog"
            assert entry["group"] == "smart_params"
            assert entry["paramType"] == "integer"
            assert entry["modes"] == ["fixed", "range"]


class TestAuditCatalogDefaults:
    def test_smart_params_have_defaults(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            expected_defaults = {
                "auto_num_leaves": True,
                "num_leaves_ratio": 1.0,
                "min_data_in_leaf_ratio": 0.01,
                "min_data_in_bin_ratio": 0.01,
                "balanced": False,
            }
            for key, expected in expected_defaults.items():
                entry = next((e for e in catalog if e["key"] == key), None)
                assert entry is not None, f"{key} missing from catalog"
                assert "default" in entry, f"{key} missing 'default' field"
                assert entry["default"] == expected, (
                    f"{key}: expected default={expected}, got {entry.get('default')}"
                )

    def test_training_params_have_defaults(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            expected_defaults = {
                "seed": 42,
                "early_stopping.enabled": True,
                "early_stopping.rounds": 150,
                "validation_ratio": 0.1,
            }
            for key, expected in expected_defaults.items():
                entry = next((e for e in catalog if e["key"] == key), None)
                assert entry is not None, f"{key} missing from catalog"
                assert "default" in entry, f"{key} missing 'default' field"
                assert entry["default"] == expected, (
                    f"{key}: expected default={expected}, got {entry.get('default')}"
                )


class TestAuditMetricOrdering:
    def test_binary_eval_metrics_start_with_auc(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            binary_metrics = contract.ui_schema["option_sets"]["metric"]["binary"]
            assert binary_metrics[0] == "auc", (
                f"binary metrics should start with 'auc', got '{binary_metrics[0]}'"
            )

    def test_regression_eval_metrics_start_with_rmse(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            reg_metrics = contract.ui_schema["option_sets"]["metric"]["regression"]
            assert reg_metrics[0] == "rmse", (
                f"regression metrics should start with 'rmse', got '{reg_metrics[0]}'"
            )


class TestAuditEarlyStoppingVisibility:
    def test_early_stopping_rounds_conditional_visibility(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            cv = contract.ui_schema["conditional_visibility"]
            assert "early_stopping.rounds" in cv, (
                "early_stopping.rounds must have conditional_visibility rule"
            )
            assert cv["early_stopping.rounds"] == {"early_stopping.enabled": True}

    def test_validation_ratio_conditional_visibility(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            cv = contract.ui_schema["conditional_visibility"]
            assert "validation_ratio" in cv, (
                "validation_ratio must have conditional_visibility rule"
            )
            assert cv["validation_ratio"] == {"early_stopping.enabled": True}

    def test_inner_valid_conditional_visibility(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            cv = contract.ui_schema["conditional_visibility"]
            assert "inner_valid" in cv


class TestAuditInnerValidDefault:
    def test_inner_valid_catalog_default_is_holdout(self) -> None:
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            contract = adapter.get_backend_contract()
            catalog = contract.ui_schema["search_space_catalog"]
            entry = next((e for e in catalog if e["key"] == "inner_valid"), None)
            assert entry is not None
            assert entry.get("default") == "holdout", (
                f"inner_valid default should be 'holdout', got {entry.get('default')}"
            )
