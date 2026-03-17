"""Tests for LizyMLAdapter — Schema and normalization operations."""

from __future__ import annotations

import pytest

from lizyml_widget import adapter_params, adapter_schema
from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import ConfigPatchOp


@pytest.fixture(autouse=True)
def _reset_eval_metrics_cache() -> None:  # type: ignore[misc]
    """Reset the module-level eval metrics cache between tests."""
    adapter_params._eval_metrics_cache = None
    yield  # type: ignore[misc]
    adapter_params._eval_metrics_cache = None


class TestInnerValidNormalization:
    """Phase 26: inner_valid legacy strings are normalized to object/null."""

    def test_holdout_string_to_object(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "holdout"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "holdout"}

    def test_group_holdout_string_to_object(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "group_holdout"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "group_holdout"}

    def test_time_holdout_string_to_object(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "time_holdout"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "time_holdout"}

    def test_null_preserved(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": None}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] is None

    def test_object_preserved(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": {"method": "holdout"}}},
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert isinstance(iv, dict)
        assert iv["method"] == "holdout"

    def test_fold_string_becomes_null(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {"early_stopping": {"inner_valid": "fold_0"}},
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] is None

    def test_unrecognized_type_becomes_null(self) -> None:
        """Integer, list, or boolean inner_valid should be discarded as None."""
        adapter = LizyMLAdapter()
        for bad_value in [42, [1, 2], True]:
            config = {
                "training": {"early_stopping": {"inner_valid": bad_value}},
            }
            result = adapter.canonicalize_config(config)
            assert result["training"]["early_stopping"]["inner_valid"] is None, (
                f"inner_valid={bad_value!r} should become None"
            )

    def test_normalization_in_apply_config_patch(self) -> None:
        adapter = LizyMLAdapter()
        config = {"model": {"name": "lgbm", "params": {}}, "training": {"early_stopping": {}}}
        ops = [ConfigPatchOp(op="set", path="training.early_stopping.inner_valid", value="holdout")]
        result = adapter.apply_config_patch(config, ops)
        assert result["training"]["early_stopping"]["inner_valid"] == {"method": "holdout"}


class TestInnerValidFieldStripping:
    """inner_valid normalization must strip fields not allowed by the method."""

    def test_time_holdout_strips_random_state_and_stratify(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "time_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert iv == {"method": "time_holdout", "ratio": 0.1}

    def test_group_holdout_strips_stratify(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "group_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert iv == {"method": "group_holdout", "ratio": 0.1, "random_state": 42}

    def test_holdout_preserves_all_fields(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": True,
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert iv == {
            "method": "holdout",
            "ratio": 0.1,
            "random_state": 42,
            "stratify": True,
        }

    def test_prepare_run_config_strips_inner_valid_extras(self) -> None:
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {}},
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "time_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    }
                }
            },
            "data": {"target": "y"},
            "features": {"categorical": [], "exclude": []},
            "split": {"method": "kfold", "n_splits": 5},
            "task": "regression",
            "config_version": 1,
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        iv = result["training"]["early_stopping"]["inner_valid"]
        assert "random_state" not in iv
        assert "stratify" not in iv

    def test_validate_config_normalizes_inner_valid(self) -> None:
        """validate_config should normalize inner_valid before LizyML validation."""
        adapter = LizyMLAdapter()
        config = {
            "model": {"name": "lgbm", "params": {"n_estimators": 100}},
            "task": "regression",
            "config_version": 1,
            "data": {"target": "y"},
            "features": {"categorical": [], "exclude": []},
            "split": {"method": "kfold", "n_splits": 5},
            "training": {
                "seed": 42,
                "early_stopping": {
                    "enabled": True,
                    "inner_valid": {
                        "method": "time_holdout",
                        "ratio": 0.1,
                        "random_state": 42,
                        "stratify": False,
                    },
                    "rounds": 150,
                },
            },
        }
        # Raw load_config would reject extra fields
        from lizyml.config.loader import load_config

        with pytest.raises(Exception, match="CONFIG_INVALID"):
            load_config(config)

        # But validate_config normalizes first, so it passes
        errors = adapter.validate_config(config)
        assert errors == []

    def test_unknown_method_nulled_out(self) -> None:
        """Unknown inner_valid method should be set to None."""
        adapter = LizyMLAdapter()
        config = {
            "training": {
                "early_stopping": {
                    "inner_valid": {
                        "method": "unknown_method",
                        "ratio": 0.1,
                        "extra_field": "should_be_dropped",
                    }
                }
            },
        }
        result = adapter.canonicalize_config(config)
        assert result["training"]["early_stopping"]["inner_valid"] is None


class TestStripForBackend:
    """strip_for_backend removes non-schema fields at all nesting levels."""

    def test_strips_extra_top_level(self) -> None:
        LizyMLAdapter()
        config = {"config_version": 1, "task": "binary", "extra": "bad"}
        result = adapter_schema.strip_for_backend(config)
        assert "extra" not in result
        assert result["config_version"] == 1

    def test_strips_extra_in_model(self) -> None:
        LizyMLAdapter()
        config = {
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 100},
                "widget_only_field": True,
            }
        }
        result = adapter_schema.strip_for_backend(config)
        assert "widget_only_field" not in result["model"]
        assert result["model"]["name"] == "lgbm"
        assert result["model"]["params"]["n_estimators"] == 100

    def test_strips_extra_in_training(self) -> None:
        LizyMLAdapter()
        config = {
            "training": {"seed": 42, "ui_state": "running"},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "ui_state" not in result["training"]
        assert result["training"]["seed"] == 42

    def test_strips_extra_in_data(self) -> None:
        LizyMLAdapter()
        config = {
            "data": {"target": "y", "widget_managed": True},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "widget_managed" not in result["data"]
        assert result["data"]["target"] == "y"

    def test_strips_extra_in_features(self) -> None:
        LizyMLAdapter()
        config = {
            "features": {"exclude": [], "categorical": [], "js_extra": True},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "js_extra" not in result["features"]

    def test_preserves_model_params_additionalProperties(self) -> None:
        """model.params has additionalProperties=true, so custom keys survive."""
        LizyMLAdapter()
        config = {
            "model": {
                "name": "lgbm",
                "params": {"n_estimators": 100, "custom_lgbm_param": 42},
            }
        }
        result = adapter_schema.strip_for_backend(config)
        assert result["model"]["params"]["custom_lgbm_param"] == 42

    def test_multiple_levels_stripped_at_once(self) -> None:
        LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "top_extra": "x",
            "data": {"target": "y", "data_extra": "x"},
            "model": {"name": "lgbm", "params": {}, "model_extra": "x"},
            "training": {"seed": 42, "training_extra": "x"},
        }
        result = adapter_schema.strip_for_backend(config)
        assert "top_extra" not in result
        assert "data_extra" not in result["data"]
        assert "model_extra" not in result["model"]
        assert "training_extra" not in result["training"]

    def test_prepare_run_config_strips_extra_fields(self) -> None:
        """Integration: prepare_run_config output has no extra fields."""
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "ui_state": "running",
            "data": {"target": "y"},
            "features": {"exclude": [], "categorical": []},
            "split": {"method": "kfold", "n_splits": 5},
            "model": {"name": "lgbm", "params": {}, "js_flag": True},
            "training": {"seed": 42, "widget_extra": True},
        }
        result = adapter.prepare_run_config(config, job_type="fit")
        assert "ui_state" not in result
        assert "js_flag" not in result.get("model", {})
        assert "widget_extra" not in result.get("training", {})

    def test_validate_config_ignores_extra_fields(self) -> None:
        """Integration: validate_config passes even with extra fields."""
        adapter = LizyMLAdapter()
        config = {
            "config_version": 1,
            "task": "binary",
            "extra_field": "from_js",
            "data": {"target": "y"},
            "features": {"exclude": [], "categorical": []},
            "split": {"method": "kfold", "n_splits": 5},
            "model": {
                "name": "lgbm",
                "params": {"objective": "binary", "metric": ["auc"]},
                "widget_flag": True,
            },
            "training": {"seed": 42},
        }
        errors = adapter.validate_config(config)
        assert errors == []


class TestNormalizeInnerValidExclusivity:
    """Tests for validation_ratio / inner_valid mutual exclusivity in apply_best_params."""

    def test_validation_ratio_set_inner_valid_none_removes_inner_valid_key(self) -> None:
        """When validation_ratio is set and inner_valid is None,
        inner_valid key must be removed entirely (not kept as None).

        Pydantic's model_fields_set treats explicit None as "set", causing
        the 'Specify either validation_ratio or inner_valid' error.
        """
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                    "validation_ratio": 0.2,
                    "inner_valid": None,
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert "inner_valid" not in es, (
            "inner_valid key must be removed when validation_ratio is set"
        )
        assert es["validation_ratio"] == 0.2

    def test_inner_valid_set_validation_ratio_present_removes_validation_ratio(self) -> None:
        """When inner_valid is explicitly set (non-None), validation_ratio
        must be removed to avoid mutual exclusivity error.
        """
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                    "validation_ratio": 0.2,
                    "inner_valid": {"method": "holdout", "ratio": 0.1},
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert "validation_ratio" not in es, (
            "validation_ratio must be removed when inner_valid is set"
        )
        assert es["inner_valid"] == {"method": "holdout", "ratio": 0.1}

    def test_neither_set_leaves_config_unchanged(self) -> None:
        """When neither validation_ratio nor inner_valid is set, config is unchanged."""
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert "inner_valid" not in es
        assert "validation_ratio" not in es

    def test_validation_ratio_only_no_inner_valid_key_unchanged(self) -> None:
        """When only validation_ratio is present (no inner_valid key at all), unchanged."""
        from lizyml_widget.adapter_schema import enforce_iv_exclusivity

        config = {
            "training": {
                "early_stopping": {
                    "rounds": 150,
                    "validation_ratio": 0.2,
                }
            }
        }
        result = enforce_iv_exclusivity(config)
        es = result["training"]["early_stopping"]
        assert es["validation_ratio"] == 0.2
        assert "inner_valid" not in es
