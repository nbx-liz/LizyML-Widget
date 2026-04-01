"""Regression tests for config generation bugs.

Each test class targets a specific discovered bug and follows TDD:
write the failing test first, then fix the implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget import adapter_params, adapter_schema
from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.service import WidgetService


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """Reset module-level caches between tests."""
    adapter_params._eval_metrics_cache = None
    adapter_schema.reset_schema_cache()
    yield
    adapter_params._eval_metrics_cache = None
    adapter_schema.reset_schema_cache()


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
                    "name": {"type": "string", "default": "lgbm"},
                    "params": {"type": "object", "default": {}},
                    "auto_num_leaves": {"type": "boolean", "default": True},
                    "num_leaves_ratio": {"type": "number", "default": 1.0},
                    "min_data_in_leaf_ratio": {"type": "number", "default": 0.01},
                    "min_data_in_bin_ratio": {"type": "number", "default": 0.01},
                    "feature_weights": {"default": None},
                    "balanced": {"type": "boolean", "default": False},
                },
            },
            "training": {
                "type": "object",
                "properties": {
                    "seed": {"type": "integer", "default": 42},
                    "early_stopping": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "rounds": {"type": "integer", "default": 150},
                            "validation_ratio": {"type": "number", "default": 0.1},
                        },
                    },
                },
            },
            "evaluation": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "default": []},
                },
            },
            "output_dir": {"type": "string", "default": "outputs/"},
            "tuning": {"type": "object", "properties": {}},
            "calibration": {"type": "object", "properties": {}},
        },
    }
    mock_module = MagicMock()
    mock_module.LizyMLConfig = mock_config_cls
    return {
        "lizyml": MagicMock(),
        "lizyml.config": MagicMock(),
        "lizyml.config.schema": mock_module,
    }


# ── Bug 1: SMART_PARAMS missing auto_num_leaves, feature_weights, balanced ──


class TestBug1SmartParamsSync:
    """SMART_PARAMS in adapter_params.py must include all smart param keys."""

    def test_classify_auto_num_leaves_as_smart(self) -> None:
        """auto_num_leaves should be classified as smart, not model."""
        model_p, smart_p, _ = adapter_params.classify_best_params(
            {"auto_num_leaves": True, "learning_rate": 0.01}
        )
        assert "auto_num_leaves" in smart_p
        assert "auto_num_leaves" not in model_p

    def test_classify_feature_weights_as_smart(self) -> None:
        """feature_weights should be classified as smart, not model."""
        model_p, smart_p, _ = adapter_params.classify_best_params(
            {"feature_weights": {"col_a": 1.0}, "learning_rate": 0.01}
        )
        assert "feature_weights" in smart_p
        assert "feature_weights" not in model_p

    def test_classify_balanced_as_smart(self) -> None:
        """balanced should be classified as smart, not model."""
        model_p, smart_p, _ = adapter_params.classify_best_params(
            {"balanced": True, "learning_rate": 0.01}
        )
        assert "balanced" in smart_p
        assert "balanced" not in model_p

    def test_smart_params_covers_all_known_keys(self) -> None:
        """SMART_PARAMS (classify) must include all known smart param keys."""
        expected = {
            "auto_num_leaves",
            "num_leaves_ratio",
            "min_data_in_leaf_ratio",
            "min_data_in_bin_ratio",
            "feature_weights",
            "balanced",
        }
        assert expected <= adapter_params.SMART_PARAMS


# ── Bug 4: prepare_tune_overrides replaces training instead of merging ──


class TestBug4TrainingMerge:
    """prepare_tune_overrides should shallow-merge tune training, not replace."""

    def test_tune_training_preserves_seed(self) -> None:
        """Fields not in tune_training (e.g. seed) must survive the override."""
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {"learning_rate": 0.01}},
            "training": {
                "seed": 42,
                "early_stopping": {
                    "enabled": True,
                    "rounds": 150,
                    "validation_ratio": 0.1,
                },
            },
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "training": {
                    "early_stopping": {
                        "enabled": True,
                        "rounds": 100,
                        "validation_ratio": 0.2,
                    },
                },
            },
        }
        result = adapter_schema.prepare_tune_overrides(config)
        # seed should still be present (not lost by complete replacement)
        assert result.get("training", {}).get("seed") == 42

    def test_tune_training_overrides_rounds(self) -> None:
        """Tune-specific rounds value should take precedence."""
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "training": {
                "seed": 42,
                "early_stopping": {"enabled": True, "rounds": 150},
            },
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "training": {
                    "early_stopping": {"enabled": True, "rounds": 80},
                },
            },
        }
        result = adapter_schema.prepare_tune_overrides(config)
        es = result.get("training", {}).get("early_stopping", {})
        assert es.get("rounds") == 80


# ── Bug 5: metric string→list normalization in apply_best_params ──


class TestBug5MetricNormalization:
    """best_params.metric (single string from Optuna) must become a list."""

    def test_metric_string_wrapped_in_list(self) -> None:
        """A single metric string from best_params must be wrapped in a list."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            service = _make_service(adapter)
            base_config: dict[str, Any] = {
                "config_version": 1,
                "model": {
                    "name": "lgbm",
                    "params": {"metric": ["auc", "binary_logloss"]},
                },
                "training": {"seed": 42, "early_stopping": {"rounds": 150}},
            }
            result = service.apply_best_params(
                {"metric": "auc", "learning_rate": 0.05},
                base_config,
            )
            metric = result.get("model", {}).get("params", {}).get("metric")
            assert isinstance(metric, list), f"metric should be list, got {type(metric)}"


# ── Bug 6: direction fallback when eval_metrics is empty ──


class TestBug6DirectionFallback:
    """Direction must be resolved even when evaluation.metrics is empty."""

    def test_direction_set_when_eval_metrics_empty(self) -> None:
        """tuning.evaluation exists with metrics=[] — direction should fall back
        to model.params.metric[0] via MODEL_METRIC_TO_EVAL mapping."""
        config: dict[str, Any] = {
            "model": {
                "name": "lgbm",
                "params": {"metric": ["auc", "binary_logloss"]},
            },
            "training": {"seed": 42},
            "evaluation": {"metrics": []},
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
                "evaluation": {"metrics": []},
            },
        }
        result = adapter_schema.prepare_tune_overrides(config)
        direction = result.get("tuning", {}).get("optuna", {}).get("params", {}).get("direction")
        assert direction is not None, "direction should be set even with empty eval metrics"


# ── Bug 2+3: Dual snapshot — tune_config_snapshot contains stripped config ──


class TestBug2DualSnapshot:
    """apply_best_params must restore calibration from UI snapshot.

    After the fix for Bug 7, smart params are no longer stripped from
    tune_snapshot, so only calibration needs restoration from UI snapshot.
    """

    def test_apply_best_params_preserves_smart_params_from_snapshot(self) -> None:
        """Smart params in tune_snapshot must survive apply_best_params."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            service = _make_service(adapter)
            ui_config: dict[str, Any] = {
                "config_version": 1,
                "model": {
                    "name": "lgbm",
                    "auto_num_leaves": True,
                    "num_leaves_ratio": 0.8,
                    "min_data_in_leaf_ratio": 0.02,
                    "min_data_in_bin_ratio": 0.01,
                    "balanced": False,
                    "params": {
                        "learning_rate": 0.001,
                        "n_estimators": 1500,
                        "metric": ["auc", "binary_logloss"],
                    },
                },
                "training": {"seed": 42, "early_stopping": {"rounds": 150}},
                "calibration": {"method": "platt"},
            }
            # tune_snapshot now includes smart params (calibration still stripped)
            run_snapshot: dict[str, Any] = {
                "config_version": 1,
                "model": {
                    "name": "lgbm",
                    "auto_num_leaves": True,
                    "num_leaves_ratio": 0.8,
                    "min_data_in_leaf_ratio": 0.02,
                    "min_data_in_bin_ratio": 0.01,
                    "balanced": False,
                    "params": {"learning_rate": 0.001, "n_estimators": 1500},
                },
                "training": {"seed": 42, "early_stopping": {"rounds": 100}},
            }
            result = service.apply_best_params(
                {"learning_rate": 0.05},
                ui_config,
                tune_snapshot=run_snapshot,
                tune_ui_snapshot=ui_config,
            )
            model = result.get("model", {})
            assert model.get("auto_num_leaves") is True
            assert model.get("num_leaves_ratio") == 0.8

    def test_apply_best_params_preserves_calibration(self) -> None:
        """calibration from UI config must survive apply_best_params."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            service = _make_service(adapter)
            ui_config: dict[str, Any] = {
                "config_version": 1,
                "model": {
                    "name": "lgbm",
                    "params": {"learning_rate": 0.001},
                },
                "training": {"seed": 42, "early_stopping": {"rounds": 150}},
                "calibration": {"method": "platt"},
            }
            # calibration is stripped from tune_snapshot
            run_snapshot: dict[str, Any] = {
                "config_version": 1,
                "model": {"name": "lgbm", "params": {"learning_rate": 0.001}},
                "training": {"seed": 42, "early_stopping": {"rounds": 100}},
            }
            result = service.apply_best_params(
                {"learning_rate": 0.05},
                ui_config,
                tune_snapshot=run_snapshot,
                tune_ui_snapshot=ui_config,
            )
            assert result.get("calibration") == {"method": "platt"}


# ── Bug 7: prepare_tune_overrides strips smart params that LizyML supports ──


class TestBug7TuneSmartParamsPreserved:
    """prepare_tune_overrides must NOT strip smart params.

    LizyML backend supports smart params during tuning (search space can
    include category='smart' dimensions, and resolve_smart_params() is
    called per trial).  Stripping them causes Tune→Apply→Fit score mismatch.
    """

    def test_prepare_tune_overrides_preserves_smart_params(self) -> None:
        """Smart params in model.* must survive prepare_tune_overrides."""
        config: dict[str, Any] = {
            "model": {
                "name": "lgbm",
                "auto_num_leaves": True,
                "num_leaves_ratio": 0.8,
                "min_data_in_leaf_ratio": 0.02,
                "min_data_in_bin_ratio": 0.01,
                "balanced": False,
                "feature_weights": {"col_a": 2.0},
                "params": {"learning_rate": 0.01, "metric": ["auc"]},
            },
            "training": {"seed": 42},
            "tuning": {"optuna": {"params": {"n_trials": 20}, "space": {}}},
        }
        result = adapter_schema.prepare_tune_overrides(config)
        model = result.get("model", {})
        assert model.get("auto_num_leaves") is True
        assert model.get("num_leaves_ratio") == 0.8
        assert model.get("min_data_in_leaf_ratio") == 0.02
        assert model.get("min_data_in_bin_ratio") == 0.01
        assert model.get("balanced") is False
        assert model.get("feature_weights") == {"col_a": 2.0}

    def test_prepare_tune_overrides_still_strips_calibration(self) -> None:
        """Calibration must still be stripped (LizyML tune does not use it)."""
        config: dict[str, Any] = {
            "model": {"name": "lgbm", "params": {}},
            "training": {"seed": 42},
            "calibration": {"method": "platt"},
            "tuning": {"optuna": {"params": {"n_trials": 20}, "space": {}}},
        }
        result = adapter_schema.prepare_tune_overrides(config)
        assert "calibration" not in result


class TestBug7TuneApplyFitConfigIdentity:
    """Tune→Apply to Fit→Fit must produce the same effective config as Tune.

    When best_params from Tune are applied to Fit config, the resulting
    config (after prepare_run_config) must match the config that Tune
    actually used for its best trial, except for the tuning section itself.
    """

    def test_tune_apply_fit_config_matches(self) -> None:
        """Config used by Tune's best trial == config used by Fit after Apply."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            service = _make_service(adapter)

            # User's original config with smart params
            user_config: dict[str, Any] = {
                "config_version": 1,
                "model": {
                    "name": "lgbm",
                    "auto_num_leaves": True,
                    "num_leaves_ratio": 0.8,
                    "min_data_in_leaf_ratio": 0.02,
                    "balanced": False,
                    "params": {
                        "learning_rate": 0.001,
                        "n_estimators": 1500,
                        "metric": ["auc", "binary_logloss"],
                    },
                },
                "training": {
                    "seed": 42,
                    "early_stopping": {"enabled": True, "rounds": 150},
                },
            }

            # Step 1: Simulate prepare_run_config for tune
            tune_config = adapter.prepare_run_config(user_config, job_type="tune", task="binary")

            # Step 2: Simulate best_params from Optuna
            best_params: dict[str, Any] = {
                "learning_rate": 0.05,
                "num_leaves_ratio": 0.7,
                "early_stopping_rounds": 80,
            }

            # Step 3: Apply best params (as Widget does)
            applied_config = service.apply_best_params(
                best_params,
                user_config,
                tune_snapshot=tune_config,
                tune_ui_snapshot=user_config,
            )

            # Step 4: Prepare fit config from applied result
            fit_config = adapter.prepare_run_config(applied_config, job_type="fit", task="binary")

            # The model section (smart params + best params) should match
            # what Tune would have used for the best trial.
            # Key invariant: smart params present in tune_config must also
            # be present in fit_config with the same values (unless overridden
            # by best_params).
            tune_model = tune_config.get("model", {})
            fit_model = fit_config.get("model", {})

            # Smart params must be present in BOTH configs
            for key in (
                "auto_num_leaves",
                "num_leaves_ratio",
                "min_data_in_leaf_ratio",
                "balanced",
            ):
                assert key in tune_model, f"{key} missing from tune config"
                assert key in fit_model, f"{key} missing from fit config"

            # Values from best_params should be applied to fit
            assert fit_model.get("num_leaves_ratio") == 0.7  # from best_params

            # Values NOT in best_params should match tune config
            assert fit_model.get("auto_num_leaves") == tune_model.get("auto_num_leaves")
            assert fit_model.get("balanced") == tune_model.get("balanced")

            # Training best params should be applied
            fit_es = fit_config.get("training", {}).get("early_stopping", {})
            assert fit_es.get("rounds") == 80  # from best_params


# ── Helpers ──


def _make_service(adapter: LizyMLAdapter) -> WidgetService:
    """Create a WidgetService with the given adapter for testing."""
    return WidgetService(adapter=adapter)
