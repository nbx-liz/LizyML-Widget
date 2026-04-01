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

    def test_smart_params_matches_smart_param_keys(self) -> None:
        """SMART_PARAMS (classify) must be a superset of SMART_PARAM_KEYS (strip)."""
        assert adapter_schema.SMART_PARAM_KEYS <= adapter_params.SMART_PARAMS


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
    """apply_best_params must restore Tune execution config accurately,
    including smart params and calibration that are stripped in run config."""

    def test_apply_best_params_preserves_smart_params(self) -> None:
        """Smart params from UI config must survive apply_best_params."""
        with patch.dict("sys.modules", _mock_schema_modules()):
            adapter = LizyMLAdapter()
            service = _make_service(adapter)
            # UI config has smart params
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
            # Simulate tune_snapshot that is the run config (stripped):
            # - smart params removed, calibration removed (as prepare_tune_overrides does)
            run_snapshot: dict[str, Any] = {
                "config_version": 1,
                "model": {
                    "name": "lgbm",
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
            # Smart params must be restored from UI snapshot
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


# ── Helpers ──


def _make_service(adapter: LizyMLAdapter) -> WidgetService:
    """Create a WidgetService with the given adapter for testing."""
    return WidgetService(adapter=adapter)
