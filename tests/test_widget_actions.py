"""Tests for LizyWidget action dispatch and handler edge cases."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo, PredictionSummary


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
        adapter.classify_best_params.side_effect = real_adapter.classify_best_params

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


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


class TestApplyBestParamsRouting:
    """Test that _handle_apply_best_params routes params to correct config locations."""

    def test_smart_params_at_model_level(self) -> None:
        """num_leaves_ratio should be at model.num_leaves_ratio, not model.params."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        best_params = {
            "learning_rate": 0.05,
            "num_leaves_ratio": 0.7,
            "min_data_in_leaf_ratio": 0.05,
        }
        w._handle_apply_best_params({"params": best_params})

        model = dict(w.config.get("model", {}))
        assert model.get("num_leaves_ratio") == 0.7
        assert model.get("min_data_in_leaf_ratio") == 0.05
        assert "num_leaves_ratio" not in model.get("params", {})
        assert "min_data_in_leaf_ratio" not in model.get("params", {})

    def test_training_params_at_training_level(self) -> None:
        """early_stopping_rounds -> training.early_stopping.rounds."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        best_params = {
            "learning_rate": 0.05,
            "early_stopping_rounds": 80,
            "validation_ratio": 0.2,
        }
        w._handle_apply_best_params({"params": best_params})

        config = dict(w.config)
        es = config.get("training", {}).get("early_stopping", {})
        assert es.get("rounds") == 80
        assert es.get("validation_ratio") == 0.2
        model_params = config.get("model", {}).get("params", {})
        assert "early_stopping_rounds" not in model_params
        assert "validation_ratio" not in model_params

    def test_model_params_still_in_model_params(self) -> None:
        """Model-category params should remain in model.params."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        best_params = {
            "learning_rate": 0.05,
            "max_depth": 7,
            "feature_fraction": 0.8,
        }
        w._handle_apply_best_params({"params": best_params})

        model_params = dict(w.config.get("model", {}).get("params", {}))
        assert model_params.get("learning_rate") == 0.05
        assert model_params.get("max_depth") == 7
        assert model_params.get("feature_fraction") == 0.8

    def test_validation_ratio_clears_inner_valid(self) -> None:
        """When validation_ratio is set from best_params, inner_valid must be
        set to None in the config traitlet so that it overrides the default
        dict value.  enforce_iv_exclusivity in prepare_run_config strips the
        None key before backend validation.
        """
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        best_params = {
            "learning_rate": 0.05,
            "validation_ratio": 0.2,
        }
        w._handle_apply_best_params({"params": best_params})

        config = dict(w.config)
        es = config.get("training", {}).get("early_stopping", {})
        assert es.get("validation_ratio") == 0.2
        # inner_valid must be None (overrides default dict);
        # enforce_iv_exclusivity strips it at prepare_run_config time
        assert es.get("inner_valid") is None, (
            f"inner_valid should be None when validation_ratio is set, got {es.get('inner_valid')}"
        )

    def test_apply_best_params_config_validates_cleanly(self) -> None:
        """Config after apply_best_params with validation_ratio must pass validation."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        best_params = {
            "learning_rate": 0.05,
            "early_stopping_rounds": 80,
            "validation_ratio": 0.2,
        }
        w._handle_apply_best_params({"params": best_params})

        # Build run config and validate — should not raise
        full_config = w._service.prepare_run_config(dict(w.config), job_type="fit")
        errors = w._service.validate_config(full_config)
        assert errors == [], f"Validation errors: {errors}"

    def test_apply_best_params_with_validation_ratio_passes_real_validation(self) -> None:
        """Full flow: apply best_params with validation_ratio, then validate
        against the REAL LizyML backend validator (not mocked).

        Reproduces: VALIDATION_ERROR 'Specify either validation_ratio or inner_valid, not both.'
        """
        real_adapter = LizyMLAdapter()
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        best_params = {
            "learning_rate": 0.05,
            "early_stopping_rounds": 80,
            "validation_ratio": 0.2,
        }
        w._handle_apply_best_params({"params": best_params})

        # Use REAL adapter validate_config (not mock)
        full_config = w._service.prepare_run_config(dict(w.config), job_type="fit")
        errors = real_adapter.validate_config(full_config)
        assert errors == [], f"Real validation errors after apply_best_params: {errors}"


class TestApplyBestParamsExceptionHandling:
    """_handle_apply_best_params errors should be caught."""

    def test_classify_error_returns_apply_error(self) -> None:
        """If classify_best_params raises, error traitlet should have APPLY_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.classify_best_params = MagicMock(side_effect=ValueError("bad params"))

        w._handle_apply_best_params({"params": {"learning_rate": 0.1}})
        assert w.error.get("code") == "APPLY_ERROR"

    def test_canonicalize_error_returns_apply_error(self) -> None:
        """If canonicalize_config raises, error traitlet should have APPLY_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.canonicalize_config = MagicMock(side_effect=RuntimeError("schema error"))

        w._handle_apply_best_params({"params": {"learning_rate": 0.1}})
        assert w.error.get("code") == "APPLY_ERROR"


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


class TestPatchConfigPathValidation:
    """patch_config must validate path format before processing."""

    def test_valid_dotted_path_accepted(self) -> None:
        """Normal dotted paths like 'model.params.learning_rate' should work."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.action = {
            "type": "patch_config",
            "payload": {
                "ops": [{"op": "set", "path": "model.params.learning_rate", "value": 0.01}]
            },
        }
        # Should succeed — no error
        assert w.error.get("code") != "INVALID_PATCH"
        assert w.config["model"]["params"]["learning_rate"] == 0.01

    def test_dunder_path_rejected(self) -> None:
        """Paths containing '__' (dunder) should be rejected."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "set", "path": "__class__.__init__", "value": "bad"}]},
        }
        assert w.error.get("code") == "INVALID_PATCH"

    def test_empty_path_rejected(self) -> None:
        """Empty path should be rejected."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "set", "path": "", "value": "bad"}]},
        }
        assert w.error.get("code") == "INVALID_PATCH"

    def test_path_with_special_chars_rejected(self) -> None:
        """Paths with special characters should be rejected."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "set", "path": "model/../etc/passwd", "value": "bad"}]},
        }
        assert w.error.get("code") == "INVALID_PATCH"


class TestPatchConfigOpValidation:
    """patch_config must validate op field before processing."""

    def test_missing_op_returns_error(self) -> None:
        """Missing 'op' key should return INVALID_PATCH, not raise KeyError."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"path": "model.params.learning_rate", "value": 0.01}]},
        }
        assert w.error.get("code") == "INVALID_PATCH"

    def test_invalid_op_value_returns_error(self) -> None:
        """Unknown op value (e.g. 'delete') should return INVALID_PATCH."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "delete", "path": "model.params.x", "value": 1}]},
        }
        assert w.error.get("code") == "INVALID_PATCH"

    def test_valid_ops_accepted(self) -> None:
        """Valid ops ('set', 'unset', 'merge') should be accepted."""
        for op in ("set", "unset", "merge"):
            w = _make_widget()
            df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
            w.load(df, target="y")

            w.action = {
                "type": "patch_config",
                "payload": {
                    "ops": [{"op": op, "path": "model.params.learning_rate", "value": 0.01}]
                },
            }
            assert w.error.get("code") != "INVALID_PATCH", f"op={op!r} should be valid"


class TestActionHandlerExceptionBranches:
    """Cover except branches in action handlers (widget.py lines 260-335)."""

    def test_set_target_service_error_sets_target_error(self) -> None:
        """widget.py:260-261: service.set_target raises → TARGET_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df)
        w._service.set_target = MagicMock(side_effect=ValueError("column not found"))
        w.action = {"type": "set_target", "payload": {"target": "y"}}
        assert w.error["code"] == "TARGET_ERROR"
        assert "column not found" in w.error["message"]

    def test_set_task_service_error_sets_task_error(self) -> None:
        """widget.py:272-273: service.set_task raises → TASK_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w._service.set_task = MagicMock(side_effect=ValueError("Invalid task"))
        w.action = {"type": "set_task", "payload": {"task": "bad"}}
        assert w.error["code"] == "TASK_ERROR"
        assert "Invalid task" in w.error["message"]

    def test_update_column_service_error_sets_column_error(self) -> None:
        """widget.py:293-294: service.update_column raises → COLUMN_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w._service.update_column = MagicMock(side_effect=RuntimeError("db error"))
        w.action = {
            "type": "update_column",
            "payload": {"name": "x", "excluded": True, "col_type": "numeric"},
        }
        assert w.error["code"] == "COLUMN_ERROR"
        assert "db error" in w.error["message"]

    def test_update_cv_service_error_sets_cv_error(self) -> None:
        """widget.py:334-335: service.update_cv raises → CV_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        w._service.update_cv = MagicMock(side_effect=ValueError("bad cv config"))
        w.action = {
            "type": "update_cv",
            "payload": {"strategy": "kfold", "n_splits": 5},
        }
        assert w.error["code"] == "CV_ERROR"
        assert "bad cv config" in w.error["message"]


class TestPatchConfigExceptionHandling:
    """apply_config_patch errors should be caught, not propagated."""

    def test_adapter_error_returns_patch_error(self) -> None:
        """If apply_config_patch raises, error traitlet should have PATCH_ERROR."""
        w = _make_widget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")

        w._service.apply_config_patch = MagicMock(side_effect=TypeError("deepcopy failed"))

        w.action = {
            "type": "patch_config",
            "payload": {"ops": [{"op": "set", "path": "model.params.x", "value": 1}]},
        }
        assert w.error.get("code") == "PATCH_ERROR"
        assert "deepcopy" in w.error.get("message", "")
