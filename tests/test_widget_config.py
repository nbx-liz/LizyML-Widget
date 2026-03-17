"""Tests for LizyWidget config flow, backend contract, and canonical config."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo


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
