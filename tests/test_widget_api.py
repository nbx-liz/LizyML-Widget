"""Tests for LizyWidget Python API methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from lizyml_widget.types import BackendInfo, ConfigSchema


def _make_widget() -> Any:
    """Create a LizyWidget with mocked LizyML backend."""
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = ConfigSchema(json_schema={"type": "object"})
        adapter.validate_config.return_value = []

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    return w


class TestLoadData:
    def test_load_sets_status(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        assert w.status == "data_loaded"
        assert w.df_info["target"] == "y"
        assert w.df_info["task"] == "binary"

    def test_load_without_target(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        w.load(df)
        assert w.status == "data_loaded"
        assert w.df_info["target"] is None

    def test_load_resets_summaries(self) -> None:
        w = _make_widget()
        w.fit_summary = {"some": "data"}
        w.tune_summary = {"some": "data"}
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df)
        assert w.fit_summary == {}
        assert w.tune_summary == {}


class TestConfig:
    def test_set_get_config(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm"}})
        assert w.get_config() == {"model": {"name": "lgbm"}}

    def test_get_config_returns_copy(self) -> None:
        w = _make_widget()
        w.set_config({"model": {"name": "lgbm"}})
        cfg = w.get_config()
        cfg["extra"] = True
        assert "extra" not in w.get_config()


class TestSummaryAPI:
    def test_get_fit_summary_empty(self) -> None:
        w = _make_widget()
        assert w.get_fit_summary() == {}

    def test_get_tune_summary_empty(self) -> None:
        w = _make_widget()
        assert w.get_tune_summary() == {}

    def test_get_model_none(self) -> None:
        w = _make_widget()
        assert w.get_model() is None


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
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df)
        w.action = {"type": "set_target", "payload": {"target": "y"}}
        assert w.df_info["target"] == "y"

    def test_update_column_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_column",
            "payload": {"name": "x", "excluded": True, "col_type": "numeric"},
        }
        x_col = next(c for c in w.df_info["columns"] if c["name"] == "x")
        assert x_col["excluded"] is True

    def test_update_cv_action(self) -> None:
        w = _make_widget()
        df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
        w.load(df, target="y")
        w.action = {
            "type": "update_cv",
            "payload": {"strategy": "group_kfold", "n_splits": 3, "group_column": "x"},
        }
        assert w.df_info["cv"]["strategy"] == "group_kfold"
        assert w.df_info["cv"]["n_splits"] == 3

    def test_update_config_action(self) -> None:
        w = _make_widget()
        w.action = {
            "type": "update_config",
            "payload": {"config": {"model": {"name": "rf"}}},
        }
        assert w.config == {"model": {"name": "rf"}}

    def test_unknown_action_ignored(self) -> None:
        w = _make_widget()
        # Should not raise
        w.action = {"type": "nonexistent", "payload": {}}

    def test_empty_action_ignored(self) -> None:
        w = _make_widget()
        w.action = {}
