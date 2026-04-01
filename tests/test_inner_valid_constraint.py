"""Tests for inner_valid constraint validation.

group_holdout requires group_column set in cv config.
time_holdout requires time_column set in cv config.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.service import WidgetService
from lizyml_widget.types import BackendInfo


def _mock_adapter() -> Any:
    """Create a mock adapter that delegates config lifecycle to LizyMLAdapter."""
    real_adapter = LizyMLAdapter()
    adapter = MagicMock()
    adapter.info = BackendInfo(name="mock", version="0.0.0")
    adapter.get_config_schema.return_value = {}

    adapter.initialize_config.side_effect = real_adapter.initialize_config
    adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
    adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
    adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
    adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
    adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
    adapter.validate_config.return_value = []
    return adapter


def _make_service(
    *,
    group_column: str | None = None,
    time_column: str | None = None,
) -> WidgetService:
    """Create a WidgetService with data loaded and optional column settings."""
    adapter = _mock_adapter()
    svc = WidgetService(adapter=adapter)
    df = pd.DataFrame({"x": range(50), "grp": [0, 1] * 25, "ts": range(50), "y": [0, 1] * 25})
    svc.load_data(df, target="y")

    # Set group/time columns via CV update
    svc.update_cv(
        "kfold",
        5,
        group_column=group_column,
        time_column=time_column,
    )
    return svc


class TestInnerValidGroupHoldout:
    """group_holdout requires group_column."""

    def test_validate_rejects_group_holdout_without_group_column(self) -> None:
        svc = _make_service(group_column=None)
        config = svc.initialize_config()
        config["training"] = {
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": "group_holdout"},
            },
        }
        errors = svc.validate_config(config)
        assert any("group_holdout" in e["message"] for e in errors)

    def test_validate_accepts_group_holdout_with_group_column(self) -> None:
        svc = _make_service(group_column="grp")
        config = svc.initialize_config()
        config["training"] = {
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": "group_holdout"},
            },
        }
        errors = svc.validate_config(config)
        assert not any("group_holdout" in e.get("message", "") for e in errors)


class TestInnerValidTimeHoldout:
    """time_holdout requires time_column."""

    def test_validate_rejects_time_holdout_without_time_column(self) -> None:
        svc = _make_service(time_column=None)
        config = svc.initialize_config()
        config["training"] = {
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": "time_holdout"},
            },
        }
        errors = svc.validate_config(config)
        assert any("time_holdout" in e["message"] for e in errors)

    def test_validate_accepts_time_holdout_with_time_column(self) -> None:
        svc = _make_service(time_column="ts")
        config = svc.initialize_config()
        config["training"] = {
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": "time_holdout"},
            },
        }
        errors = svc.validate_config(config)
        assert not any("time_holdout" in e.get("message", "") for e in errors)


class TestInnerValidHoldoutAlwaysOk:
    """plain holdout always accepted regardless of column settings."""

    def test_validate_accepts_holdout_without_any_columns(self) -> None:
        svc = _make_service()
        config = svc.initialize_config()
        config["training"] = {
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": "holdout"},
            },
        }
        errors = svc.validate_config(config)
        assert not any("holdout" in e.get("message", "") for e in errors)
