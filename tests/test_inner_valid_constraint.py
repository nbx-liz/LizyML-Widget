"""Tests for inner_valid constraint validation.

group_holdout requires a group-based CV strategy.
time_holdout requires a time-based CV strategy.
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


def _make_service(strategy: str = "kfold") -> WidgetService:
    """Create a WidgetService with data loaded and given CV strategy."""
    adapter = _mock_adapter()
    svc = WidgetService(adapter=adapter)
    df = pd.DataFrame({"x": range(50), "grp": [0, 1] * 25, "ts": range(50), "y": [0, 1] * 25})
    svc.load_data(df, target="y")
    svc.update_cv(strategy, 5)
    return svc


def _config_with_inner_valid(method: str) -> dict[str, Any]:
    """Build a config dict with the given inner_valid method."""
    return {
        "training": {
            "early_stopping": {
                "enabled": True,
                "inner_valid": {"method": method},
            },
        },
    }


class TestInnerValidGroupHoldout:
    """group_holdout requires a group-based CV strategy."""

    def test_rejects_group_holdout_with_kfold(self) -> None:
        svc = _make_service("kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("group_holdout")}
        errors = svc.validate_config(config)
        assert any("group_holdout" in e["message"] for e in errors)

    def test_rejects_group_holdout_with_time_series(self) -> None:
        svc = _make_service("time_series")
        config = {**svc.initialize_config(), **_config_with_inner_valid("group_holdout")}
        errors = svc.validate_config(config)
        assert any("group_holdout" in e["message"] for e in errors)

    def test_accepts_group_holdout_with_group_kfold(self) -> None:
        svc = _make_service("group_kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("group_holdout")}
        errors = svc.validate_config(config)
        assert not any("group_holdout" in e.get("message", "") for e in errors)

    def test_accepts_group_holdout_with_stratified_group_kfold(self) -> None:
        svc = _make_service("stratified_group_kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("group_holdout")}
        errors = svc.validate_config(config)
        assert not any("group_holdout" in e.get("message", "") for e in errors)

    def test_accepts_group_holdout_with_group_time_series(self) -> None:
        svc = _make_service("group_time_series")
        config = {**svc.initialize_config(), **_config_with_inner_valid("group_holdout")}
        errors = svc.validate_config(config)
        assert not any("group_holdout" in e.get("message", "") for e in errors)

    def test_accepts_group_holdout_with_blocked_group_kfold(self) -> None:
        svc = _make_service("blocked_group_kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("group_holdout")}
        errors = svc.validate_config(config)
        assert not any("group_holdout" in e.get("message", "") for e in errors)


class TestInnerValidTimeHoldout:
    """time_holdout requires a time-based CV strategy."""

    def test_rejects_time_holdout_with_kfold(self) -> None:
        svc = _make_service("kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("time_holdout")}
        errors = svc.validate_config(config)
        assert any("time_holdout" in e["message"] for e in errors)

    def test_rejects_time_holdout_with_group_kfold(self) -> None:
        svc = _make_service("group_kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("time_holdout")}
        errors = svc.validate_config(config)
        assert any("time_holdout" in e["message"] for e in errors)

    def test_accepts_time_holdout_with_time_series(self) -> None:
        svc = _make_service("time_series")
        config = {**svc.initialize_config(), **_config_with_inner_valid("time_holdout")}
        errors = svc.validate_config(config)
        assert not any("time_holdout" in e.get("message", "") for e in errors)

    def test_accepts_time_holdout_with_purged_time_series(self) -> None:
        svc = _make_service("purged_time_series")
        config = {**svc.initialize_config(), **_config_with_inner_valid("time_holdout")}
        errors = svc.validate_config(config)
        assert not any("time_holdout" in e.get("message", "") for e in errors)

    def test_accepts_time_holdout_with_group_time_series(self) -> None:
        svc = _make_service("group_time_series")
        config = {**svc.initialize_config(), **_config_with_inner_valid("time_holdout")}
        errors = svc.validate_config(config)
        assert not any("time_holdout" in e.get("message", "") for e in errors)


class TestInnerValidHoldoutAlwaysOk:
    """plain holdout always accepted regardless of strategy."""

    def test_accepts_holdout_with_kfold(self) -> None:
        svc = _make_service("kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("holdout")}
        errors = svc.validate_config(config)
        assert not any("holdout" in e.get("message", "") for e in errors)

    def test_accepts_holdout_with_group_kfold(self) -> None:
        svc = _make_service("group_kfold")
        config = {**svc.initialize_config(), **_config_with_inner_valid("holdout")}
        errors = svc.validate_config(config)
        assert not any("holdout" in e.get("message", "") for e in errors)
