"""Tests for P-025: CV strategy metadata in backend contract + service delegation.

Verifies that:
- build_capabilities() includes cv_strategy_fields, cv_defaults, cv_default_strategy
- build_ui_schema() includes special_search_space_fields
- Service._default_strategy_for_task reads from adapter contract
- Service._default_cv_state reads cv_defaults from adapter contract
- DataTab reads cv_strategy_fields from backendContract prop (static contract test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Python contract tests
# ---------------------------------------------------------------------------


class TestCvStrategyFieldsCapabilities:
    """build_capabilities() must expose cv_strategy_fields, cv_defaults, cv_default_strategy."""

    def test_cv_strategy_fields_present(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        assert "cv_strategy_fields" in caps

    def test_cv_strategy_fields_has_all_strategies(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        fields = caps["cv_strategy_fields"]
        expected_strategies = {
            "kfold",
            "stratified_kfold",
            "group_kfold",
            "stratified_group_kfold",
            "time_series",
            "purged_time_series",
            "group_time_series",
            "blocked_group_kfold",
        }
        assert set(fields.keys()) == expected_strategies

    def test_cv_strategy_fields_group_strategies_have_group_col(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        fields = caps["cv_strategy_fields"]
        for strategy in ("group_kfold", "stratified_group_kfold", "group_time_series"):
            assert "group_col" in fields[strategy], f"{strategy} should have group_col"

    def test_cv_strategy_fields_time_strategies_have_time_or_gap(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        fields = caps["cv_strategy_fields"]
        for strategy in ("time_series", "group_time_series"):
            assert "gap" in fields[strategy], f"{strategy} should have gap"

    def test_cv_strategy_fields_purged_has_purge_gap(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        fields = caps["cv_strategy_fields"]
        assert "purge_gap" in fields["purged_time_series"]

    def test_cv_defaults_present(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        assert "cv_defaults" in caps
        defaults = caps["cv_defaults"]
        assert defaults["n_splits"] == 5
        assert defaults["shuffle"] is True
        assert defaults["random_state"] == 42
        assert defaults["gap"] == 0

    def test_cv_default_strategy_present(self) -> None:
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        assert "cv_default_strategy" in caps
        ds = caps["cv_default_strategy"]
        assert ds["binary"] == "stratified_kfold"
        assert ds["multiclass"] == "stratified_kfold"
        assert ds["regression"] == "kfold"

    def test_cv_strategy_labels_present(self) -> None:
        """#119: every cv_strategies entry must have a human-readable label."""
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        assert "cv_strategy_labels" in caps
        labels = caps["cv_strategy_labels"]
        for strategy in caps["cv_strategies"]:
            assert strategy in labels, f"{strategy} missing from cv_strategy_labels"
            assert isinstance(labels[strategy], str)
            assert labels[strategy]  # non-empty

    def test_additional_params_hidden_keys_present(self) -> None:
        """#119: backend declares which param keys to hide from Additional Params UI."""
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        assert "additional_params_hidden_keys" in caps
        hidden = caps["additional_params_hidden_keys"]
        assert isinstance(hidden, list)
        # verbose / num_threads stay backend-internal even when set on params
        assert "verbose" in hidden
        assert "num_threads" in hidden


class TestSpecialSearchSpaceFields:
    """build_ui_schema() must expose special_search_space_fields."""

    def test_special_search_space_fields_present(self) -> None:
        from lizyml_widget.adapter_contract import build_ui_schema
        from lizyml_widget.adapter_params import get_eval_metrics_by_task

        schema = build_ui_schema(get_eval_metrics_by_task())
        assert "special_search_space_fields" in schema

    def test_special_search_space_fields_values(self) -> None:
        from lizyml_widget.adapter_contract import build_ui_schema
        from lizyml_widget.adapter_params import get_eval_metrics_by_task

        schema = build_ui_schema(get_eval_metrics_by_task())
        fields = schema["special_search_space_fields"]
        assert fields["objective"] == "objective"
        assert fields["metric"] == "model_metric"
        assert fields["inner_valid"] == "inner_valid_picker"


# ---------------------------------------------------------------------------
# Service delegation tests
# ---------------------------------------------------------------------------


class TestServiceCvDelegation:
    """Service reads CV defaults from adapter contract."""

    @pytest.fixture()
    def service_with_adapter(self) -> Any:
        """Create a WidgetService with a mock adapter returning custom contract."""
        from lizyml_widget.service import WidgetService
        from lizyml_widget.types import BackendContract

        adapter = MagicMock()
        adapter.info = {"name": "test", "version": "0.0.0"}
        adapter.get_backend_contract.return_value = BackendContract(
            schema_version=1,
            config_schema={},
            ui_schema={},
            capabilities={
                "cv_default_strategy": {
                    "binary": "group_kfold",
                    "multiclass": "group_kfold",
                    "regression": "time_series",
                },
                "cv_defaults": {
                    "n_splits": 10,
                    "shuffle": False,
                    "random_state": 99,
                    "gap": 5,
                },
            },
        )
        adapter.initialize_config.return_value = {
            "model": {"name": "lgbm", "params": {}},
            "training": {},
        }
        return WidgetService(adapter)

    def test_default_strategy_from_contract(self, service_with_adapter: Any) -> None:
        svc = service_with_adapter
        assert svc._default_strategy_for_task("binary") == "group_kfold"
        assert svc._default_strategy_for_task("regression") == "time_series"

    def test_default_strategy_fallback_on_missing_task(self, service_with_adapter: Any) -> None:
        svc = service_with_adapter
        # Task not in contract defaults -> fallback
        assert svc._default_strategy_for_task("unknown_task") == "kfold"

    def test_default_strategy_fallback_on_adapter_error(self) -> None:
        from lizyml_widget.service import WidgetService

        adapter = MagicMock()
        adapter.info = {"name": "test", "version": "0.0.0"}
        adapter.get_backend_contract.side_effect = RuntimeError("no contract")
        adapter.initialize_config.return_value = {
            "model": {"name": "lgbm", "params": {}},
            "training": {},
        }
        svc = WidgetService(adapter)
        assert svc._default_strategy_for_task("binary") == "stratified_kfold"
        assert svc._default_strategy_for_task("regression") == "kfold"

    def test_default_cv_state_from_contract(self, service_with_adapter: Any) -> None:
        svc = service_with_adapter
        state = svc._default_cv_state(strategy="kfold", n_splits=3)
        assert state["random_state"] == 99
        assert state["shuffle"] is False
        assert state["gap"] == 5

    def test_default_cv_state_fallback_on_error(self) -> None:
        from lizyml_widget.service import WidgetService

        adapter = MagicMock()
        adapter.info = {"name": "test", "version": "0.0.0"}
        adapter.get_backend_contract.side_effect = RuntimeError("no contract")
        adapter.initialize_config.return_value = {
            "model": {"name": "lgbm", "params": {}},
            "training": {},
        }
        svc = WidgetService(adapter)
        state = svc._default_cv_state(strategy="kfold", n_splits=5)
        # Fallback defaults
        assert state["random_state"] == 42
        assert state["shuffle"] is True
        assert state["gap"] == 0


# ---------------------------------------------------------------------------
# Frontend static contract tests (P-025)
# ---------------------------------------------------------------------------

JS_SRC = Path(__file__).resolve().parent.parent / "js" / "src"


class TestDataTabContractDriven:
    """DataTab must read CV strategy fields from backend_contract."""

    def test_data_tab_has_backend_contract_prop(self) -> None:
        content = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        assert "backendContract" in content, "DataTab should accept backendContract prop"

    def test_data_tab_reads_cv_strategy_fields(self) -> None:
        content = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        assert "cv_strategy_fields" in content, (
            "DataTab should read cv_strategy_fields from contract capabilities"
        )

    def test_data_tab_keeps_fallback_sets(self) -> None:
        content = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        assert "FALLBACK_NEEDS_GROUP" in content, "DataTab should keep fallback sets"
        assert "FALLBACK_NEEDS_TIME" in content

    def test_app_passes_backend_contract_to_data_tab(self) -> None:
        content = (JS_SRC / "App.tsx").read_text()
        # Find DataTab usage and verify backendContract prop
        data_tab_idx = content.index("<DataTab")
        data_tab_section = content[data_tab_idx : data_tab_idx + 500]
        assert "backendContract" in data_tab_section, (
            "App.tsx should pass backendContract to DataTab"
        )

    def test_data_tab_has_no_static_cv_strategies_array(self) -> None:
        """#119: the CV_STRATEGIES literal array is removed; dropdown is contract-driven."""
        content = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        assert "const CV_STRATEGIES" not in content, (
            "DataTab should no longer ship a static CV_STRATEGIES literal — "
            "read from backend_contract.capabilities.cv_strategies instead."
        )

    def test_data_tab_reads_cv_strategies_from_contract(self) -> None:
        """#119: dropdown options sourced from backend contract."""
        content = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        assert "capabilities.cv_strategies" in content, (
            "DataTab should read cv_strategies from contract capabilities."
        )
        assert "cv_strategy_labels" in content, (
            "DataTab should read cv_strategy_labels from contract capabilities."
        )


class TestModelEditorsContractDriven:
    """#119: ModelEditors must source LightGBM-specific keys from the backend contract."""

    def test_model_editors_no_static_handled_model_fields(self) -> None:
        content = (JS_SRC / "components" / "ModelEditors.tsx").read_text()
        # The legacy literal `const HANDLED_MODEL_FIELDS = new Set([...])` must be gone.
        assert "const HANDLED_MODEL_FIELDS = new Set" not in content, (
            "Static HANDLED_MODEL_FIELDS literal should be replaced by a "
            "catalog-derived set (smart_params + structural keys)."
        )

    def test_model_editors_no_num_leaves_default_literal(self) -> None:
        content = (JS_SRC / "components" / "ModelEditors.tsx").read_text()
        # `??\s*256` literal default is gone — default flows from contract default.
        assert "?? 256" not in content, (
            "num_leaves default 256 literal must be removed; the default lives "
            "in backend search_space_catalog and propagates via props."
        )

    def test_model_editors_excludes_via_contract_keys(self) -> None:
        content = (JS_SRC / "components" / "ModelEditors.tsx").read_text()
        # Hardcoded "verbose" / "num_threads" exclusions in the Set literal must be gone.
        # Acceptable references: only the prop name `additionalParamsHiddenKeys`.
        assert '"verbose", "num_leaves", "num_threads"' not in content, (
            "Hardcoded LightGBM-specific exclusion list should route through "
            "additionalParamsHiddenKeys / smart_params keys from contract."
        )
        assert "additionalParamsHiddenKeys" in content, (
            "ModelEditors should accept the additional_params_hidden_keys list "
            "from the backend contract."
        )


class TestSearchSpaceContractDriven:
    """SearchSpace must read special fields from ui_schema."""

    def test_search_space_reads_special_fields(self) -> None:
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "special_search_space_fields" in content, (
            "SearchSpace should read special_search_space_fields from uiSchema"
        )

    def test_search_space_keeps_key_fallback(self) -> None:
        """SearchSpace should fall back to key-based matching when contract lacks special fields."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        # The fallback block should still reference entry.key for backward compat
        assert 'entry.key === "objective"' in content or "entry.key === 'objective'" in content, (
            "SearchSpace should have key-based fallback for objective"
        )
