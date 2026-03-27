"""Static contract tests for frontend layout and UI conventions.

These tests grep the TypeScript/CSS source to prevent regressions
in layout structure, input conventions, and class usage.
"""

from __future__ import annotations

from pathlib import Path

JS_SRC = Path(__file__).resolve().parent.parent / "js" / "src"


def _read_config_files() -> str:
    """Read all Config tab files (coordinator + sub-tabs) as a single string."""
    parts = []
    for name in ("ConfigTab.tsx", "FitSubTab.tsx", "TuneSubTab.tsx"):
        path = JS_SRC / "tabs" / name
        if path.exists():
            parts.append(path.read_text())
    return "\n".join(parts)


class TestLayoutContract:
    """Phase 15: Ensure .lzw-root is applied only once (in index.tsx)."""

    def test_lzw_root_only_in_index(self) -> None:
        index_tsx = (JS_SRC / "index.tsx").read_text()
        assert 'classList.add("lzw-root")' in index_tsx

    def test_app_tsx_no_lzw_root(self) -> None:
        app_tsx = (JS_SRC / "App.tsx").read_text()
        assert "lzw-root" not in app_tsx

    def test_css_no_fixed_height_620(self) -> None:
        css = (JS_SRC / "widget.css").read_text()
        import re

        # Match standalone "height: 620px" but not "min-height: 620px"
        fixed_height = re.findall(r"(?<!min-)height:\s*620px", css)
        assert not fixed_height, f"Found fixed height: {fixed_height}"
        # min-height: 620px is expected
        assert "min-height: 620px" in css


class TestNumericInputContract:
    """Phase 17: All type='number' inputs should be inside NumericStepper only."""

    def test_no_raw_number_inputs_outside_stepper(self) -> None:
        stepper_path = JS_SRC / "components" / "NumericStepper.tsx"
        for tsx_file in JS_SRC.rglob("*.tsx"):
            if tsx_file == stepper_path:
                continue
            # Skip test files — they may contain type="number" in test assertions
            if "__tests__" in tsx_file.parts:
                continue
            content = tsx_file.read_text()
            assert 'type="number"' not in content, (
                f'{tsx_file.name} contains raw type="number" input '
                f"— should use NumericStepper component instead"
            )


class TestChipSelectionContract:
    """Phase 21: Task and CV Strategy should use chip/segment selection, not <select>."""

    def test_data_tab_task_uses_segment(self) -> None:
        data_tab = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        # Task section should use segment buttons, not <select>
        assert "set_task" in data_tab
        # Find the Task label and verify segment button follows
        task_idx = data_tab.index(">Task<")
        task_section = data_tab[task_idx : task_idx + 500]
        assert "lzw-segment" in task_section, "Task should use segment buttons"
        assert "<select" not in task_section, "Task should not use <select>"

    def test_data_tab_cv_strategy_uses_segment(self) -> None:
        data_tab = (JS_SRC / "tabs" / "DataTab.tsx").read_text()
        strategy_idx = data_tab.index(">Strategy<")
        strategy_section = data_tab[strategy_idx : strategy_idx + 500]
        assert "lzw-segment" in strategy_section, "CV Strategy should use segment buttons"
        assert "<select" not in strategy_section, "CV Strategy should not use <select>"


class TestSearchSpaceContract:
    """Phase 22: SearchSpace must convert mode→type format for LizyML."""

    def test_handle_update_outputs_type_based_format(self) -> None:
        """SearchSpace handleUpdate should output type-based format."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        # handleUpdate must produce type: "float"/"int"/"categorical"
        assert '"float"' in content
        assert '"int"' in content
        assert '"categorical"' in content

    def test_handle_update_does_not_pass_mode_to_on_change(self) -> None:
        """handleUpdate should set type, not mode, in the stored object."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        handle_idx = content.index("handleUpdate")
        # P-014: function is larger with group routing; search 1500 chars
        handle_block = content[handle_idx : handle_idx + 1500]
        # The block should reference "type:" for stored values, not "mode:"
        assert "type:" in handle_block or "type: spaceType" in handle_block

    def test_tune_button_checks_type_format(self) -> None:
        """ConfigTab Tune button enablement should check type-based format."""
        content = _read_config_files()
        # hasSearchParam should check p.type values
        assert 'p.type === "float"' in content or "p.type ==" in content


class TestGridLayoutContract:
    """Phase 18: ColumnTable and SearchSpace should use CSS Grid, not <table>."""

    def test_column_table_no_html_table(self) -> None:
        content = (JS_SRC / "components" / "ColumnTable.tsx").read_text()
        assert "<table" not in content, "ColumnTable should use CSS Grid, not <table>"
        assert "lzw-columns-grid" in content

    def test_search_space_no_html_table(self) -> None:
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "<table" not in content, "SearchSpace should use CSS Grid, not <table>"
        assert "lzw-search-space-grid" in content


class TestColumnTypeSegmentContract:
    """Phase 23: Column Type should use segment buttons, not <select>."""

    def test_column_table_type_uses_segment(self) -> None:
        content = (JS_SRC / "components" / "ColumnTable.tsx").read_text()
        assert "lzw-segment" in content, "ColumnTable Type should use segment buttons"
        assert "lzw-table__type-select" not in content, "ColumnTable should not use type <select>"


class TestChipButtonContract:
    """Phase 23: Metric selections should use chip buttons, not checkbox groups."""

    def test_config_tab_metric_uses_chip(self) -> None:
        content = _read_config_files()
        assert "lzw-chip" in content, "Config tab metric should use chip buttons"

    def test_search_space_choice_uses_chip(self) -> None:
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "lzw-chip" in content, "SearchSpace choice should use chip buttons"

    def test_css_has_chip_styles(self) -> None:
        css = (JS_SRC / "widget.css").read_text()
        assert ".lzw-chip " in css or ".lzw-chip{" in css or ".lzw-chip-" in css
        assert ".lzw-chip--active" in css
        assert ".lzw-chip-group" in css


class TestInnerValidDefaultContract:
    """inner_valid should default to holdout, not show 'Default' or 'auto'."""

    def test_inner_valid_defaults_to_holdout(self) -> None:
        content = _read_config_files()
        assert ">Default<" not in content, "inner_valid should not show 'Default' label"
        assert ">auto<" not in content, "inner_valid should not show 'auto'"
        assert '"holdout"' in content, "inner_valid should default to holdout"


class TestStepperWidthContract:
    """Phase 23: NumericStepper input width should be 75px."""

    def test_stepper_width_75px(self) -> None:
        css = (JS_SRC / "widget.css").read_text()
        assert "width: 75px" in css, "Stepper input width should be 75px"
        assert "width: 50px" not in css, "Old 50px width should be removed"


class TestBackendContractMigration:
    """Phase 25: Frontend reads from backend_contract, not hardcoded constants."""

    def test_config_tab_no_hardcoded_objective_options(self) -> None:
        content = _read_config_files()
        assert "OBJECTIVE_OPTIONS" not in content, (
            "Config tab should not have hardcoded OBJECTIVE_OPTIONS"
        )

    def test_config_tab_no_hardcoded_metric_options(self) -> None:
        content = _read_config_files()
        assert "METRIC_OPTIONS" not in content, (
            "Config tab should not have hardcoded METRIC_OPTIONS"
        )

    def test_config_tab_no_hardcoded_step_map(self) -> None:
        content = _read_config_files()
        assert "STEP_MAP" not in content, "Config tab should not have hardcoded STEP_MAP"

    def test_config_tab_no_hardcoded_typed_params(self) -> None:
        content = _read_config_files()
        assert "TYPED_PARAMS" not in content, "Config tab should not have hardcoded TYPED_PARAMS"

    def test_config_tab_no_hardcoded_fit_sections(self) -> None:
        content = _read_config_files()
        assert "FIT_SECTIONS" not in content, "Config tab should not have hardcoded FIT_SECTIONS"

    def test_search_space_no_hardcoded_constants(self) -> None:
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        for const_name in ("OBJECTIVE_OPTIONS", "METRIC_OPTIONS", "STEP_MAP", "LGBM_PARAMS"):
            assert const_name not in content, f"SearchSpace should not have hardcoded {const_name}"

    def test_config_tab_uses_patch_config(self) -> None:
        content = _read_config_files()
        assert "patch_config" in content, "Config tab should use patch_config action"

    def test_app_uses_backend_contract(self) -> None:
        content = (JS_SRC / "App.tsx").read_text()
        assert "backend_contract" in content, "App should read backend_contract traitlet"

    def test_config_tab_receives_backend_contract(self) -> None:
        content = (JS_SRC / "tabs" / "ConfigTab.tsx").read_text()
        assert "backendContract" in content, "ConfigTab should receive backendContract prop"


class TestInnerValidCanonicalContract:
    """Phase 26: inner_valid sends canonical object/null, not bare strings."""

    def test_inner_valid_sends_method_object(self) -> None:
        content = _read_config_files()
        assert "method:" in content or '"method"' in content, (
            "inner_valid onChange should produce {method: ...} object"
        )

    def test_inner_valid_reads_method_field(self) -> None:
        content = _read_config_files()
        assert "inner_valid?.method" in content, (
            "inner_valid display value should read .method from object"
        )

    def test_inner_valid_options_from_ui_schema(self) -> None:
        content = _read_config_files()
        assert "inner_valid_options" in content, (
            "inner_valid options should come from uiSchema.inner_valid_options"
        )

    def test_no_fold_options_generated(self) -> None:
        content = _read_config_files()
        assert "fold_" not in content, (
            "Config tab should not generate fold_N options for inner_valid"
        )


class TestSearchSpaceAuditFixes:
    """Audit fixes: feature_weights toggle, [+ Add] button, conditional visibility."""

    def test_feature_weights_uses_toggle(self) -> None:
        """feature_weights must render lzw-toggle, not text input."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "lzw-toggle" in content, "SearchSpace must use lzw-toggle for feature_weights"

    def test_add_button_exists(self) -> None:
        """SearchSpace must have a [+ Add] select for additional params."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "additional_params" in content or "additionalParamsList" in content, (
            "SearchSpace must reference additional_params"
        )
        assert "+ Add" in content, "SearchSpace must contain '+ Add' label"

    def test_conditional_visibility_referenced(self) -> None:
        """SearchSpace must use conditional_visibility for num_leaves etc."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "conditional_visibility" in content or "conditionalVisibility" in content, (
            "SearchSpace must use conditional_visibility"
        )
        assert "isParamVisible" in content, "SearchSpace must have isParamVisible helper"

    def test_group_wrapper_uses_display_contents(self) -> None:
        """Group wrappers must use display:contents CSS class for proper grid layout."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "lzw-search-space-grid__group-wrap" in content
        css = (JS_SRC / "widget.css").read_text()
        assert "lzw-search-space-grid__group-wrap" in css
        assert "display: contents" in css

    def test_catalog_entry_default_field_used(self) -> None:
        """SearchSpace CatalogEntry interface must include default field."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "default?" in content or "default?:" in content, (
            "CatalogEntry must have optional default field"
        )


class TestSearchSpaceAtomicUpdate:
    """SearchSpace must use a single atomic onChange callback, not separate callbacks."""

    def test_no_separate_callbacks(self) -> None:
        """SearchSpace must NOT have separate onSpaceChange callbacks."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        for old_cb in ("onSpaceChange", "onFixedModelParamsChange", "onFixedTrainingChange"):
            assert old_cb not in content, (
                f"SearchSpace must use single atomic onChange, not {old_cb}"
            )

    def test_has_atomic_on_change(self) -> None:
        """SearchSpace must expose SearchSpaceUpdate type and single onChange prop."""
        content = (JS_SRC / "components" / "SearchSpace.tsx").read_text()
        assert "SearchSpaceUpdate" in content, "SearchSpace must export SearchSpaceUpdate type"
        assert "onChange: (update: SearchSpaceUpdate)" in content, (
            "SearchSpace must accept onChange with SearchSpaceUpdate parameter"
        )

    def test_config_tab_uses_atomic_callback(self) -> None:
        """Config tab must call SearchSpace with single onChange, not separate callbacks."""
        content = _read_config_files()
        assert "onFixedModelParamsChange" not in content, (
            "Config tab must not pass onFixedModelParamsChange to SearchSpace"
        )
        assert "onChange={" in content, "Config tab must pass onChange to SearchSpace"


class TestTuneCapabilitiesContract:
    """Phase 26: Tune button reads allow_empty_space from backend_contract.capabilities."""

    def test_tune_reads_allow_empty_space(self) -> None:
        content = _read_config_files()
        assert "allow_empty_space" in content, (
            "Tune condition should read allow_empty_space from capabilities"
        )


class TestContractNewFields:
    """M-1/M-2: backend contract exposes cv_strategies and calibration_methods."""

    def test_capabilities_returns_cv_strategies(self) -> None:
        """build_capabilities() must return cv_strategies list."""
        from lizyml_widget.adapter_contract import build_capabilities

        caps = build_capabilities()
        strategies = caps["cv_strategies"]
        assert isinstance(strategies, list)
        assert len(strategies) >= 4
        for expected in ("kfold", "stratified_kfold", "group_kfold", "time_series"):
            assert expected in strategies, f"{expected} missing from cv_strategies"

    def test_ui_schema_returns_calibration_methods(self) -> None:
        """build_ui_schema() must return calibration_methods list."""
        from lizyml_widget.adapter_contract import build_ui_schema
        from lizyml_widget.adapter_params import get_eval_metrics_by_task

        schema = build_ui_schema(get_eval_metrics_by_task())
        methods = schema["calibration_methods"]
        assert isinstance(methods, list)
        assert len(methods) >= 2
        for expected in ("platt", "isotonic", "beta"):
            assert expected in methods, f"{expected} missing from calibration_methods"

    def test_widget_update_cv_reads_strategies_from_contract(self) -> None:
        """Widget._handle_update_cv should accept strategies from backend_contract."""
        from unittest.mock import patch

        from lizyml_widget.adapter import LizyMLAdapter
        from lizyml_widget.adapter_contract import build_capabilities

        real_adapter = LizyMLAdapter()
        with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
            adapter = MockAdapter.return_value
            adapter.info = real_adapter.info
            adapter.get_config_schema.return_value = {}
            adapter.initialize_config.side_effect = real_adapter.initialize_config
            adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
            adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
            adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
            adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
            adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults

            from lizyml_widget.widget import LizyWidget

            w = LizyWidget()

        import pandas as pd

        w.load(pd.DataFrame({"x": range(50), "y": [0, 1] * 25}), target="y")

        # Verify backend_contract has cv_strategies
        caps = build_capabilities()
        contract_strategies = caps["cv_strategies"]
        assert len(contract_strategies) > 0

        # All contract strategies should be accepted by _handle_update_cv
        for strategy in contract_strategies:
            w.action = {"op": "update_cv", "strategy": strategy, "n_splits": 3}
            assert w.error == {} or w.error.get("code") != "CV_ERROR", (
                f"Strategy {strategy!r} from contract was rejected"
            )
