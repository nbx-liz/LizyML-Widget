"""E2E tests — complete user flows through the widget."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestFitFlow:
    """Verify the complete Fit workflow from Data → Model → Results."""

    def test_complete_fit_flow(self, widget_page: Page) -> None:
        page = widget_page

        # Navigate to Model tab
        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)

        # Click Fit button
        fit_btn = page.locator(".lzw-btn--primary:has-text('Fit'), button:has-text('Fit')")
        expect(fit_btn.first).to_be_visible()
        fit_btn.first.click()

        # Wait for completion — the widget should auto-switch to Results tab
        # and display a success badge.  Training may take up to 120 s.
        page.wait_for_selector(
            ".lzw-badge--success, .lzw-badge--completed",
            timeout=120_000,
        )

        # Verify a results table is present
        results_table = page.locator(".lzw-table, table")
        assert results_table.count() > 0, "Expected a results table after Fit"


class TestTabNavigation:
    """Verify tab switching works correctly."""

    def test_switch_all_tabs(self, widget_page: Page) -> None:
        page = widget_page

        tab_names = ["Data", "Model", "Results"]
        for name in tab_names:
            tab = page.locator(".lzw-tabs__btn", has_text=name)
            if tab.count() > 0:
                tab.click()
                page.wait_for_timeout(300)
                expect(tab).to_have_attribute("aria-selected", "true")


class TestDataTabInteractions:
    """Verify Data tab interactions work correctly."""

    def test_data_preview_visible(self, widget_page: Page) -> None:
        page = widget_page

        # Data tab should show a preview of the loaded dataframe
        data_tab = page.locator(".lzw-tabs__btn", has_text="Data")
        data_tab.click()
        page.wait_for_timeout(300)

        # Check for data preview elements (table, stats, or summary)
        data_content = page.locator(
            ".lzw-data-preview, .lzw-data-summary, .lzw-table, .lzw-stats, .lzw-data-tab"
        )
        expect(data_content.first).to_be_visible()


class TestTuneApplyRetuneFlow:
    """#114 Phase B: Tune → Apply to Fit → Re-tune (resume) happy path."""

    def test_tune_apply_then_retune_renders_boundary_panel(self, widget_page: Page) -> None:
        page = widget_page

        # Switch to the Model tab and pick the Tune sub-tab.
        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)
        tune_subtab = page.locator(
            ".lzw-tabs__btn--sub, .lzw-subtabs__btn",
            has_text="Tune",
        )
        if tune_subtab.count() > 0:
            tune_subtab.first.click()
            page.wait_for_timeout(300)

        # Trigger Tune — the notebook ships with default tuning config so
        # this exercises the search-space → optuna → fit pipeline.
        tune_btn = page.locator(".lzw-btn--primary:has-text('Tune'), button:has-text('Tune')")
        if tune_btn.count() == 0:
            pytest.skip("Tune button not visible — backend may not expose tune")
        tune_btn.first.click()

        # Tune completion: the widget auto-switches to Results.
        page.wait_for_selector(
            ".lzw-badge--success, .lzw-badge--completed",
            timeout=180_000,
        )

        # Best Params accordion + Apply to Fit button must be present.
        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)
        best_params = page.locator(".lzw-accordion__header", has_text="Best Params")
        if best_params.count() > 0:
            best_params.first.click()
            page.wait_for_timeout(200)

        apply_btn = page.locator("button", has_text="Apply to Fit")
        if apply_btn.count() == 0:
            pytest.skip("Apply to Fit button not visible after Tune")
        apply_btn.first.click()
        page.wait_for_timeout(500)

        # The widget should switch to the Model tab so the user can fit
        # with the applied params.
        active_tab_text = page.locator(".lzw-tabs__btn--active").inner_text()
        assert "Model" in active_tab_text, (
            f"Expected switch to Model tab after Apply to Fit; got {active_tab_text!r}"
        )

        # Now exercise Re-tune (resume): switch back to Results, find the
        # Re-tune button, click, and assert the Boundary Expansion panel
        # renders after the resume completes.
        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)
        retune_btn = page.locator("button", has_text="Re-tune")
        if retune_btn.count() == 0:
            pytest.skip("Re-tune button not visible — feature may be hidden")
        retune_btn.first.click()
        page.wait_for_selector(
            ".lzw-badge--success, .lzw-badge--completed",
            timeout=180_000,
        )

        # Boundary Expansion panel should render after re-tune. The
        # selector matches either a dedicated panel class or the
        # accordion header, depending on rendering mode.
        boundary_panel = page.locator(
            ".lzw-boundary-expansion, .lzw-accordion__header:has-text('Boundary Expansion')"
        )
        assert boundary_panel.count() > 0, "Expected the Boundary Expansion panel after Re-tune"
