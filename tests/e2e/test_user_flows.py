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
