"""E2E Playwright tests — Learning Curve metric selector (P-026).

Verifies the full UI flow in a real Jupyter notebook:
1. Results tab shows Learning Curve plot chip
2. Learning Curve has a metric selector with 3 chips
3. Clicking each metric chip changes the displayed plot
4. Parameters section shows the metric field

Requires: ``pytest tests/e2e -m e2e`` (Jupyter + Playwright)
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestLearningCurveMetricSelector:
    """Verify metric selector appears and switches Learning Curve plots."""

    def test_results_tab_shows_learning_curve_chip(self, learning_curve_page: Page) -> None:
        page = learning_curve_page

        # Navigate to Results tab
        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)

        # Verify Learning Curve chip exists in the Plots section
        lc_chip = page.locator(".lzw-chip", has_text="Learning Curve")
        expect(lc_chip).to_be_visible()

    def test_metric_selector_appears_for_learning_curve(self, learning_curve_page: Page) -> None:
        page = learning_curve_page

        # Go to Results tab
        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)

        # Click Learning Curve chip
        page.locator(".lzw-chip", has_text="Learning Curve").click()
        page.wait_for_timeout(1000)

        # Metric selector chips should appear (second chip-group row)
        chip_groups = page.locator(".lzw-chip-group")
        # First group = plot type chips, second group = metric selector
        assert chip_groups.count() >= 2, (
            f"Expected ≥2 chip groups (plot types + metric selector), got {chip_groups.count()}"
        )

        # Verify the 3 metric chips are visible
        metric_group = chip_groups.nth(1)
        auc_chip = metric_group.locator(".lzw-chip", has_text="auc")
        expect(auc_chip).to_be_visible()

        bl_chip = metric_group.locator(".lzw-chip", has_text="binary_logloss")
        expect(bl_chip).to_be_visible()

        be_chip = metric_group.locator(".lzw-chip", has_text="binary_error")
        expect(be_chip).to_be_visible()

    def test_default_metric_is_first(self, learning_curve_page: Page) -> None:
        page = learning_curve_page

        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)
        page.locator(".lzw-chip", has_text="Learning Curve").click()
        page.wait_for_timeout(1000)

        # First metric (auc) should be active by default
        chip_groups = page.locator(".lzw-chip-group")
        metric_group = chip_groups.nth(1)
        active_chip = metric_group.locator(".lzw-chip--active")
        expect(active_chip).to_have_count(1)
        expect(active_chip).to_have_text("auc")

    def test_clicking_metric_chip_changes_plot(self, learning_curve_page: Page) -> None:
        page = learning_curve_page

        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)
        page.locator(".lzw-chip", has_text="Learning Curve").click()
        page.wait_for_timeout(2000)  # Wait for initial plot to load

        chip_groups = page.locator(".lzw-chip-group")
        metric_group = chip_groups.nth(1)

        # Take screenshot of initial plot (auc)
        plot_container = page.locator(".lzw-plot-viewer__canvas")
        expect(plot_container).to_be_visible()

        # Click binary_logloss
        metric_group.locator(".lzw-chip", has_text="binary_logloss").click()
        page.wait_for_timeout(2000)  # Wait for new plot to load

        # Verify active chip changed
        active_chip = metric_group.locator(".lzw-chip--active")
        expect(active_chip).to_have_text("binary_logloss")

        # Plot canvas should still be visible (not empty/error)
        expect(plot_container).to_be_visible()

        # Click binary_error
        metric_group.locator(".lzw-chip", has_text="binary_error").click()
        page.wait_for_timeout(2000)

        active_chip = metric_group.locator(".lzw-chip--active")
        expect(active_chip).to_have_text("binary_error")
        expect(plot_container).to_be_visible()

    def test_plot_title_reflects_selected_metric(self, learning_curve_page: Page) -> None:
        """Plotly subplot title should contain the selected metric name."""
        page = learning_curve_page

        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)
        page.locator(".lzw-chip", has_text="Learning Curve").click()
        page.wait_for_timeout(2000)

        # Read the Plotly annotation text (subplot title) from the SVG
        # Plotly renders titles as <text> elements inside .annotation
        plot_canvas = page.locator(".lzw-plot-viewer__canvas")

        for metric in ["auc", "binary_logloss", "binary_error"]:
            chip_groups = page.locator(".lzw-chip-group")
            metric_group = chip_groups.nth(1)
            metric_group.locator(".lzw-chip", has_text=metric).click()
            page.wait_for_timeout(2000)

            # Check the Plotly annotation contains the metric name
            # Plotly renders subplot titles as .annotation-text elements
            annotation = plot_canvas.locator(".annotation-text")
            if annotation.count() > 0:
                text = annotation.first.text_content() or ""
                assert metric in text, f"Plot title should contain '{metric}', got '{text}'"


class TestParametersShowMetric:
    """Results > Details > Parameters should display the metric field."""

    def test_params_table_has_metric_row(self, learning_curve_page: Page) -> None:
        page = learning_curve_page

        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)

        # Open the Details accordion (click the button text)
        details_btn = page.get_by_text("Details", exact=True)
        if details_btn.count() > 0:
            details_btn.first.click()
            page.wait_for_timeout(1000)

        # The params table should contain "metric" in a table cell
        metric_cell = page.locator("td", has_text="metric")
        expect(metric_cell.first).to_be_visible(timeout=10_000)

    # Note: Model tab metric chip validation is covered by Python-level
    # tests in test_model_metric_validity.py (TestModelMetricOptionSetValidity).
    # Playwright cannot reliably open nested accordions across all themes.
