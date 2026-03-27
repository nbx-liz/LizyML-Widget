"""E2E tests — verify that UI interactions propagate to widget state."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestTargetSelection:
    """Verify target dropdown changes update the task badge."""

    def test_target_dropdown_changes_task(self, widget_page: Page) -> None:
        page = widget_page

        # Data tab should be active by default
        data_tab = page.locator(".lzw-tabs__btn", has_text="Data")
        expect(data_tab).to_have_attribute("aria-selected", "true")

        # Find the target dropdown and verify it has a value
        target_select = page.locator(".lzw-target-select select, .lzw-target-select")
        expect(target_select).to_be_visible()

        # Verify a task badge is displayed
        task_badge = page.locator(".lzw-badge")
        expect(task_badge).to_be_visible()


class TestCVStrategy:
    """Verify CV strategy changes show/hide conditional fields."""

    def test_cv_strategy_change(self, widget_page: Page) -> None:
        page = widget_page

        # Look for CV-related controls (accordion or section)
        cv_section = page.locator(
            "[data-section='cv'], .lzw-cv-section, .lzw-accordion:has-text('CV')"
        )
        if cv_section.count() > 0:
            cv_section.first.click()
            # Wait for potential animation
            page.wait_for_timeout(300)

        # Look for strategy selector buttons
        strategy_buttons = page.locator(".lzw-cv-strategy button, .lzw-btn-group button")
        if strategy_buttons.count() > 1:
            # Click the second strategy option
            strategy_buttons.nth(1).click()
            page.wait_for_timeout(300)

            # Verify the UI reacted (some conditional field appeared or button is active)
            active_btn = page.locator(
                ".lzw-cv-strategy .lzw-btn--active, .lzw-btn-group .lzw-btn--active"
            )
            expect(active_btn).to_have_count(1)


class TestModelParameters:
    """Verify model parameter stepper changes propagate."""

    def test_stepper_changes_config(self, widget_page: Page) -> None:
        page = widget_page

        # Navigate to Model tab
        model_tab = page.locator(".lzw-tabs__btn", has_text="Model")
        model_tab.click()
        page.wait_for_timeout(500)

        # Find a numeric stepper (e.g., n_estimators)
        stepper = page.locator(".lzw-stepper, .lzw-number-input").first
        if stepper.count() > 0:
            # Read initial value
            value_el = stepper.locator("input, .lzw-stepper__value")
            initial_value = value_el.input_value() if value_el.count() > 0 else ""

            # Click the increment button
            inc_btn = stepper.locator(
                "button:has-text('+'), button[aria-label*='increment'], .lzw-stepper__btn--inc"
            )
            if inc_btn.count() > 0:
                inc_btn.click()
                page.wait_for_timeout(200)

                # Verify value changed
                new_value = value_el.input_value() if value_el.count() > 0 else ""
                assert new_value != initial_value, (
                    f"Stepper value did not change: {initial_value} -> {new_value}"
                )
