"""E2E visual regression tests — screenshot comparison via Playwright."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestLightMode:
    """Visual regression snapshots in the default (light) theme."""

    def test_data_tab(self, widget_page: Page) -> None:
        page = widget_page
        widget = page.locator(".lzw-app")
        expect(widget).to_have_screenshot("data-tab-light.png", threshold=0.1)

    def test_model_tab(self, widget_page: Page) -> None:
        page = widget_page
        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)
        widget = page.locator(".lzw-app")
        expect(widget).to_have_screenshot("model-tab-light.png", threshold=0.1)


class TestDarkMode:
    """Visual regression snapshots in dark theme."""

    def test_data_tab_dark(self, widget_page: Page) -> None:
        page = widget_page

        # Toggle dark mode
        theme_toggle = page.locator(
            ".lzw-theme-toggle, button[aria-label*='theme'], button[aria-label*='dark']"
        )
        if theme_toggle.count() > 0:
            theme_toggle.first.click()
            page.wait_for_timeout(300)

        widget = page.locator(".lzw-app")
        expect(widget).to_have_screenshot("data-tab-dark.png", threshold=0.1)

    def test_model_tab_dark(self, widget_page: Page) -> None:
        page = widget_page

        # Toggle dark mode
        theme_toggle = page.locator(
            ".lzw-theme-toggle, button[aria-label*='theme'], button[aria-label*='dark']"
        )
        if theme_toggle.count() > 0:
            theme_toggle.first.click()
            page.wait_for_timeout(300)

        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)
        widget = page.locator(".lzw-app")
        expect(widget).to_have_screenshot("model-tab-dark.png", threshold=0.1)
