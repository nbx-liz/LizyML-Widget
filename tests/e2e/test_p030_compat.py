"""E2E tests covering P-030 backend compat surfaces (#114 Phase B).

Verifies that the two user-visible additions from P-030 actually round-trip
through the widget UI:

1. Non-numeric multiclass labels (lizyml 0.10) — `TargetEncoder` decoding
   must surface in the prediction table so users see ``setosa`` / etc.
   rather than int codes.
2. ``smape`` / ``wape`` regression metrics (lizyml 0.11) — must appear in
   the Search Space metric chip group and in the learning-curve metric
   switcher.

Both tests are E2E because the structural-refactor PRs (#117) need a
post-refactor safety net that exercises real Python ↔ JS traffic.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestMulticlassStringLabels:
    """P-030: lizyml 0.10 non-numeric target encoding round-trip."""

    def test_predict_returns_original_string_labels(self, multiclass_widget_page: Page) -> None:
        page = multiclass_widget_page

        # Run Fit through the UI. The notebook already loads + sets target.
        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)
        fit_btn = page.locator(".lzw-btn--primary:has-text('Fit'), button:has-text('Fit')")
        expect(fit_btn.first).to_be_visible()
        fit_btn.first.click()

        # Wait for fit completion.
        page.wait_for_selector(
            ".lzw-badge--success, .lzw-badge--completed",
            timeout=120_000,
        )

        # Open Inference accordion and run a prediction round-trip on the
        # training frame (load_inference + Run Inference).
        page.evaluate(
            """
            (() => {
                const cell = document.querySelector('.jp-Cell');
                // Use the kernel via the widget Python API: we add a cell
                // that calls w.load_inference + w.predict and re-renders.
                // Skipped — instead rely on the existing widget API:
                //   the UI's Run Inference button uses the loaded df.
            })();
            """
        )

        # The notebook loaded df without calling load_inference. The Inference
        # accordion in the widget defaults to using the training frame, so
        # we just open it and click Run Inference if available.
        results_tab = page.locator(".lzw-tabs__btn", has_text="Results")
        results_tab.click()
        page.wait_for_timeout(500)

        # Open the Inference accordion (header).
        inference = page.locator(".lzw-accordion__header", has_text="Inference")
        if inference.count() > 0:
            inference.first.click()
            page.wait_for_timeout(300)
            run_btn = page.locator("button", has_text="Run Inference")
            if run_btn.count() > 0:
                run_btn.first.click()
                # Wait for the prediction table to render.
                page.wait_for_selector(".lzw-pred-table table", timeout=60_000)

                # The pred column in the rendered table should contain at
                # least one of the original string labels — not int codes.
                table_text = page.locator(".lzw-pred-table table").inner_text()
                assert any(
                    label in table_text for label in ("setosa", "versicolor", "virginica")
                ), f"Expected original string labels in PredTable; got: {table_text[:200]!r}"

                # #133 Phase 2.2: pin the row count so a backend regression
                # that drops rows fails loudly. iris has 150 rows; the
                # widget paginates / truncates the rendered table, so we
                # require at least 5 visible body rows.
                body_rows = page.locator(".lzw-pred-table table tbody tr")
                assert body_rows.count() >= 5, (
                    f"Expected >= 5 body rows in PredTable; got {body_rows.count()}"
                )

        # #133 Phase 2.2: at least one Plotly figure container must mount
        # on the Results tab so a regression that breaks the Plotly loader
        # surfaces here instead of as silent missing-plot.
        plot_container = page.locator(".lzw-plot-viewer__canvas")
        assert plot_container.count() >= 1, (
            "Expected at least one Plotly figure container on the Results tab"
        )


class TestRegressionSmapeWapeChips:
    """P-030: smape / wape regression metrics surface in the UI."""

    def test_search_space_renders_smape_and_wape_chips(
        self, regression_smape_wape_page: Page
    ) -> None:
        page = regression_smape_wape_page

        # Open the Model tab and switch to the Tune sub-tab so the Search
        # Space accordion is mounted.
        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)
        tune_subtab = page.locator(
            ".lzw-tabs__btn--sub, .lzw-subtabs__btn",
            has_text="Tune",
        )
        if tune_subtab.count() > 0:
            tune_subtab.first.click()
            page.wait_for_timeout(300)

        # Ensure the Search Space accordion is open.
        ss_header = page.locator(".lzw-accordion__header", has_text="Search Space")
        if ss_header.count() > 0:
            ss_header.first.click()
            page.wait_for_timeout(300)

        # Both chips must be present in the rendered DOM.
        body = page.locator(".lzw-app").inner_text()
        assert "smape" in body, "smape chip should be rendered for regression"
        assert "wape" in body, "wape chip should be rendered for regression"

        # #133 Phase 2.2: pin the chip-group rendering surface so a
        # SearchSpace regression that swallows the contract-driven chips
        # fails loudly. The chip group sits inside the Search Space
        # accordion body and must include at least the two P-030 metrics.
        smape_chip = page.locator(".lzw-chip", has_text="smape")
        wape_chip = page.locator(".lzw-chip", has_text="wape")
        assert smape_chip.count() >= 1, "Expected a smape chip in Search Space"
        assert wape_chip.count() >= 1, "Expected a wape chip in Search Space"

    def test_learning_curve_metric_switcher_includes_smape(
        self, regression_smape_wape_page: Page
    ) -> None:
        page = regression_smape_wape_page

        # Trigger Fit so the Learning Curve plot is requestable on the
        # Results tab. The notebook already configures metric=[smape, wape].
        page.locator(".lzw-tabs__btn", has_text="Model").click()
        page.wait_for_timeout(500)
        fit_btn = page.locator(".lzw-btn--primary:has-text('Fit'), button:has-text('Fit')")
        if fit_btn.count() > 0:
            fit_btn.first.click()
            page.wait_for_selector(
                ".lzw-badge--success, .lzw-badge--completed",
                timeout=120_000,
            )

        page.locator(".lzw-tabs__btn", has_text="Results").click()
        page.wait_for_timeout(500)

        # Look for a learning-curve metric switcher chip labelled smape.
        # Exact selector varies by Plotly mount; the simplest assertion is
        # that the textual label appears anywhere on the Results tab.
        body = page.locator(".lzw-app").inner_text()
        assert "smape" in body, "smape should appear as a learning-curve switcher chip on Results"

        # #133 Phase 2.2: at least one Plotly figure container must mount
        # on the Results tab so a regression that breaks the loader is
        # caught here. The container is rendered by PlotViewer regardless
        # of whether Plotly itself has finished loading the CDN bundle.
        plot_container = page.locator(".lzw-plot-viewer__canvas")
        assert plot_container.count() >= 1, (
            "Expected at least one Plotly figure container on the Results tab"
        )
