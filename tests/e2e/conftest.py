"""E2E test fixtures — Jupyter server lifecycle and widget page setup."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import requests
from playwright.sync_api import Page


@pytest.fixture(scope="session")
def jupyter_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[dict[str, str | int], None, None]:
    """Start a Jupyter Lab server for E2E tests.

    The server runs in the background and is terminated when the test session
    ends.  A health-check loop waits up to 30 s for the ``/api/status``
    endpoint to become responsive.
    """
    port = 18888
    token = "test-token-e2e"
    notebook_dir = str(Path(__file__).parent)

    # Use .venv/bin/jupyter directly to avoid uv sync overwriting
    # editable-installed packages (e.g. lizyml dev version)
    venv_jupyter = str(Path(__file__).parents[2] / ".venv" / "bin" / "jupyter")
    proc = subprocess.Popen(
        [
            venv_jupyter,
            "lab",
            f"--port={port}",
            f"--IdentityProvider.token={token}",
            "--no-browser",
            f"--notebook-dir={notebook_dir}",
            "--ServerApp.disable_check_xsrf=True",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Jupyter to become ready
    url = f"http://localhost:{port}"
    ready = False
    for _ in range(30):
        try:
            r = requests.get(f"{url}/api/status", params={"token": token}, timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)

    if not ready:
        proc.terminate()
        proc.wait(timeout=10)
        pytest.fail("Jupyter server did not start within 30 seconds")

    yield {"url": url, "token": token, "port": port}

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture()
def widget_page(jupyter_server: dict[str, str | int], page: Page) -> Page:
    """Open the test notebook, execute all cells, and wait for the widget.

    Returns a Playwright ``Page`` with the LizyML widget rendered.
    """
    base = jupyter_server["url"]
    token = jupyter_server["token"]

    # Navigate to the test notebook (JupyterLab URL format)
    page.goto(f"{base}/lab/tree/test_widget.ipynb?token={token}")

    # Wait for the JupyterLab notebook UI to load
    page.wait_for_selector(".jp-Notebook", timeout=30_000)
    page.wait_for_timeout(3_000)  # Wait for kernel connection

    # Run all cells via JupyterLab menu
    page.locator(".lm-MenuBar-itemLabel", has_text="Run").click()
    page.wait_for_timeout(500)
    page.get_by_text("Run All Cells", exact=True).click()

    # Wait for the widget root to appear
    page.wait_for_selector(".lzw-app", timeout=60_000)

    return page


@pytest.fixture(scope="session")
def learning_curve_page(
    jupyter_server: dict[str, str | int], browser: Any
) -> Generator[Page, None, None]:
    """Open the learning curve test notebook, execute all cells, wait for widget.

    The notebook runs fit() with 3 custom metrics (auc, binary_logloss,
    binary_error), so the Results tab shows a metric selector for Learning
    Curve plots.  Session-scoped so the expensive fit runs only once.
    """
    base = jupyter_server["url"]
    token = jupyter_server["token"]

    page = browser.new_page()
    page.goto(f"{base}/lab/tree/test_learning_curve.ipynb?token={token}")
    page.wait_for_selector(".jp-Notebook", timeout=30_000)
    page.wait_for_timeout(3_000)  # Wait for kernel connection

    # Run all cells via JupyterLab menu
    page.locator(".lm-MenuBar-itemLabel", has_text="Run").click()
    page.wait_for_timeout(500)
    page.get_by_text("Run All Cells", exact=True).click()

    # Wait for widget + fit completion (status badge appears)
    page.wait_for_selector(".lzw-app", timeout=60_000)
    page.wait_for_selector(
        ".lzw-badge--success, .lzw-badge--completed",
        timeout=120_000,
    )

    yield page
    page.close()


def _open_widget_notebook(base: str, token: str, page: Page, notebook: str) -> Page:
    """Open a notebook, run all cells, wait for the widget root."""
    page.goto(f"{base}/lab/tree/{notebook}?token={token}")
    page.wait_for_selector(".jp-Notebook", timeout=30_000)
    page.wait_for_timeout(3_000)  # Wait for kernel connection

    page.locator(".lm-MenuBar-itemLabel", has_text="Run").click()
    page.wait_for_timeout(500)
    page.get_by_text("Run All Cells", exact=True).click()

    page.wait_for_selector(".lzw-app", timeout=60_000)
    return page


@pytest.fixture()
def multiclass_widget_page(jupyter_server: dict[str, str | int], page: Page) -> Page:
    """#114 Phase B: open the multiclass-with-string-labels notebook.

    Used by `test_p030_compat.py` to verify lizyml 0.10's TargetEncoder
    decoding round-trips through the widget so prediction labels render
    as the original strings (e.g. ``setosa`` / ``versicolor`` /
    ``virginica``) rather than int codes.
    """
    return _open_widget_notebook(
        str(jupyter_server["url"]),
        str(jupyter_server["token"]),
        page,
        "test_multiclass_strings.ipynb",
    )


@pytest.fixture()
def regression_smape_wape_page(jupyter_server: dict[str, str | int], page: Page) -> Page:
    """#114 Phase B: open the regression notebook configured with smape/wape.

    Used by `test_p030_compat.py` to verify P-030's smape / wape regression
    metrics surface in the Search Space chip group and learning-curve
    switcher.
    """
    return _open_widget_notebook(
        str(jupyter_server["url"]),
        str(jupyter_server["token"]),
        page,
        "test_regression_smape_wape.ipynb",
    )


@pytest.fixture()
def failed_state_page(jupyter_server: dict[str, str | int], page: Page) -> Page:
    """#133 Phase 2.2: open a notebook that puts the widget into a failed state.

    The notebook writes ``w.status = "failed"`` plus a synthetic ``w.error``
    so the failed-state UI (error banner + Re-run button) renders without
    depending on a brittle backend rejection. The supervisor takes the same
    write path on real failures, so the surface under test is identical.
    """
    return _open_widget_notebook(
        str(jupyter_server["url"]),
        str(jupyter_server["token"]),
        page,
        "test_failed_state.ipynb",
    )


@pytest.fixture()
def long_tune_page(jupyter_server: dict[str, str | int], page: Page) -> Page:
    """#133 Phase 2.2: open a notebook that launches a 200-trial tune.

    The first cell builds a widget and large search space. The second cell
    calls ``w.tune()``; that call returns immediately because the widget
    runs the tune on a background thread. The E2E test then clicks the
    Cancel button mid-flight to exercise the running -> cancelled
    transition (INV-D / INV-F).
    """
    return _open_widget_notebook(
        str(jupyter_server["url"]),
        str(jupyter_server["token"]),
        page,
        "test_long_tune.ipynb",
    )
