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
