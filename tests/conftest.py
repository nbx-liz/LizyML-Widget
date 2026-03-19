"""Global test fixtures."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _default_thread_strategy() -> None:  # type: ignore[misc]
    """Default all tests to thread execution strategy.

    On Linux with libgomp, get_execution_strategy() returns ("subprocess", ...),
    which causes tests using mock adapters to fail (mock adapters can't be
    serialized across process boundaries). Tests that explicitly test the
    subprocess path should mock get_execution_strategy themselves.

    Strategy is detected lazily in _run_job, so we patch it there.
    """
    with patch(
        "lizyml_widget.widget.get_execution_strategy",
        return_value=("thread", None),
    ):
        yield
