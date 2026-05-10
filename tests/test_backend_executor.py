"""Unit tests for ``BackendExecutor`` (P-039 Phase 3).

The executor is the caller-thread chokepoint for ML library calls.
Phase 3's behavioral guarantee is narrow: ``run_ml`` invokes the
provided callable, returns its result (or re-raises), and marks the
service's libgomp owner state to ``"main"`` for kinds that bind
libgomp.

Phase 4 (lint rule) will enforce that no ML library call site exists
outside the executor module — these tests pin the executor's own
contract so that enforcement has a stable target.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from lizyml_widget.backend_executor import BackendExecutor


def _make_executor() -> tuple[BackendExecutor, MagicMock]:
    svc = MagicMock()
    svc._libgomp_pool_owner = "unknown"

    def _mark(owner: str) -> None:
        if svc._libgomp_pool_owner == "main" and owner != "main":
            return
        svc._libgomp_pool_owner = owner

    svc.mark_libgomp_owner.side_effect = _mark
    return BackendExecutor(svc), svc


# ---------------------------------------------------------------------------
# Return value / exception transparency
# ---------------------------------------------------------------------------


class TestRunMlBehavior:
    def test_returns_callable_result(self) -> None:
        ex, _ = _make_executor()
        assert ex.run_ml(lambda: 42, ml_kind="plot_other") == 42

    def test_runs_callable_exactly_once(self) -> None:
        ex, _ = _make_executor()
        op = MagicMock(return_value="ok")
        ex.run_ml(op, ml_kind="predict")
        assert op.call_count == 1

    def test_propagates_callable_exception(self) -> None:
        ex, _ = _make_executor()

        def boom() -> None:
            raise RuntimeError("op failed")

        with pytest.raises(RuntimeError, match="op failed"):
            ex.run_ml(boom, ml_kind="predict")


# ---------------------------------------------------------------------------
# Owner-state marking — only kinds that bind libgomp transition to "main"
# ---------------------------------------------------------------------------


class TestRunMlOwnerMarking:
    @pytest.mark.parametrize("kind", ["predict", "explain", "plot_shap"])
    def test_libgomp_binding_kinds_mark_main(self, kind: str) -> None:
        ex, svc = _make_executor()
        ex.run_ml(lambda: None, ml_kind=kind)  # type: ignore[arg-type]
        svc.mark_libgomp_owner.assert_called_once_with("main")

    def test_plot_other_does_not_mark_main(self) -> None:
        ex, svc = _make_executor()
        ex.run_ml(lambda: None, ml_kind="plot_other")
        svc.mark_libgomp_owner.assert_not_called()

    def test_marks_main_even_when_callable_raises(self) -> None:
        """LightGBM may have already entered the parallel region by the
        time the exception unwinds — defensively mark "main" so the
        next worker-thread Tune/Fit re-routes regardless."""
        ex, svc = _make_executor()

        def boom() -> None:
            raise RuntimeError("op failed")

        with pytest.raises(RuntimeError):
            ex.run_ml(boom, ml_kind="predict")

        svc.mark_libgomp_owner.assert_called_once_with("main")


# ---------------------------------------------------------------------------
# Concurrency — multiple threads each get their own state transition.
# ---------------------------------------------------------------------------


class TestRunMlThreadSafety:
    def test_concurrent_run_ml_calls_all_record_main(self) -> None:
        ex, svc = _make_executor()

        def work() -> int:
            ex.run_ml(lambda: 1, ml_kind="predict")
            return 1

        threads = [threading.Thread(target=work) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every thread should have hit mark_libgomp_owner("main").
        assert svc.mark_libgomp_owner.call_count == 20
        for call in svc.mark_libgomp_owner.call_args_list:
            assert call.args == ("main",)
