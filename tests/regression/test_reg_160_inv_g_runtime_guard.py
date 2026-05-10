"""Regression tests for P-039 Phase 2 / INV-G — runtime libgomp owner guard.

Phase 1 (PR #162) added a CI gate that detects libgomp pool-affinity
catastrophes on the cross-product of historical regressions. Phase 2
adds a *runtime* guard so the catastrophe is averted in production
without waiting for a CI run on a future PR.

INV-G (BLUEPRINT.md §6.4): when ``WidgetService._libgomp_pool_owner``
is ``"main"``, ``ThreadJobRunner`` must NOT be used to launch
Tune/Fit/Retune. The widget guard re-routes to ``SubprocessJobRunner``
unless ``LZW_FORCE_THREAD=1`` is set, in which case the user opt-out
is honored with a WARN-level log.

State transitions encoded:

- ``__init__``                                   → ``"unknown"``
- ``SubprocessJobRunner.run`` success            → ``"subprocess"``
- ``ThreadJobRunner.run`` success                → ``"worker"``
- ``service.predict(...)``                       → ``"main"``
- ``service.get_plot("feature-importance-shap")`` → ``"main"``
- ``service.get_inference_plot(..., "shap-summary")`` → ``"main"``
- ``"main"`` is sticky in-process (load_data does NOT reset)

These tests pin the state-machine itself plus the widget guard's
re-route behaviour. They are fast (mocks) so they run in the default
pytest tier. The slow perf-grid in
``test_reg_160_libgomp_perf_grid.py`` covers end-to-end timing.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo

# ---------------------------------------------------------------------------
# Helpers — mirror test_reg_154 / test_reg_156 harness so canonical config
# flows but ML library calls are intercepted.
# ---------------------------------------------------------------------------


def _make_widget() -> Any:
    real_adapter = LizyMLAdapter()
    with patch("lizyml_widget.widget.LizyMLAdapter") as MockAdapter:
        adapter = MockAdapter.return_value
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.get_config_schema.return_value = {"type": "object"}
        adapter.validate_config.return_value = []
        adapter.initialize_config.side_effect = real_adapter.initialize_config
        adapter.apply_config_patch.side_effect = real_adapter.apply_config_patch
        adapter.prepare_run_config.side_effect = real_adapter.prepare_run_config
        adapter.get_backend_contract.side_effect = real_adapter.get_backend_contract
        adapter.canonicalize_config.side_effect = real_adapter.canonicalize_config
        adapter.apply_task_defaults.side_effect = real_adapter.apply_task_defaults
        adapter.classify_best_params.side_effect = real_adapter.classify_best_params

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
    df = pd.DataFrame(
        {
            "f1": list(range(40)),
            "f2": [i * 0.5 for i in range(40)],
            "f3": [i % 5 for i in range(40)],
            "y": [0, 1] * 20,
        }
    )
    w.load(df, target="y")
    return w


def _stub_runner_result() -> Any:
    from lizyml_widget.subprocess_runner import SubprocessJobResult

    return SubprocessJobResult(
        job_type="fit",
        fit_summary={"metrics": {}, "fold_count": 0, "fold_details": [], "params": []},
        tune_summary={},
        eval_table=[],
        split_summary=[],
        available_plots=[],
        model_path=None,
        tune_state_path=None,
    )


# ---------------------------------------------------------------------------
# State machine — Service.mark_libgomp_owner
# ---------------------------------------------------------------------------


class TestLibgompOwnerStateMachine:
    """Service-side state transitions for INV-G."""

    def _make_service(self) -> Any:
        from lizyml_widget.service import WidgetService

        return WidgetService(adapter=LizyMLAdapter())

    def test_default_state_is_unknown(self) -> None:
        svc = self._make_service()
        assert svc._libgomp_pool_owner == "unknown"

    def test_mark_owner_sets_state(self) -> None:
        svc = self._make_service()
        svc.mark_libgomp_owner("subprocess")
        assert svc._libgomp_pool_owner == "subprocess"
        svc.mark_libgomp_owner("worker")
        assert svc._libgomp_pool_owner == "worker"

    def test_main_is_sticky_against_subprocess_downgrade(self) -> None:
        """Once main thread bound libgomp, the bind does not unbind in
        this process — subsequent subprocess success must NOT downgrade
        the state to "subprocess" (which would re-enable thread runner
        for the *next* job and re-trigger catastrophe).
        """
        svc = self._make_service()
        svc.mark_libgomp_owner("main")
        svc.mark_libgomp_owner("subprocess")
        assert svc._libgomp_pool_owner == "main"

    def test_main_is_sticky_against_worker_downgrade(self) -> None:
        svc = self._make_service()
        svc.mark_libgomp_owner("main")
        svc.mark_libgomp_owner("worker")
        assert svc._libgomp_pool_owner == "main"

    def test_main_is_idempotent(self) -> None:
        svc = self._make_service()
        svc.mark_libgomp_owner("main")
        svc.mark_libgomp_owner("main")
        assert svc._libgomp_pool_owner == "main"

    def test_load_data_does_not_reset_libgomp_owner(self) -> None:
        """libgomp pool affinity is process-state — reloading data does
        not unbind it. The state must persist so the guard catches
        the next thread-runner job.
        """
        from lizyml_widget.service import WidgetService

        svc = WidgetService(adapter=LizyMLAdapter())
        svc.mark_libgomp_owner("main")
        df = pd.DataFrame({"f1": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
        svc.load_data(df, target="y")
        assert svc._libgomp_pool_owner == "main"


# ---------------------------------------------------------------------------
# State machine — predict / SHAP plot transitions
# ---------------------------------------------------------------------------


class TestServiceCallSitesMarkMain:
    """Caller-thread ML library calls must mark the state to "main".

    The actual adapter calls are mocked because we are testing the
    state-transition wiring, not the LightGBM/SHAP behavior.
    """

    def _make_service_with_model(self) -> Any:
        from lizyml_widget.service import WidgetService

        adapter = MagicMock()
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.predict.return_value = MagicMock()
        adapter.plot.return_value = MagicMock()
        adapter.plot_inference.return_value = MagicMock()
        svc = WidgetService(adapter=adapter)
        svc._model = MagicMock()
        return svc, adapter

    def test_predict_marks_main(self) -> None:
        svc, _ = self._make_service_with_model()
        assert svc._libgomp_pool_owner == "unknown"
        svc.predict(pd.DataFrame({"f1": [1, 2]}))
        assert svc._libgomp_pool_owner == "main"

    def test_predict_marks_main_even_when_adapter_raises(self) -> None:
        """LightGBM may have already entered the parallel region by the
        time the exception unwinds — mark "main" defensively."""
        svc, adapter = self._make_service_with_model()
        adapter.predict.side_effect = RuntimeError("predict failed")
        with pytest.raises(RuntimeError):
            svc.predict(pd.DataFrame({"f1": [1, 2]}))
        assert svc._libgomp_pool_owner == "main"

    def test_get_plot_shap_marks_main(self) -> None:
        svc, _ = self._make_service_with_model()
        svc.get_plot("feature-importance-shap")
        assert svc._libgomp_pool_owner == "main"

    def test_get_plot_non_shap_does_not_mark_main(self) -> None:
        """Non-SHAP plots (e.g., learning-curve) do not run TreeExplainer
        on the caller thread — state must remain unchanged."""
        svc, _ = self._make_service_with_model()
        svc.get_plot("optimization-history")
        assert svc._libgomp_pool_owner == "unknown"

    def test_get_inference_plot_shap_summary_marks_main(self) -> None:
        svc, _ = self._make_service_with_model()
        svc.get_inference_plot(pd.DataFrame({"pred": [0.1]}), "shap-summary")
        assert svc._libgomp_pool_owner == "main"

    def test_get_inference_plot_distribution_does_not_mark_main(self) -> None:
        svc, _ = self._make_service_with_model()
        svc.get_inference_plot(
            pd.DataFrame({"pred": [0.1]}),
            "prediction-distribution",
        )
        assert svc._libgomp_pool_owner == "unknown"


# ---------------------------------------------------------------------------
# State machine — runner completion transitions (in widget._supervise)
# ---------------------------------------------------------------------------


class TestRunnerCompletionMarksOwner:
    """``_supervise`` marks the owner after a successful runner.run()."""

    def test_subprocess_runner_success_marks_subprocess(self) -> None:
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", "/usr/lib/libomp5.so"),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
        ):
            mock_inst = mock_sp.return_value
            mock_inst.kind = "subprocess"
            mock_inst.run.return_value = _stub_runner_result()

            w = _make_widget()
            assert w._service._libgomp_pool_owner == "unknown"
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)
            assert w.status == "completed", f"Fit failed: {w.error!r}"
            assert w._service._libgomp_pool_owner == "subprocess"

    def test_thread_runner_success_marks_worker(self) -> None:
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("thread", None),
            ),
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_inst = mock_thread.return_value
            mock_inst.kind = "thread"
            mock_inst.run.return_value = _stub_runner_result()

            w = _make_widget()
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)
            assert w.status == "completed", f"Fit failed: {w.error!r}"
            assert w._service._libgomp_pool_owner == "worker"

    def test_failed_run_does_not_mark_owner(self) -> None:
        """A failed job may have aborted before any parallel region
        executed — leave state unchanged so the guard does not over-
        constrain future jobs."""
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("thread", None),
            ),
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_inst = mock_thread.return_value
            mock_inst.kind = "thread"
            mock_inst.run.side_effect = RuntimeError("boom")

            w = _make_widget()
            with pytest.raises(RuntimeError, match="boom"):
                w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)
            assert w.status == "failed"
            assert w._service._libgomp_pool_owner == "unknown"


# ---------------------------------------------------------------------------
# Widget guard — re-route thread → subprocess when state is "main"
# ---------------------------------------------------------------------------


class TestWidgetGuardReroutesToSubprocess:
    """``widget._run_job`` upgrades thread → subprocess when state is "main".

    The actual reroute decision is observable via which runner class
    gets instantiated. We pre-seed the service state and assert
    SubprocessJobRunner is constructed even though the strategy
    detector returned "thread".
    """

    def test_thread_strategy_with_main_state_reroutes_to_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LZW_FORCE_THREAD", raising=False)
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("thread", None),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_sp_inst = mock_sp.return_value
            mock_sp_inst.kind = "subprocess"
            mock_sp_inst.run.return_value = _stub_runner_result()

            w = _make_widget()
            # Pre-seed: prior predict on parent main thread bound libgomp.
            w._service.mark_libgomp_owner("main")

            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)

            (
                mock_sp.assert_called_once(),
                ("INV-G: thread+main state must re-route to SubprocessJobRunner"),
            )
            mock_thread.assert_not_called()

    def test_force_thread_env_honored_when_state_main(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """LZW_FORCE_THREAD=1 is an explicit user opt-out. The guard
        must respect it (only WARN, no auto-reroute)."""
        monkeypatch.setenv("LZW_FORCE_THREAD", "1")
        caplog.set_level(logging.WARNING, logger="lizyml_widget.widget")

        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("thread", None),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_thread_inst = mock_thread.return_value
            mock_thread_inst.kind = "thread"
            mock_thread_inst.run.return_value = _stub_runner_result()

            w = _make_widget()
            w._service.mark_libgomp_owner("main")
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)

            (
                mock_thread.assert_called_once(),
                ("LZW_FORCE_THREAD=1 must keep thread runner under main state"),
            )
            mock_sp.assert_not_called()
            warn_msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("INV-G" in m for m in warn_msgs), (
                f"Expected INV-G WARN under FORCE_THREAD; got: {warn_msgs}"
            )

    def test_thread_strategy_with_unknown_state_uses_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without prior main-thread parallel region, thread strategy
        runs as-is — no spurious subprocess startup overhead."""
        monkeypatch.delenv("LZW_FORCE_THREAD", raising=False)
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("thread", None),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_thread_inst = mock_thread.return_value
            mock_thread_inst.kind = "thread"
            mock_thread_inst.run.return_value = _stub_runner_result()

            w = _make_widget()
            assert w._service._libgomp_pool_owner == "unknown"
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)

            mock_thread.assert_called_once()
            mock_sp.assert_not_called()

    def test_subprocess_strategy_does_not_log_inv_g_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The guard only fires for thread strategy with main state.
        Subprocess + main is the safe combination — no WARN should fire."""
        monkeypatch.delenv("LZW_FORCE_THREAD", raising=False)
        caplog.set_level(logging.WARNING, logger="lizyml_widget.widget")

        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
        ):
            mock_inst = mock_sp.return_value
            mock_inst.kind = "subprocess"
            mock_inst.run.return_value = _stub_runner_result()

            w = _make_widget()
            w._service.mark_libgomp_owner("main")
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)

            inv_g_warns = [r for r in caplog.records if "INV-G" in r.getMessage()]
            assert not inv_g_warns, (
                f"Subprocess+main should not emit INV-G warn; got: {inv_g_warns}"
            )

    def test_concurrent_jobs_each_apply_guard_independently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second job, after a first job that did NOT bind main, must
        consult the live state — not a frozen snapshot. This catches a
        future refactor that captures state once at __init__ time."""
        monkeypatch.delenv("LZW_FORCE_THREAD", raising=False)
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("thread", None),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_thread_inst = mock_thread.return_value
            mock_thread_inst.kind = "thread"
            mock_thread_inst.run.return_value = _stub_runner_result()
            mock_sp_inst = mock_sp.return_value
            mock_sp_inst.kind = "subprocess"
            mock_sp_inst.run.return_value = _stub_runner_result()

            w = _make_widget()

            # Job 1: state is "unknown" → thread runner.
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)
            assert mock_thread.call_count == 1
            assert mock_sp.call_count == 0

            # Simulate user predict between jobs.
            w._service.mark_libgomp_owner("main")

            # Job 2: state is now "main" → must re-route to subprocess.
            w.fit()
            if w._job_thread:
                w._job_thread.join(timeout=10)
            assert mock_sp.call_count == 1, "Second job must re-route to subprocess after predict"


# ---------------------------------------------------------------------------
# Threading sanity: mark_libgomp_owner is safe under concurrent calls.
# ---------------------------------------------------------------------------


class TestMarkLibgompOwnerThreadSafety:
    def test_concurrent_marks_converge_to_main_when_main_observed(self) -> None:
        """If any thread observes main, the final state is main —
        regardless of concurrent worker/subprocess transitions."""
        from lizyml_widget.service import WidgetService

        svc = WidgetService(adapter=LizyMLAdapter())

        def hammer_subprocess() -> None:
            for _ in range(200):
                svc.mark_libgomp_owner("subprocess")

        def hammer_worker() -> None:
            for _ in range(200):
                svc.mark_libgomp_owner("worker")

        def hammer_main() -> None:
            for _ in range(50):
                svc.mark_libgomp_owner("main")

        threads = [
            threading.Thread(target=hammer_subprocess),
            threading.Thread(target=hammer_worker),
            threading.Thread(target=hammer_main),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert svc._libgomp_pool_owner == "main"
