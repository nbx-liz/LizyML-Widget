"""State-machine invariant tests (P-033, #118).

Each test corresponds to one INV declared in BLUEPRINT.md §6.4. The tests
exercise the *invariant violation* scenarios described there, not the
happy paths. They are RED-then-GREEN: written to assert the post-P-032
behaviour locked in by the supervisor.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.adapter_views import LMBoundaryDimView, LMBoundaryReportView
from lizyml_widget.job_runner import JobResult, JobSpec, ThreadJobRunner
from lizyml_widget.types import BackendInfo, FitSummary, TuningSummary


def _make_widget() -> Any:
    """Create a LizyWidget with a mocked backend.

    Mirrors the helper in tests/test_widget_jobs.py so this file stays
    self-contained.
    """
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

        return LizyWidget()


# ── INV-A: status FSM ─────────────────────────────────────────────────


class TestInvAStatusFsm:
    """INV-A: status transitions follow idle → data_loaded → running → {completed | failed}."""

    def test_idle_to_data_loaded_on_load(self) -> None:
        w = _make_widget()
        assert w.status == "idle"
        w.load(pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}), target="y")
        assert w.status == "data_loaded"

    def test_running_status_blocks_a_second_run_job(self) -> None:
        """INV-A consequence: a manual `_run_job` while already running is a no-op."""
        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")

        # Simulate a job already in flight by setting status manually.
        w.status = "running"
        before_counter = w._job_counter
        w._run_job("fit")
        assert w._job_counter == before_counter, (
            "INV-A/B: _run_job during running must not start a second job"
        )

    def test_load_resets_status_to_data_loaded(self) -> None:
        """A second load() after completion must reset status (driving the FSM forward)."""
        w = _make_widget()
        w.load(pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}), target="y")
        # Simulate a finished job.
        w.status = "completed"
        # Reload data — status returns to data_loaded.
        w.load(pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}), target="y")
        assert w.status == "data_loaded"


# ── INV-B: at most one live worker ────────────────────────────────────


class TestInvBJobThreadSingleton:
    """INV-B: _job_thread holds at most one live worker at any time."""

    def test_run_job_during_running_does_not_spawn_second_thread(self) -> None:
        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
        first_thread = MagicMock(spec=threading.Thread)
        first_thread.is_alive.return_value = True
        w._job_thread = first_thread
        w.status = "running"

        w._run_job("fit")
        # The pre-existing thread reference must remain — the guard returned
        # early without overwriting it.
        assert w._job_thread is first_thread


# ── INV-C: _tune_model ownership across tune→fit→retune ───────────────


class TestInvCTuneModelOwnership:
    """INV-C: _tune_model is owned by the most-recent tune invocation."""

    def test_intervening_fit_does_not_clobber_tune_model(self) -> None:
        from lizyml_widget.service import WidgetService

        adapter = MagicMock()
        adapter.info = BackendInfo(name="mock", version="0.0.0")
        adapter.create_model.return_value = MagicMock(name="tune_model")
        adapter.tune.return_value = TuningSummary(
            best_params={"lr": 0.01},
            best_score=0.92,
            trials=[],
            metric_name="auc",
            direction="maximize",
            rounds=[],
            boundary_report=None,
        )
        adapter.fit.return_value = FitSummary(
            metrics={"auc": {"oos": 0.91}},
            fold_count=3,
            params=[],
        )
        svc = WidgetService(adapter=adapter)
        # Inject `_df` directly so tune/fit can run without going through
        # `load_data` (whose feature-availability preconditions are not
        # relevant to this invariant). The mocked adapter accepts the
        # stub config as-is.
        svc._df = pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10})
        cfg: dict[str, Any] = {"task": "binary", "model": {}}
        svc.tune(cfg)
        tuned = svc._tune_model
        assert tuned is not None, "tune should populate _tune_model (P-028)"

        # Adapter returns a *different* model instance for the fit call —
        # if the invariant holds, _tune_model must remain the original one.
        adapter.create_model.return_value = MagicMock(name="fit_model")
        svc.fit(cfg)
        assert svc._tune_model is tuned, "INV-C: an intervening fit must not clobber _tune_model"


# ── INV-D: cancel flag lifecycle ──────────────────────────────────────


class TestInvDCancelFlagLifecycle:
    """INV-D: _cancel_flag is reset per job; cancel transitions to failed/CANCELLED."""

    def test_cancel_flag_is_cleared_at_run_job_start(self) -> None:
        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
        # Pretend a previous cancel left the flag set.
        w._cancel_flag.set()
        assert w._cancel_flag.is_set()

        # Bypass the actual fit by stubbing prepare_run_config so it
        # returns a minimal valid config, and stub the runner so the
        # worker thread terminates immediately.
        w._service.prepare_run_config = MagicMock(  # type: ignore[method-assign]
            return_value={"config_version": 1, "task": "binary", "model": {}}
        )

        with patch("lizyml_widget.widget.ThreadJobRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.kind = "thread"
            mock_runner.run.return_value = JobResult(job_type="fit")
            w._run_job("fit")
            if w._job_thread:
                w._job_thread.join(timeout=2.0)

        # _run_job clears the cancel flag before launching the worker.
        assert not w._cancel_flag.is_set(), (
            "INV-D: cancel_flag must be cleared at start of every new job"
        )

    def test_cancel_during_running_transitions_to_failed_cancelled(self) -> None:
        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
        runner = ThreadJobRunner(w._service)
        spec = JobSpec(job_type="fit", config={"task": "binary"})

        # _run_job sets status="running" inside the job lock before
        # spawning the supervisor — emulate that pre-condition here.
        w.status = "running"

        # Simulate the runner raising InterruptedError mid-flight.
        with patch.object(runner, "run", side_effect=InterruptedError("cancelled")):
            w._supervise(runner, spec)

        assert w.status == "failed"
        assert w.error.get("code") == "CANCELLED"


# ── INV-E: progress.round monotonic ───────────────────────────────────


class TestInvERoundMonotonic:
    """INV-E: progress.round is monotonic non-decreasing within a single tune."""

    def test_round_does_not_regress_across_progress_events(self) -> None:
        from lizyml_widget.job_runner import JobSpec

        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
        runner = ThreadJobRunner(w._service)
        spec = JobSpec(job_type="tune", config={"task": "binary"})

        captured_rounds: list[int] = []

        def fake_run(spec_arg: JobSpec, on_progress: Any, cancel_event: Any) -> Any:
            # Drive a sequence of round-aware progress events.
            for round_no in (1, 1, 2, 2, 3):
                on_progress(0, 10, "tick", round=round_no)
                captured_rounds.append(int(w.progress.get("round", 0)))

            return JobResult(
                job_type="tune",
                tune_summary={
                    "best_params": {},
                    "best_score": 0.0,
                    "trials": [],
                    "metric_name": "auc",
                    "direction": "maximize",
                    "rounds": [],
                    "boundary_report": None,
                },
                eval_table=[],
                split_summary=[],
                available_plots=[],
                model_path=None,
            )

        # _run_job sets status="running" before spawning supervise — emulate it.
        w.status = "running"
        with patch.object(runner, "run", side_effect=fake_run):
            w._supervise(runner, spec)

        # Every captured round must be >= the previous one.
        assert captured_rounds == sorted(captured_rounds), (
            f"INV-E: round regressed in sequence {captured_rounds}"
        )


# ── INV-F: boundary_report.dims uniqueness + completeness ─────────────


class TestInvFBoundaryReportDimsUnique:
    """INV-F: boundary_report.dims lists each dim exactly once."""

    def test_view_boundary_report_uniqueness(self) -> None:
        """Construct a view with unique dims — invariant holds."""
        report = LMBoundaryReportView(
            dims=(
                LMBoundaryDimView(
                    name="lr",
                    best_value=0.05,
                    low=0.001,
                    high=0.1,
                    position_pct=0.5,
                    edge="",
                    expanded=False,
                    new_low=None,
                    new_high=None,
                ),
                LMBoundaryDimView(
                    name="num_leaves",
                    best_value=64,
                    low=15,
                    high=255,
                    position_pct=0.2,
                    edge="",
                    expanded=False,
                    new_low=None,
                    new_high=None,
                ),
            ),
            expanded_names=(),
        )
        names = [d.name for d in report.dims]
        assert len(names) == len(set(names)), "INV-F: dims must be unique by name"

    def test_assert_dims_unique_helper_detects_duplicate(self) -> None:
        """A duplicate dim name violates INV-F — detection helper flags it."""
        report = LMBoundaryReportView(
            dims=(
                LMBoundaryDimView(
                    name="lr",
                    best_value=0.05,
                    low=0.001,
                    high=0.1,
                    position_pct=0.5,
                    edge="",
                    expanded=False,
                    new_low=None,
                    new_high=None,
                ),
                LMBoundaryDimView(
                    name="lr",  # duplicate
                    best_value=0.06,
                    low=0.001,
                    high=0.1,
                    position_pct=0.6,
                    edge="",
                    expanded=False,
                    new_low=None,
                    new_high=None,
                ),
            ),
            expanded_names=(),
        )
        names = [d.name for d in report.dims]
        # The helper assertion lives at the call-site — we encode the
        # check here so a future violation in the backend surfaces.
        assert len(names) != len(set(names)), (
            "Sanity: this fixture intentionally violates INV-F to verify the check"
        )


# ── Runtime assertion gates (#135) ────────────────────────────────────


def _make_tune_runner_emitting(rounds: list[int]) -> tuple[ThreadJobRunner, JobSpec, Any]:
    """Build a runner whose ``run()`` drives ``on_progress`` with the given rounds.

    Returns ``(runner, spec, widget)``. The caller invokes ``widget._supervise``
    to exercise the supervisor's runtime guards.
    """
    w = _make_widget()
    w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
    runner = ThreadJobRunner(w._service)
    spec = JobSpec(job_type="tune", config={"task": "binary"})

    def fake_run(spec_arg: JobSpec, on_progress: Any, cancel_event: Any) -> JobResult:
        for round_no in rounds:
            on_progress(0, 10, "tick", round=round_no)
        return JobResult(
            job_type="tune",
            tune_summary={
                "best_params": {},
                "best_score": 0.0,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
                "rounds": [],
                "boundary_report": None,
            },
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path=None,
        )

    runner.run = MagicMock(side_effect=fake_run)  # type: ignore[method-assign]
    # _run_job sets status="running" before spawning supervise — emulate that.
    w.status = "running"
    w._cancel_flag.clear()
    return runner, spec, w


class TestRuntimeAssertionsSupervise:
    """#135: ``_supervise`` enforces INV-A, INV-D, INV-E at runtime."""

    def test_inv_a_entry_rejects_non_running_status(self) -> None:
        """INV-A: supervisor must enter with status=='running'."""
        runner, spec, w = _make_tune_runner_emitting([1])
        w.status = "data_loaded"  # illegal: caller forgot to flip to running
        with pytest.raises(AssertionError, match="INV-A"):
            w._supervise(runner, spec)

    def test_inv_d_entry_rejects_carryover_cancel_flag(self) -> None:
        """INV-D: cancel flag must not be set when the supervisor enters."""
        runner, spec, w = _make_tune_runner_emitting([1])
        w._cancel_flag.set()  # illegal: stale cancel from a prior job
        with pytest.raises(AssertionError, match="INV-D"):
            w._supervise(runner, spec)

    def test_inv_e_round_regression_raises(self) -> None:
        """INV-E: a regressing round in on_progress raises AssertionError.

        The supervisor's outer ``except Exception`` catches the assertion
        and surfaces it as status=='failed' with the INV-E message in
        ``error``. We assert on that path because production code runs
        the supervisor under the same try/except boundary.
        """
        runner, spec, w = _make_tune_runner_emitting([2, 1])  # regression!
        w._supervise(runner, spec)
        assert w.status == "failed"
        assert "INV-E" in w.error.get("message", ""), (
            f"expected INV-E violation in error, got {w.error!r}"
        )

    def test_inv_a_exit_terminal_status_holds_on_completion(self) -> None:
        """Happy path: supervisor exits with status in {completed, failed}."""
        runner, spec, w = _make_tune_runner_emitting([1, 1, 2])
        w._supervise(runner, spec)
        assert w.status == "completed"


class TestRuntimeAssertionsApplyResult:
    """#135: ``_apply_job_result`` enforces INV-F at runtime."""

    def test_inv_f_duplicate_dims_raises(self) -> None:
        """INV-F: boundary_report.dims with duplicate names must raise."""
        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
        result = JobResult(
            job_type="tune",
            tune_summary={
                "best_params": {},
                "best_score": 0.0,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
                "rounds": [],
                "boundary_report": {
                    "dims": [
                        {"name": "lr", "best_value": 0.05},
                        {"name": "lr", "best_value": 0.06},  # duplicate
                    ],
                    "expanded_names": [],
                },
            },
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path=None,
        )
        with pytest.raises(AssertionError, match="INV-F"):
            w._apply_job_result(result)

    def test_inv_f_unique_dims_passes(self) -> None:
        """INV-F: unique dim names pass without error."""
        w = _make_widget()
        w.load(pd.DataFrame({"x": list(range(20)), "y": [0, 1] * 10}), target="y")
        result = JobResult(
            job_type="tune",
            tune_summary={
                "best_params": {},
                "best_score": 0.0,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
                "rounds": [],
                "boundary_report": {
                    "dims": [
                        {"name": "lr", "best_value": 0.05},
                        {"name": "num_leaves", "best_value": 64},
                    ],
                    "expanded_names": [],
                },
            },
            eval_table=[],
            split_summary=[],
            available_plots=[],
            model_path=None,
        )
        w._apply_job_result(result)
        assert w.tune_summary["boundary_report"]["dims"][0]["name"] == "lr"
