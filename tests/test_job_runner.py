"""Unit tests for JobRunner Protocol + ThreadJobRunner / SubprocessJobRunner.

Covers the four cases the issue's acceptance criteria call out for each
runner:
- normal completion
- user cancellation mid-run
- exception mid-run
- abnormal exit (subprocess only)

The tests use lightweight mocks so they run on the unit-test tier; the
end-to-end path is exercised by tests/e2e/.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget.job_runner import (
    JobResult,
    JobSpec,
    SubprocessJobRunner,
    ThreadJobRunner,
)
from lizyml_widget.subprocess_runner import SubprocessJobResult
from lizyml_widget.types import FitSummary, TuningSummary


@pytest.fixture()
def mock_service() -> Any:
    svc = MagicMock()
    svc.fit.return_value = FitSummary(
        metrics={"auc": {"oos": 0.9}},
        fold_count=5,
        params=[{"param": "v"}],
    )
    svc.tune.return_value = TuningSummary(
        best_params={"lr": 0.01},
        best_score=0.92,
        trials=[],
        metric_name="auc",
        direction="maximize",
        rounds=[],
        boundary_report=None,
    )
    svc.get_evaluate_table.return_value = [{"metric": "auc", "OOS": 0.9}]
    svc.get_split_summary.return_value = []
    svc.get_available_plots.return_value = ["learning-curve"]
    svc.get_dataframe.return_value = MagicMock()
    svc.get_df_info.return_value = {"target": "y"}
    return svc


# ── ThreadJobRunner ────────────────────────────────────────────────────


class TestThreadJobRunnerNormal:
    def test_fit_returns_fit_summary(self, mock_service: Any) -> None:
        runner = ThreadJobRunner(mock_service)
        result = runner.run(
            JobSpec(job_type="fit", config={}),
            on_progress=lambda *a, **kw: None,
            cancel_event=threading.Event(),
        )
        assert isinstance(result, JobResult)
        assert result.job_type == "fit"
        assert result.fit_summary["fold_count"] == 5
        assert result.available_plots == ["learning-curve"]

    def test_tune_returns_tune_summary(self, mock_service: Any) -> None:
        runner = ThreadJobRunner(mock_service)
        result = runner.run(
            JobSpec(job_type="tune", config={}),
            on_progress=lambda *a, **kw: None,
            cancel_event=threading.Event(),
        )
        assert result.tune_summary["best_score"] == 0.92
        assert result.eval_table == [{"metric": "auc", "OOS": 0.9}]

    def test_kind_is_thread(self, mock_service: Any) -> None:
        assert ThreadJobRunner(mock_service).kind == "thread"


class TestThreadJobRunnerCancel:
    def test_cooperative_cancel_via_on_progress(self, mock_service: Any) -> None:
        """The runner doesn't poll cancel itself; the adapter's on_progress
        consumer raises InterruptedError when cancel_event is set."""
        # Simulate a cancelled fit by making the service raise InterruptedError.
        mock_service.fit.side_effect = InterruptedError("cancelled")
        runner = ThreadJobRunner(mock_service)
        with pytest.raises(InterruptedError):
            runner.run(
                JobSpec(job_type="fit", config={}),
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )


class TestThreadJobRunnerException:
    def test_backend_exception_propagates(self, mock_service: Any) -> None:
        mock_service.fit.side_effect = RuntimeError("backend boom")
        runner = ThreadJobRunner(mock_service)
        with pytest.raises(RuntimeError, match="backend boom"):
            runner.run(
                JobSpec(job_type="fit", config={}),
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

    def test_unknown_job_type_raises_value_error(self, mock_service: Any) -> None:
        runner = ThreadJobRunner(mock_service)
        with pytest.raises(ValueError, match="Unknown job_type"):
            runner.run(
                JobSpec(job_type="banana", config={}),
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )


class TestThreadJobRunnerTuneOnly:
    """P-004 R3 / #147: tune may complete without a fitted model. The
    adapter layer (LizyMLAdapter) now returns ``[]`` from
    evaluate_table/split_summary on an unfit model, so the runner does
    not need exception-based control flow. Pin that the runner forwards
    the empty list through unchanged.
    """

    def test_tune_with_unfit_model_returns_empty_eval_table(self, mock_service: Any) -> None:
        mock_service.get_evaluate_table.return_value = []
        mock_service.get_split_summary.return_value = []
        runner = ThreadJobRunner(mock_service)
        result = runner.run(
            JobSpec(job_type="tune", config={}),
            on_progress=lambda *a, **kw: None,
            cancel_event=threading.Event(),
        )
        assert result.eval_table == []
        assert result.split_summary == []
        assert result.tune_summary["best_score"] == 0.92  # tune still succeeded


# ── SubprocessJobRunner ────────────────────────────────────────────────


class TestSubprocessJobRunnerNormal:
    def test_fit_returns_subprocess_result_fields(self, mock_service: Any) -> None:
        runner = SubprocessJobRunner(mock_service)
        sp_result = SubprocessJobResult(
            job_type="fit",
            fit_summary={"metrics": {"auc": {"oos": 0.9}}, "fold_count": 5, "params": []},
            tune_summary={},
            eval_table=[],
            split_summary=[],
            available_plots=["learning-curve"],
            model_path="/tmp/model.pkl",
        )
        with patch("lizyml_widget.job_runner.run_job_subprocess", return_value=sp_result):
            result = runner.run(
                JobSpec(job_type="fit", config={}),
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )
        assert result.fit_summary["fold_count"] == 5
        assert result.model_path == "/tmp/model.pkl"
        # The runner reloads the model into the service.
        mock_service.load_model_from_path.assert_called_once_with("/tmp/model.pkl")

    def test_kind_is_subprocess(self, mock_service: Any) -> None:
        assert SubprocessJobRunner(mock_service).kind == "subprocess"


class TestSubprocessJobRunnerRetuneResume:
    """P-038: subprocess retune resume forwards retune_kwargs and
    tune_state_in_path to ``run_job_subprocess`` after exporting the
    parent's current tune state.
    """

    def test_retune_forwards_kwargs_and_exports_tune_state(self, mock_service: Any) -> None:
        from lizyml_widget.subprocess_runner import SubprocessJobResult

        runner = SubprocessJobRunner(mock_service)
        spec = JobSpec(
            job_type="tune",
            config={},
            retune_kwargs={"resume": True, "n_trials": 5},
        )
        captured: dict[str, Any] = {}

        def fake_run(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SubprocessJobResult(
                job_type="tune",
                fit_summary={},
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

        with patch("lizyml_widget.job_runner.run_job_subprocess", side_effect=fake_run):
            runner.run(
                spec,
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )

        # The parent must call export_tune_state_to_path before spawning.
        mock_service.export_tune_state_to_path.assert_called_once()
        # The path written by the parent must reach the subprocess input.
        export_path = mock_service.export_tune_state_to_path.call_args.args[0]
        assert captured.get("tune_state_in_path") == export_path
        assert captured.get("retune_kwargs") == {"resume": True, "n_trials": 5}


class TestSubprocessJobRunnerCancel:
    def test_subprocess_interrupted_error_propagates(self, mock_service: Any) -> None:
        runner = SubprocessJobRunner(mock_service)
        with (
            patch(
                "lizyml_widget.job_runner.run_job_subprocess",
                side_effect=InterruptedError("cancelled"),
            ),
            pytest.raises(InterruptedError),
        ):
            runner.run(
                JobSpec(job_type="fit", config={}),
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )


class TestSubprocessJobRunnerAbnormalExit:
    def test_runtime_error_propagates(self, mock_service: Any) -> None:
        """Subprocess crash (no result message) surfaces as RuntimeError."""
        runner = SubprocessJobRunner(mock_service)
        with (
            patch(
                "lizyml_widget.job_runner.run_job_subprocess",
                side_effect=RuntimeError("Subprocess exited with code 1 without result"),
            ),
            pytest.raises(RuntimeError, match="without result"),
        ):
            runner.run(
                JobSpec(job_type="fit", config={}),
                on_progress=lambda *a, **kw: None,
                cancel_event=threading.Event(),
            )
