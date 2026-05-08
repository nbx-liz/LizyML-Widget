"""JobRunner Protocol + thread / subprocess implementations (P-032).

Extracts the job execution surface out of ``widget.py`` so the widget's
state-machine ( ``_supervise`` ) lives in one place and is shared by
both runner strategies. This prepares the ground for invariant-first
work in #118 — once status transitions are governed by a single
supervisor, INV-A / INV-B / INV-D / INV-E can be encoded as runtime
guards there rather than scattered across two near-duplicate worker
methods.

A ``JobSpec`` is the immutable description of work to do; a ``JobResult``
is what each runner produces. The supervisor in ``widget.py`` consumes
either dataclass uniformly.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .subprocess_runner import run_job_subprocess

if TYPE_CHECKING:
    from collections.abc import Callable

    from .service import WidgetService

_log = logging.getLogger(__name__)

__all__ = [
    "JobResult",
    "JobRunner",
    "JobSpec",
    "RetuneSubprocessUnsupportedError",
    "SubprocessJobRunner",
    "ThreadJobRunner",
]


class RetuneSubprocessUnsupportedError(RuntimeError):
    """Raised when retune is requested via the subprocess runner.

    The subprocess path cannot pick up an existing Optuna study from a
    fresh process today; supervisors translate this into a structured
    ``RETUNE_SUBPROCESS_UNSUPPORTED`` error code on the widget.
    """


@dataclass(frozen=True)
class JobSpec:
    """Immutable description of one job to execute.

    Each ``_run_job`` invocation builds exactly one ``JobSpec`` and hands
    it to the selected runner. Snapshotting ``config`` and
    ``retune_kwargs`` per-spec eliminates the cross-thread state handoff
    that the P-028 HIGH-2 review fix originally addressed.
    """

    job_type: str
    config: dict[str, Any]
    retune_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class JobResult:
    """Result of one runner execution.

    The fields parallel ``SubprocessJobResult`` so the supervisor can
    handle either runner identically. ``model_path`` is set only by the
    subprocess runner — when present, the supervisor reloads the model
    via ``WidgetService.load_model_from_path``.
    """

    job_type: str
    fit_summary: dict[str, Any] = field(default_factory=dict)
    tune_summary: dict[str, Any] = field(default_factory=dict)
    eval_table: list[dict[str, Any]] = field(default_factory=list)
    split_summary: list[dict[str, Any]] = field(default_factory=list)
    available_plots: list[str] = field(default_factory=list)
    model_path: str | None = None


class JobRunner(Protocol):
    """Single execution strategy for one job.

    Implementations must:

    - Run *spec* synchronously in the calling thread (the supervisor
      already runs each invocation on a background thread of its own).
    - Forward progress events to *on_progress* with the same arity the
      adapter uses (``current, total, message, **extra``).
    - Honour *cancel_event* and raise ``InterruptedError`` when set.
    - Return a populated :class:`JobResult` on success or raise an
      exception (including ``InterruptedError`` for cancel) on failure.

    Attributes:
        kind: Stable identifier the supervisor uses to choose between
            in-process cooperative-cancel semantics ("thread") and
            out-of-process SIGTERM-cancel semantics ("subprocess"),
            instead of `isinstance` (which breaks under test patching).
    """

    kind: str

    def run(
        self,
        spec: JobSpec,
        on_progress: Callable[..., None],
        cancel_event: threading.Event,
    ) -> JobResult: ...


class ThreadJobRunner:
    """In-process runner that drives ``WidgetService.fit/tune`` directly.

    The current default for the widget. Cancellation is cooperative —
    ``on_progress`` raises ``InterruptedError`` when ``cancel_event`` is
    set, and the adapter's polling helper picks that up.
    """

    kind: str = "thread"

    def __init__(self, service: WidgetService) -> None:
        self._service = service

    def run(
        self,
        spec: JobSpec,
        on_progress: Callable[..., None],
        cancel_event: threading.Event,  # noqa: ARG002 — adapter polls cancel via on_progress
    ) -> JobResult:
        if spec.job_type == "fit":
            return self._run_fit(spec, on_progress)
        if spec.job_type == "tune":
            return self._run_tune(spec, on_progress)
        msg = f"Unknown job_type: {spec.job_type}"
        raise ValueError(msg)

    def _run_fit(self, spec: JobSpec, on_progress: Callable[..., None]) -> JobResult:
        summary = self._service.fit(spec.config, on_progress=on_progress)
        eval_table = self._service.get_evaluate_table()
        split_summary = self._service.get_split_summary()
        fit_summary = {
            "metrics": summary.metrics,
            "fold_count": summary.fold_count,
            "fold_details": split_summary,
            "params": summary.params,
        }
        return JobResult(
            job_type="fit",
            fit_summary=fit_summary,
            eval_table=eval_table,
            split_summary=split_summary,
            available_plots=self._service.get_available_plots(),
        )

    def _run_tune(self, spec: JobSpec, on_progress: Callable[..., None]) -> JobResult:
        tune_kwargs = spec.retune_kwargs or {}
        is_resume = bool(tune_kwargs.get("resume"))
        n_trials = tune_kwargs.get("n_trials") or spec.config.get("tuning", {}).get(
            "optuna", {}
        ).get("params", {}).get("n_trials", 10)
        # Show round 2+ badge eagerly when resuming so the user sees
        # immediate feedback before the first trial fires.
        initial_round = 2 if is_resume else 1
        msg = (
            f"Resuming tune with {n_trials} more trials..."
            if is_resume
            else f"Tuning {n_trials} trials..."
        )
        on_progress(0, n_trials, msg, round=initial_round)
        summary_t = self._service.tune(spec.config, on_progress=on_progress, **tune_kwargs)
        tune_summary = {
            "best_params": summary_t.best_params,
            "best_score": summary_t.best_score,
            "trials": summary_t.trials,
            "metric_name": summary_t.metric_name,
            "direction": summary_t.direction,
            "rounds": summary_t.rounds,
            "boundary_report": summary_t.boundary_report,
        }
        # After tune, model MAY be fitted — guard evaluate/split calls (P-004 R3).
        eval_table: list[dict[str, Any]] = []
        split_summary: list[dict[str, Any]] = []
        try:
            eval_table = self._service.get_evaluate_table()
            split_summary = self._service.get_split_summary()
        except (AttributeError, RuntimeError, ValueError) as exc:
            # lizyml raises RuntimeError("Model has not been fitted") when
            # evaluate_table is called on an unfitted model (P-004 R3).
            _log.debug("Tune-only fit_summary skipped (model not fitted): %s", exc)
        return JobResult(
            job_type="tune",
            tune_summary=tune_summary,
            eval_table=eval_table,
            split_summary=split_summary,
            available_plots=self._service.get_available_plots(),
        )


class SubprocessJobRunner:
    """Out-of-process runner for OpenMP-safe execution.

    Spawns ``lizyml_widget._subprocess_entry`` via
    :func:`subprocess_runner.run_job_subprocess`. Cancellation is
    delivered as ``SIGTERM`` to the child process; the parent's
    ``on_progress`` does not need to raise ``InterruptedError`` itself.

    Re-tune is not supported on this path yet — the subprocess starts
    with a fresh interpreter and cannot resume the previous Optuna
    study. The supervisor turns
    :class:`RetuneSubprocessUnsupportedError` into the
    ``RETUNE_SUBPROCESS_UNSUPPORTED`` error code.
    """

    kind: str = "subprocess"

    def __init__(
        self,
        service: WidgetService,
        *,
        libomp_path: str | None = None,
    ) -> None:
        self._service = service
        self._libomp_path = libomp_path

    def run(
        self,
        spec: JobSpec,
        on_progress: Callable[..., None],
        cancel_event: threading.Event,
    ) -> JobResult:
        if spec.retune_kwargs:
            msg = (
                "Re-tune is not supported in subprocess execution mode. "
                "Unset LZW_FORCE_SUBPROCESS=1 or use w.tune() for a "
                "fresh study."
            )
            raise RetuneSubprocessUnsupportedError(msg)

        df = self._service.get_dataframe()
        target = self._service.get_df_info().get("target", "")
        model_out_path = tempfile.mkdtemp(prefix="lzw_model_")

        try:
            sp_result = run_job_subprocess(
                job_type=spec.job_type,
                config=spec.config,
                df=df,
                target=target,
                libomp_path=self._libomp_path,
                on_progress=on_progress,
                cancel_flag=cancel_event,
                model_out_path=model_out_path,
            )
            # Load model back from subprocess so the widget side can
            # serve plots / inference.
            if sp_result.model_path:
                try:
                    self._service.load_model_from_path(sp_result.model_path)
                except Exception as load_err:  # noqa: BLE001
                    _log.warning("Model load from subprocess failed: %s", load_err)

            return JobResult(
                job_type=spec.job_type,
                fit_summary=sp_result.fit_summary,
                tune_summary=sp_result.tune_summary,
                eval_table=sp_result.eval_table,
                split_summary=sp_result.split_summary,
                available_plots=sp_result.available_plots,
                model_path=sp_result.model_path,
            )
        finally:
            with contextlib.suppress(OSError):
                shutil.rmtree(model_out_path, ignore_errors=True)
