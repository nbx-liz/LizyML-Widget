"""LizyWidget - anywidget-based notebook UI for LizyML."""

from __future__ import annotations

import copy
import importlib.resources
import logging
import os
import statistics
import threading
import time
from typing import Any

import anywidget
import pandas as pd
import traitlets

from .adapter import BackendAdapter, LizyMLAdapter
from .job_runner import (
    JobResult,
    JobRunner,
    JobSpec,
    RetuneSubprocessUnsupportedError,
    SubprocessJobRunner,
    ThreadJobRunner,
)
from .openmp_detect import get_execution_strategy
from .service import WidgetService
from .types import FitSummary, PredictionSummary, TuningSummary

_log = logging.getLogger(__name__)

#: Progress traitlet keys that the job workers forward from the adapter's
#: ``on_progress`` callback ``**extra`` payload (P-027 round-aware fields
#: plus ``fold_results`` from the fit path).  Shared between
#: ``_job_worker`` and ``_subprocess_job_worker`` so new keys only need
#: to be added in one place.
_PROGRESS_EXTRA_KEYS: tuple[str, ...] = (
    "round",
    "total_rounds",
    "cumulative_trials",
    "expanded_dims",
    "latest_score",
    "latest_state",
    "best_score",
    "fold_results",
)


def _build_progress_payload(
    current: int,
    total: int,
    message: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable progress dict from an adapter callback.

    Only the keys in :data:`_PROGRESS_EXTRA_KEYS` are forwarded, and only
    when their value is non-``None``, so the widget ``progress`` traitlet
    stays minimal and round-trippable through anywidget's JSON bridge.
    """
    payload: dict[str, Any] = {
        "current": current,
        "total": total,
        "message": message,
    }
    for key in _PROGRESS_EXTRA_KEYS:
        if key in extra and extra[key] is not None:
            payload[key] = extra[key]
    return payload


class LizyWidget(anywidget.AnyWidget):
    """LizyML notebook UI widget."""

    _esm = importlib.resources.files("lizyml_widget") / "static/widget.js"
    _css = importlib.resources.files("lizyml_widget") / "static/widget.css"

    # ── Python → JS traitlets ────────────────────────────────
    backend_info = traitlets.Dict({}).tag(sync=True)
    df_info = traitlets.Dict({}).tag(sync=True)
    backend_contract = traitlets.Dict({}).tag(sync=True)
    config = traitlets.Dict({}).tag(sync=True)
    status = traitlets.Unicode("idle").tag(sync=True)
    job_type = traitlets.Unicode("").tag(sync=True)
    job_index = traitlets.Int(0).tag(sync=True)
    progress = traitlets.Dict({}).tag(sync=True)
    elapsed_sec = traitlets.Float(0.0).tag(sync=True)
    fit_summary = traitlets.Dict({}).tag(sync=True)
    tune_summary = traitlets.Dict({}).tag(sync=True)
    available_plots: list[str] = traitlets.List([]).tag(sync=True)  # type: ignore[assignment]
    inference_result = traitlets.Dict({}).tag(sync=True)
    error = traitlets.Dict({}).tag(sync=True)

    # ── JS → Python traitlet ─────────────────────────────────
    action = traitlets.Dict({}).tag(sync=True)

    def __init__(self, *, adapter: BackendAdapter | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._service = WidgetService(adapter=adapter or LizyMLAdapter())
        self._execution_strategy: str | None = None
        self._libomp_path: str | None = None
        self._job_thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()
        self._job_lock = threading.Lock()
        self._job_counter = 0
        self._inference_df: pd.DataFrame | None = None
        # #154: warn-once flag for the retune→thread fallback notice. The
        # check/set in ``_run_job`` runs after ``_job_lock`` is released
        # (the lock only protects the FSM transition, not runner choice),
        # so a race between two near-simultaneous retunes could log the
        # notice twice. That is harmless and we accept it; the cost of
        # adding a separate lock is not justified for an info log.
        self._retune_fallback_warned: bool = False
        # P-035: tune snapshots now live on TuningSummary inside
        # WidgetService, not on the widget. Removed _tune_config_snapshot
        # and _tune_ui_snapshot.
        # #127: action handlers extracted to widget_actions.WidgetActionDispatcher.
        # The dispatcher holds a back-reference to this widget and reads/writes
        # traitlets through it; widget owns the state machine, dispatcher owns
        # the routing.
        from .widget_actions import WidgetActionDispatcher

        self._dispatcher = WidgetActionDispatcher(self)

        try:
            info = self._service.info
            self.backend_info = {"name": info.name, "version": info.version}
        except Exception:
            self.backend_info = {}

    # ── Public Python API ─────────────────────────────────────

    def load(self, df: pd.DataFrame, target: str | None = None) -> LizyWidget:
        """Load a DataFrame into the widget."""
        df_info = self._service.load_data(df, target=target)
        self.df_info = df_info
        self.status = "data_loaded"
        self.error = {}
        self.fit_summary = {}
        self.tune_summary = {}
        self.available_plots = []

        contract = self._service.get_backend_contract()
        self.backend_contract = {
            "schema_version": contract.schema_version,
            "config_schema": contract.config_schema,
            "ui_schema": contract.ui_schema,
            "capabilities": contract.capabilities,
        }
        self.config = self._service.initialize_config()
        # Apply task defaults (metrics, search space) if task is known
        task = df_info.get("task")
        if task:
            self.config = self._service.apply_task_params(dict(self.config), task)
        return self

    def set_target(self, col: str) -> LizyWidget:
        """Set target column and trigger auto-detection."""
        df_info = self._service.set_target(col)
        self.df_info = df_info
        self.status = "data_loaded"
        return self

    def fit(self, *, timeout: float | None = None) -> LizyWidget:
        """Run Fit in a background thread and block until complete.

        Raises RuntimeError if the job fails.
        """
        return self._run_blocking_job("fit", timeout=timeout)

    def tune(self, *, timeout: float | None = None) -> LizyWidget:
        """Run Tune in a background thread and block until complete.

        Raises RuntimeError if the job fails.
        """
        return self._run_blocking_job("tune", timeout=timeout)

    def retune(
        self,
        *,
        n_trials: int | None = None,
        expand_boundary: bool | None = None,
        boundary_threshold: float = 0.05,
        timeout: float | None = None,
    ) -> LizyWidget:
        """Resume the last tuning run with additional trials (P-028).

        Requires a prior ``tune()`` call on this widget: the underlying
        lizyml ``Model`` is reused so its Optuna study continues in place.
        When ``expand_boundary`` is True the backend may widen search-space
        boundaries that the best trial pushed against.

        Parameters
        ----------
        n_trials
            Extra trials to run during the resume round.  ``None`` defers
            to the backend default.
        expand_boundary
            Allow boundary expansion on the resumed round.  ``None`` defers
            to the backend default.
        boundary_threshold
            Fraction of the dim range that counts as "at the edge" when
            ``expand_boundary`` is active.
        timeout
            Maximum seconds to block; forwarded to
            :meth:`_run_blocking_job`.  ``None`` blocks indefinitely.

        Raises
        ------
        ValueError
            If no prior tune result exists, or if any argument is out of
            its documented range (e.g. non-positive ``n_trials``,
            ``boundary_threshold`` outside ``[0.0, 1.0]``).
        RuntimeError
            If the resumed tune fails.
        """
        if not self.tune_summary:
            msg = "Cannot retune: no prior tune result on this widget. Call w.tune() first."
            raise ValueError(msg)
        # Reject bool subclasses up-front so ``w.retune(n_trials=True)``
        # cannot silently pass as ``int``.
        if n_trials is not None and (isinstance(n_trials, bool) or n_trials <= 0):
            msg = f"retune(): n_trials must be a positive int or None, got {n_trials!r}"
            raise ValueError(msg)
        if expand_boundary is not None and not isinstance(expand_boundary, bool):
            msg = (
                f"retune(): expand_boundary must be bool or None, "
                f"got {type(expand_boundary).__name__}"
            )
            raise ValueError(msg)
        if (
            isinstance(boundary_threshold, bool)
            or not isinstance(boundary_threshold, (int, float))
            or not (0.0 <= boundary_threshold <= 1.0)
        ):
            msg = (
                f"retune(): boundary_threshold must be a float in "
                f"[0.0, 1.0], got {boundary_threshold!r}"
            )
            raise ValueError(msg)
        retune_kwargs: dict[str, Any] = {
            "resume": True,
            "n_trials": n_trials,
            "expand_boundary": expand_boundary,
            "boundary_threshold": float(boundary_threshold),
        }
        return self._run_blocking_job("tune", timeout=timeout, retune_kwargs=retune_kwargs)

    def _run_blocking_job(
        self,
        job_type: str,
        *,
        timeout: float | None,
        retune_kwargs: dict[str, Any] | None = None,
    ) -> LizyWidget:
        """Run a job in a background thread and block until complete."""
        done = threading.Event()

        def _watch(change: dict[str, Any]) -> None:
            if change["new"] in ("completed", "failed"):
                done.set()

        self.observe(_watch, names=["status"])
        self._run_job(job_type, retune_kwargs=retune_kwargs)
        if self.status in ("completed", "failed"):
            done.set()
        finished = done.wait(timeout=timeout)
        self.unobserve(_watch, names=["status"])
        if not finished and timeout is not None:
            raise TimeoutError(f"{job_type}() timed out after {timeout}s")
        if self.status == "failed":
            msg = self.error.get("message", f"{job_type.title()} failed")
            raise RuntimeError(msg)
        return self

    @property
    def task(self) -> str | None:
        """Auto-detected task type (binary / multiclass / regression)."""
        return self.df_info.get("task")

    @property
    def cv_method(self) -> str:
        """Current CV strategy name."""
        return str(self.df_info.get("cv", {}).get("strategy", "kfold"))

    @property
    def cv_n_splits(self) -> int:
        """Number of CV splits."""
        return int(self.df_info.get("cv", {}).get("n_splits", 5))

    @property
    def df_shape(self) -> list[int]:
        """Shape [rows, cols] of the loaded DataFrame."""
        return list(self.df_info.get("shape", []))

    @property
    def df_columns(self) -> list[dict[str, Any]]:
        """Column metadata list from the loaded DataFrame."""
        return list(self.df_info.get("columns", []))

    def load_model(self, path: str) -> LizyWidget:
        """Load a trained model from file for inference without re-fitting."""
        self._service.load_model_from_path(path)
        self.status = "completed"
        self.available_plots = self._service.get_available_plots()
        return self

    @property
    def model_info(self) -> dict[str, Any] | None:
        """Return metadata about the currently loaded model, or None if no model."""
        model = self._service.get_model()
        if model is None:
            return None
        try:
            return self._service.model_info(model)
        except Exception as exc:
            _log.debug("model_info delegate failed, returning minimal info: %s", exc)
            return {"loaded": True}

    def load_inference(self, df: pd.DataFrame) -> LizyWidget:
        """Load a DataFrame for inference."""
        self._inference_df = df
        self.inference_result = {"status": "ready", "rows": len(df)}
        return self

    def set_config(self, config: dict[str, Any]) -> LizyWidget:
        """Set config programmatically. Canonicalizes via adapter defaults."""
        if "config_version" not in config:
            existing_version = self.config.get("config_version", 1)
            config = {**config, "config_version": existing_version}
        self.config = self._service.canonicalize_config(config)
        return self

    def get_config(self) -> dict[str, Any]:
        """Get current config."""
        return dict(self.config)

    def get_fit_summary(self) -> FitSummary | None:
        """Get last fit summary."""
        if not self.fit_summary:
            return None
        return FitSummary(
            metrics=self.fit_summary["metrics"],
            fold_count=self.fit_summary["fold_count"],
            params=self.fit_summary["params"],
        )

    def get_tune_summary(self) -> TuningSummary | None:
        """Get last tune summary."""
        if not self.tune_summary:
            return None
        return TuningSummary(
            best_params=self.tune_summary["best_params"],
            best_score=self.tune_summary["best_score"],
            trials=self.tune_summary["trials"],
            metric_name=self.tune_summary["metric_name"],
            direction=self.tune_summary["direction"],
            rounds=self.tune_summary.get("rounds", []),
            boundary_report=self.tune_summary.get("boundary_report"),
        )

    def get_model(self) -> Any:
        """Get the underlying trained model object."""
        return self._service.get_model()

    def predict(self, df: pd.DataFrame, *, return_shap: bool = False) -> PredictionSummary:
        """Run prediction and return results as PredictionSummary."""
        return self._service.predict(df, return_shap=return_shap)

    def save_model(self, path: str) -> str:
        """Save the trained model to the given path. Returns the path."""
        return self._service.save_model(path)

    def export_code(self, path: str | None = None) -> Any:
        """Export inference code for the trained model.

        Parameters
        ----------
        path:
            Output directory. If None, a temporary directory is created.

        Returns
        -------
        Path to the generated code directory (as returned by the adapter).

        Raises
        ------
        ValueError
            If no model has been trained yet.
        """
        return self._service.export_code(path)

    def save_config(self, path: str) -> None:
        """Save current full config to YAML file."""
        import yaml  # type: ignore[import-untyped]

        full_config = self._service.build_config(dict(self.config))
        with open(path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False)

    def load_config(self, path: str) -> LizyWidget:
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            loaded: dict[str, Any] = yaml.safe_load(f)

        self._apply_loaded_config(loaded)
        return self

    def _apply_loaded_config(self, loaded: dict[str, Any]) -> None:
        """Apply a parsed config dict to widget state (canonicalized via adapter)."""
        canonical = self._service.apply_loaded_config(loaded)
        self.df_info = self._service.get_df_info()
        self.config = canonical

    # ── Custom message handler (Colab polling) ──────────────

    def _handle_custom_msg(self, content: dict[str, Any], buffers: list[Any]) -> None:
        """Handle msg:custom messages from JS (e.g. poll and action requests).

        Overrides ipywidgets.Widget._handle_custom_msg(content, buffers).
        Runs on the main (shell) thread, so self.send() reliably reaches JS
        even on Google Colab where BG-thread comm is blocked.

        Message types:
        - ``{type: "poll"}``: return current job state (Colab polling fallback)
        - ``{type: "action", action_type: str, payload: dict}``: dispatch an
          action (P-023 — replaces traitlet-based JS→Python action sync which
          breaks on Colab ipywidgets 7.x)

        Note: the msg:custom schema uses ``action_type`` to name the action,
        while the traitlet path (``_on_action``) uses ``type``.  This is
        intentional — msg:custom already uses ``type`` to distinguish message
        categories ("action" vs "poll"), so the action name lives in
        ``action_type`` to avoid collision.
        """
        msg_type = content.get("type")

        if msg_type == "action":
            action_type: str = content.get("action_type", "")
            raw_payload = content.get("payload", {})
            payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
            self._dispatcher.dispatch(action_type, payload)
            return

        if msg_type != "poll":
            super()._handle_custom_msg(content, buffers)
            return

        state: dict[str, Any] = {
            "type": "job_state",
            "status": self.status,
            "progress": dict(self.progress),
            "elapsed_sec": self.elapsed_sec,
            "job_type": self.job_type,
            "job_index": self.job_index,
            "error": dict(self.error),
        }

        # Include result payloads on terminal states
        if self.status in ("completed", "failed"):
            state["fit_summary"] = dict(self.fit_summary)
            state["tune_summary"] = dict(self.tune_summary)
            state["available_plots"] = list(self.available_plots)

        try:
            self.send(state)
        except Exception as exc:
            _log.debug("poll reply failed (comm likely closed): %s", exc)

    # ── Action dispatcher ─────────────────────────────────────

    @traitlets.observe("action")
    def _on_action(self, change: dict[str, Any]) -> None:
        action: dict[str, Any] = change["new"]
        if not action:
            return
        action_type: str = action.get("type", "")
        raw_payload = action.get("payload", {})
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        self._dispatcher.dispatch(action_type, payload)

    # ── Back-compat shims (#127) ─────────────────────────────
    # Existing tests reach into ``_handle_*`` methods directly. They now
    # live on ``WidgetActionDispatcher`` as ``handle_*``; ``__getattr__``
    # below proxies the legacy private names. ``_send_plot_response`` is
    # a single explicit shim because tests call it as a positional helper.
    def _send_plot_response(
        self,
        plot_type: str,
        plotly_json: str,
        request_id: str | None,
    ) -> None:
        self._dispatcher._send_plot_response(plot_type, plotly_json, request_id)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_handle_") and name not in {"_handle_custom_msg"}:
            try:
                dispatcher = self.__dict__["_dispatcher"]
            except KeyError:
                raise AttributeError(name) from None
            target = "handle_" + name[len("_handle_") :]
            handler = getattr(dispatcher, target, None)
            if handler is not None:
                return handler
        raise AttributeError(name)

    # ── Job execution ─────────────────────────────────────────

    def _run_job(
        self,
        job_type: str,
        *,
        retune_kwargs: dict[str, Any] | None = None,
    ) -> None:
        with self._job_lock:
            # INV-A / INV-B (BLUEPRINT §6.4): a job is already running,
            # so reject this re-entry silently. Holds _job_thread to one
            # live worker and keeps status FSM linear.
            if self.status == "running":
                return

            # Pre-execution data/target checks (BLUEPRINT §6.1)
            if not self._service.has_data():
                self.error = {"code": "NO_DATA", "message": "No data loaded. Call load(df) first."}
                self.status = "failed"
                return

            if not self._service.has_target():
                self.error = {
                    "code": "NO_TARGET",
                    "message": "No target selected. Call load(df, target=...) or set_target(col).",
                }
                self.status = "failed"
                return

            # Build and validate config before committing to "running"
            # so that _job_counter is only incremented for valid jobs.
            try:
                full_config = self._service.prepare_run_config(dict(self.config), job_type=job_type)
            except Exception as exc:
                _log.error("Config build failed: %s", exc, exc_info=True)
                self.error = {"code": "CONFIG_ERROR", "message": str(exc)}
                self.status = "failed"
                return

            errors = self._service.validate_config(full_config)
            if errors:
                self.error = {
                    "code": "VALIDATION_ERROR",
                    "message": errors[0]["message"],
                    "details": errors,
                }
                self.status = "failed"
                return

            # P-035: tune snapshots are now carried inside TuningSummary
            # via JobSpec.ui_snapshot → service.tune (or
            # service.record_subprocess_tune_summary for the subprocess
            # path). The widget no longer keeps private snapshot attrs.
            tune_ui_snapshot = copy.deepcopy(dict(self.config)) if job_type == "tune" else None

            # INV-D (BLUEPRINT §6.4): cancel flag is cleared exactly once
            # per job at startup; the worker / supervisor never write it
            # back. Reading it later is safe even if a previous job ended
            # via cancel.
            self._cancel_flag.clear()
            self._job_counter += 1
            self.job_type = job_type
            self.job_index = self._job_counter
            self.status = "running"
            self.progress = {"current": 0, "total": 0, "message": f"Starting {job_type}..."}
            self.elapsed_sec = 0.0
            self.error = {}

        # Detect execution strategy lazily. ``get_execution_strategy`` forces
        # a ``lightgbm`` import on first call so ``/proc/self/maps`` reflects
        # the OpenMP runtime that the data path will actually use. On Linux
        # with libgomp this returns ``("subprocess", libomp_path)`` — the
        # default — because the worker-thread path hits libgomp's pool-
        # affinity bug (#147 reproducer: Fit ~30x slower in a worker thread,
        # multi-trial Tune compounds to 20-50x; ~30 OS threads leak per job).
        # Set ``LZW_FORCE_THREAD=1`` to opt back into the legacy in-process
        # path (e.g. for debugging or when subprocess startup overhead
        # dominates a tiny Fit). The historical ``LZW_FORCE_SUBPROCESS=1``
        # gate is retained as a no-op for backward compatibility but is no
        # longer required to enable subprocess execution.
        if self._execution_strategy is None:
            if os.environ.get("LZW_FORCE_THREAD") == "1":
                self._execution_strategy = "thread"
                self._libomp_path = None
            else:
                self._execution_strategy, self._libomp_path = get_execution_strategy()

        # Join previous worker thread to ensure its OpenMP thread pool is
        # fully cleaned up.  Without this, repeated Fit/Tune cycles accumulate
        # orphaned OS threads (one libgomp pool per worker), causing severe CPU
        # thrashing once thread count exceeds core count.
        prev = self._job_thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=5.0)

        # P-032: pick the runner strategy and hand it the immutable JobSpec.
        # The supervisor in ``_supervise`` owns the state machine + traitlet
        # plumbing for *both* runners, so the worker logic lives once.
        #
        # #154: ``SubprocessJobRunner`` cannot resume an Optuna study from a
        # fresh process today (#128 tracks the long-term fix using P-037's
        # tune-state IPC). Until that lands, transparently fall back to the
        # thread runner for re-tune jobs when subprocess is the default —
        # otherwise the happy-path ``w.tune() → w.retune()`` flow surfaces
        # ``RETUNE_SUBPROCESS_UNSUPPORTED`` on every default install. Initial
        # tune (``retune_kwargs is None``) still uses subprocess to keep the
        # P-036 / #147 perf win.
        runner: JobRunner
        if self._execution_strategy == "subprocess" and retune_kwargs is None:
            runner = SubprocessJobRunner(self._service, libomp_path=self._libomp_path)
        elif self._execution_strategy == "subprocess" and retune_kwargs is not None:
            if not self._retune_fallback_warned:
                _log.info(
                    "re-tune temporarily falls back to thread runner; "
                    "subprocess re-tune resume is tracked by issue #128"
                )
                self._retune_fallback_warned = True
            runner = ThreadJobRunner(self._service)
        else:
            runner = ThreadJobRunner(self._service)
        spec = JobSpec(
            job_type=job_type,
            config=full_config,
            retune_kwargs=retune_kwargs,
            ui_snapshot=tune_ui_snapshot,
        )
        thread = threading.Thread(
            target=self._supervise,
            args=(runner, spec),
            daemon=False,
        )
        self._job_thread = thread
        thread.start()

    def _supervise(self, runner: JobRunner, spec: JobSpec) -> None:
        """Run *spec* through *runner* and own the state-machine transitions.

        Single source of truth for status / traitlet updates during job
        execution — both ``ThreadJobRunner`` and ``SubprocessJobRunner``
        use this supervisor. Runtime invariants (BLUEPRINT §6.4 INV-A..F)
        are encoded inline as ``assert`` guards; running under
        ``python -O`` strips them, so production behaviour is unchanged.
        """
        # INV-A entry: a supervisor only runs when status was set to "running"
        # by ``_run_job`` under the job lock. Any other status means the
        # caller violated the FSM (e.g., a stray supervisor thread).
        assert self.status == "running", (
            f"INV-A violated: supervisor entered with status={self.status!r}"
        )
        # INV-D entry: ``_run_job`` clears the cancel flag inside the job lock
        # immediately before spawning this thread, so it must be clear here.
        # If it's set we'd cancel the new job before its first tick.
        assert not self._cancel_flag.is_set(), (
            "INV-D violated: cancel flag carried over into a new job"
        )

        start = time.monotonic()
        timer_stop = threading.Event()
        # INV-E: track the highest progress.round seen so we can assert
        # monotonicity across ``on_progress`` callbacks within this job.
        last_round: list[int] = [0]

        def tick_elapsed() -> None:
            while not timer_stop.is_set():
                self.elapsed_sec = round(time.monotonic() - start, 1)
                timer_stop.wait(1.0)

        timer = threading.Thread(target=tick_elapsed, daemon=True)
        timer.start()

        is_subprocess = getattr(runner, "kind", "thread") == "subprocess"

        def on_progress(
            current: int,
            total: int,
            message: str,
            **extra: Any,
        ) -> None:
            """Forward progress to the traitlet; raise to cancel (thread runner)."""
            # Subprocess runner cancels via SIGTERM, so the parent
            # process does not need to raise InterruptedError here. The
            # in-process thread runner relies on this raise to abort the
            # adapter's polling loop.
            if not is_subprocess and self._cancel_flag.is_set():
                raise InterruptedError("Job cancelled by user")
            # INV-E: round must be monotonic non-decreasing within a job.
            round_no = extra.get("round")
            if isinstance(round_no, int):
                assert round_no >= last_round[0], (
                    f"INV-E violated: round regressed {last_round[0]} -> {round_no}"
                )
                last_round[0] = round_no
            self.progress = _build_progress_payload(current, total, message, extra)
            self.elapsed_sec = round(time.monotonic() - start, 1)

        try:
            result: JobResult = runner.run(spec, on_progress, self._cancel_flag)
            self._apply_job_result(result)
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.status = "completed"
        except RetuneSubprocessUnsupportedError as exc:
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.error = {
                "code": "RETUNE_SUBPROCESS_UNSUPPORTED",
                "message": str(exc),
            }
            self.status = "failed"
        except InterruptedError:
            # INV-D (BLUEPRINT §6.4): cancel during running -> failed/CANCELLED.
            self.elapsed_sec = round(time.monotonic() - start, 1)
            self.error = {"code": "CANCELLED", "message": "Job cancelled by user"}
            self.status = "failed"
        except Exception as exc:  # noqa: BLE001 — outer-most boundary
            self.elapsed_sec = round(time.monotonic() - start, 1)
            code = self._classify_job_error(exc, subprocess=is_subprocess)
            _log.error("Job %s failed (%s): %s", spec.job_type, code, exc, exc_info=True)
            self.error = {"code": code, "message": str(exc)}
            self.status = "failed"
        finally:
            timer_stop.set()
            timer.join(timeout=2.0)
            # INV-A exit: status FSM only allows terminal states once the
            # supervisor returns. Catches any future code path that forgets
            # to write the status.
            assert self.status in {"completed", "failed"}, (
                f"INV-A violated: terminal status invalid ({self.status!r})"
            )

    def _apply_job_result(self, result: JobResult) -> None:
        """Project a runner's :class:`JobResult` onto traitlets."""
        if result.fit_summary:
            normalized = self._normalize_metrics(result.eval_table)
            self.fit_summary = {
                "metrics": normalized if normalized else result.fit_summary.get("metrics", {}),
                "fold_count": result.fit_summary.get("fold_count", 0),
                "fold_details": result.split_summary or result.fit_summary.get("fold_details", []),
                "params": result.fit_summary.get("params", []),
            }
        if result.tune_summary:
            # INV-F (BLUEPRINT §6.4): boundary_report.dims must list each
            # search-space dim exactly once. Check before publishing so a
            # backend regression surfaces here instead of as silent UI weirdness.
            br = result.tune_summary.get("boundary_report")
            if isinstance(br, dict):
                dims = br.get("dims") or []
                names = [d.get("name") for d in dims if isinstance(d, dict) and d.get("name")]
                assert len(names) == len(set(names)), (
                    f"INV-F violated: boundary_report.dims has duplicates: {names}"
                )
            self.tune_summary = result.tune_summary
            # After tune, evaluate_table may exist if the model was
            # implicitly fitted on the best params — surface it as a
            # fit_summary too so the Results tab can render the score
            # table even without a separate fit run.
            if result.eval_table:
                normalized = self._normalize_metrics(result.eval_table)
                if normalized:
                    self.fit_summary = {
                        "metrics": normalized,
                        "fold_count": len(result.split_summary),
                        "fold_details": result.split_summary,
                        "params": [],
                    }
        if result.available_plots:
            self.available_plots = result.available_plots

    @staticmethod
    def _classify_job_error(exc: BaseException, *, subprocess: bool) -> str:
        """Map a worker exception to an error code for the widget banner."""
        try:
            mod = getattr(type(exc), "__module__", "") or ""
            if "lizyml" in mod.lower():
                return "BACKEND_ERROR"
            # Subprocess errors arrive wrapped as RuntimeError("[ExcType] msg")
            # — the prefix carries the original module name so we can still
            # detect lizyml drift inside that envelope.
            msg = str(exc).lower()
        except Exception:  # noqa: BLE001
            msg = ""
        if "lizyml" in msg:
            return "BACKEND_ERROR"
        return "SUBPROCESS_ERROR" if subprocess else "INTERNAL_ERROR"

    # ── Config helpers ─────────────────────────────────────────

    @staticmethod
    def _normalize_metrics(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert evaluate_table records to {metric: {is, oos, oos_std}} for ScoreTable.

        evaluate_table() returns records with 'index' (metric name), 'if_mean', 'oof',
        and 'fold_0'...'fold_N-1' columns.
        """
        result: dict[str, Any] = {}
        if not eval_records:
            return result
        for record in eval_records:
            metric_name = str(record.get("index", record.get("metric", "")))
            if not metric_name:
                continue
            entry: dict[str, Any] = {}
            if "if_mean" in record:
                entry["is"] = record["if_mean"]
            if "oof" in record:
                entry["oos"] = record["oof"]
            # Compute OOS Std from per-fold columns
            fold_values = [
                v
                for k, v in record.items()
                if k.startswith("fold_") and isinstance(v, (int, float))
            ]
            if len(fold_values) > 1:
                entry["oos_std"] = statistics.stdev(fold_values)
            result[metric_name] = entry
        return result
