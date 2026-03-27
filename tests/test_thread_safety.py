"""Tests for thread safety in widget job execution (C-1, C-2, C-3).

C-1: _run_job TOCTOU guard — double invocation must not spawn two workers.
C-2: WidgetService._model must be protected during concurrent access.
C-3: _tune_config_snapshot read in apply_best_params must be safe.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo


def _make_widget() -> Any:
    """Create a LizyWidget with mocked LizyML backend."""
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
    return w


def _load_data(w: Any) -> None:
    """Load sample data and set target."""
    df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
    w.load(df, target="y")


class TestRunJobAtomicGuard:
    """C-1: Only one job can be spawned even under concurrent calls."""

    def test_double_fit_only_starts_one_job(self) -> None:
        """Calling _handle_fit while a job is running should be rejected."""
        w = _make_widget()
        _load_data(w)

        # Block adapter.fit so the first job stays in "running" state
        fit_entered = threading.Event()
        fit_release = threading.Event()

        def blocking_fit(_model: Any, *, on_progress: Any = None) -> Any:
            fit_entered.set()
            fit_release.wait(timeout=10)
            from lizyml_widget.types import FitSummary

            return FitSummary(metrics={}, fold_count=1, fold_details=[], params=[])

        w._service._adapter.fit = blocking_fit  # type: ignore[assignment]

        # Start first fit (will block in adapter.fit)
        w._handle_fit({})
        fit_entered.wait(timeout=5)
        assert w.status == "running", f"Expected running, got {w.status}"

        # Second fit while first is running — must be rejected
        w._handle_fit({})

        # Release the first job
        fit_release.set()
        if w._job_thread is not None:
            w._job_thread.join(timeout=10)

        assert w._job_counter == 1, f"Only one job should have started, got {w._job_counter}"

    def test_concurrent_fit_from_two_threads(self) -> None:
        """Two threads calling _run_job must result in only one executing."""
        w = _make_widget()
        _load_data(w)

        # Block adapter.fit so jobs stay in "running"
        fit_entered = threading.Event()
        fit_release = threading.Event()

        def blocking_fit(_model: Any, *, on_progress: Any = None) -> Any:
            fit_entered.set()
            fit_release.wait(timeout=10)
            from lizyml_widget.types import FitSummary

            return FitSummary(metrics={}, fold_count=1, fold_details=[], params=[])

        w._service._adapter.fit = blocking_fit  # type: ignore[assignment]

        t1 = threading.Thread(target=w._handle_fit, args=({},))
        t2 = threading.Thread(target=w._handle_fit, args=({},))

        t1.start()
        # Wait for first job to actually enter adapter.fit
        fit_entered.wait(timeout=5)
        t2.start()
        t2.join(timeout=10)

        # Release the first job and wait
        fit_release.set()
        t1.join(timeout=10)
        if w._job_thread is not None:
            w._job_thread.join(timeout=10)

        assert w._job_counter == 1, f"Expected 1 job, got {w._job_counter}"


class TestModelLock:
    """C-2: Service._model access should be thread-safe."""

    def test_service_has_model_lock(self) -> None:
        """WidgetService should have a _model_lock attribute."""
        w = _make_widget()
        assert hasattr(w._service, "_model_lock"), (
            "WidgetService must have _model_lock for thread-safe model access"
        )

    def test_predict_acquires_model_lock(self) -> None:
        """predict() must acquire _model_lock before reading _model."""
        w = _make_widget()
        _load_data(w)

        model_lock = getattr(w._service, "_model_lock", None)
        if model_lock is None:
            pytest.skip("_model_lock not yet implemented")

        # Hold the lock on the main thread, then try predict on another thread
        lock_acquired = threading.Event()
        lock_released = threading.Event()
        predict_completed = threading.Event()
        predict_blocked = True

        def hold_lock() -> None:
            with model_lock:
                lock_acquired.set()
                lock_released.wait(timeout=5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        lock_acquired.wait(timeout=2)

        def try_predict() -> None:
            nonlocal predict_blocked
            with contextlib.suppress(Exception):
                # This should block on _model_lock inside predict()
                w._service.predict(pd.DataFrame({"x": range(10)}))
            predict_blocked = False
            predict_completed.set()

        t = threading.Thread(target=try_predict)
        t.start()

        # Give predict thread time to attempt lock acquisition
        time.sleep(0.3)

        # predict should still be blocked (lock held by holder)
        assert predict_blocked, "predict() should block while _model_lock is held"

        # Release the lock — predict can proceed
        lock_released.set()
        predict_completed.wait(timeout=5)
        holder.join(timeout=2)
        t.join(timeout=5)

        assert not predict_blocked, "predict() should have completed after lock released"


class TestTuneConfigSnapshotProtection:
    """C-3: _tune_config_snapshot should be safely read."""

    def test_apply_best_params_reads_snapshot_safely(self) -> None:
        """apply_best_params should read a consistent snapshot."""
        w = _make_widget()
        _load_data(w)

        # Set a snapshot as if tune completed
        w._tune_config_snapshot = {"model": {"params": {"learning_rate": 0.1}}}

        # apply_best_params should read the snapshot
        w._handle_apply_best_params({"params": {"learning_rate": 0.05}})

        # Config should be updated
        lr = w.config.get("model", {}).get("params", {}).get("learning_rate")
        assert lr is not None, "learning_rate should have been applied"
