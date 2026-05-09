"""Regression tests for issue #154 — subprocess-default retune regressions.

After P-036 made subprocess the default execution strategy on libgomp hosts,
``w.tune() → w.retune()`` happy path failed by default with
``RETUNE_SUBPROCESS_UNSUPPORTED``. The remediation message also pointed
users to the post-P-036 no-op ``LZW_FORCE_SUBPROCESS=1`` env var.

These tests pin the post-fix contract:

- INV-#154-A: under the default strategy, ``w.retune(...)`` after a successful
  ``w.tune()`` completes via an automatic fallback to the thread runner.
- INV-#154-B: any remaining subprocess-rejection error message must point to
  ``LZW_FORCE_THREAD=1`` (the actual opt-out), not the no-op env var.
- INV-#154-C: subprocess Fit overhead is bounded (slow regression — pinned
  numerically so future strategy changes that double the cost fail CI).

Test failures on commit ``8d169f5`` (current ``develop`` tip), pass after the
hotfix lands.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.types import BackendInfo


def _make_widget_with_data() -> Any:
    """Build a LizyWidget with the real LizyMLAdapter wrapped behind a mock so
    canonical config / backend-contract calls work but ML library calls can
    be intercepted by the test."""
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


# ---------------------------------------------------------------------------
# INV-#154-A: retune auto-fallback under default subprocess strategy
# ---------------------------------------------------------------------------


class TestRetuneAutoFallbackUnderDefaultStrategy:
    """When strategy is detected as ``subprocess``, ``_run_job(retune)`` must
    transparently fall back to ``ThreadJobRunner`` so the user-visible flow
    succeeds without env-var fiddling."""

    def test_retune_with_subprocess_default_uses_thread_runner(self) -> None:
        """The supervisor must end in ``completed``, not ``failed`` with
        ``RETUNE_SUBPROCESS_UNSUPPORTED``."""
        with patch(
            "lizyml_widget.widget.get_execution_strategy",
            return_value=("subprocess", "/usr/lib/libomp5.so"),
        ):
            w = _make_widget_with_data()

            # Seed prior tune so retune() passes the precondition.
            w.tune_summary = {
                "best_params": {"learning_rate": 0.05},
                "best_score": 0.9,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
                "rounds": [],
                "boundary_report": None,
            }
            # Replace service.tune so retune doesn't hit lightgbm.
            captured: list[dict[str, Any]] = []

            def mock_tune(config: dict[str, Any], *, on_progress: Any = None, **kwargs: Any) -> Any:
                captured.append(kwargs)
                from lizyml_widget.types import TuningSummary

                return TuningSummary(
                    best_params={},
                    best_score=0.91,
                    trials=[],
                    metric_name="auc",
                    direction="maximize",
                    rounds=[],
                    boundary_report=None,
                )

            w._service.tune = mock_tune  # type: ignore[assignment]
            w._service._tune_model = MagicMock()  # P-028 prerequisite

            w.retune(n_trials=5)

            # Wait for the supervisor thread to complete.
            if w._job_thread:
                w._job_thread.join(timeout=10)

            assert w.status == "completed", (
                f"Expected retune to complete via thread fallback; "
                f"got status={w.status!r} error={w.error!r}"
            )
            assert w.error.get("code") != "RETUNE_SUBPROCESS_UNSUPPORTED"
            # The thread runner forwards resume kwargs via _run_tune.
            assert captured, "service.tune was never called"
            assert captured[0].get("resume") is True
            assert captured[0].get("n_trials") == 5

    def test_initial_tune_still_uses_subprocess_under_default(self) -> None:
        """The fallback must be retune-only — initial tune must NOT regress
        back to thread mode (that would re-introduce the #147 30x slowdown)."""

        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", None),
            ),
            patch("lizyml_widget.widget.SubprocessJobRunner") as mock_sp,
            patch("lizyml_widget.widget.ThreadJobRunner") as mock_thread,
        ):
            mock_sp_inst = mock_sp.return_value
            mock_sp_inst.kind = "subprocess"
            mock_sp_inst.run.return_value = MagicMock(
                job_type="tune",
                fit_summary={},
                tune_summary={
                    "best_params": {},
                    "best_score": 0.9,
                    "trials": [],
                    "metric_name": "auc",
                    "direction": "maximize",
                },
                eval_table=[],
                split_summary=[],
                available_plots=[],
                model_path=None,
            )

            w = _make_widget_with_data()
            w._run_job("tune")
            if w._job_thread:
                w._job_thread.join(timeout=10)

            # Initial tune used subprocess runner.
            mock_sp.assert_called_once()
            mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# INV-#154-B: error message points to the actual opt-out env var
# ---------------------------------------------------------------------------


class TestRejectionMessageRemediation:
    """Even when subprocess retune is rejected (e.g. user explicitly forces
    subprocess), the remediation guidance must reference ``LZW_FORCE_THREAD``,
    not the no-op ``LZW_FORCE_SUBPROCESS``."""

    def test_rejection_message_mentions_lzw_force_thread(self) -> None:
        from lizyml_widget.job_runner import (
            JobSpec,
            RetuneSubprocessUnsupportedError,
            SubprocessJobRunner,
        )

        w = _make_widget_with_data()
        runner = SubprocessJobRunner(w._service)
        spec = JobSpec(
            job_type="tune",
            config={"config_version": 1},
            retune_kwargs={"resume": True, "n_trials": 10},
        )
        with pytest.raises(RetuneSubprocessUnsupportedError) as exc_info:
            runner.run(spec, on_progress=lambda *a, **kw: None, cancel_event=threading.Event())

        msg = str(exc_info.value)
        # The historical wording referenced the no-op env var; the new
        # message must point users to the actual opt-out.
        assert "LZW_FORCE_THREAD" in msg, (
            f"Expected remediation message to mention LZW_FORCE_THREAD; got: {msg!r}"
        )
        assert "LZW_FORCE_SUBPROCESS" not in msg, (
            f"LZW_FORCE_SUBPROCESS is a no-op after P-036 and must not appear "
            f"in the remediation message; got: {msg!r}"
        )


# ---------------------------------------------------------------------------
# INV-#154-C: subprocess Fit overhead is bounded
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_subprocess_fit_overhead_is_bounded() -> None:
    """End-to-end: under default strategy on a libgomp host, a small Fit
    must complete within a generous upper bound. This pins the overhead so
    a catastrophic strategy regression (e.g. double-spawning, recursive
    process tree) fails CI.

    Threshold rationale: measured 1.6-1.9s standalone on a WSL2 host
    (5000 rows x 30 cols). When this test runs immediately after
    ``test_reg_147_openmp_perf.py``, ``subprocess.Popen`` slows
    dramatically (observed 22s) because the parent process retains
    ~120 leaked libgomp threads from the perf test's worker-thread
    sequence — the cost is in fork()/Popen, not in the actual Fit. The
    bound is set high enough (30s) to absorb that interaction while still
    catching a genuine 15x+ regression in the subprocess code path
    (BLUEPRINT §3.7.1 design budget is ~500ms-2s; anything past 30s
    indicates a fundamentally broken strategy).
    """
    if not __import__("sys").platform.startswith("linux"):
        pytest.skip("subprocess default applies only on Linux + libgomp")

    from lizyml_widget.openmp_detect import _reset_libgomp_cache, get_execution_strategy

    _reset_libgomp_cache()
    os.environ.pop("LZW_FORCE_THREAD", None)
    strategy, _ = get_execution_strategy()
    if strategy != "subprocess":
        pytest.skip(f"strategy is {strategy}; bound applies only to subprocess default")

    import numpy as np

    n = 5000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, 30), dtype=np.float32))
    df.columns = [f"f{i}" for i in range(30)]
    df["y"] = (rng.random(n) > 0.5).astype(int)

    from lizyml_widget import LizyWidget

    w = LizyWidget()
    w.load(df, target="y")
    t0 = time.perf_counter()
    w.fit()
    if w._job_thread:
        w._job_thread.join(timeout=60)
    elapsed = time.perf_counter() - t0

    UPPER_BOUND_S = 30.0
    assert w.status == "completed", f"Fit failed: error={w.error!r}"
    assert elapsed < UPPER_BOUND_S, (
        f"Subprocess Fit overhead regressed: elapsed={elapsed:.2f}s "
        f"exceeds bound {UPPER_BOUND_S}s. Standalone baseline is ~2s; "
        f"a 15x regression indicates a fundamentally broken strategy "
        f"(double-spawn, recursive process tree, or massive thread leak "
        f"in the parent). Investigate strategy detection / subprocess "
        f"startup / DataFrame pickling."
    )
