"""Regression tests for issue #154 — subprocess-default retune regressions.

After P-036 made subprocess the default execution strategy on libgomp hosts,
``w.tune() → w.retune()`` happy path failed by default with
``RETUNE_SUBPROCESS_UNSUPPORTED``. PR #155 patched this with a thread
fallback; #156 / P-038 then replaced the fallback with subprocess retune
resume because the thread fallback was unsafe whenever any libgomp
parallel region had been entered on the parent main thread.

These tests pin the post-P-038 contract:

- INV-#154-A: under the default strategy, ``w.retune(...)`` after a
  successful ``w.tune()`` completes via the SAME ``SubprocessJobRunner``
  used by initial tune (no thread fallback any more).
- INV-#154-C: subprocess Fit overhead is bounded (slow regression — pinned
  numerically so future strategy changes that double the cost fail CI).

INV-#154-B (rejection-message remediation) was retired alongside
``RetuneSubprocessUnsupportedError`` itself in P-038. The full
subprocess retune contract now lives in
``test_reg_156_subprocess_retune.py``.
"""

from __future__ import annotations

import os
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
# INV-#154-A: retune routes through the SubprocessJobRunner under default
# (P-038 replaced PR #155's thread fallback with subprocess retune resume).
# ---------------------------------------------------------------------------


class TestRetuneRoutesToSubprocessRunnerUnderDefault:
    """Under the default subprocess strategy, ``w.retune(...)`` must use the
    same ``SubprocessJobRunner`` as initial tune. PR #155's thread fallback
    is gone."""

    def test_retune_with_subprocess_default_uses_subprocess_runner(self) -> None:
        with (
            patch(
                "lizyml_widget.widget.get_execution_strategy",
                return_value=("subprocess", "/usr/lib/libomp5.so"),
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
                    "best_score": 0.91,
                    "trials": [],
                    "metric_name": "auc",
                    "direction": "maximize",
                    "rounds": [],
                    "boundary_report": None,
                },
                eval_table=[],
                split_summary=[],
                available_plots=["optimization-history"],
                model_path=None,
            )

            w = _make_widget_with_data()
            w.tune_summary = {
                "best_params": {"learning_rate": 0.05},
                "best_score": 0.9,
                "trials": [],
                "metric_name": "auc",
                "direction": "maximize",
                "rounds": [],
                "boundary_report": None,
            }
            w._service._tune_model = MagicMock()  # P-028 prerequisite

            w.retune(n_trials=5)
            if w._job_thread:
                w._job_thread.join(timeout=10)

            mock_sp.assert_called_once()
            mock_thread.assert_not_called()
            assert w.status == "completed", (
                f"Expected retune to complete via subprocess runner; "
                f"got status={w.status!r} error={w.error!r}"
            )
            assert w.error.get("code") != "RETUNE_SUBPROCESS_UNSUPPORTED"

    def test_initial_tune_still_uses_subprocess_under_default(self) -> None:
        """Initial tune must continue to use subprocess (the #147 perf win
        must not regress)."""

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
