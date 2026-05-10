"""Parameterised libgomp perf regression grid for P-039 / issue #160.

Phase 1 / Layer 3 of the systemic prevention plan declared in P-039
(see HISTORY.md). The four historical regressions in the libgomp
pool-affinity family (#147 / #154 / #156 / #158) all share one root
cause — GCC libgomp binds the OpenMP thread pool to the first thread
that enters a parallel region, and any subsequent worker-thread call
gets a 10-50x penalty (GCC bug #108494). Each prior regression test
pinned a *specific* path; none of them covered the cross-product of
"what happened on the parent main thread before the next ML op".

This grid asserts the catastrophe-class invariant directly:

INV-#160-A: for every combination of {intermediate parent-thread op}
× {next ML op}, the per-trial wall-clock of the next op stays within
1.5x of the baseline subprocess tune. A failure indicates that some
parent-main-thread libgomp parallel region has bound the pool
affinity such that a follow-up worker-thread Tune/Fit/Retune is
running on the wrong thread.

The dataset is intentionally smaller than the existing
``test_reg_147_openmp_perf.py`` / ``test_reg_156_subprocess_retune.py``
tests so the grid fits inside the GitHub Actions ``libgomp-perf`` CI
job budget (~5-10 min total). Catastrophe is *not* dataset-size
dependent (libgomp pool affinity is a per-region constant cost), so
the 1.5x ratio remains expressive on small data.

The grid is gated on Linux + libgomp; non-libgomp hosts skip
because the bound is meaningless without the underlying GCC bug.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Skip helpers — mirror the openmp_detect logic so the grid doesn't silently
# pass on a fresh kernel where lightgbm has not been imported yet.
# ---------------------------------------------------------------------------


def _libgomp_loaded() -> bool:
    if sys.platform != "linux":
        return False
    try:
        import lightgbm  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        with open("/proc/self/maps") as f:
            return any("libgomp" in line for line in f)
    except OSError:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="libgomp pool-affinity bug is Linux-only",
    ),
    pytest.mark.skipif(
        not _libgomp_loaded(),
        reason="libgomp not loaded — 1.5x bound only meaningful on libgomp hosts",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_perf_widget(n_rows: int = 5_000, n_trials: int = 2) -> Any:
    """Build a minimal LizyWidget for the grid.

    Smaller than ``test_reg_156_subprocess_retune._make_perf_widget`` so
    the grid fits the CI job budget. Catastrophe-class regressions are
    not dataset-size dependent — a 5k row tune is enough to drive 10-50x
    pool-affinity slowdowns on libgomp hosts.
    """
    import numpy as np

    from lizyml_widget.widget import LizyWidget

    rng = np.random.default_rng(0)
    n_cols = 20
    cols = {f"f{i}": rng.random(n_rows, dtype=np.float64) for i in range(n_cols)}
    cols["y"] = (rng.random(n_rows) > 0.5).astype(int)
    df = pd.DataFrame(cols)

    w = LizyWidget()
    w.load(df, target="y")
    cfg = dict(w.config)
    tuning = dict(cfg.get("tuning") or {})
    optuna = dict(tuning.get("optuna") or {})
    params = dict(optuna.get("params") or {})
    params["n_trials"] = n_trials
    optuna["params"] = params
    tuning["optuna"] = optuna
    cfg["tuning"] = tuning
    w.set_config(cfg)
    return w


def _real_strategy_patch() -> Any:
    """Override conftest's autouse thread-strategy patch.

    ``conftest.py`` patches ``lizyml_widget.widget.get_execution_strategy`` to
    return ``("thread", None)`` for the entire suite. The grid needs the
    real production strategy ``("subprocess", libomp_path)`` on libgomp
    hosts.
    """
    from lizyml_widget.openmp_detect import _reset_libgomp_cache, get_execution_strategy

    _reset_libgomp_cache()
    return patch(
        "lizyml_widget.widget.get_execution_strategy",
        side_effect=get_execution_strategy,
    )


# ---------------------------------------------------------------------------
# Intermediate-op helpers — drive the parent main thread into a libgomp
# parallel region in the same way real users would (predict / fit + predict).
# ---------------------------------------------------------------------------


def _run_intermediate_noop(_w: Any) -> None:
    """No parent-main-thread libgomp activity. Baseline cell."""


def _run_intermediate_main_thread_predict(_w: Any) -> None:
    """Reproduce ``w.predict(test_df)`` semantics without holding a model.

    A ``booster.predict`` call on the parent main thread is sufficient to
    bind libgomp's pool affinity to that thread (#156 root cause).
    """
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.random((500, 20))
    y = (rng.random(500) > 0.5).astype(int)

    booster_holder: dict[str, Any] = {}

    def _fit_in_worker() -> None:
        ds = lgb.Dataset(X, label=y)
        booster_holder["b"] = lgb.train(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbose": -1,
                "num_threads": 0,
            },
            ds,
            num_boost_round=20,
        )

    th = threading.Thread(target=_fit_in_worker, daemon=False)
    th.start()
    th.join()
    booster = booster_holder["b"]
    for _ in range(3):
        _ = booster.predict(X)


def _run_intermediate_main_thread_fit_predict(_w: Any) -> None:
    """Heavier variant: both fit AND predict on the parent main thread.

    Some users call ``lgb.train`` directly from notebook cells (outside
    the widget). This cell pins that path: no worker offload at all.
    """
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(7)
    X = rng.random((500, 20))
    y = (rng.random(500) > 0.5).astype(int)
    ds = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "num_threads": 0,
        },
        ds,
        num_boost_round=20,
    )
    for _ in range(3):
        _ = booster.predict(X)


_INTERMEDIATES = {
    "noop": _run_intermediate_noop,
    "main_thread_predict": _run_intermediate_main_thread_predict,
    "main_thread_fit_predict": _run_intermediate_main_thread_fit_predict,
}


# ---------------------------------------------------------------------------
# Op runners — both baseline and measured runs use the SAME op type so the
# comparison cancels out fixed subprocess startup / dataset reload / model
# serialisation overhead. Only catastrophe-level (libgomp pool affinity)
# slowdowns remain visible in the ratio.
# ---------------------------------------------------------------------------


_RETUNE_TRIALS = 2
_TUNE_PRIMER_TRIALS = 2


def _prime_for_retune_cell(w: Any) -> None:
    """Run a clean tune so the widget has tune state for the retune cells.

    The actual baseline measurement is the FIRST retune (post-tune); the
    measured run is the SECOND retune (post-intermediate). Both retunes
    pay identical fixed overheads (subprocess startup, study restore,
    model serialisation), so the ratio cleanly isolates libgomp
    pool-affinity effects.
    """
    w.tune(timeout=600)
    assert w.status == "completed", f"Primer tune failed: {w.error!r}"
    assert w._execution_strategy == "subprocess", (
        f"Pre-condition: strategy must be subprocess on libgomp host, got {w._execution_strategy!r}"
    )


def _run_retune_per_trial(w: Any) -> float:
    t0 = time.perf_counter()
    w.retune(n_trials=_RETUNE_TRIALS, timeout=600)
    elapsed = time.perf_counter() - t0
    assert w.status == "completed", f"Retune failed: {w.error!r}"
    return elapsed / _RETUNE_TRIALS


def _run_fit_seconds(w: Any) -> float:
    t0 = time.perf_counter()
    w.fit(timeout=600)
    elapsed = time.perf_counter() - t0
    assert w.status == "completed", f"Fit failed: {w.error!r}"
    return elapsed


_OP_RUNNERS = {
    "retune": _run_retune_per_trial,
    "fit": _run_fit_seconds,
}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intermediate",
    ["noop", "main_thread_predict", "main_thread_fit_predict"],
)
@pytest.mark.parametrize("next_op", ["retune", "fit"])
def test_no_libgomp_perf_catastrophe(intermediate: str, next_op: str) -> None:
    """INV-#160-A: cross-product perf grid.

    For every combination of ``{intermediate parent-thread op}`` ×
    ``{next ML op}``, per-unit wall-clock of the post-intermediate op
    must stay within 1.5x of a pre-intermediate op-matched baseline.
    Both runs are the SAME op type on the SAME widget, so subprocess
    startup / dataset reload / model serialisation overheads cancel
    out and only catastrophe-level (libgomp pool affinity) slowdowns
    remain visible in the ratio.

    Test shape:

      [tune primer if next_op == retune]
      run #1 (clean baseline)            <- baseline_per_unit
      intermediate parent-main-thread op
      run #2 (measured)                  <- next_per_unit
      assert next / baseline < 1.5

    Cells that exercise historical regressions:

    * ``(noop, retune)`` — #154 clean retune path; PR #155 broke this
      with the spurious ``RetuneSubprocessUnsupportedError``. P-038
      pins it within 1.2x in the dedicated #156 test; this grid uses
      the looser 1.5x bound for CI runner variance.
    * ``(main_thread_predict, retune)`` — #156 catastrophe path;
      ``w.tune() → w.predict(test_df) → w.retune()`` regressed to ~11x
      under PR #155 thread fallback. P-038 keeps it within 1.2x; the
      grid pins the 1.5x catastrophe ceiling.
    * ``(main_thread_fit_predict, fit)`` — covers a pattern where the
      user runs lightgbm directly in a cell, then later kicks off a
      widget Fit that gets stuck on the bound pool.
    """
    os.environ.pop("LZW_FORCE_THREAD", None)

    with _real_strategy_patch():
        w = _make_perf_widget()

        if next_op == "retune":
            _prime_for_retune_cell(w)

        baseline_per_unit = _OP_RUNNERS[next_op](w)

        # Intermediate parent-main-thread op (or noop). After this point
        # the libgomp pool may be bound to the parent main thread.
        _INTERMEDIATES[intermediate](w)

        # Measured run — same op as baseline so fixed overheads amortise
        # identically and only catastrophe-level slowdowns survive.
        next_per_unit = _OP_RUNNERS[next_op](w)

    ratio = next_per_unit / baseline_per_unit if baseline_per_unit > 0 else float("inf")

    UPPER_BOUND = 1.5
    assert ratio < UPPER_BOUND, (
        f"INV-#160-A regression: next_op={next_op!r} after "
        f"intermediate={intermediate!r} per-unit wall {next_per_unit:.2f}s "
        f"is {ratio:.2f}x op-matched clean baseline {baseline_per_unit:.2f}s "
        f"(bound {UPPER_BOUND}x). "
        f"This indicates a libgomp pool-affinity catastrophe — most "
        f"likely a new code path that runs an OpenMP parallel region "
        f"on the parent main thread before the next ML op fires. See "
        f"P-039 in HISTORY.md."
    )
