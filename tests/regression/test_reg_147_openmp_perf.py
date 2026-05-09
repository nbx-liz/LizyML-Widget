"""Regression tests for issue #147 / P-036.

Pins the libgomp pool-affinity numbers that justify defaulting to
subprocess execution on Linux + libgomp. These tests are marked ``slow``
and are excluded from the default ``pytest`` run; opt-in via
``pytest -m slow``.

Two assertions:

* ``test_libgomp_pool_affinity_still_manifests_in_thread`` — the bug we
  work around still exists (worker-thread Fit is dramatically slower
  than main-thread Fit). If this ever flips to "fast", investigate
  whether the workaround is still needed.
* ``test_default_strategy_is_subprocess_on_libgomp`` — the resulting
  default execution strategy must be ``"subprocess"`` on this platform,
  i.e., the env-var gate has not regressed back to forcing ``"thread"``.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest


def _count_threads() -> int:
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return -1


def _libgomp_loaded() -> bool:
    try:
        with open("/proc/self/maps") as f:
            return any("libgomp" in line for line in f)
    except OSError:
        return False


@pytest.mark.slow
def test_libgomp_pool_affinity_still_manifests_in_thread() -> None:
    """Confirm issue #147 reproducer: worker-thread Fit is much slower than
    main-thread Fit when libgomp is loaded.

    Fail-mode: ratio < 5.0x → either lightgbm/libgomp fixed the affinity
    bug or the test environment changed (LD_PRELOAD libomp, libomp-only
    install, etc.). Re-evaluate whether subprocess default is still needed.
    """
    if sys.platform != "linux":
        pytest.skip("libgomp affinity bug is Linux-only")

    import lightgbm as lgb
    import numpy as np

    if not _libgomp_loaded():
        pytest.skip(
            "libgomp not loaded in this environment — bug cannot manifest "
            "(likely LD_PRELOAD libomp or libomp-only build of lightgbm)"
        )

    n = 5000
    rng = np.random.default_rng(0)
    X = rng.random((n, 50), dtype=np.float32)
    y = (rng.random(n) > 0.5).astype(int)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "num_threads": 0,
        "num_iterations": 100,
    }

    # Main-thread baseline — first one binds the libgomp pool.
    ds = lgb.Dataset(X, label=y)
    t0 = time.perf_counter()
    lgb.train(params, ds, num_boost_round=100)
    main = time.perf_counter() - t0
    threads_after_main = _count_threads()

    # Worker-thread Fit — affinity bug means a fresh team cannot use the
    # bound pool and falls back to a serial-ish path.
    result: dict[str, float] = {}

    def _worker() -> None:
        wds = lgb.Dataset(X, label=y)
        t = time.perf_counter()
        lgb.train(params, wds, num_boost_round=100)
        result["t"] = time.perf_counter() - t

    th = threading.Thread(target=_worker, daemon=False)
    th.start()
    th.join()

    worker = result["t"]
    threads_after_worker = _count_threads()
    ratio = worker / main if main > 0 else float("inf")

    msg = (
        f"main={main:.2f}s (threads_after={threads_after_main}), "
        f"worker={worker:.2f}s (threads_after={threads_after_worker}), "
        f"ratio={ratio:.1f}x"
    )

    # The published issue saw ~30x. We pin a conservative >=5x bar so
    # noise on slower CI hosts does not flake — but if the bug ever
    # regresses to within 2-3x we want to know, because the workaround
    # cost (subprocess startup) becomes harder to justify.
    assert ratio >= 5.0, (
        f"Worker/main ratio dropped below 5x — libgomp affinity bug may "
        f"have been fixed or the test environment changed. {msg}"
    )


@pytest.mark.slow
def test_default_strategy_is_subprocess_on_libgomp() -> None:
    """The env-var gate must default to subprocess when libgomp is loaded.

    Catches the specific regression that motivated issue #147: a thread
    default would re-introduce the 30x worker-thread slowdown for every
    Tune trial. Tests in ``test_subprocess_integration.py`` already pin
    the gate logic; this test pins the *runtime* outcome on a real
    libgomp-affected host.
    """
    if sys.platform != "linux":
        pytest.skip("subprocess default only applies on Linux + libgomp")

    if not _libgomp_loaded():
        pytest.skip("libgomp not loaded — strategy is expected to be 'thread'")

    # Reset cache so the deferred-import path actually runs.
    from lizyml_widget import openmp_detect

    openmp_detect._reset_libgomp_cache()

    # The opt-out env var must be unset for this assertion to be meaningful.
    os.environ.pop("LZW_FORCE_THREAD", None)

    strategy, _ = openmp_detect.get_execution_strategy()
    assert strategy == "subprocess", (
        f"Expected subprocess default on libgomp host but got {strategy!r}. "
        f"Issue #147 regression: the LZW_FORCE_THREAD opt-out gate may have "
        f"flipped polarity again."
    )
