"""Detect OpenMP runtime and execution strategy for background jobs.

On Linux with libgomp (GCC OpenMP), the thread pool is bound to the first
thread that enters an OpenMP parallel region (GCC bug #108494). Worker
threads suffer ~30x slowdown for a single Fit, and 20-50x for a multi-trial
Tune (Optuna re-enters parallel regions per trial). This module detects the
affected environment and recommends subprocess execution with optional
``LD_PRELOAD=libomp``.

The detection (``is_libgomp_affected``) is deferred until the first call and
cached for the rest of the process lifetime: at ``LizyWidget.__init__`` time
``lightgbm`` has not been imported yet, so ``/proc/self/maps`` does not yet
contain ``libgomp``. We force-import ``lightgbm`` before inspecting the
maps file so the check observes the real OpenMP runtime once a data path
needs it.
"""

from __future__ import annotations

import logging
import sys
from ctypes.util import find_library as ctypes_find_library
from pathlib import Path

_log = logging.getLogger(__name__)

# Well-known libomp paths to check before ctypes fallback.
_LIBOMP_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libomp5.so",
    "/usr/lib/x86_64-linux-gnu/libomp.so",
    "/usr/lib/x86_64-linux-gnu/libomp.so.5",
    "/usr/lib/llvm-18/lib/libomp.so",
    "/usr/lib/llvm-17/lib/libomp.so",
    "/usr/lib/llvm-16/lib/libomp.so",
)

# Module-level cache for ``is_libgomp_affected``. Set on first call and never
# changes within the process (the affected runtime is determined by the OS +
# installed libraries, not by anything mutable). Tests reset this via the
# ``_reset_libgomp_cache`` helper.
_libgomp_cache: bool | None = None


def _reset_libgomp_cache() -> None:
    """Internal helper for tests — clear the affinity cache."""
    global _libgomp_cache
    _libgomp_cache = None


def _ensure_lightgbm_imported() -> None:
    """Best-effort import of ``lightgbm`` to load libgomp into the process.

    The libgomp affinity check reads ``/proc/self/maps``, which only lists
    shared objects that are actually loaded. ``LizyWidget.__init__`` runs
    long before any code path imports ``lightgbm``, so the maps file would
    be empty of libgomp and the detection would silently mis-classify the
    environment as unaffected (issue #147).

    Failures are swallowed: if ``lightgbm`` is missing or import errors,
    we fall back to whatever is already in the address space.
    """
    if "lightgbm" in sys.modules:
        return
    try:
        import lightgbm  # noqa: F401
    except Exception:  # pragma: no cover - defensive: import side-effects
        _log.debug("lightgbm import failed during libgomp detection", exc_info=True)


def is_libgomp_affected() -> bool:
    """Return True if running on Linux with libgomp loaded.

    Forces a ``lightgbm`` import on first call so ``/proc/self/maps``
    reflects the OpenMP runtime that the data path will actually use, then
    caches the result for the rest of the process. Returns False on
    non-Linux platforms, when libomp is loaded instead, or when
    ``/proc/self/maps`` cannot be read (safe fallback).
    """
    global _libgomp_cache
    if _libgomp_cache is not None:
        return _libgomp_cache

    if sys.platform != "linux":
        _libgomp_cache = False
        return _libgomp_cache

    _ensure_lightgbm_imported()

    try:
        with open("/proc/self/maps") as f:
            result = any("libgomp" in line for line in f)
    except OSError:
        _log.debug("/proc/self/maps not readable; assuming unaffected")
        result = False

    _libgomp_cache = result
    return result


def find_libomp_path() -> str | None:
    """Find libomp shared library path. Returns None if not installed."""
    for candidate in _LIBOMP_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return str(p)

    found = ctypes_find_library("omp")
    if found:
        return found

    return None


def get_execution_strategy() -> tuple[str, str | None]:
    """Return execution strategy for background jobs.

    Returns:
        ("thread", None) on unaffected platforms (Windows, macOS, or libomp).
        ("subprocess", libomp_path) on Linux with libgomp and libomp available.
        ("subprocess", None) on Linux with libgomp but no libomp.
    """
    if not is_libgomp_affected():
        return ("thread", None)

    libomp_path = find_libomp_path()
    if libomp_path is None:
        _log.warning(
            "libgomp detected but libomp not found. "
            "Training in subprocess will be ~1.5x slower than optimal. "
            "Install libomp for best performance: sudo apt install libomp-dev"
        )

    return ("subprocess", libomp_path)
