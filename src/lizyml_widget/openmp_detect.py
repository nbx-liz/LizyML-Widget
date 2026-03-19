"""Detect OpenMP runtime and execution strategy for background jobs.

On Linux with libgomp (GCC OpenMP), the thread pool is bound to the first
thread that enters an OpenMP parallel region (GCC bug #108494). Worker
threads suffer ~50x slowdown. This module detects the affected environment
and recommends subprocess execution with optional LD_PRELOAD=libomp.
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


def is_libgomp_affected() -> bool:
    """Return True if running on Linux with libgomp loaded.

    Returns False on non-Linux platforms, when libomp is loaded instead,
    or when /proc/self/maps cannot be read (safe fallback).
    """
    if sys.platform != "linux":
        return False

    try:
        with open("/proc/self/maps") as f:
            return any("libgomp" in line for line in f)
    except OSError:
        _log.debug("/proc/self/maps not readable; assuming unaffected")
        return False


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
