"""Tests for openmp_detect module (TDD — RED phase)."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from lizyml_widget.openmp_detect import (
    find_libomp_path,
    get_execution_strategy,
    is_libgomp_affected,
)

# ---------------------------------------------------------------------------
# Sample /proc/self/maps content
# ---------------------------------------------------------------------------

MAPS_LIBGOMP = """\
7f1234560000-7f1234570000 r-xp 00000000 08:01 12345  /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7f1234570000-7f1234580000 rw-p 00010000 08:01 12345  /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7f1234600000-7f1234650000 r-xp 00000000 08:01 99999  /usr/lib/x86_64-linux-gnu/libc.so.6
"""

MAPS_LIBOMP = """\
7f1234560000-7f1234570000 r-xp 00000000 08:01 12345  /usr/lib/x86_64-linux-gnu/libomp.so.5
7f1234600000-7f1234650000 r-xp 00000000 08:01 99999  /usr/lib/x86_64-linux-gnu/libc.so.6
"""

MAPS_NO_OMP = """\
7f1234600000-7f1234650000 r-xp 00000000 08:01 99999  /usr/lib/x86_64-linux-gnu/libc.so.6
"""


# ===========================================================================
# is_libgomp_affected
# ===========================================================================


class TestIsLibgompAffected:
    """Test detection of libgomp-affected environments."""

    def test_linux_with_libgomp_loaded(self) -> None:
        """Linux + libgomp in /proc/self/maps → True."""
        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch("builtins.open", mock_open(read_data=MAPS_LIBGOMP)),
        ):
            mock_sys.platform = "linux"
            assert is_libgomp_affected() is True

    def test_linux_with_libomp_loaded(self) -> None:
        """Linux + libomp (not libgomp) → False."""
        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch("builtins.open", mock_open(read_data=MAPS_LIBOMP)),
        ):
            mock_sys.platform = "linux"
            assert is_libgomp_affected() is False

    def test_linux_no_openmp(self) -> None:
        """Linux with no OpenMP library loaded → False."""
        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch("builtins.open", mock_open(read_data=MAPS_NO_OMP)),
        ):
            mock_sys.platform = "linux"
            assert is_libgomp_affected() is False

    def test_macos(self) -> None:
        """macOS → False (uses libomp by default)."""
        with patch("lizyml_widget.openmp_detect.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert is_libgomp_affected() is False

    def test_windows(self) -> None:
        """Windows → False (uses vcomp)."""
        with patch("lizyml_widget.openmp_detect.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert is_libgomp_affected() is False

    def test_proc_not_readable(self) -> None:
        """If /proc/self/maps is not readable → False (safe fallback)."""
        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch("builtins.open", side_effect=OSError("No /proc")),
        ):
            mock_sys.platform = "linux"
            assert is_libgomp_affected() is False


# ===========================================================================
# find_libomp_path
# ===========================================================================


class TestFindLibompPath:
    """Test libomp shared library discovery."""

    def test_finds_libomp5_so(self) -> None:
        """Finds /usr/lib/x86_64-linux-gnu/libomp5.so."""
        with patch("lizyml_widget.openmp_detect.Path") as mock_path_cls:
            p1 = mock_path_cls("/usr/lib/x86_64-linux-gnu/libomp5.so")
            p1.exists.return_value = True
            p1.__str__ = lambda self: "/usr/lib/x86_64-linux-gnu/libomp5.so"
            # Make Path() return our mock for the first candidate
            instances = []

            def path_side_effect(arg: str):  # noqa: ANN202
                m = type(
                    "MockPath",
                    (),
                    {
                        "exists": lambda s: arg == "/usr/lib/x86_64-linux-gnu/libomp5.so",
                        "__str__": lambda s: arg,
                    },
                )()
                instances.append(m)
                return m

            mock_path_cls.side_effect = path_side_effect
            result = find_libomp_path()
            assert result is not None
            assert "libomp" in result

    def test_finds_via_ctypes_find_library(self) -> None:
        """Falls back to ctypes.util.find_library("omp")."""
        with (
            patch("lizyml_widget.openmp_detect.Path") as mock_path_cls,
            patch("lizyml_widget.openmp_detect.ctypes_find_library") as mock_find,
        ):
            # No file found at known paths
            mock_path_cls.side_effect = lambda arg: type(
                "P", (), {"exists": lambda s: False, "__str__": lambda s: arg}
            )()
            mock_find.return_value = "libomp.so.5"
            result = find_libomp_path()
            assert result == "libomp.so.5"

    def test_not_installed(self) -> None:
        """No libomp found anywhere → None."""
        with (
            patch("lizyml_widget.openmp_detect.Path") as mock_path_cls,
            patch("lizyml_widget.openmp_detect.ctypes_find_library") as mock_find,
        ):
            mock_path_cls.side_effect = lambda arg: type(
                "P", (), {"exists": lambda s: False, "__str__": lambda s: arg}
            )()
            mock_find.return_value = None
            assert find_libomp_path() is None


# ===========================================================================
# get_execution_strategy
# ===========================================================================


class TestGetExecutionStrategy:
    """Test combined strategy selection."""

    def test_unaffected_platform(self) -> None:
        """macOS/Windows → ("thread", None)."""
        with patch("lizyml_widget.openmp_detect.is_libgomp_affected", return_value=False):
            strategy, path = get_execution_strategy()
            assert strategy == "thread"
            assert path is None

    def test_affected_with_libomp(self) -> None:
        """Linux + libgomp + libomp available → ("subprocess", path)."""
        with (
            patch("lizyml_widget.openmp_detect.is_libgomp_affected", return_value=True),
            patch(
                "lizyml_widget.openmp_detect.find_libomp_path",
                return_value="/usr/lib/x86_64-linux-gnu/libomp5.so",
            ),
        ):
            strategy, path = get_execution_strategy()
            assert strategy == "subprocess"
            assert path == "/usr/lib/x86_64-linux-gnu/libomp5.so"

    def test_affected_without_libomp_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Linux + libgomp + no libomp → ("subprocess", None) + warning."""
        import logging

        with (
            caplog.at_level(logging.WARNING),
            patch("lizyml_widget.openmp_detect.is_libgomp_affected", return_value=True),
            patch("lizyml_widget.openmp_detect.find_libomp_path", return_value=None),
        ):
            strategy, path = get_execution_strategy()
            assert strategy == "subprocess"
            assert path is None
            assert "libomp" in caplog.text.lower()
