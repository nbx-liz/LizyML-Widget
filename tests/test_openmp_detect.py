"""Tests for openmp_detect module."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from lizyml_widget.openmp_detect import (
    _reset_libgomp_cache,
    find_libomp_path,
    get_execution_strategy,
    is_libgomp_affected,
)


@pytest.fixture(autouse=True)
def _isolated_libgomp_cache() -> None:
    """Reset module-level cache so test order does not leak state."""
    _reset_libgomp_cache()
    yield
    _reset_libgomp_cache()


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


# ===========================================================================
# Issue #147: deferred lightgbm-aware detection + caching
# ===========================================================================


class TestLibgompDeferredDetection:
    """Detection must force lightgbm to be importable so /proc/self/maps
    reflects the OpenMP runtime that the data path will use, and must cache
    the result so we don't pay the cost on every job dispatch."""

    def test_force_imports_lightgbm_when_missing(self) -> None:
        """If lightgbm is not yet in sys.modules, _ensure_lightgbm_imported
        runs to make /proc/self/maps reflect the loaded runtime."""
        from unittest.mock import MagicMock

        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch(
                "lizyml_widget.openmp_detect._ensure_lightgbm_imported",
                MagicMock(),
            ) as mock_ensure,
            patch("builtins.open", mock_open(read_data=MAPS_LIBGOMP)),
        ):
            mock_sys.platform = "linux"
            assert is_libgomp_affected() is True
            mock_ensure.assert_called_once()

    def test_skips_lightgbm_import_on_non_linux(self) -> None:
        """Non-linux platforms short-circuit before touching lightgbm."""
        from unittest.mock import MagicMock

        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch(
                "lizyml_widget.openmp_detect._ensure_lightgbm_imported",
                MagicMock(),
            ) as mock_ensure,
        ):
            mock_sys.platform = "darwin"
            assert is_libgomp_affected() is False
            mock_ensure.assert_not_called()

    def test_caches_result_across_calls(self) -> None:
        """Second call must use the cached value — /proc/self/maps is read once."""
        from unittest.mock import MagicMock

        opener = mock_open(read_data=MAPS_LIBGOMP)
        with (
            patch("lizyml_widget.openmp_detect.sys") as mock_sys,
            patch(
                "lizyml_widget.openmp_detect._ensure_lightgbm_imported",
                MagicMock(),
            ),
            patch("builtins.open", opener),
        ):
            mock_sys.platform = "linux"
            assert is_libgomp_affected() is True
            assert is_libgomp_affected() is True
            assert opener.call_count == 1

    def test_ensure_lightgbm_imported_swallows_import_errors(self) -> None:
        """The helper is contractually best-effort — it must never raise.

        is_libgomp_affected delegates to this helper outside any try/except,
        so the helper is the swallow point. We pin that contract directly.
        """
        import sys as real_sys

        from lizyml_widget.openmp_detect import _ensure_lightgbm_imported

        original = real_sys.modules.pop("lightgbm", None)
        real_sys.modules["lightgbm"] = None  # type: ignore[assignment]
        try:
            # ``import lightgbm`` against a None entry raises ImportError.
            try:
                _ensure_lightgbm_imported()
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"_ensure_lightgbm_imported raised {exc!r}")
        finally:
            if original is None:
                real_sys.modules.pop("lightgbm", None)
            else:
                real_sys.modules["lightgbm"] = original

    def test_ensure_lightgbm_imported_is_no_op_when_loaded(self) -> None:
        """If lightgbm is already in sys.modules, no re-import attempt."""
        import sys as real_sys

        from lizyml_widget.openmp_detect import _ensure_lightgbm_imported

        sentinel = object()
        original = real_sys.modules.get("lightgbm")
        real_sys.modules["lightgbm"] = sentinel  # type: ignore[assignment]
        try:
            _ensure_lightgbm_imported()
            # Still our sentinel — function did not re-import.
            assert real_sys.modules["lightgbm"] is sentinel
        finally:
            if original is None:
                real_sys.modules.pop("lightgbm", None)
            else:
                real_sys.modules["lightgbm"] = original
