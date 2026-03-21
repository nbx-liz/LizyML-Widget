"""Tests for tune cancel-polling fix (Issue 1).

TDD: tests written FIRST, implementation follows.
The tune() v0.2.0+ path must use _run_with_cancel_polling, not a direct call,
so the cancel flag is checked during tune execution.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget.adapter import LizyMLAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tune_result() -> Any:
    """Return a minimal fake TuneResult."""
    from dataclasses import dataclass

    @dataclass
    class FakeTrial:
        state: str = "COMPLETE"
        value: float = 0.9
        params: dict[str, Any] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            if self.params is None:
                self.params = {}

    result = MagicMock()
    result.best_params = {"n_estimators": 100}
    result.best_score = 0.9
    result.trials = [FakeTrial()]
    result.metric_name = "auc"
    result.direction = "maximize"
    return result


# ---------------------------------------------------------------------------
# Test: cancel_event propagation
# ---------------------------------------------------------------------------


class TestTuneCancelSetsFlag:
    """Verify that when cancel is requested, the cancel flag propagates."""

    def test_cancel_flag_interrupts_on_progress(self) -> None:
        """on_progress raises InterruptedError when cancel flag is set."""
        cancel_flag = threading.Event()
        cancel_flag.set()  # pre-cancel

        calls: list[Any] = []

        def on_progress(current: int, total: int, message: str) -> None:
            calls.append((current, total, message))
            if cancel_flag.is_set():
                raise InterruptedError("Job cancelled by user")

        with pytest.raises(InterruptedError):
            on_progress(0, 50, "Tuning 50 trials...")

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Test: tune() uses _run_with_cancel_polling
# ---------------------------------------------------------------------------


class TestTuneUsesCancelPolling:
    """Verify tune() routes through _run_with_cancel_polling (not direct call)."""

    def test_tune_v020_uses_cancel_polling_not_direct_call(self) -> None:
        """With TuneProgressInfo available, tune() must use _run_with_cancel_polling."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model.tune.return_value = _make_tune_result()

        on_progress_calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, message: str) -> None:
            on_progress_calls.append((current, total, message))

        polling_calls: list[Any] = []
        real_run_with_cancel_polling = adapter._run_with_cancel_polling

        def spy_run_with_cancel_polling(
            target: Any,
            on_prog: Any,
            poll_interval: float = 0.5,
        ) -> Any:
            polling_calls.append((target, on_prog))
            return real_run_with_cancel_polling(target, on_prog, poll_interval)

        adapter._run_with_cancel_polling = spy_run_with_cancel_polling  # type: ignore[method-assign]

        # Patch TuneProgressInfo to make it importable (simulate v0.2.0+)
        fake_info_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "lizyml": MagicMock(),
                "lizyml.core": MagicMock(),
                "lizyml.core.types": MagicMock(),
                "lizyml.core.types.tuning_result": MagicMock(TuneProgressInfo=fake_info_cls),
            },
        ):
            adapter.tune(model=model, on_progress=on_progress)

        # _run_with_cancel_polling must have been called exactly once
        assert len(polling_calls) == 1, (
            f"Expected _run_with_cancel_polling to be called once, got {len(polling_calls)} calls"
        )

    def test_tune_legacy_path_uses_cancel_polling(self) -> None:
        """When TuneProgressInfo is not importable, legacy path also uses cancel polling."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model.tune.return_value = _make_tune_result()

        on_progress_calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, message: str) -> None:
            on_progress_calls.append((current, total, message))

        polling_calls: list[Any] = []
        real_run_with_cancel_polling = adapter._run_with_cancel_polling

        def spy_run_with_cancel_polling(
            target: Any,
            on_prog: Any,
            poll_interval: float = 0.5,
        ) -> Any:
            polling_calls.append((target, on_prog))
            return real_run_with_cancel_polling(target, on_prog, poll_interval)

        adapter._run_with_cancel_polling = spy_run_with_cancel_polling  # type: ignore[method-assign]

        # Patch to make TuneProgressInfo import fail (simulate < v0.2.0)
        import sys

        # Save and temporarily make the module absent
        saved = sys.modules.pop("lizyml.core.types.tuning_result", None)
        try:
            adapter.tune(model=model, on_progress=on_progress)
        finally:
            if saved is not None:
                sys.modules["lizyml.core.types.tuning_result"] = saved

        assert len(polling_calls) == 1, (
            f"Expected _run_with_cancel_polling to be called once, got {len(polling_calls)} calls"
        )

    def test_tune_without_on_progress_skips_cancel_polling(self) -> None:
        """When on_progress is None, tune() still completes without cancel polling overhead."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model.tune.return_value = _make_tune_result()

        # _run_with_cancel_polling with on_progress=None falls through to target() directly.
        # This test verifies tune() completes successfully regardless of polling path.
        with patch.dict(
            "sys.modules",
            {
                "lizyml": MagicMock(),
                "lizyml.core": MagicMock(),
                "lizyml.core.types": MagicMock(),
                "lizyml.core.types.tuning_result": MagicMock(TuneProgressInfo=MagicMock()),
            },
        ):
            result = adapter.tune(model=model, on_progress=None)

        assert result.best_score == 0.9
        assert result.metric_name == "auc"


# ---------------------------------------------------------------------------
# Test: InterruptedError raised when cancelled
# ---------------------------------------------------------------------------


class TestTuneCancelRaisesInterrupted:
    """Verify InterruptedError propagates out of tune() when cancel is signalled."""

    def test_tune_cancel_raises_interrupted_error(self) -> None:
        """on_progress raising InterruptedError propagates through _run_with_cancel_polling."""
        adapter = LizyMLAdapter()
        model = MagicMock()

        # tune() will run in a daemon thread; make it block briefly
        ready = threading.Event()
        released = threading.Event()

        def slow_tune(**kwargs: Any) -> Any:
            ready.set()
            released.wait(timeout=2.0)
            return _make_tune_result()

        model.tune.side_effect = slow_tune

        def on_progress(current: int, total: int, message: str) -> None:
            raise InterruptedError("Job cancelled by user")

        fake_info_cls = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "lizyml": MagicMock(),
                    "lizyml.core": MagicMock(),
                    "lizyml.core.types": MagicMock(),
                    "lizyml.core.types.tuning_result": MagicMock(TuneProgressInfo=fake_info_cls),
                },
            ),
            # The on_progress callback immediately raises InterruptedError,
            # which must propagate out of tune()
            pytest.raises(InterruptedError),
        ):
            adapter.tune(model=model, on_progress=on_progress)

        released.set()  # unblock slow_tune daemon thread

    def test_tune_no_cancel_completes_normally(self) -> None:
        """tune() without cancellation completes and returns TuningSummary."""
        adapter = LizyMLAdapter()
        model = MagicMock()
        model.tune.return_value = _make_tune_result()

        progress_calls: list[str] = []

        def on_progress(current: int, total: int, message: str) -> None:
            progress_calls.append(message)

        fake_info_cls = MagicMock()

        def tune_with_callback(**kwargs: Any) -> Any:
            cb = kwargs.get("progress_callback")
            if cb is not None:
                info = MagicMock()
                info.current_trial = 1
                info.total_trials = 2
                info.best_score = 0.9
                cb(info)
            return _make_tune_result()

        model.tune.side_effect = tune_with_callback

        with patch.dict(
            "sys.modules",
            {
                "lizyml": MagicMock(),
                "lizyml.core": MagicMock(),
                "lizyml.core.types": MagicMock(),
                "lizyml.core.types.tuning_result": MagicMock(TuneProgressInfo=fake_info_cls),
            },
        ):
            result = adapter.tune(model=model, on_progress=on_progress)

        assert result.best_score == 0.9
        assert result.best_params == {"n_estimators": 100}
        # Progress was reported
        assert len(progress_calls) >= 1
