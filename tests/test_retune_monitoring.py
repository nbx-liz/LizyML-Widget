"""Tests for P-027 re-tune monitoring (round-aware progress + boundary expansion).

Exercises three seams:

1. Version guard — adapter rejects out-of-range lizyml versions.
2. Progress callback payload — lizyml 0.9.0+ `TuneProgressInfo.round`,
   `cumulative_trials` and `expanded_dims` are forwarded to `on_progress`
   as keyword arguments.
3. TuningSummary serialization — `rounds` and `boundary_report` are
   round-tripped through the Widget dict traitlet shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lizyml_widget.adapter import (
    LIZYML_MAX_VERSION,
    LIZYML_MIN_VERSION,
    LizyMLAdapter,
    _check_lizyml_version,
    _parse_lizyml_version,
    _serialize_boundary_report,
    _serialize_rounds,
    _serialize_trials,
)
from lizyml_widget.types import TuningSummary


class TestVersionGuard:
    """P-027: LizyMLAdapter rejects lizyml versions outside the supported range."""

    def test_parse_version_handles_rc_suffix(self) -> None:
        assert _parse_lizyml_version("0.9.0rc1") == (0, 9, 0)
        assert _parse_lizyml_version("0.9.0.dev3") == (0, 9, 0)
        assert _parse_lizyml_version("0.10.2") == (0, 10, 2)
        assert _parse_lizyml_version("1") == (1, 0, 0)

    def test_parse_version_pads_short_strings(self) -> None:
        assert _parse_lizyml_version("0.9") == (0, 9, 0)

    def test_version_range_constants(self) -> None:
        assert LIZYML_MIN_VERSION == (0, 9, 0)
        assert LIZYML_MAX_VERSION == (0, 10, 0)

    def test_check_passes_for_supported_version(self) -> None:
        with patch.dict(
            "sys.modules",
            {"lizyml": MagicMock(__version__="0.9.5")},
        ):
            _check_lizyml_version()  # must not raise

    def test_check_rejects_too_old(self) -> None:
        with (
            patch.dict(
                "sys.modules",
                {"lizyml": MagicMock(__version__="0.7.3")},
            ),
            pytest.raises(ImportError, match=r"lizyml>=0\.9\.0,<0\.10\.0"),
        ):
            _check_lizyml_version()

    def test_check_rejects_too_new(self) -> None:
        with (
            patch.dict(
                "sys.modules",
                {"lizyml": MagicMock(__version__="0.10.0")},
            ),
            pytest.raises(ImportError, match=r"lizyml>=0\.9\.0,<0\.10\.0"),
        ):
            _check_lizyml_version()

    def test_check_is_silent_for_mocked_version(self) -> None:
        """When lizyml is a MagicMock (unit-test patching) the version guard
        skips silently so existing tests don't need to know about the check."""
        mock_lizyml = MagicMock()
        # __version__ access returns yet another MagicMock, not a str.
        with patch.dict("sys.modules", {"lizyml": mock_lizyml}):
            _check_lizyml_version()  # must not raise

    def test_adapter_init_applies_guard(self) -> None:
        with (
            patch.dict(
                "sys.modules",
                {"lizyml": MagicMock(__version__="0.1.0")},
            ),
            pytest.raises(ImportError),
        ):
            LizyMLAdapter()


class TestProgressCallbackRoundPayload:
    """P-027: Adapter forwards round-aware fields to on_progress as kwargs."""

    def test_progress_callback_forwards_round_fields(self) -> None:
        from lizyml.core.types.tuning_result import TuneProgressInfo

        adapter = LizyMLAdapter()
        model = MagicMock()

        def fake_tune(**kwargs: Any) -> MagicMock:
            cb = kwargs.get("progress_callback")
            assert cb is not None
            cb(
                TuneProgressInfo(
                    current_trial=17,
                    total_trials=30,
                    elapsed_seconds=42.0,
                    best_score=0.9,
                    latest_score=0.88,
                    latest_state="complete",
                    round=2,
                    cumulative_trials=67,
                    expanded_dims=("learning_rate", "num_leaves"),
                ),
            )
            # Return a minimal lizyml-shaped result for serialization.
            result = MagicMock()
            result.best_params = {"lr": 0.01}
            result.best_score = 0.9
            result.trials = ()
            result.metric_name = "auc"
            result.direction = "maximize"
            result.rounds = ()
            result.boundary_report = None
            return result

        model.tune.side_effect = fake_tune

        captured: list[tuple[int, int, str, dict[str, Any]]] = []

        def on_progress(current: int, total: int, message: str, **extra: Any) -> None:
            captured.append((current, total, message, extra))

        adapter.tune(model, on_progress=on_progress)

        assert len(captured) == 1
        cur, total, msg, extra = captured[0]
        assert cur == 17
        assert total == 30
        assert "Trial 17/30" in msg
        assert "best: 0.9000" in msg
        assert extra["round"] == 2
        assert extra["cumulative_trials"] == 67
        assert extra["expanded_dims"] == ["learning_rate", "num_leaves"]
        assert extra["latest_score"] == 0.88
        assert extra["latest_state"] == "complete"
        assert extra["best_score"] == 0.9

    def test_single_round_defaults_to_round_one(self) -> None:
        """Without an explicit round the callback payload defaults to round=1."""
        from lizyml.core.types.tuning_result import TuneProgressInfo

        adapter = LizyMLAdapter()
        model = MagicMock()

        def fake_tune(**kwargs: Any) -> MagicMock:
            cb = kwargs["progress_callback"]
            cb(
                TuneProgressInfo(
                    current_trial=1,
                    total_trials=10,
                    elapsed_seconds=1.0,
                    best_score=None,
                    latest_score=None,
                    latest_state="running",
                ),
            )
            result = MagicMock()
            result.best_params = {}
            result.best_score = 0.0
            result.trials = ()
            result.metric_name = "rmse"
            result.direction = "minimize"
            result.rounds = ()
            result.boundary_report = None
            return result

        model.tune.side_effect = fake_tune

        captured: list[dict[str, Any]] = []

        def on_progress(current: int, total: int, message: str, **extra: Any) -> None:
            captured.append(extra)

        adapter.tune(model, on_progress=on_progress)
        assert captured[0]["round"] == 1
        assert captured[0]["cumulative_trials"] == 1
        assert captured[0]["expanded_dims"] == []


class TestTuningSummarySerialization:
    """P-027: rounds / boundary_report must be JSON-friendly dicts."""

    def test_tuning_summary_defaults(self) -> None:
        s = TuningSummary(
            best_params={"a": 1},
            best_score=0.5,
            trials=[],
            metric_name="rmse",
            direction="minimize",
        )
        assert s.rounds == []
        assert s.boundary_report is None

    def test_serialize_rounds_drops_snapshot(self) -> None:
        """`space_snapshot` is intentionally dropped to keep the dict JSON-friendly."""
        round_obj = MagicMock()
        round_obj.round = 2
        round_obj.n_trials = 30
        round_obj.best_score_before = 0.28
        round_obj.best_score_after = 0.279
        round_obj.expanded_dims = ("learning_rate",)
        round_obj.space_snapshot = (MagicMock(), MagicMock())

        out = _serialize_rounds((round_obj,))
        assert len(out) == 1
        assert out[0] == {
            "round": 2,
            "n_trials": 30,
            "best_score_before": 0.28,
            "best_score_after": 0.279,
            "expanded_dims": ["learning_rate"],
        }

    def test_serialize_rounds_handles_none_before(self) -> None:
        r = MagicMock()
        r.round = 1
        r.n_trials = 50
        r.best_score_before = None
        r.best_score_after = 0.92
        r.expanded_dims = ()
        r.space_snapshot = ()
        out = _serialize_rounds((r,))
        assert out[0]["best_score_before"] is None
        assert out[0]["expanded_dims"] == []

    def test_serialize_boundary_report_none(self) -> None:
        assert _serialize_boundary_report(None) is None

    def test_serialize_boundary_report_expands_dims(self) -> None:
        dim = MagicMock()
        dim.name = "learning_rate"
        dim.best_value = 0.01
        dim.low = 0.001
        dim.high = 0.1
        dim.position_pct = 0.0
        dim.edge = "lower"
        dim.expanded = True
        dim.new_low = 0.00001
        dim.new_high = 0.1

        report = MagicMock()
        report.dims = (dim,)
        report.expanded_names = ("learning_rate",)

        out = _serialize_boundary_report(report)
        assert out is not None
        assert out["expanded_names"] == ["learning_rate"]
        assert out["dims"][0] == {
            "name": "learning_rate",
            "best_value": 0.01,
            "low": 0.001,
            "high": 0.1,
            "position_pct": 0.0,
            "edge": "lower",
            "expanded": True,
            "new_low": 0.00001,
            "new_high": 0.1,
        }

    def test_serialize_trials_sets_default_round(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class Trial:
            number: int
            params: dict[str, Any]
            score: float
            state: str

        trials = (Trial(number=1, params={}, score=0.5, state="COMPLETE"),)
        out = _serialize_trials(trials)
        assert out[0]["round"] == 1
        assert out[0]["number"] == 1
        assert out[0]["state"] == "COMPLETE"

    def test_serialize_trials_preserves_round_field(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class Trial:
            number: int
            params: dict[str, Any]
            score: float
            state: str
            round: int

        trials = (
            Trial(number=1, params={}, score=0.5, state="COMPLETE", round=1),
            Trial(number=2, params={}, score=0.6, state="COMPLETE", round=2),
        )
        out = _serialize_trials(trials)
        assert [t["round"] for t in out] == [1, 2]


class TestWidgetProgressTraitletRoundFields:
    """P-027: `progress` traitlet propagates round-aware fields through the widget layer."""

    def _make_widget(self) -> Any:
        """Create a widget using the same pattern as tests/test_widget_jobs.py."""
        import pandas as pd

        from lizyml_widget.widget import LizyWidget

        w = LizyWidget()
        df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
        w.load(df, target="y")
        return w

    def test_progress_traitlet_carries_round_fields(self) -> None:
        """When the tune service forwards round kwargs, the widget payload
        must contain them (omitted keys stay absent)."""
        from lizyml_widget.types import TuningSummary

        w = self._make_widget()

        def mock_tune(config: Any, *, on_progress: Any = None) -> Any:
            assert on_progress is not None
            # Simulate a re-tune round 2 progress update.
            on_progress(
                17,
                30,
                "Trial 17/30 (best: 0.2790)",
                round=2,
                cumulative_trials=67,
                expanded_dims=["learning_rate", "num_leaves"],
                latest_score=0.275,
                latest_state="complete",
                best_score=0.279,
            )
            return TuningSummary(
                best_params={"lr": 0.01},
                best_score=0.279,
                trials=[],
                metric_name="auc",
                direction="maximize",
                rounds=[
                    {
                        "round": 1,
                        "n_trials": 50,
                        "best_score_before": None,
                        "best_score_after": 0.274,
                        "expanded_dims": [],
                    },
                    {
                        "round": 2,
                        "n_trials": 30,
                        "best_score_before": 0.274,
                        "best_score_after": 0.279,
                        "expanded_dims": ["learning_rate", "num_leaves"],
                    },
                ],
                boundary_report={
                    "dims": [],
                    "expanded_names": ["learning_rate", "num_leaves"],
                },
            )

        w._service.tune = mock_tune  # type: ignore[assignment]

        w.config = {
            **dict(w.config),
            "tuning": {
                "optuna": {"params": {"n_trials": 30}, "space": {}},
            },
        }

        import time

        w.action = {"type": "tune", "payload": {}}

        # Wait for completion — our mock is instant but the widget runs it
        # on a worker thread.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert w.status == "completed", f"expected completed, got {w.status} err={w.error}"

        # tune_summary has the round + boundary_report fields.
        assert w.tune_summary["rounds"][1]["expanded_dims"] == [
            "learning_rate",
            "num_leaves",
        ]
        assert w.tune_summary["boundary_report"]["expanded_names"] == [
            "learning_rate",
            "num_leaves",
        ]
        # get_tune_summary() must round-trip the new fields.
        ts = w.get_tune_summary()
        assert ts is not None
        assert ts.rounds == w.tune_summary["rounds"]
        assert ts.boundary_report == w.tune_summary["boundary_report"]
