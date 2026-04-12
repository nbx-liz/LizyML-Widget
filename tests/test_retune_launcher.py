"""Tests for P-028 Re-tune Launcher (`w.retune()` API + retune action).

P-027 covered the *monitoring* side of lizyml 0.9.0 re-tune.  P-028 covers the
*launcher* side: the widget must be able to kick off `Model.tune(resume=True, ...)`
on an existing study, reuse the previously-fitted model, and propagate the
extra kwargs (``n_trials``, ``expand_boundary``, ``boundary_threshold``)
through every layer of the stack.

Exercised seams:

1. ``LizyMLAdapter.tune`` forwards ``resume`` and friends to ``model.tune``.
2. ``WidgetService.tune(resume=True)`` reuses ``self._model`` instead of
   calling ``create_model`` again; raises ``ValueError`` when no prior model.
3. ``LizyWidget.retune`` refuses to run before any initial tune completed.
4. The ``retune`` action routes through ``_run_job`` with the same kwargs.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from lizyml_widget.adapter import LizyMLAdapter
from lizyml_widget.service import WidgetService
from lizyml_widget.types import BackendInfo, TuningSummary
from lizyml_widget.widget import LizyWidget

# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def _dummy_summary(**overrides: Any) -> TuningSummary:
    """Build a minimal TuningSummary so the widget 'completed' path runs."""
    base = {
        "best_params": {"lr": 0.01},
        "best_score": 0.9,
        "trials": [],
        "metric_name": "auc",
        "direction": "maximize",
        "rounds": [
            {
                "round": 1,
                "n_trials": 50,
                "best_score_before": None,
                "best_score_after": 0.88,
                "expanded_dims": [],
            }
        ],
        "boundary_report": None,
    }
    base.update(overrides)
    return TuningSummary(**base)  # type: ignore[arg-type]


def _make_widget() -> LizyWidget:
    w = LizyWidget()
    df = pd.DataFrame({"x": [i % 10 for i in range(50)], "y": [0, 1] * 25})
    w.load(df, target="y")
    return w


# ──────────────────────────────────────────────────────────────
#  1) Adapter: forwards resume kwargs to Model.tune
# ──────────────────────────────────────────────────────────────


class TestAdapterForwardsResumeKwargs:
    """LizyMLAdapter.tune must pass resume/n_trials/expand_boundary/boundary_threshold
    straight through to the underlying ``model.tune`` call."""

    def test_default_call_keeps_legacy_contract(self) -> None:
        adapter = LizyMLAdapter()
        model = MagicMock()

        def fake_tune(**kwargs: Any) -> MagicMock:
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
        adapter.tune(model)  # no resume kwargs

        call_kwargs = model.tune.call_args.kwargs
        assert call_kwargs.get("resume", False) is False
        assert call_kwargs.get("n_trials") is None

    def test_resume_kwargs_are_forwarded_verbatim(self) -> None:
        adapter = LizyMLAdapter()
        model = MagicMock()

        def fake_tune(**kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.best_params = {}
            result.best_score = 0.0
            result.trials = ()
            result.metric_name = "auc"
            result.direction = "maximize"
            result.rounds = ()
            result.boundary_report = None
            return result

        model.tune.side_effect = fake_tune

        adapter.tune(
            model,
            resume=True,
            n_trials=30,
            expand_boundary=True,
            boundary_threshold=0.03,
        )

        call_kwargs = model.tune.call_args.kwargs
        assert call_kwargs["resume"] is True
        assert call_kwargs["n_trials"] == 30
        assert call_kwargs["expand_boundary"] is True
        assert call_kwargs["boundary_threshold"] == 0.03


# ──────────────────────────────────────────────────────────────
#  2) Service: resume path reuses existing model
# ──────────────────────────────────────────────────────────────


class _FakeAdapter:
    """Minimal BackendAdapter double that records tune() invocations."""

    def __init__(self) -> None:
        self.info = BackendInfo(name="fake", version="0.0.0")
        self.created_models: list[Any] = []
        self.tune_calls: list[dict[str, Any]] = []
        self.next_summary: TuningSummary = _dummy_summary()

    def get_backend_contract(self) -> Any:
        from lizyml_widget.types import BackendContract

        return BackendContract(schema_version=1, config_schema={}, ui_schema={}, capabilities={})

    def initialize_config(self, *, task: str | None = None) -> dict[str, Any]:
        return {"config_version": 1}

    def apply_config_patch(self, config: Any, ops: Any, *, task: Any = None) -> Any:
        return dict(config)

    def prepare_run_config(self, config: Any, *, job_type: Any, task: Any = None) -> Any:
        return dict(config)

    def canonicalize_config(self, config: Any, *, task: Any = None) -> Any:
        return dict(config)

    def apply_task_defaults(self, config: Any, *, task: str) -> Any:
        return dict(config)

    def validate_config(self, config: Any) -> list[Any]:
        return []

    def create_model(self, config: Any, dataframe: Any) -> Any:
        model = MagicMock(name=f"model_{len(self.created_models)}")
        self.created_models.append(model)
        return model

    def fit(self, model: Any, *, params: Any = None, on_progress: Any = None) -> Any:
        from lizyml_widget.types import FitSummary

        return FitSummary(metrics={}, fold_count=1, params=[])

    def tune(
        self,
        model: Any,
        *,
        on_progress: Any = None,
        resume: bool = False,
        n_trials: int | None = None,
        expand_boundary: bool | None = None,
        boundary_threshold: float = 0.05,
    ) -> TuningSummary:
        self.tune_calls.append(
            {
                "model": model,
                "resume": resume,
                "n_trials": n_trials,
                "expand_boundary": expand_boundary,
                "boundary_threshold": boundary_threshold,
            }
        )
        return self.next_summary

    def predict(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover - not used
        raise NotImplementedError

    def evaluate_table(self, model: Any) -> list[Any]:
        return []

    def split_summary(self, model: Any) -> list[Any]:
        return []

    def importance(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        return {}

    def plot(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def available_plots(self, model: Any) -> list[str]:
        return []

    def export_model(self, *a: Any, **kw: Any) -> str:  # pragma: no cover
        return ""

    def export_code(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        return b""

    def load_model(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        return MagicMock()

    def model_info(self, *a: Any, **kw: Any) -> dict[str, Any]:  # pragma: no cover
        return {}

    def classify_best_params(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return params, {}, {}

    def plot_inference(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class TestServiceResumePath:
    def test_first_tune_creates_new_model(self) -> None:
        adapter = _FakeAdapter()
        service = WidgetService(adapter=adapter)  # type: ignore[arg-type]
        service.load_data(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), target="y")

        service.tune({"config_version": 1})
        assert len(adapter.created_models) == 1
        assert adapter.tune_calls[0]["resume"] is False

    def test_resume_reuses_previously_tuned_model(self) -> None:
        adapter = _FakeAdapter()
        service = WidgetService(adapter=adapter)  # type: ignore[arg-type]
        service.load_data(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), target="y")

        service.tune({"config_version": 1})
        first_model = adapter.created_models[-1]

        service.tune(
            {"config_version": 1},
            resume=True,
            n_trials=20,
            expand_boundary=True,
        )

        # No new model created: second call must pass the existing model.
        assert len(adapter.created_models) == 1
        assert adapter.tune_calls[-1]["model"] is first_model
        assert adapter.tune_calls[-1]["resume"] is True
        assert adapter.tune_calls[-1]["n_trials"] == 20
        assert adapter.tune_calls[-1]["expand_boundary"] is True

    def test_resume_without_prior_model_raises(self) -> None:
        adapter = _FakeAdapter()
        service = WidgetService(adapter=adapter)  # type: ignore[arg-type]
        service.load_data(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), target="y")

        with pytest.raises(ValueError, match=r"resume.*prior"):
            service.tune({"config_version": 1}, resume=True)

    def test_resume_forwards_kwargs_through_adapter(self) -> None:
        adapter = _FakeAdapter()
        service = WidgetService(adapter=adapter)  # type: ignore[arg-type]
        service.load_data(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), target="y")
        service.tune({"config_version": 1})  # establish prior model

        service.tune(
            {"config_version": 1},
            resume=True,
            n_trials=10,
            expand_boundary=False,
            boundary_threshold=0.1,
        )
        call = adapter.tune_calls[-1]
        assert call["resume"] is True
        assert call["n_trials"] == 10
        assert call["expand_boundary"] is False
        assert call["boundary_threshold"] == 0.1

    def test_resume_survives_intervening_fit(self) -> None:
        """Regression: tune() → fit() → retune() must resume the tune study,
        not the fresh fit model.  Fix for P-028 HIGH-1 review finding."""
        adapter = _FakeAdapter()
        service = WidgetService(adapter=adapter)  # type: ignore[arg-type]
        service.load_data(pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}), target="y")

        # 1. Initial tune → establishes _tune_model.
        service.tune({"config_version": 1})
        tune_model = adapter.created_models[-1]

        # 2. fit() creates a fresh model and stores it in _model.
        service.fit({"config_version": 1})
        assert len(adapter.created_models) == 2
        fit_model = adapter.created_models[-1]
        assert fit_model is not tune_model

        # 3. retune() must target the original tune model, not the fit model.
        service.tune({"config_version": 1}, resume=True, n_trials=5)
        # No extra model was created.
        assert len(adapter.created_models) == 2
        # The adapter received the tune model, not the fit model.
        assert adapter.tune_calls[-1]["model"] is tune_model
        assert adapter.tune_calls[-1]["model"] is not fit_model


# ──────────────────────────────────────────────────────────────
#  3) Widget Python API: LizyWidget.retune()
# ──────────────────────────────────────────────────────────────


class TestWidgetRetuneApi:
    def test_retune_before_initial_tune_raises(self) -> None:
        w = _make_widget()
        with pytest.raises(ValueError, match=r"(?i)no prior tune"):
            w.retune(n_trials=10)

    def test_retune_triggers_tune_action_with_resume_kwargs(self) -> None:
        w = _make_widget()

        captured: list[dict[str, Any]] = []

        def mock_tune(config: Any, *, on_progress: Any = None, **kwargs: Any) -> Any:
            captured.append(kwargs)
            return _dummy_summary()

        w._service.tune = mock_tune  # type: ignore[assignment]

        # Seed tune_summary so retune() passes the precondition.
        w.tune_summary = {
            "best_params": {"lr": 0.01},
            "best_score": 0.9,
            "trials": [],
            "metric_name": "auc",
            "direction": "maximize",
            "rounds": [],
            "boundary_report": None,
        }

        w.config = {
            **dict(w.config),
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
            },
        }

        w.retune(n_trials=20, expand_boundary=True, boundary_threshold=0.03)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert w.status == "completed", f"got {w.status} err={w.error}"
        assert len(captured) == 1
        call = captured[0]
        assert call.get("resume") is True
        assert call.get("n_trials") == 20
        assert call.get("expand_boundary") is True
        assert call.get("boundary_threshold") == 0.03


# ──────────────────────────────────────────────────────────────
#  4) Retune action (UI path)
# ──────────────────────────────────────────────────────────────


class TestRetuneAction:
    def test_retune_action_invokes_service_with_resume(self) -> None:
        w = _make_widget()

        captured: list[dict[str, Any]] = []

        def mock_tune(config: Any, *, on_progress: Any = None, **kwargs: Any) -> Any:
            captured.append(kwargs)
            return _dummy_summary()

        w._service.tune = mock_tune  # type: ignore[assignment]

        # Preseed so the action passes its tune_summary precondition.
        w.tune_summary = {
            "best_params": {"lr": 0.01},
            "best_score": 0.9,
            "trials": [],
            "metric_name": "auc",
            "direction": "maximize",
            "rounds": [],
            "boundary_report": None,
        }
        w.config = {
            **dict(w.config),
            "tuning": {
                "optuna": {"params": {"n_trials": 50}, "space": {}},
            },
        }

        w.action = {
            "type": "retune",
            "payload": {
                "n_trials": 15,
                "expand_boundary": True,
                "boundary_threshold": 0.02,
            },
        }

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if w.status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert w.status == "completed", f"got {w.status} err={w.error}"
        assert len(captured) == 1
        call = captured[0]
        assert call["resume"] is True
        assert call["n_trials"] == 15
        assert call["expand_boundary"] is True
        assert call["boundary_threshold"] == 0.02

    def test_retune_action_before_tune_sets_error(self) -> None:
        w = _make_widget()
        w.action = {"type": "retune", "payload": {}}
        # Synchronous precondition check: no job thread is spawned.
        assert w.error.get("code") == "NO_PRIOR_TUNE"
        assert w.status == "failed"
