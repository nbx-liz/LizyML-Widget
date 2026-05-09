"""Typed views over lizyml result objects (#116).

This module is the **only** place in the widget that reads internal
attributes off lizyml result types. The rest of the adapter consumes
``LM*View`` dataclasses, which means a contract drift in lizyml fails
loudly (``LizyMLContractError``) at the boundary instead of silently
degrading to ``None`` / ``[]`` deeper in the stack.

The views deliberately enumerate exactly the fields the widget consumes.
If a future widget feature needs another field, add it here in a single
place rather than spreading new ``getattr(result, "...", default)``
chains across ``adapter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LMBoundaryDimView",
    "LMBoundaryReportView",
    "LMFitResultView",
    "LMPredictionResultView",
    "LMRoundView",
    "LMTuneProgressView",
    "LMTuningResultView",
    "LizyMLContractError",
    "view_boundary_report",
    "view_fit_result",
    "view_prediction_result",
    "view_rounds",
    "view_tune_progress",
    "view_tuning_result",
]


class LizyMLContractError(RuntimeError):
    """Raised when a lizyml result object is missing a required field.

    The version guard in ``adapter.py`` (``LIZYML_MIN_VERSION`` /
    ``LIZYML_MAX_VERSION``) only catches major-version drift. Field-level
    contract changes between supported minors surface here as a fail-fast
    error rather than silent ``None``s deeper in the widget.
    """


def _require(obj: Any, name: str, *, optional: bool = False) -> Any:
    """Read ``obj.<name>``; raise ``LizyMLContractError`` if missing."""
    if not hasattr(obj, name):
        if optional:
            return None
        msg = (
            f"lizyml object {type(obj).__name__!r} is missing required "
            f"attribute {name!r} — backend contract drift?"
        )
        raise LizyMLContractError(msg)
    return getattr(obj, name)


# ── Tune ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LMRoundView:
    """One Round entry from a ``TuningResult.rounds`` sequence."""

    round: int
    n_trials: int
    best_score_before: float | None
    best_score_after: float
    expanded_dims: tuple[str, ...]


@dataclass(frozen=True)
class LMBoundaryDimView:
    """One dimension entry from a ``BoundaryReport.dims`` sequence."""

    name: str
    best_value: Any
    low: Any
    high: Any
    position_pct: float | None
    edge: str
    expanded: bool
    new_low: Any
    new_high: Any


@dataclass(frozen=True)
class LMBoundaryReportView:
    """Wraps lizyml's ``BoundaryReport`` for the widget.

    INV (#118): every dim in :attr:`dims` is unique by ``name``. Construction
    here does not enforce uniqueness — that's a backend invariant — but the
    field-level type makes it explicit so a future invariant test can lock
    it in.
    """

    dims: tuple[LMBoundaryDimView, ...]
    expanded_names: tuple[str, ...]


@dataclass(frozen=True)
class LMTuneProgressView:
    """Per-trial progress info from ``TuneProgressInfo``."""

    current_trial: int
    total_trials: int
    round: int
    cumulative_trials: int
    expanded_dims: tuple[str, ...]
    latest_score: float | None
    latest_state: str | None
    best_score: float | None


@dataclass(frozen=True)
class LMTuningResultView:
    """Wraps lizyml's ``TuningResult``."""

    best_params: dict[str, Any]
    best_score: float
    metric_name: str
    direction: str
    rounds: tuple[LMRoundView, ...]
    boundary_report: LMBoundaryReportView | None
    raw_trials: Any  # opaque — passed through to _serialize_trials


# ── Fit / Prediction ────────────────────────────────────────────────────


@dataclass(frozen=True)
class LMFitResultView:
    """Wraps lizyml's ``FitResult`` for the fields the widget uses."""

    metrics: dict[str, Any]
    fold_count: int


@dataclass(frozen=True)
class LMPredictionResultView:
    """Wraps lizyml's ``PredictionResult``."""

    pred: Any
    proba: Any | None
    shap_values: Any | None
    warnings: list[str] = field(default_factory=list)


# ── Factory functions ───────────────────────────────────────────────────


def view_rounds(rounds: Any) -> tuple[LMRoundView, ...]:
    """Wrap a tuple of ``RoundSummary`` into ``LMRoundView`` instances."""
    if not rounds:
        return ()
    out: list[LMRoundView] = []
    for r in rounds:
        out.append(
            LMRoundView(
                round=int(_require(r, "round")),
                n_trials=int(_require(r, "n_trials")),
                best_score_before=_require(r, "best_score_before", optional=True),
                best_score_after=float(_require(r, "best_score_after")),
                expanded_dims=tuple(_require(r, "expanded_dims") or ()),
            )
        )
    return tuple(out)


def view_boundary_report(report: Any) -> LMBoundaryReportView | None:
    """Wrap a ``BoundaryReport`` (or None) into ``LMBoundaryReportView``."""
    if report is None:
        return None
    raw_dims = _require(report, "dims") or ()
    dims = tuple(
        LMBoundaryDimView(
            name=str(_require(d, "name")),
            best_value=_require(d, "best_value", optional=True),
            low=_require(d, "low", optional=True),
            high=_require(d, "high", optional=True),
            position_pct=_require(d, "position_pct", optional=True),
            edge=str(_require(d, "edge", optional=True) or ""),
            expanded=bool(_require(d, "expanded", optional=True) or False),
            new_low=_require(d, "new_low", optional=True),
            new_high=_require(d, "new_high", optional=True),
        )
        for d in raw_dims
    )
    expanded_names = tuple(_require(report, "expanded_names", optional=True) or ())
    return LMBoundaryReportView(dims=dims, expanded_names=expanded_names)


def view_tune_progress(info: Any) -> LMTuneProgressView:
    """Wrap a ``TuneProgressInfo`` per-trial event."""
    current_trial = int(_require(info, "current_trial"))
    total_trials = int(_require(info, "total_trials"))
    # Re-tune fields are optional on older minors but lizyml>=0.9.0 always
    # supplies them. Default to single-round values when absent so the
    # widget traitlet contract stays stable.
    round_no = int(_require(info, "round", optional=True) or 1)
    cumulative = int(_require(info, "cumulative_trials", optional=True) or current_trial)
    return LMTuneProgressView(
        current_trial=current_trial,
        total_trials=total_trials,
        round=round_no,
        cumulative_trials=cumulative,
        expanded_dims=tuple(_require(info, "expanded_dims", optional=True) or ()),
        latest_score=_require(info, "latest_score", optional=True),
        latest_state=_require(info, "latest_state", optional=True),
        best_score=_require(info, "best_score", optional=True),
    )


def view_tuning_result(result: Any) -> LMTuningResultView:
    """Wrap a ``TuningResult`` into the typed widget view."""
    return LMTuningResultView(
        best_params=dict(_require(result, "best_params")),
        best_score=float(_require(result, "best_score")),
        metric_name=str(_require(result, "metric_name")),
        direction=str(_require(result, "direction")),
        rounds=view_rounds(_require(result, "rounds", optional=True) or ()),
        boundary_report=view_boundary_report(_require(result, "boundary_report", optional=True)),
        raw_trials=_require(result, "trials"),
    )


def view_fit_result(result: Any) -> LMFitResultView:
    """Wrap a ``FitResult`` for the fields the widget uses."""
    metrics = _require(result, "metrics")
    splits = _require(result, "splits", optional=True)
    outer = _require(splits, "outer", optional=True) if splits is not None else None
    fold_count = len(outer) if outer is not None else 0
    return LMFitResultView(metrics=metrics, fold_count=fold_count)


def view_prediction_result(result: Any) -> LMPredictionResultView:
    """Wrap a ``PredictionResult`` into the typed widget view."""
    return LMPredictionResultView(
        pred=_require(result, "pred"),
        proba=_require(result, "proba", optional=True),
        shap_values=_require(result, "shap_values", optional=True),
        warnings=list(_require(result, "warnings", optional=True) or []),
    )
