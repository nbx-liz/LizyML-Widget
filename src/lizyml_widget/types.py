"""Common types shared between Widget, Service, and Adapter layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BackendInfo:
    """ML backend identification."""

    name: str  # e.g. "lizyml"
    version: str  # e.g. "0.1.0"


@dataclass
class BackendContract:
    """Backend contract containing schema, UI metadata, and capabilities."""

    schema_version: int
    config_schema: dict[str, Any]  # JSON Schema for config validation
    ui_schema: dict[str, Any]  # UI-specific metadata (sections, option_sets, parameter_hints, etc.)
    capabilities: dict[str, Any]  # Feature flags (e.g. tune.allow_empty_space)


@dataclass
class ConfigPatchOp:
    """A single config patch operation."""

    op: str  # "set" | "unset" | "merge"
    path: str  # dot-separated path (e.g. "model.params.learning_rate")
    value: Any | None = None


@dataclass
class FitSummary:
    """Summary of a fit run."""

    metrics: dict[str, Any]  # nested: {"raw": {"oof": {...}, ...}, "calibrated": ...}
    fold_count: int
    params: list[dict[str, Any]]  # per-fold or global params


@dataclass
class TuningSummary:
    """Summary of a tuning run.

    Fields ``rounds`` and ``boundary_report`` require lizyml>=0.9.0 (re-tune
    support, proposal P-027). For single-round tuning both collapse to a
    single-element ``rounds`` list and ``boundary_report=None``.
    """

    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]]  # [{number, params, score, state, round}, ...]
    metric_name: str
    direction: str  # "minimize" | "maximize"
    rounds: list[dict[str, Any]] = field(default_factory=list)  # per-round summary
    boundary_report: dict[str, Any] | None = None  # None when no resume rounds ran


@dataclass
class PredictionSummary:
    """Summary of a prediction run."""

    predictions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlotData:
    """Plotly figure as JSON string."""

    plotly_json: str  # fig.to_json()
