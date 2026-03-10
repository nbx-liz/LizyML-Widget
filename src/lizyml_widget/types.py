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
class ConfigSchema:
    """JSON Schema for backend config."""

    json_schema: dict[str, Any]


@dataclass
class FitSummary:
    """Summary of a fit run."""

    metrics: dict[str, Any]  # nested: {"raw": {"oof": {...}, ...}, "calibrated": ...}
    fold_count: int
    params: list[dict[str, Any]]  # per-fold or global params


@dataclass
class TuningSummary:
    """Summary of a tuning run."""

    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]]  # [{number, params, score, state}, ...]
    metric_name: str
    direction: str  # "minimize" | "maximize"


@dataclass
class PredictionSummary:
    """Summary of a prediction run."""

    predictions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlotData:
    """Plotly figure as JSON string."""

    plotly_json: str  # fig.to_json()
