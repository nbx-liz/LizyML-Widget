"""Result-side helpers for LizyMLAdapter (plots + inference rendering).

Pure functions extracted from adapter.py (#137) so the adapter module
stays under the 800-line ceiling. These helpers are stateless: ``model``
and any registry data are passed in by the caller.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import pandas as pd

from .types import PlotData


def task_for_model(model: Any, model_configs: dict[int, dict[str, Any]]) -> str:
    """Return the task ('binary'/'multiclass'/'regression') for *model*.

    Centralises the legacy ``_cfg.task`` private read with a graceful
    fallback to the adapter-side config registry. Returns ``""`` if no
    source can be resolved (caller treats this as "unknown task").
    """
    try:
        return str(model._cfg.task)  # noqa: SLF001
    except AttributeError:
        pass
    cfg = model_configs.get(id(model))
    if cfg is not None:
        task = cfg.get("task", "")
        if isinstance(task, str):
            return task
    return ""


def render_plot(model: Any, plot_type: str, **kwargs: Any) -> PlotData:
    """Dispatch to the model's plotting method for *plot_type*."""
    plot_methods: dict[str, Callable[..., Any]] = {
        "learning-curve": model.plot_learning_curve,
        "oof-distribution": model.plot_oof_distribution,
        "residuals": model.residuals_plot,
        "roc-curve": model.roc_curve_plot,
        "calibration": model.calibration_plot,
        "probability-histogram": model.probability_histogram_plot,
        "feature-importance-split": lambda: model.importance_plot(kind="split"),
        "feature-importance-gain": lambda: model.importance_plot(kind="gain"),
        "feature-importance-shap": lambda: model.importance_plot(kind="shap"),
        "optimization-history": model.tuning_plot,
    }
    method = plot_methods.get(plot_type)
    if method is None:
        msg = f"Unknown plot type: {plot_type}"
        raise ValueError(msg)
    # Forward metrics filter to learning-curve only (P-026)
    if plot_type == "learning-curve":
        lc_kwargs: dict[str, Any] = {}
        metrics = kwargs.get("metrics")
        if metrics:  # skip None and empty list → fall back to all metrics
            lc_kwargs["metrics"] = metrics
        fig = method(**lc_kwargs) if lc_kwargs else method()
    else:
        fig = method()
    return PlotData(plotly_json=fig.to_json())


def list_available_plots(model: Any, task: str) -> list[str]:
    """Return the list of plot types available for the current model + task."""
    is_fitted = False
    with contextlib.suppress(Exception):
        is_fitted = model.fit_result is not None

    has_calibration = False
    with contextlib.suppress(Exception):
        has_calibration = is_fitted and model.fit_result.calibrator is not None

    # Feature-detection probe — model._tuning_result is a private slot
    # lizyml exposes for tuning state. The presence check is intentional
    # and does not read data off the value, so it is allowed by #116.
    has_tuning = (
        hasattr(model, "_tuning_result") and model._tuning_result is not None  # noqa: SLF001
    )

    plots: list[str] = []

    if is_fitted:
        plots.extend(["learning-curve", "oof-distribution"])
        if task == "regression":
            plots.append("residuals")
        if task == "binary":
            plots.append("roc-curve")
            if has_calibration:
                plots.append("calibration")
                plots.append("probability-histogram")
        if task == "multiclass":
            plots.append("roc-curve")
        plots.extend(
            [
                "feature-importance-split",
                "feature-importance-gain",
                "feature-importance-shap",
            ]
        )

    if has_tuning:
        plots.append("optimization-history")
    return plots


def render_inference_plot(predictions: pd.DataFrame, plot_type: str) -> PlotData:
    """Generate a Plotly plot from inference results (post-predict)."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError as e:
        msg = "plotly is required for inference plots"
        raise ImportError(msg) from e

    if plot_type == "prediction-distribution":
        # Find prediction column: prefer columns starting with "pred",
        # fallback to last column.
        pred_col = next(
            (c for c in predictions.columns if c.startswith("pred")),
            predictions.columns[-1],
        )
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=predictions[pred_col], name="Predictions"))
        fig.update_layout(
            title="Prediction Distribution",
            xaxis_title="Predicted Value",
            yaxis_title="Count",
        )
        return PlotData(plotly_json=fig.to_json())

    if plot_type == "shap-summary":
        shap_cols = [c for c in predictions.columns if c.startswith("shap_")]
        if not shap_cols:
            msg = "No SHAP values available. Run inference with return_shap=True."
            raise ValueError(msg)
        mean_abs_shap = predictions[shap_cols].abs().mean().sort_values(ascending=True)
        feature_names = [c.replace("shap_", "", 1) for c in mean_abs_shap.index]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=mean_abs_shap.values,
                y=feature_names,
                orientation="h",
                name="Mean |SHAP|",
            )
        )
        fig.update_layout(
            title="SHAP Summary",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            height=max(300, len(shap_cols) * 25),
        )
        return PlotData(plotly_json=fig.to_json())

    msg = f"Unknown inference plot type: {plot_type}"
    raise ValueError(msg)
