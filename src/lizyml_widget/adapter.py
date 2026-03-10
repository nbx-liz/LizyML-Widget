"""BackendAdapter protocol and LizyML implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from .types import (
    BackendInfo,
    ConfigSchema,
    FitSummary,
    PlotData,
    PredictionSummary,
    TuningSummary,
)


class BackendAdapter(Protocol):
    """Protocol for ML backend adapters."""

    @property
    def info(self) -> BackendInfo: ...

    def get_config_schema(self) -> ConfigSchema: ...

    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]: ...

    def create_model(self, config: dict[str, Any], dataframe: pd.DataFrame) -> Any: ...

    def fit(
        self,
        model: Any,
        *,
        params: dict[str, Any] | None = None,
        on_progress: Callable[..., Any] | None = None,
    ) -> FitSummary: ...

    def tune(
        self,
        model: Any,
        *,
        on_progress: Callable[..., Any] | None = None,
    ) -> TuningSummary: ...

    def predict(
        self,
        model: Any,
        data: pd.DataFrame,
        *,
        return_shap: bool = False,
    ) -> PredictionSummary: ...

    def evaluate_table(self, model: Any) -> list[dict[str, Any]]: ...

    def split_summary(self, model: Any) -> list[dict[str, Any]]: ...

    def importance(self, model: Any, kind: str) -> dict[str, float]: ...

    def plot(self, model: Any, plot_type: str) -> PlotData: ...

    def available_plots(self, model: Any) -> list[str]: ...

    def export_model(self, model: Any, path: str) -> str: ...

    def load_model(self, path: str) -> Any: ...

    def model_info(self, model: Any) -> dict[str, Any]: ...


class LizyMLAdapter:
    """Adapter for the LizyML backend library."""

    @property
    def info(self) -> BackendInfo:
        import lizyml

        return BackendInfo(name="lizyml", version=lizyml.__version__)

    def get_config_schema(self) -> ConfigSchema:
        from lizyml.config.schema import LizyMLConfig

        return ConfigSchema(json_schema=LizyMLConfig.model_json_schema())

    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        from lizyml.config.schema import LizyMLConfig
        from pydantic import ValidationError

        try:
            LizyMLConfig(**config)
            return []
        except ValidationError as e:
            return [
                {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
                for err in e.errors()
            ]

    def create_model(self, config: dict[str, Any], dataframe: pd.DataFrame) -> Any:
        from lizyml.core.model import Model

        return Model(config, data=dataframe)

    def fit(
        self,
        model: Any,
        *,
        params: dict[str, Any] | None = None,
        on_progress: Callable[..., Any] | None = None,
    ) -> FitSummary:
        result = model.fit(params=params)
        return FitSummary(
            metrics=result.metrics,
            fold_count=len(result.splits.outer),
            params=model.params_table().reset_index().to_dict(orient="records"),
        )

    def tune(
        self,
        model: Any,
        *,
        on_progress: Callable[..., Any] | None = None,
    ) -> TuningSummary:
        from dataclasses import asdict

        result = model.tune()
        return TuningSummary(
            best_params=result.best_params,
            best_score=result.best_score,
            trials=[asdict(t) for t in result.trials],
            metric_name=result.metric_name,
            direction=result.direction,
        )

    def predict(
        self,
        model: Any,
        data: pd.DataFrame,
        *,
        return_shap: bool = False,
    ) -> PredictionSummary:
        result = model.predict(data, return_shap=return_shap)
        df = pd.DataFrame({"pred": result.pred})
        if result.proba is not None:
            df["proba"] = result.proba
        return PredictionSummary(predictions=df, warnings=result.warnings)

    def evaluate_table(self, model: Any) -> list[dict[str, Any]]:
        df: pd.DataFrame = model.evaluate_table()
        records: list[dict[str, Any]] = df.reset_index().to_dict(orient="records")
        return records

    def split_summary(self, model: Any) -> list[dict[str, Any]]:
        df: pd.DataFrame = model.split_summary()
        return list(df.to_dict(orient="records"))  # type: ignore[arg-type]

    def importance(self, model: Any, kind: str) -> dict[str, float]:
        return model.importance(kind=kind)  # type: ignore[no-any-return]

    def plot(self, model: Any, plot_type: str) -> PlotData:
        plot_methods: dict[str, Callable[..., Any]] = {
            "learning-curve": model.plot_learning_curve,
            "oof-distribution": model.plot_oof_distribution,
            "residuals": model.residuals_plot,
            "roc-curve": model.roc_curve_plot,
            "calibration": model.calibration_plot,
            "probability-histogram": model.probability_histogram_plot,
            "importance": lambda: model.importance_plot(kind="split"),
            "tuning": model.tuning_plot,
        }
        method = plot_methods.get(plot_type)
        if method is None:
            msg = f"Unknown plot type: {plot_type}"
            raise ValueError(msg)
        fig = method()
        return PlotData(plotly_json=fig.to_json())

    def available_plots(self, model: Any) -> list[str]:
        task: str = model._config.task  # noqa: SLF001
        has_calibration = model.fit_result.calibrator is not None
        has_tuning = hasattr(model, "_tuning_result") and model._tuning_result is not None  # noqa: SLF001

        plots = ["learning-curve", "oof-distribution"]
        if task == "regression":
            plots.append("residuals")
        if task == "binary":
            plots.append("roc-curve")
            if has_calibration:
                plots.append("calibration")
                plots.append("probability-histogram")
        if task == "multiclass":
            plots.append("roc-curve")
        plots.append("importance")
        if has_tuning:
            plots.append("tuning")
        return plots

    def export_model(self, model: Any, path: str) -> str:
        raise NotImplementedError("export_model not yet implemented")

    def load_model(self, path: str) -> Any:
        raise NotImplementedError("load_model not yet implemented")

    def model_info(self, model: Any) -> dict[str, Any]:
        raise NotImplementedError("model_info not yet implemented")
