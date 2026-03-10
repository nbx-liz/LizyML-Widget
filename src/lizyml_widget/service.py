"""WidgetService — state management and adapter orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .adapter import BackendAdapter
from .types import (
    BackendInfo,
    ConfigSchema,
    FitSummary,
    PlotData,
    PredictionSummary,
    TuningSummary,
)


class WidgetService:
    """Manages widget state and delegates to BackendAdapter.

    This layer owns data management, auto-detection logic,
    and config construction. It does NOT touch traitlets or
    any frontend concepts.
    """

    def __init__(self, adapter: BackendAdapter) -> None:
        self._adapter = adapter
        self._df: pd.DataFrame | None = None
        self._df_info: dict[str, Any] = {}
        self._model: Any = None

    @property
    def info(self) -> BackendInfo:
        return self._adapter.info

    # ── Data management ──────────────────────────────────────

    def load_data(self, df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
        """Load a DataFrame and compute column metadata."""
        self._df = df
        self._model = None

        columns: list[dict[str, Any]] = []
        for col in df.columns:
            columns.append(
                {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                }
            )

        self._df_info = {
            "shape": list(df.shape),
            "target": None,
            "task": None,
            "columns": columns,
            "cv": {"strategy": "kfold", "n_splits": 5, "group_column": None},
            "feature_summary": self._calc_feature_summary(columns),
        }

        if target:
            self.set_target(target)

        return self._df_info

    def set_target(self, target: str) -> dict[str, Any]:
        """Set target column, auto-detect task, auto-configure columns."""
        df = self._df
        if df is None:
            msg = "No data loaded"
            raise ValueError(msg)

        col = df[target]
        n_rows = len(df)
        n_unique = int(col.nunique())

        task = self._detect_task(col, n_rows, n_unique)

        updated_columns: list[dict[str, Any]] = []
        for c in self._df_info["columns"]:
            if c["name"] == target:
                continue  # exclude target from feature columns
            updated_columns.append(self._auto_configure_column(c, n_rows))

        cv_strategy = "stratified_kfold" if task in ("binary", "multiclass") else "kfold"

        self._df_info.update(
            {
                "target": target,
                "task": task,
                "columns": updated_columns,
                "cv": {**self._df_info["cv"], "strategy": cv_strategy},
                "feature_summary": self._calc_feature_summary(updated_columns),
            }
        )
        return self._df_info

    def update_column(self, name: str, *, excluded: bool, col_type: str) -> dict[str, Any]:
        """Update a single column's settings."""
        for c in self._df_info["columns"]:
            if c["name"] == name:
                c["excluded"] = excluded
                c["col_type"] = col_type
                if not excluded:
                    c["exclude_reason"] = None
                break
        self._df_info["feature_summary"] = self._calc_feature_summary(self._df_info["columns"])
        return self._df_info

    def update_cv(
        self,
        strategy: str,
        n_splits: int,
        group_column: str | None = None,
    ) -> dict[str, Any]:
        """Update cross-validation settings."""
        self._df_info["cv"] = {
            "strategy": strategy,
            "n_splits": n_splits,
            "group_column": group_column,
        }
        return self._df_info

    # ── Config ───────────────────────────────────────────────

    def get_config_schema(self) -> ConfigSchema:
        return self._adapter.get_config_schema()

    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return self._adapter.validate_config(config)

    def build_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """Merge df_info settings with user config for the adapter."""
        info = self._df_info
        active_cols = [c for c in info["columns"] if not c.get("excluded", False)]

        data_section = {
            "target": info["target"],
            "task": info["task"],
        }
        features_section = {
            "categorical": [c["name"] for c in active_cols if c.get("col_type") == "categorical"],
            "exclude": [c["name"] for c in info["columns"] if c.get("excluded", False)],
        }

        cv = info["cv"]
        split_section: dict[str, Any] = {
            "method": cv["strategy"],
            "n_splits": cv["n_splits"],
        }
        if cv.get("group_column"):
            split_section["group_col"] = cv["group_column"]

        return {
            **user_config,
            "data": {**user_config.get("data", {}), **data_section},
            "features": {**user_config.get("features", {}), **features_section},
            "split": {**user_config.get("split", {}), **split_section},
        }

    # ── Execution ────────────────────────────────────────────

    def fit(
        self,
        config: dict[str, Any],
        *,
        on_progress: Callable[..., Any] | None = None,
    ) -> FitSummary:
        """Create model and run fit."""
        if self._df is None:
            msg = "No data loaded"
            raise ValueError(msg)
        model = self._adapter.create_model(config, self._df)
        result = self._adapter.fit(model, on_progress=on_progress)
        self._model = model
        return result

    def tune(
        self,
        config: dict[str, Any],
        *,
        on_progress: Callable[..., Any] | None = None,
    ) -> TuningSummary:
        """Create model and run tune."""
        if self._df is None:
            msg = "No data loaded"
            raise ValueError(msg)
        model = self._adapter.create_model(config, self._df)
        result = self._adapter.tune(model, on_progress=on_progress)
        self._model = model
        return result

    def predict(self, data: pd.DataFrame, *, return_shap: bool = False) -> PredictionSummary:
        """Run prediction on new data."""
        if self._model is None:
            msg = "No trained model. Run fit or tune first."
            raise ValueError(msg)
        return self._adapter.predict(self._model, data, return_shap=return_shap)

    # ── Results ──────────────────────────────────────────────

    def get_plot(self, plot_type: str) -> PlotData:
        if self._model is None:
            msg = "No trained model"
            raise ValueError(msg)
        return self._adapter.plot(self._model, plot_type)

    def get_available_plots(self) -> list[str]:
        if self._model is None:
            return []
        return self._adapter.available_plots(self._model)

    def get_evaluate_table(self) -> list[dict[str, Any]]:
        if self._model is None:
            return []
        return self._adapter.evaluate_table(self._model)

    def get_split_summary(self) -> list[dict[str, Any]]:
        if self._model is None:
            return []
        return self._adapter.split_summary(self._model)

    def get_importance(self, kind: str = "split") -> dict[str, float]:
        if self._model is None:
            return {}
        return self._adapter.importance(self._model, kind)

    def get_model(self) -> Any:
        return self._model

    # ── Auto-detection internals ─────────────────────────────

    @staticmethod
    def _detect_task(col: pd.Series[Any], n_rows: int, n_unique: int) -> str:
        """Auto-detect task type from target column."""
        if n_unique == 2:
            return "binary"
        if str(col.dtype) in ("object", "str", "string", "category"):
            return "multiclass"
        if pd.api.types.is_numeric_dtype(col):
            threshold = max(20, int(n_rows * 0.05))
            if n_unique <= threshold:
                return "multiclass"
            return "regression"
        return "multiclass"

    def _auto_configure_column(self, col_info: dict[str, Any], n_rows: int) -> dict[str, Any]:
        """Auto-configure a single column (exclusion + type)."""
        name = col_info["name"]
        dtype = col_info["dtype"]
        unique = col_info["unique_count"]

        excluded = False
        exclude_reason: str | None = None
        col_type = "numeric"

        # ID detection
        if unique == n_rows:
            excluded = True
            exclude_reason = "id"
        # Constant detection
        elif unique == 1:
            excluded = True
            exclude_reason = "constant"

        # Type detection
        if dtype in ("object", "str", "string", "category", "bool"):
            col_type = "categorical"
        elif self._df is not None and pd.api.types.is_numeric_dtype(self._df[name]):
            threshold = max(20, int(n_rows * 0.05))
            if unique <= threshold:
                col_type = "categorical"

        return {
            **col_info,
            "suggested_type": col_type,
            "suggested_excluded": excluded,
            "exclude_reason": exclude_reason,
            "excluded": excluded,
            "col_type": col_type,
        }

    @staticmethod
    def _calc_feature_summary(columns: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate feature summary counts."""
        active = [c for c in columns if not c.get("excluded", False)]
        excluded = [c for c in columns if c.get("excluded", False)]
        return {
            "total": len(active),
            "numeric": sum(1 for c in active if c.get("col_type") == "numeric"),
            "categorical": sum(1 for c in active if c.get("col_type") == "categorical"),
            "excluded": len(excluded),
            "excluded_id": sum(1 for c in excluded if c.get("exclude_reason") == "id"),
            "excluded_const": sum(1 for c in excluded if c.get("exclude_reason") == "constant"),
            "excluded_manual": sum(1 for c in excluded if c.get("exclude_reason") is None),
        }
