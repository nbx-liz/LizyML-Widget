"""WidgetService — state management and adapter orchestration."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from .adapter import BackendAdapter
from .types import (
    BackendContract,
    BackendInfo,
    ConfigPatchOp,
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

    def has_data(self) -> bool:
        """Return whether training data has been loaded."""
        return self._df is not None

    def has_target(self) -> bool:
        """Return whether a target column is currently selected."""
        return bool(self._df_info.get("target"))

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
            "auto_task": None,
            "columns": columns,
            "cv": self._default_cv_state(strategy="kfold", n_splits=5),
            "feature_summary": self._calc_feature_summary(columns),
        }

        if target:
            self.set_target(target)

        return copy.deepcopy(self._df_info)

    def set_target(self, target: str) -> dict[str, Any]:
        """Set target column, auto-detect task, auto-configure columns."""
        df = self._df
        if df is None:
            msg = "No data loaded"
            raise ValueError(msg)
        if target not in df.columns:
            msg = f"Unknown target column: {target}"
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

        self._df_info.update(
            {
                "target": target,
                "columns": updated_columns,
                "feature_summary": self._calc_feature_summary(updated_columns),
            }
        )
        self._apply_task_defaults(task, update_auto_task=True)
        return copy.deepcopy(self._df_info)

    def set_task(self, task: str) -> dict[str, Any]:
        """Override auto-detected task type."""
        valid_tasks = {"binary", "multiclass", "regression"}
        if task not in valid_tasks:
            msg = f"Invalid task: {task}. Must be one of {valid_tasks}"
            raise ValueError(msg)

        self._apply_task_defaults(task, update_auto_task=False)
        return copy.deepcopy(self._df_info)

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
        return copy.deepcopy(self._df_info)

    def update_cv(
        self,
        strategy: str,
        n_splits: int,
        *,
        group_column: str | None = None,
        time_column: str | None = None,
        random_state: int | None = 42,
        shuffle: bool | None = True,
        gap: int = 0,
        purge_gap: int = 0,
        embargo: int = 0,
        train_size_max: int | None = None,
        test_size_max: int | None = None,
    ) -> dict[str, Any]:
        """Update cross-validation settings."""
        self._df_info["cv"] = {
            "strategy": strategy,
            "n_splits": n_splits,
            "group_column": group_column,
            "time_column": time_column,
            "random_state": random_state,
            "shuffle": shuffle,
            "gap": gap,
            "purge_gap": purge_gap,
            "embargo": embargo,
            "train_size_max": train_size_max,
            "test_size_max": test_size_max,
        }
        return copy.deepcopy(self._df_info)

    def get_df_info(self) -> dict[str, Any]:
        """Return a copy of the current df_info state."""
        return copy.deepcopy(self._df_info)

    def get_backend_contract(self) -> BackendContract:
        """Return the backend contract (delegated to adapter)."""
        return self._adapter.get_backend_contract()

    def initialize_config(self) -> dict[str, Any]:
        """Build the initial widget config (delegated to adapter)."""
        return self._adapter.initialize_config(task=self._df_info.get("task"))

    def apply_config_patch(
        self, config: dict[str, Any], ops: Sequence[ConfigPatchOp]
    ) -> dict[str, Any]:
        """Apply config patch operations (delegated to adapter)."""
        return self._adapter.apply_config_patch(config, ops, task=self._df_info.get("task"))

    def canonicalize_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize a partial/full config via adapter defaults (delegated to adapter)."""
        return self._adapter.canonicalize_config(config, task=self._df_info.get("task"))

    def apply_task_params(self, config: dict[str, Any], task: str) -> dict[str, Any]:
        """Re-initialize config with new task to apply task-dependent defaults."""
        new_config = self._adapter.initialize_config(task=task)
        # Extract task-dependent params from initialized config
        new_params = new_config.get("model", {}).get("params", {})
        task_keys = {"objective", "metric"}
        task_params = {k: v for k, v in new_params.items() if k in task_keys}

        if not task_params:
            return copy.deepcopy(config)

        ops = [
            ConfigPatchOp(op="set", path=f"model.params.{k}", value=v)
            for k, v in task_params.items()
        ]
        return self._adapter.apply_config_patch(config, ops, task=task)

    # ── Config ───────────────────────────────────────────────

    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return self._adapter.validate_config(config)

    def build_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """Merge df_info settings with user config for the adapter."""
        info = self._df_info
        active_cols = [c for c in info["columns"] if not c.get("excluded", False)]

        data_section: dict[str, Any] = {
            "target": info["target"],
        }

        cv = info["cv"]
        strategy = cv["strategy"]

        # group_col / time_col belong in data section per BLUEPRINT §5.2
        if cv.get("group_column"):
            data_section["group_col"] = cv["group_column"]
        if cv.get("time_column"):
            data_section["time_col"] = cv["time_column"]

        features_section = {
            "categorical": [c["name"] for c in active_cols if c.get("col_type") == "categorical"],
            "exclude": [c["name"] for c in info["columns"] if c.get("excluded", False)],
        }

        split_section: dict[str, Any] = {
            "method": strategy,
            "n_splits": cv["n_splits"],
        }

        # Strategy-dependent split fields per BLUEPRINT §5.2
        if strategy in ("kfold", "stratified_kfold"):
            split_section["random_state"] = cv.get("random_state", 42)
        if strategy == "kfold":
            split_section["shuffle"] = cv.get("shuffle", True)
        if strategy in ("time_series", "group_time_series"):
            split_section["gap"] = cv.get("gap", 0)
        if strategy == "purged_time_series":
            split_section["purge_gap"] = cv.get("purge_gap", 0)
            split_section["embargo"] = cv.get("embargo", 0)
        if strategy in ("time_series", "purged_time_series", "group_time_series"):
            if cv.get("train_size_max") is not None:
                split_section["train_size_max"] = cv["train_size_max"]
            if cv.get("test_size_max") is not None:
                split_section["test_size_max"] = cv["test_size_max"]

        result = {
            **user_config,
            "task": info["task"],
            "data": {**user_config.get("data", {}), **data_section},
            "features": {**user_config.get("features", {}), **features_section},
            "split": {**user_config.get("split", {}), **split_section},
        }
        # Remove task from data section if it leaked from user_config
        result.get("data", {}).pop("task", None)

        # Ensure required top-level fields are present (BLUEPRINT §5.3)
        result.setdefault("config_version", 1)
        result.setdefault("model", {}).setdefault("name", "lgbm")

        return result

    def apply_loaded_config(self, loaded_config: dict[str, Any]) -> dict[str, Any]:
        """Apply parsed config to service state; return canonicalized widget-owned keys."""
        loaded = copy.deepcopy(loaded_config)
        data_section = loaded.pop("data", {})
        loaded.pop("features", {})  # consumed by build_config from df_info
        split_section = loaded.pop("split", {})
        loaded.pop("task", None)

        if "target" in data_section and self.has_data():
            self.set_target(data_section["target"])

        if split_section:
            strategy = split_section.get("method", split_section.get("strategy", "kfold"))
            n_splits = split_section.get("n_splits", 5)
            self.update_cv(
                strategy,
                n_splits,
                group_column=split_section.get("group_col", split_section.get("group_column")),
                time_column=data_section.get("time_col"),
                random_state=split_section.get("random_state", 42),
                shuffle=split_section.get("shuffle", True),
                gap=split_section.get("gap", 0),
                purge_gap=split_section.get("purge_gap", 0),
                embargo=split_section.get("embargo", 0),
                train_size_max=split_section.get("train_size_max"),
                test_size_max=split_section.get("test_size_max"),
            )

        return self.canonicalize_config(loaded)

    def prepare_run_config(self, user_config: dict[str, Any], *, job_type: str) -> dict[str, Any]:
        """Build the executable config for a job, delegating backend defaults to adapter."""
        full_config = self.build_config(copy.deepcopy(user_config))
        return self._adapter.prepare_run_config(
            full_config,
            job_type=job_type,  # type: ignore[arg-type]
            task=self._df_info.get("task"),
        )

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

    def get_inference_plot(self, predictions: pd.DataFrame, plot_type: str) -> PlotData:
        """Generate an inference plot (prediction-distribution or shap-summary)."""
        plot_fn = getattr(self._adapter, "plot_inference", None)
        if plot_fn is None:
            msg = "Inference plots not supported by this adapter"
            raise TypeError(msg)
        return plot_fn(predictions, plot_type)  # type: ignore[no-any-return]

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

    def save_model(self, path: str) -> str:
        """Persist the current trained model using the active adapter."""
        if self._model is None:
            msg = "No trained model. Run fit or tune first."
            raise ValueError(msg)
        return self._adapter.export_model(self._model, path)

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

    @staticmethod
    def _default_strategy_for_task(task: str) -> str:
        return "stratified_kfold" if task in ("binary", "multiclass") else "kfold"

    @staticmethod
    def _default_cv_state(*, strategy: str, n_splits: int) -> dict[str, Any]:
        return {
            "strategy": strategy,
            "n_splits": n_splits,
            "group_column": None,
            "time_column": None,
            "random_state": 42,
            "shuffle": True,
            "gap": 0,
            "purge_gap": 0,
            "embargo": 0,
            "train_size_max": None,
            "test_size_max": None,
        }

    def _apply_task_defaults(self, task: str, *, update_auto_task: bool) -> None:
        cv = self._df_info.get("cv", {})
        n_splits = int(cv.get("n_splits", 5))
        strategy = self._default_strategy_for_task(task)

        self._df_info["task"] = task
        if update_auto_task:
            self._df_info["auto_task"] = task
        self._df_info["cv"] = self._default_cv_state(strategy=strategy, n_splits=n_splits)

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
