"""WidgetService — state management and adapter orchestration."""

from __future__ import annotations

import copy
import logging
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

_log = logging.getLogger(__name__)


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

        # Map existing column settings to detect manual overrides
        existing_settings = {c["name"]: c for c in self._df_info["columns"]}

        # Rebuild from original DataFrame columns to restore previously excluded targets
        updated_columns: list[dict[str, Any]] = []
        for col_name in df.columns:
            if col_name == target:
                continue  # exclude new target from feature columns
            col_info = {
                "name": str(col_name),
                "dtype": str(df[col_name].dtype),
                "unique_count": int(df[col_name].nunique()),
            }
            configured = self._auto_configure_column(col_info, n_rows)
            # Preserve manual overrides from update_column()
            prev = existing_settings.get(col_name)
            if prev is not None:
                if prev.get("excluded") != prev.get("suggested_excluded"):
                    configured["excluded"] = prev["excluded"]
                    configured["exclude_reason"] = prev.get("exclude_reason")
                if prev.get("col_type") != prev.get("suggested_type"):
                    configured["col_type"] = prev["col_type"]
            updated_columns.append(configured)

        self._df_info = {
            **self._df_info,
            "target": target,
            "columns": updated_columns,
            "feature_summary": self._calc_feature_summary(updated_columns),
        }
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
        new_columns = [
            {
                **c,
                "excluded": excluded,
                "col_type": col_type,
                "exclude_reason": None if not excluded else c.get("exclude_reason"),
            }
            if c["name"] == name
            else c
            for c in self._df_info["columns"]
        ]
        self._df_info = {
            **self._df_info,
            "columns": new_columns,
            "feature_summary": self._calc_feature_summary(new_columns),
        }
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
        self._df_info = {
            **self._df_info,
            "cv": {
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
            },
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
        """Apply task-dependent defaults via adapter (no backend-specific keys here)."""
        return self._adapter.apply_task_defaults(config, task=task)

    # ── Config ───────────────────────────────────────────────

    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return self._adapter.validate_config(config)

    def build_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """Merge df_info settings with user config for the adapter."""
        info = self._df_info
        active_cols = [c for c in info["columns"] if not c.get("excluded", False)]

        if not active_cols:
            msg = "No features available. Un-exclude at least one column."
            raise ValueError(msg)

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
        if strategy in ("kfold", "stratified_kfold", "stratified_group_kfold"):
            split_section["random_state"] = cv.get("random_state", 42)
        if strategy in ("kfold", "stratified_group_kfold"):
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

        # Ensure required top-level fields via adapter canonicalization
        # (no backend-specific model name hardcoded here)
        defaults = self._adapter.initialize_config(task=info.get("task"))
        result.setdefault("config_version", defaults.get("config_version", 1))
        model_name = defaults.get("model", {}).get("name")
        if not model_name:
            msg = "Adapter's initialize_config() must return a non-empty model.name"
            raise ValueError(msg)
        result.setdefault("model", {}).setdefault("name", model_name)

        return result

    def apply_loaded_config(self, loaded_config: dict[str, Any]) -> dict[str, Any]:
        """Apply parsed config to service state; return canonicalized widget-owned keys."""
        loaded = copy.deepcopy(loaded_config)
        data_section = loaded.pop("data", {})
        features_section = loaded.pop("features", {})
        split_section = loaded.pop("split", {})
        task_override = loaded.pop("task", None)

        if "target" in data_section and self.has_data():
            self.set_target(data_section["target"])

        if split_section:
            strategy = split_section.get("method", split_section.get("strategy", "kfold"))
            n_splits = split_section.get("n_splits", 5)
            self.update_cv(
                strategy,
                n_splits,
                group_column=(
                    split_section.get("group_col")
                    or split_section.get("group_column")
                    or data_section.get("group_col")
                ),
                time_column=data_section.get("time_col"),
                random_state=split_section.get("random_state", 42),
                shuffle=split_section.get("shuffle", True),
                gap=split_section.get("gap", 0),
                purge_gap=split_section.get("purge_gap", 0),
                embargo=split_section.get("embargo", 0),
                train_size_max=split_section.get("train_size_max"),
                test_size_max=split_section.get("test_size_max"),
            )

        # Restore feature exclusions and categorical overrides (batch update)
        if features_section and self.has_data():
            exclude_set = set(features_section.get("exclude", []))
            categorical_set = set(features_section.get("categorical", []))
            target = self._df_info.get("target")
            new_columns = [
                (
                    {
                        **col,
                        "excluded": col["name"] in exclude_set,
                        "col_type": (
                            "categorical"
                            if col["name"] in categorical_set
                            else col.get("col_type", "numeric")
                        ),
                    }
                    if col["name"] != target
                    else col
                )
                for col in self._df_info["columns"]
            ]
            self._df_info = {
                **self._df_info,
                "columns": new_columns,
                "feature_summary": self._calc_feature_summary(new_columns),
            }

        # Restore explicit task override
        if task_override and self.has_data():
            self._df_info = {**self._df_info, "task": task_override}

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
        try:
            return self._adapter.plot_inference(predictions, plot_type)
        except AttributeError:
            msg = "Inference plots not supported by this adapter"
            raise TypeError(msg) from None

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

    def get_model(self) -> Any:
        return self._model

    def classify_best_params(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Classify best_params into (model, smart, training) categories.

        Delegates to the adapter (part of BackendAdapter Protocol per P-013).
        """
        return self._adapter.classify_best_params(params)

    def apply_best_params(
        self,
        params: dict[str, Any],
        current_config: dict[str, Any],
        *,
        tune_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply best_params from Tune to config, routing by category.

        Returns a canonicalized config with best params applied.
        Uses tune_snapshot as base if available, otherwise current_config.
        """
        if not params:
            return self.canonicalize_config(copy.deepcopy(current_config))

        # 1. Choose base config
        if tune_snapshot is not None:
            base = copy.deepcopy(tune_snapshot)
            for key in ("data", "features", "split", "task"):
                base.pop(key, None)
        else:
            base = copy.deepcopy(current_config)

        # 2. Classify params
        model_p, smart_p, training_p = self.classify_best_params(params)

        # 3. Merge model params → model.params
        model_section = dict(base.get("model", {}))
        model_params = {**dict(model_section.get("params", {})), **model_p}
        model_section = {**model_section, "params": model_params}

        # 4. Merge smart params → model.* level
        # smart_p already contains only smart-category keys (classified by adapter)
        base = {**base, "model": {**model_section, **smart_p}}

        # 5. Merge training params → training.early_stopping.*
        if training_p:
            training = dict(base.get("training", {}))
            es = dict(training.get("early_stopping", {}))
            es_updates: dict[str, Any] = {}
            if "early_stopping_rounds" in training_p:
                es_updates["rounds"] = training_p["early_stopping_rounds"]
            if "validation_ratio" in training_p:
                es_updates["validation_ratio"] = training_p["validation_ratio"]
                es_updates["inner_valid"] = None
            new_es = {**es, **es_updates}
            base = {**base, "training": {**training, "early_stopping": new_es}}

        return self.canonicalize_config(base)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the loaded DataFrame (for subprocess serialization)."""
        if self._df is None:
            msg = "No data loaded"
            raise ValueError(msg)
        return self._df

    def load_model_from_path(self, path: str) -> None:
        """Load a model from file path (for subprocess result recovery)."""
        self._model = self._adapter.load_model(path)

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

        new_cv = self._default_cv_state(strategy=strategy, n_splits=n_splits)
        # Preserve user-configured CV fields that are strategy-independent
        for key in (
            "group_column",
            "time_column",
            "random_state",
            "shuffle",
            "gap",
            "purge_gap",
            "embargo",
            "train_size_max",
            "test_size_max",
        ):
            if cv.get(key) is not None:
                new_cv[key] = cv[key]

        self._df_info = {
            **self._df_info,
            "task": task,
            **({"auto_task": task} if update_auto_task else {}),
            "cv": new_cv,
        }

    def _auto_configure_column(self, col_info: dict[str, Any], n_rows: int) -> dict[str, Any]:
        """Auto-configure a single column (exclusion + type)."""
        name = col_info["name"]
        dtype = col_info["dtype"]
        unique = col_info["unique_count"]

        excluded = False
        exclude_reason: str | None = None
        col_type = "numeric"

        # ID detection — only for integer/string columns, not floats
        is_float = dtype.startswith("float") or dtype.startswith("Float")
        if unique == n_rows and not is_float:
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
