"""BackendAdapter protocol and LizyML implementation."""

from __future__ import annotations

import contextlib
import copy
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import pandas as pd

from .types import (
    BackendContract,
    BackendInfo,
    ConfigPatchOp,
    FitSummary,
    PlotData,
    PredictionSummary,
    TuningSummary,
)


class BackendAdapter(Protocol):
    """Protocol for ML backend adapters."""

    @property
    def info(self) -> BackendInfo: ...

    def get_config_schema(self) -> dict[str, Any]: ...

    def get_backend_contract(self) -> BackendContract: ...

    def initialize_config(self, *, task: str | None = None) -> dict[str, Any]: ...

    def apply_config_patch(
        self,
        config: dict[str, Any],
        ops: Sequence[ConfigPatchOp],
        *,
        task: str | None = None,
    ) -> dict[str, Any]: ...

    def prepare_run_config(
        self,
        config: dict[str, Any],
        *,
        job_type: Literal["fit", "tune"],
        task: str | None = None,
    ) -> dict[str, Any]: ...

    def canonicalize_config(
        self, config: dict[str, Any], *, task: str | None = None
    ) -> dict[str, Any]: ...

    def apply_task_defaults(self, config: dict[str, Any], *, task: str) -> dict[str, Any]: ...

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

    def get_config_schema(self) -> dict[str, Any]:
        from lizyml.config.schema import LizyMLConfig

        return LizyMLConfig.model_json_schema()

    # ── Backend Contract & Config Lifecycle (Phase 25) ─────────

    # LightGBM model.params defaults for pre-population (moved from service.py)
    _LGBM_PARAMS_TASK_INDEPENDENT: dict[str, Any] = {
        "n_estimators": 1500,
        "learning_rate": 0.001,
        "max_depth": 5,
        "max_bin": 511,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 10,
        "lambda_l1": 0.0,
        "lambda_l2": 0.000001,
        "first_metric_only": False,
        "verbose": -1,
    }

    _LGBM_PARAMS_BY_TASK: dict[str, dict[str, Any]] = {
        "regression": {"objective": "huber", "metric": ["huber", "mae", "mape"]},
        "binary": {"objective": "binary", "metric": ["auc", "binary_logloss"]},
        "multiclass": {"objective": "multiclass", "metric": ["auc_mu", "multi_logloss"]},
    }

    def get_backend_contract(self) -> BackendContract:
        """Return the full backend contract with config schema, UI metadata, and capabilities."""
        config_schema = self.get_config_schema()

        ui_schema: dict[str, Any] = {
            "sections": [
                {"key": "model", "title": "Model"},
                {"key": "training", "title": "Training"},
                {"key": "evaluation", "title": "Evaluation"},
                {"key": "calibration", "title": "Calibration"},
            ],
            "option_sets": {
                "objective": {
                    "regression": ["huber", "mse", "mae", "quantile", "mape", "cross_entropy"],
                    "binary": ["binary", "cross_entropy", "cross_entropy_lambda"],
                    "multiclass": ["multiclass", "softmax", "multiclassova"],
                },
                "metric": {
                    "regression": ["huber", "mae", "mape", "mse", "rmse", "quantile"],
                    "binary": ["auc", "binary_logloss", "binary_error", "average_precision"],
                    "multiclass": ["auc_mu", "multi_logloss", "multi_error"],
                },
            },
            "parameter_hints": [
                {"key": "objective", "label": "Objective", "kind": "objective"},
                {"key": "metric", "label": "Metric", "kind": "metric"},
                {"key": "n_estimators", "label": "N Estimators", "kind": "integer", "step": 100},
                {"key": "learning_rate", "label": "Learning Rate", "kind": "number", "step": 0.001},
                {"key": "max_depth", "label": "Max Depth", "kind": "integer", "step": 1},
                {"key": "max_bin", "label": "Max Bin", "kind": "integer", "step": 1},
                {
                    "key": "feature_fraction",
                    "label": "Feature Fraction",
                    "kind": "number",
                    "step": 0.05,
                },
                {
                    "key": "bagging_fraction",
                    "label": "Bagging Fraction",
                    "kind": "number",
                    "step": 0.05,
                },
                {"key": "bagging_freq", "label": "Bagging Freq", "kind": "integer", "step": 1},
                {"key": "lambda_l1", "label": "Lambda L1", "kind": "number", "step": 0.0001},
                {"key": "lambda_l2", "label": "Lambda L2", "kind": "number", "step": 0.0001},
                {
                    "key": "first_metric_only",
                    "label": "First Metric Only",
                    "kind": "boolean",
                },
            ],
            "search_space_catalog": [
                {
                    "key": "objective",
                    "title": "Objective",
                    "paramType": "string",
                    "modes": ["fixed", "choice"],
                },
                {
                    "key": "metric",
                    "title": "Metric",
                    "paramType": "string",
                    "modes": ["fixed", "choice"],
                },
                {
                    "key": "n_estimators",
                    "title": "N Estimators",
                    "paramType": "integer",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "learning_rate",
                    "title": "Learning Rate",
                    "paramType": "number",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "max_depth",
                    "title": "Max Depth",
                    "paramType": "integer",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "max_bin",
                    "title": "Max Bin",
                    "paramType": "integer",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "feature_fraction",
                    "title": "Feature Fraction",
                    "paramType": "number",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "bagging_fraction",
                    "title": "Bagging Fraction",
                    "paramType": "number",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "bagging_freq",
                    "title": "Bagging Freq",
                    "paramType": "integer",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "lambda_l1",
                    "title": "Lambda L1",
                    "paramType": "number",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "lambda_l2",
                    "title": "Lambda L2",
                    "paramType": "number",
                    "modes": ["fixed", "range"],
                },
                {
                    "key": "first_metric_only",
                    "title": "First Metric Only",
                    "paramType": "boolean",
                    "modes": ["fixed", "choice"],
                },
            ],
            "step_map": {
                "n_estimators": 100,
                "learning_rate": 0.001,
                "max_depth": 1,
                "max_bin": 1,
                "feature_fraction": 0.05,
                "bagging_fraction": 0.05,
                "bagging_freq": 1,
                "lambda_l1": 0.0001,
                "lambda_l2": 0.0001,
                "num_leaves_ratio": 0.05,
                "num_leaves": 1,
            },
            "conditional_visibility": {
                "calibration": {"task": ["binary"]},
                "num_leaves_ratio": {"auto_num_leaves": True},
                "num_leaves": {"auto_num_leaves": False},
            },
            "defaults": {
                "calibration": {"method": "platt", "n_splits": 5, "params": {}},
            },
            "inner_valid_options": ["holdout", "group_holdout", "time_holdout"],
        }

        capabilities: dict[str, Any] = {
            "tune": {"allow_empty_space": True},
        }

        return BackendContract(
            schema_version=1,
            config_schema=config_schema,
            ui_schema=ui_schema,
            capabilities=capabilities,
        )

    def initialize_config(self, *, task: str | None = None) -> dict[str, Any]:
        """Build the full initial config dict with backend-specific defaults."""
        schema = self.get_config_schema()
        config = self._extract_defaults(schema)
        config.setdefault("config_version", 1)

        model_section = dict(config.get("model", {}))
        model_section.setdefault("name", "lgbm")
        auto_num_leaves = model_section.get("auto_num_leaves", True)

        params: dict[str, Any] = dict(self._LGBM_PARAMS_TASK_INDEPENDENT)
        if task and task in self._LGBM_PARAMS_BY_TASK:
            params.update(self._LGBM_PARAMS_BY_TASK[task])
        if not auto_num_leaves:
            params["num_leaves"] = 256

        model_section["params"] = {**params, **dict(model_section.get("params", {}))}
        config["model"] = model_section
        return config

    def apply_config_patch(
        self,
        config: dict[str, Any],
        ops: Sequence[ConfigPatchOp],
        *,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Apply a list of ConfigPatchOp to config and return the updated config."""
        result = copy.deepcopy(config)

        for op in ops:
            parts = op.path.split(".")
            if op.op == "set":
                self._set_nested(result, parts, op.value)
            elif op.op == "unset":
                self._unset_nested(result, parts)
            elif op.op == "merge":
                existing = self._get_nested(result, parts)
                if isinstance(existing, dict) and isinstance(op.value, dict):
                    self._set_nested(result, parts, {**existing, **op.value})
                else:
                    self._set_nested(result, parts, op.value)

        # ── Final canonical pass (single, after all ops) ──────
        # 1. Re-complete required fields
        result.setdefault("config_version", 1)
        if "model" not in result:
            defaults = self.initialize_config()
            result["model"] = defaults.get("model", {"name": "lgbm"})

        # Re-read model from result to avoid stale local references
        cur_model = result["model"]
        cur_model.setdefault("name", "lgbm")

        # 2. Enforce auto_num_leaves exclusivity
        result["model"] = self._enforce_auto_num_leaves(cur_model)

        # 3. Normalize inner_valid
        result = self._normalize_inner_valid_pure(result)
        return result

    def apply_task_defaults(self, config: dict[str, Any], *, task: str) -> dict[str, Any]:
        """Apply task-dependent default params to config.

        Returns a new config with task-specific objective/metric applied
        via patch operations. Backend-specific knowledge stays here.
        """
        defaults = self._LGBM_PARAMS_BY_TASK.get(task, {})
        if not defaults:
            return copy.deepcopy(config)

        ops = [
            ConfigPatchOp(op="set", path=f"model.params.{k}", value=v) for k, v in defaults.items()
        ]
        return self.apply_config_patch(config, ops, task=task)

    @staticmethod
    def _enforce_auto_num_leaves(model: dict[str, Any]) -> dict[str, Any]:
        """Return a new model dict with auto_num_leaves exclusivity enforced."""
        auto_nl = model.get("auto_num_leaves", True)
        params = dict(model.get("params", {}))
        if auto_nl:
            params.pop("num_leaves", None)
        elif "num_leaves" not in params:
            params["num_leaves"] = 256
        return {**model, "params": params}

    # ── inner_valid normalization ─────────────────────────────

    _INNER_VALID_ALIASES: set[str] = {"holdout", "group_holdout", "time_holdout"}

    # Allowed fields per inner_valid method (LizyML Pydantic schema, extra='forbid')
    _INNER_VALID_FIELDS: dict[str, set[str]] = {
        "holdout": {"method", "ratio", "random_state", "stratify"},
        "group_holdout": {"method", "ratio", "random_state"},
        "time_holdout": {"method", "ratio"},
    }

    @classmethod
    def _normalize_iv_value(cls, iv: Any) -> Any:
        """Return a normalized inner_valid value (pure, no mutation)."""
        if iv is None:
            return None
        if isinstance(iv, str):
            return {"method": iv} if iv in cls._INNER_VALID_ALIASES else None
        if isinstance(iv, dict):
            method = iv.get("method")
            allowed = cls._INNER_VALID_FIELDS.get(method or "")
            if allowed:
                return {k: v for k, v in iv.items() if k in allowed}
            if method is not None:
                return None
        return iv

    @classmethod
    def _normalize_inner_valid_pure(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Return a new config with inner_valid normalized (no mutation).

        - Converts legacy string format ("holdout") to dict {"method": "holdout"}.
        - Strips fields not allowed by the selected method's schema to prevent
          Pydantic 'Extra inputs are not permitted' validation errors.
        """
        training = config.get("training")
        if not isinstance(training, dict):
            return config
        es = training.get("early_stopping")
        if not isinstance(es, dict):
            return config
        iv = es.get("inner_valid")
        new_iv = cls._normalize_iv_value(iv)
        if new_iv is iv:
            return config
        return {
            **config,
            "training": {
                **training,
                "early_stopping": {**es, "inner_valid": new_iv},
            },
        }

    # ── Schema-based config sanitizer ────────────────────────

    _schema_cache: dict[str, Any] | None = None
    _schema_lock: threading.Lock = threading.Lock()

    @classmethod
    def _get_schema(cls) -> dict[str, Any]:
        """Lazily load and cache the LizyML JSON schema (thread-safe)."""
        if cls._schema_cache is not None:
            return cls._schema_cache
        with cls._schema_lock:
            if cls._schema_cache is None:
                from lizyml.config.schema import LizyMLConfig

                cls._schema_cache = LizyMLConfig.model_json_schema()
        return cls._schema_cache

    @classmethod
    def _resolve_ref(cls, ref: str, defs: dict[str, Any]) -> dict[str, Any]:
        """Resolve a $ref pointer like '#/$defs/FooConfig'."""
        name = ref.rsplit("/", 1)[-1]
        result: dict[str, Any] = defs.get(name, {})
        return result

    @classmethod
    def _resolve_sub_schema(
        cls, prop_schema: dict[str, Any], defs: dict[str, Any], value: Any
    ) -> dict[str, Any] | None:
        """Resolve a property schema to a concrete object schema, handling
        $ref, oneOf, anyOf, and discriminated unions."""
        if "$ref" in prop_schema:
            return cls._resolve_ref(prop_schema["$ref"], defs)

        # oneOf with discriminator (e.g., model, split, inner_valid)
        if "oneOf" in prop_schema and isinstance(value, dict):
            discriminator = prop_schema.get("discriminator", {})
            prop_name = discriminator.get("propertyName")
            mapping = discriminator.get("mapping", {})
            if prop_name and prop_name in value:
                ref = mapping.get(value[prop_name])
                if ref:
                    return cls._resolve_ref(ref, defs)
            # Fallback: try each oneOf variant
            for variant in prop_schema["oneOf"]:
                resolved = cls._resolve_sub_schema(variant, defs, value)
                if resolved and "properties" in resolved:
                    return resolved

        # anyOf (nullable types, union types)
        if "anyOf" in prop_schema and isinstance(value, dict):
            for variant in prop_schema["anyOf"]:
                resolved = cls._resolve_sub_schema(variant, defs, value)
                if resolved and "properties" in resolved:
                    return resolved

        # Direct object with properties
        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            return prop_schema

        return None

    @classmethod
    def _strip_to_schema(
        cls,
        config: dict[str, Any],
        obj_schema: dict[str, Any],
        defs: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively strip fields not in the schema.

        Only removes keys when the schema has additionalProperties=false.
        Recurses into nested objects to strip at all levels.
        """
        props = obj_schema.get("properties", {})
        additional = obj_schema.get("additionalProperties", True)

        result: dict[str, Any] = {}
        for key, value in config.items():
            # If additionalProperties is false, skip unknown keys
            if not additional and key not in props:
                continue

            if isinstance(value, dict) and key in props:
                sub = cls._resolve_sub_schema(props[key], defs, value)
                if sub and "properties" in sub:
                    value = cls._strip_to_schema(value, sub, defs)

            result[key] = value
        return result

    @classmethod
    def strip_for_backend(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Strip fields not recognized by LizyML schema.

        Uses the Pydantic JSON schema to whitelist allowed fields at each
        nesting level. Prevents 'Extra inputs are not permitted' errors.
        """
        schema = cls._get_schema()
        defs = schema.get("$defs", {})
        return cls._strip_to_schema(config, schema, defs)

    # ── Canonicalize config ───────────────────────────────────

    # Keys managed by Service (df_info / build_config), not the widget config traitlet
    _SERVICE_MANAGED_KEYS: set[str] = {"data", "features", "split", "task"}

    def canonicalize_config(
        self, config: dict[str, Any], *, task: str | None = None
    ) -> dict[str, Any]:
        """Canonicalize a partial/full config by merging with backend defaults."""
        defaults = self.initialize_config(task=task)
        result = self._deep_merge(defaults, copy.deepcopy(config))

        # Strip Service-managed keys (data/features/split/task)
        for key in self._SERVICE_MANAGED_KEYS:
            result.pop(key, None)

        # Enforce auto_num_leaves exclusivity
        result["model"] = self._enforce_auto_num_leaves(result.get("model", {}))

        result = self._normalize_inner_valid_pure(result)
        return result

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override into base. Override values take precedence."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = LizyMLAdapter._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def prepare_run_config(
        self,
        config: dict[str, Any],
        *,
        job_type: Literal["fit", "tune"],
        task: str | None = None,
    ) -> dict[str, Any]:
        """Prepare config for execution, applying backend-specific defaults."""
        result = copy.deepcopy(config)

        # Ensure model name
        model = result.get("model", {})
        if not model.get("name"):
            model["name"] = "lgbm"
            result["model"] = model

        # Enforce auto_num_leaves exclusivity
        result["model"] = self._enforce_auto_num_leaves(model)

        # Tune defaults
        if job_type == "tune":
            if not result.get("tuning"):
                result["tuning"] = {"optuna": {"params": {"n_trials": 50}, "space": {}}}
            else:
                optuna = result["tuning"].setdefault("optuna", {})
                optuna.setdefault("params", {}).setdefault("n_trials", 50)
                optuna.setdefault("space", {})

        result = self._normalize_inner_valid_pure(result)
        return self.strip_for_backend(result)

    # ── Config patch helpers ──────────────────────────────────

    @staticmethod
    def _get_nested(obj: dict[str, Any], parts: list[str]) -> Any:
        """Get a value at a dot-path inside a nested dict."""
        current: Any = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _set_nested(obj: dict[str, Any], parts: list[str], value: Any) -> None:
        """Set a value at a dot-path inside a nested dict, creating intermediates."""
        current = obj
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    @staticmethod
    def _unset_nested(obj: dict[str, Any], parts: list[str]) -> None:
        """Remove a key at a dot-path inside a nested dict."""
        current = obj
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return
            current = current[part]
        current.pop(parts[-1], None)

    @staticmethod
    def _extract_defaults(schema: dict[str, Any]) -> dict[str, Any]:
        """Walk a JSON Schema and extract default values into a config dict."""

        def _resolve(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
            if "$ref" in node:
                parts = node["$ref"].lstrip("#/").split("/")
                ref_node: Any = root
                for part in parts:
                    ref_node = ref_node.get(part, {})
                merged = dict(ref_node)
                merged.update({k: v for k, v in node.items() if k != "$ref"})
                return merged
            if "allOf" in node and len(node["allOf"]) == 1 and "$ref" in node["allOf"][0]:
                resolved = _resolve(node["allOf"][0], root)
                resolved.update({k: v for k, v in node.items() if k != "allOf"})
                return resolved
            return node

        def _walk(node: dict[str, Any], root: dict[str, Any]) -> Any:
            node = _resolve(node, root)
            if node.get("type") == "object" and "properties" in node:
                obj: dict[str, Any] = {}
                for key, prop in node["properties"].items():
                    prop = _resolve(prop, root)
                    if "default" in prop:
                        obj[key] = prop["default"]
                    elif prop.get("type") == "object" and "properties" in prop:
                        child = _walk(prop, root)
                        if child:
                            obj[key] = child
                return obj
            if "default" in node:
                return node["default"]
            return {}

        result = _walk(schema, schema)
        return result if isinstance(result, dict) else {}

    # ── Config validation ─────────────────────────────────────

    def validate_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        from lizyml.config.loader import load_config

        # Pre-validate search space format — catch legacy 'mode' format (P-004)
        tuning = config.get("tuning") or {}
        tuning_space = tuning.get("optuna", {}).get("space", {})
        space_errors: list[dict[str, Any]] = []
        for key, spec in tuning_space.items():
            if isinstance(spec, dict) and "mode" in spec and "type" not in spec:
                space_errors.append(
                    {
                        "field": f"tuning.optuna.space.{key}",
                        "message": (
                            f"Legacy search space format for '{key}': "
                            f"'mode={spec.get('mode')}'. Expected 'type' (float/int/categorical)."
                        ),
                        "type": "search_space_format",
                    }
                )
        # Check for invalid type values in search space
        _VALID_SPACE_TYPES = {"float", "int", "categorical"}
        for key, spec in tuning_space.items():
            if isinstance(spec, dict) and "type" in spec and spec["type"] not in _VALID_SPACE_TYPES:
                space_errors.append(
                    {
                        "field": f"tuning.optuna.space.{key}",
                        "message": (
                            f"Invalid search space type '{spec['type']}' for '{key}'. "
                            f"Expected one of: {', '.join(sorted(_VALID_SPACE_TYPES))}."
                        ),
                        "type": "invalid_space_type",
                    }
                )
        if space_errors:
            return space_errors

        # Normalize and strip non-schema fields before validation
        normalized = copy.deepcopy(config)
        normalized = self._normalize_inner_valid_pure(normalized)
        normalized = self.strip_for_backend(normalized)

        try:
            load_config(normalized)
            return []
        except Exception as e:
            # Extract structured validation details when available (Pydantic)
            errors: list[dict[str, Any]] = []
            # Check the exception itself, then walk __cause__ chain
            exc: BaseException | None = e
            while exc is not None and not errors:
                if hasattr(exc, "errors") and callable(exc.errors):
                    for err in exc.errors():
                        errors.append(
                            {
                                "field": ".".join(str(loc) for loc in err.get("loc", [])),
                                "message": err.get("msg", str(exc)),
                                "type": err.get("type", ""),
                            }
                        )
                    break
                exc = exc.__cause__
            if not errors:
                errors.append({"field": "", "message": str(e), "type": "unknown"})
            return errors

    def create_model(self, config: dict[str, Any], dataframe: pd.DataFrame) -> Any:
        from lizyml.core.model import Model

        model = Model(config, data=dataframe)
        model._widget_config = config  # type: ignore[attr-defined]  # noqa: SLF001
        return model

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
            "feature-importance": lambda: model.importance_plot(kind="split"),
            "optimization-history": model.tuning_plot,
        }
        method = plot_methods.get(plot_type)
        if method is None:
            msg = f"Unknown plot type: {plot_type}"
            raise ValueError(msg)
        fig = method()
        return PlotData(plotly_json=fig.to_json())

    def available_plots(self, model: Any) -> list[str]:
        # Extract task safely: try _cfg (LizyML internal), then widget fallback
        task: str = ""
        try:
            task = model._cfg.task  # noqa: SLF001
        except AttributeError:
            cfg = getattr(model, "_widget_config", {})
            task = cfg.get("task", "")

        # Check if model has been fitted (not just tuned) — P-004 R4
        is_fitted = False
        with contextlib.suppress(Exception):
            is_fitted = model.fit_result is not None

        has_calibration = False
        with contextlib.suppress(Exception):
            has_calibration = is_fitted and model.fit_result.calibrator is not None

        has_tuning = getattr(model, "_tuning_result", None) is not None

        plots: list[str] = []

        # Fit-dependent plots only when model is fitted
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
            plots.append("feature-importance")

        # Tune-dependent plots
        if has_tuning:
            plots.append("optimization-history")
        return plots

    def plot_inference(self, predictions: pd.DataFrame, plot_type: str) -> PlotData:
        """Generate Plotly plots from inference results (not part of Protocol)."""
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]
        except ImportError as e:
            msg = "plotly is required for inference plots"
            raise ImportError(msg) from e

        if plot_type == "prediction-distribution":
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=predictions["pred"], name="Predictions"))
            fig.update_layout(
                title="Prediction Distribution",
                xaxis_title="Predicted Value",
                yaxis_title="Count",
            )
            return PlotData(plotly_json=fig.to_json())

        if plot_type == "shap-summary":
            # SHAP columns are prefixed with "shap_"
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

    def export_model(self, model: Any, path: str) -> str:
        model.save(path)
        return path

    def load_model(self, path: str) -> Any:
        raise NotImplementedError("load_model not yet implemented")

    def model_info(self, model: Any) -> dict[str, Any]:
        raise NotImplementedError("model_info not yet implemented")
