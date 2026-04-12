"""BackendAdapter protocol and LizyML implementation."""

from __future__ import annotations

import contextlib
import copy
import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import pandas as pd

from .adapter_contract import build_capabilities, build_ui_schema
from .adapter_params import (
    LGBM_PARAMS_BY_TASK,
    LGBM_PARAMS_TASK_INDEPENDENT,
    MODEL_METRIC_TO_EVAL,
    get_eval_metrics_by_task,
)
from .adapter_params import classify_best_params as _classify_best_params_impl
from .adapter_schema import (
    enforce_iv_exclusivity,
    get_default_search_space,
    normalize_inner_valid,
    prepare_tune_overrides,
    strip_for_backend,
)
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
        resume: bool = False,
        n_trials: int | None = None,
        expand_boundary: bool | None = None,
        boundary_threshold: float = 0.05,
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

    def plot(self, model: Any, plot_type: str, **kwargs: Any) -> PlotData: ...

    def available_plots(self, model: Any) -> list[str]: ...

    def export_model(self, model: Any, path: str) -> str: ...

    def export_code(self, model: Any, path: str) -> Any: ...

    def load_model(self, path: str) -> Any: ...

    def model_info(self, model: Any) -> dict[str, Any]: ...

    def classify_best_params(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: ...

    def plot_inference(self, predictions: pd.DataFrame, plot_type: str) -> PlotData: ...


#: Minimum supported lizyml version (inclusive). P-027: re-tune monitoring.
LIZYML_MIN_VERSION = (0, 9, 0)
#: Maximum supported lizyml version (exclusive).
LIZYML_MAX_VERSION = (0, 10, 0)


def _parse_lizyml_version(raw: str) -> tuple[int, ...]:
    """Parse a lizyml version string into a comparable tuple of ints.

    Pre-release / dev suffixes ("0.9.0rc1", "0.9.0.dev3") are stripped so
    they compare equal to their base release.  This keeps the version guard
    permissive for local dev installs.
    """
    import re

    core = re.split(r"[^0-9.]", raw, maxsplit=1)[0]
    parts = core.split(".")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            break
    # Pad to 3 components so comparisons work against (major, minor, patch).
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def _check_lizyml_version() -> None:
    """Validate the installed lizyml version against the widget's contract.

    Raises
    ------
    ImportError
        When ``lizyml`` is missing or outside ``[LIZYML_MIN_VERSION,
        LIZYML_MAX_VERSION)``.  The error message points the user at the
        ``[lizyml]`` extras install command.
    """
    try:
        import lizyml
    except ImportError as exc:  # pragma: no cover - exercised via dedicated test
        msg = (
            "lizyml-widget requires the 'lizyml' backend. "
            "Install it with:\n"
            "    pip install 'lizyml-widget[lizyml]'\n"
            "See docs/VERSION_COMPAT.md for details."
        )
        raise ImportError(msg) from exc

    version_str = getattr(lizyml, "__version__", None)
    # Unit tests often install ``lizyml`` as a ``MagicMock`` via
    # ``sys.modules`` patching; in that case ``__version__`` is not a
    # real string and the guard cannot reliably parse it.  Skip silently
    # so mocked tests still exercise the adapter without being blocked
    # by a version check that is irrelevant in that environment.
    if not isinstance(version_str, str):
        return

    version = _parse_lizyml_version(version_str)
    if version < LIZYML_MIN_VERSION or version >= LIZYML_MAX_VERSION:
        min_s = ".".join(str(x) for x in LIZYML_MIN_VERSION)
        max_s = ".".join(str(x) for x in LIZYML_MAX_VERSION)
        msg = (
            f"lizyml-widget requires lizyml>={min_s},<{max_s} "
            f"(found: lizyml=={version_str}). Run:\n"
            f"    pip install --upgrade "
            f"'lizyml[plots,tuning,calibration,explain]>={min_s},<{max_s}'\n"
            "See docs/VERSION_COMPAT.md for the full compatibility matrix."
        )
        raise ImportError(msg)


def _serialize_trials(trials: Any) -> list[dict[str, Any]]:
    """Convert a sequence of lizyml TrialResult into JSON-friendly dicts.

    Each dict keeps ``round`` (defaulting to 1 for pre-re-tune backends)
    so the UI can group trials by round when rendering the score history.
    """
    from dataclasses import asdict

    out: list[dict[str, Any]] = []
    for t in trials or ():
        d = asdict(t)
        d.setdefault("round", 1)
        out.append(d)
    return out


def _serialize_rounds(rounds: Any) -> list[dict[str, Any]]:
    """Convert a tuple of ``RoundSummary`` into plain dicts.

    ``space_snapshot`` is stripped because each entry is a dataclass
    (``FloatDim`` / ``IntDim`` / ``CategoricalDim``) that anywidget cannot
    serialize directly.  The UI only needs the round-level metadata
    (scores, expanded dim names); full space snapshots can be retrieved
    from ``boundary_report`` if needed in a future iteration.
    """
    out: list[dict[str, Any]] = []
    for r in rounds or ():
        out.append(
            {
                "round": int(getattr(r, "round", 0)),
                "n_trials": int(getattr(r, "n_trials", 0)),
                "best_score_before": getattr(r, "best_score_before", None),
                "best_score_after": float(getattr(r, "best_score_after", 0.0)),
                "expanded_dims": list(getattr(r, "expanded_dims", ()) or ()),
            }
        )
    return out


def _serialize_boundary_report(report: Any) -> dict[str, Any] | None:
    """Convert ``BoundaryReport`` into a JSON-friendly dict (or None)."""
    if report is None:
        return None
    dims_out: list[dict[str, Any]] = []
    for d in getattr(report, "dims", ()) or ():
        dims_out.append(
            {
                "name": getattr(d, "name", ""),
                "best_value": getattr(d, "best_value", None),
                "low": getattr(d, "low", None),
                "high": getattr(d, "high", None),
                "position_pct": getattr(d, "position_pct", None),
                "edge": getattr(d, "edge", ""),
                "expanded": bool(getattr(d, "expanded", False)),
                "new_low": getattr(d, "new_low", None),
                "new_high": getattr(d, "new_high", None),
            }
        )
    return {
        "dims": dims_out,
        "expanded_names": list(getattr(report, "expanded_names", ()) or ()),
    }


class LizyMLAdapter:
    """Adapter for the LizyML backend library."""

    def __init__(self) -> None:
        _check_lizyml_version()
        self._last_worker_thread: threading.Thread | None = None

    @property
    def info(self) -> BackendInfo:
        import lizyml

        return BackendInfo(name="lizyml", version=lizyml.__version__)

    def get_config_schema(self) -> dict[str, Any]:
        from lizyml.config.schema import LizyMLConfig

        return LizyMLConfig.model_json_schema()

    # ── Backend Contract & Config Lifecycle (Phase 25) ─────────

    # Class-level aliases for constants defined in adapter_params.py.
    # Tests and internal methods access these via self._ or cls._ ;
    # the aliases keep that interface stable.
    _MODEL_METRIC_TO_EVAL = MODEL_METRIC_TO_EVAL
    _LGBM_PARAMS_TASK_INDEPENDENT = LGBM_PARAMS_TASK_INDEPENDENT
    _LGBM_PARAMS_BY_TASK = LGBM_PARAMS_BY_TASK

    def classify_best_params(
        self,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Split best_params into (model, smart, training) category dicts."""
        return _classify_best_params_impl(params)

    def get_backend_contract(self) -> BackendContract:
        """Return the full backend contract with config schema, UI metadata, and capabilities."""
        config_schema = self.get_config_schema()
        ui_schema = build_ui_schema(self._get_eval_metrics_by_task())
        capabilities = build_capabilities()
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
        if not config.get("output_dir"):
            config["output_dir"] = "outputs/"

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

        # Override training seed to widget default (differs from LizyML schema default)
        training = dict(config.get("training", {}))
        training["seed"] = 1120
        config["training"] = training

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
        result = normalize_inner_valid(result)
        return result

    @staticmethod
    def _get_eval_metrics_by_task() -> dict[str, list[str]]:
        """Query LizyML's metric registry for available evaluation metrics per task."""
        return get_eval_metrics_by_task()

    def apply_task_defaults(self, config: dict[str, Any], *, task: str) -> dict[str, Any]:
        """Apply task-dependent default params to config via patch operations."""
        ops: list[ConfigPatchOp] = []

        # LGBM model params (may be empty for unknown tasks)
        defaults = self._LGBM_PARAMS_BY_TASK.get(task, {})
        ops.extend(
            ConfigPatchOp(op="set", path=f"model.params.{k}", value=v) for k, v in defaults.items()
        )

        # Ensure evaluation metrics are valid for current task
        eval_metrics = (config.get("evaluation") or {}).get("metrics", [])
        task_metrics = self._get_eval_metrics_by_task().get(task, [])
        if task_metrics:
            valid_set = set(task_metrics)
            if not eval_metrics:
                # Empty → populate all defaults for this task
                ops.append(
                    ConfigPatchOp(
                        op="set",
                        path="evaluation.metrics",
                        value=list(task_metrics),
                    )
                )
            else:
                # Filter to only valid metrics for current task
                filtered = [m for m in eval_metrics if m in valid_set]
                if not filtered:
                    filtered = list(task_metrics)
                if filtered != eval_metrics:
                    ops.append(
                        ConfigPatchOp(
                            op="set",
                            path="evaluation.metrics",
                            value=filtered,
                        )
                    )

        # Populate default search space (create tuning section if absent/None)
        default_space = get_default_search_space(task)
        if default_space:
            tuning = config.get("tuning")
            if not isinstance(tuning, dict):
                # tuning is None or absent → create full structure
                ops.append(
                    ConfigPatchOp(
                        op="set",
                        path="tuning",
                        value={"optuna": {"params": {"n_trials": 10}, "space": default_space}},
                    )
                )
            else:
                space = (tuning.get("optuna") or {}).get("space", {})
                if not space:
                    ops.append(
                        ConfigPatchOp(
                            op="set",
                            path="tuning.optuna.space",
                            value=default_space,
                        )
                    )

        if not ops:
            return copy.deepcopy(config)
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

    # ── Canonicalize config ───────────────────────────────────

    # Keys managed by Service (df_info / build_config), not the widget config traitlet
    _SERVICE_MANAGED_KEYS: frozenset[str] = frozenset({"data", "features", "split", "task"})

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

        result = normalize_inner_valid(result)
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
            model = {**model, "name": "lgbm"}

        # Enforce auto_num_leaves exclusivity
        result = {**result, "model": self._enforce_auto_num_leaves(model)}

        if job_type == "tune":
            result = prepare_tune_overrides(result)

        result = normalize_inner_valid(result)
        result = enforce_iv_exclusivity(result)
        result = self._convert_metric_entries(result)
        return strip_for_backend(result)

    @staticmethod
    def _convert_metric_entries(config: dict[str, Any]) -> dict[str, Any]:
        """Convert widget-only _precision_at_k_k to MetricEntry dict form.

        Transforms ``model.params.metric`` entries:
        - ``"precision_at_k"`` + ``_precision_at_k_k=20``
          → ``{"precision_at_k": {"k": 20}}``
        - Strips ``_precision_at_k_k`` from params regardless.
        """
        model = config.get("model")
        if not isinstance(model, dict):
            return config
        params = model.get("params")
        if not isinstance(params, dict):
            return config

        metric = params.get("metric")
        pak_k = params.get("_precision_at_k_k")

        new_params = {k: v for k, v in params.items() if k != "_precision_at_k_k"}

        if isinstance(metric, list) and "precision_at_k" in metric and pak_k is not None:
            new_metric = [
                {"precision_at_k": {"k": int(pak_k)}} if m == "precision_at_k" else m
                for m in metric
            ]
            new_params = {**new_params, "metric": new_metric}

        return {**config, "model": {**model, "params": new_params}}

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
        normalized = normalize_inner_valid(normalized)
        normalized = strip_for_backend(normalized)

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
        model._widget_config = copy.deepcopy(config)  # type: ignore[attr-defined]  # noqa: SLF001
        return model

    def _run_with_cancel_polling(
        self,
        target: Callable[[], Any],
        on_progress: Callable[..., Any] | None,
        poll_interval: float = 0.5,
    ) -> Any:
        """Run *target* in a non-daemon worker thread, polling for cancellation.

        If *on_progress* raises ``InterruptedError`` the worker thread is
        abandoned and the exception propagates to the caller.
        The abandoned thread continues running in the background.
        """
        # Warn if a previously abandoned thread is still running
        prev = self._last_worker_thread
        if prev is not None and prev.is_alive():
            _log.warning(
                "Previous backend worker thread is still running; "
                "OpenMP thread contention may degrade performance."
            )

        if on_progress is None:
            self._last_worker_thread = None
            return target()

        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                result_holder["value"] = target()
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=_worker, daemon=False)
        self._last_worker_thread = thread
        thread.start()

        try:
            while thread.is_alive():
                thread.join(timeout=poll_interval)
                if thread.is_alive():
                    # on_progress may raise InterruptedError → cancels the job
                    on_progress(0, 0, "Processing...")
        except InterruptedError:
            _log.warning(
                "Job cancelled; backend worker thread abandoned (will finish in background)."
            )
            raise

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["value"]

    def fit(
        self,
        model: Any,
        *,
        params: dict[str, Any] | None = None,
        on_progress: Callable[..., Any] | None = None,
    ) -> FitSummary:
        result = self._run_with_cancel_polling(
            lambda: model.fit(params=params),
            on_progress,
        )
        return FitSummary(
            metrics=result.metrics,
            fold_count=len(getattr(getattr(result, "splits", None), "outer", [])),
            params=model.params_table().reset_index().to_dict(orient="records"),
        )

    def tune(
        self,
        model: Any,
        *,
        on_progress: Callable[..., Any] | None = None,
        resume: bool = False,
        n_trials: int | None = None,
        expand_boundary: bool | None = None,
        boundary_threshold: float = 0.05,
    ) -> TuningSummary:
        """Run ``Model.tune`` with optional re-tune (Study Resume) support.

        Parameters
        ----------
        resume
            When ``True`` the backend resumes the existing Optuna study on
            ``model`` instead of starting a fresh one.  Requires that the
            caller (``WidgetService``) passes a model that was already tuned.
        n_trials
            Extra trials to run on the resumed study.  ``None`` defers to
            the backend default.
        expand_boundary
            If ``True``, allow the backend to widen search-space boundaries
            when the best trial lands against an edge.  ``None`` defers to
            the backend default.
        boundary_threshold
            Fraction of the dim range that qualifies as "against the edge"
            when ``expand_boundary`` is enabled.
        """
        from lizyml.core.types.tuning_result import TuneProgressInfo

        # lizyml>=0.9.0 (P-027) always supports progress_callback & re-tune.
        # The cancel flag is polled every 0.5s via _run_with_cancel_polling,
        # while progress_cb fires per trial with round-aware payload.
        progress_cb: Callable[[TuneProgressInfo], None] | None = None
        if on_progress is not None:

            def progress_cb(info: TuneProgressInfo) -> None:
                msg = f"Trial {info.current_trial}/{info.total_trials}"
                if info.best_score is not None:
                    msg += f" (best: {info.best_score:.4f})"
                # Re-tune fields (lizyml>=0.9.0 only).  Default to sane
                # single-round values so the widget traitlet can assume the
                # keys are always present.
                round_no = int(getattr(info, "round", 1) or 1)
                cumulative = int(
                    getattr(info, "cumulative_trials", info.current_trial) or info.current_trial
                )
                expanded = tuple(getattr(info, "expanded_dims", ()) or ())
                on_progress(
                    info.current_trial,
                    info.total_trials,
                    msg,
                    round=round_no,
                    cumulative_trials=cumulative,
                    expanded_dims=list(expanded),
                    latest_score=info.latest_score,
                    latest_state=info.latest_state,
                    best_score=info.best_score,
                )

        def _cancel_only(_c: int, _t: int, _m: str) -> None:
            # progress_cb handles real updates; this sentinel only lets
            # _run_with_cancel_polling poll the cancel flag without emitting
            # a generic "Processing..." message.
            pass

        # Build tune kwargs once so we can trace them in tests.  All
        # re-tune knobs are forwarded only when the caller opts in so the
        # default legacy call shape stays the same.
        tune_kwargs: dict[str, Any] = {"progress_callback": progress_cb}
        if resume:
            tune_kwargs["resume"] = True
        if n_trials is not None:
            tune_kwargs["n_trials"] = n_trials
        if expand_boundary is not None:
            tune_kwargs["expand_boundary"] = expand_boundary
        if resume or expand_boundary is not None:
            # boundary_threshold only matters when re-tune/expand machinery
            # is active; otherwise leave the backend default untouched.
            tune_kwargs["boundary_threshold"] = boundary_threshold

        result = self._run_with_cancel_polling(
            lambda: model.tune(**tune_kwargs),
            _cancel_only,
        )

        return TuningSummary(
            best_params=result.best_params,
            best_score=result.best_score,
            trials=_serialize_trials(result.trials),
            metric_name=result.metric_name,
            direction=result.direction,
            rounds=_serialize_rounds(getattr(result, "rounds", ())),
            boundary_report=_serialize_boundary_report(getattr(result, "boundary_report", None)),
        )

    def predict(
        self,
        model: Any,
        data: pd.DataFrame,
        *,
        return_shap: bool = False,
    ) -> PredictionSummary:
        import numpy as np

        result = model.predict(data, return_shap=return_shap)
        df = pd.DataFrame({"pred": result.pred})

        # Proba: expand 2D (multiclass) into per-class columns
        if result.proba is not None:
            proba = np.asarray(result.proba)
            if proba.ndim == 2:
                for i in range(proba.shape[1]):
                    df[f"proba_{i}"] = proba[:, i]
            else:
                df["proba"] = proba

        # SHAP values: include if available
        shap_values = getattr(result, "shap_values", None)
        if shap_values is not None:
            shap_arr = np.asarray(shap_values)
            if shap_arr.ndim == 2:
                feature_names = list(data.columns)
                for i, name in enumerate(feature_names):
                    if i < shap_arr.shape[1]:
                        df[f"shap_{name}"] = shap_arr[:, i]

        return PredictionSummary(predictions=df, warnings=result.warnings)

    def evaluate_table(self, model: Any) -> list[dict[str, Any]]:
        df: pd.DataFrame = model.evaluate_table()
        return list(df.reset_index().to_dict(orient="records"))  # type: ignore[arg-type]

    def split_summary(self, model: Any) -> list[dict[str, Any]]:
        df: pd.DataFrame = model.split_summary()
        return list(df.to_dict(orient="records"))  # type: ignore[arg-type]

    def importance(self, model: Any, kind: str) -> dict[str, float]:
        return model.importance(kind=kind)  # type: ignore[no-any-return]

    def plot(self, model: Any, plot_type: str, **kwargs: Any) -> PlotData:
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
            plots.extend(
                [
                    "feature-importance-split",
                    "feature-importance-gain",
                    "feature-importance-shap",
                ]
            )

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
            # Find prediction column: prefer columns starting with "pred", fallback to last column
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
        model.export(path)
        return path

    def export_code(self, model: Any, path: str) -> Any:
        """Export inference code for the trained model."""
        from pathlib import Path

        return model.export_code(Path(path))

    def load_model(self, path: str) -> Any:
        from lizyml.core.model import Model

        return Model.load(path)

    def model_info(self, model: Any) -> dict[str, Any]:
        """Return metadata about a loaded/trained model."""
        info: dict[str, Any] = {"loaded": True}

        # Extract model params if available
        with contextlib.suppress(Exception):
            params_df = model.params_table()
            if params_df is not None:
                info["params"] = params_df.reset_index().to_dict(orient="records")

        # Extract task from config
        with contextlib.suppress(Exception):
            info["task"] = model._cfg.task  # noqa: SLF001

        return info
