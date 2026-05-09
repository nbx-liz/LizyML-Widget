"""BackendAdapter protocol and LizyML implementation."""

from __future__ import annotations

import copy
import logging
import pickle
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import pandas as pd

from .adapter_contract import build_capabilities, build_ui_schema
from .adapter_internals import (
    LIZYML_MAX_VERSION,
    LIZYML_MIN_VERSION,
    _check_lizyml_version,
    _parse_lizyml_version,
    _serialize_boundary_report,
    _serialize_rounds,
    _serialize_trials,
    convert_metric_entries,
    deep_merge,
    enforce_auto_num_leaves,
    extract_defaults,
    get_nested,
    set_nested,
    unset_nested,
)
from .adapter_params import (
    LGBM_PARAMS_BY_TASK,
    LGBM_PARAMS_TASK_INDEPENDENT,
    MODEL_METRIC_TO_EVAL,
    get_eval_metrics_by_task,
)
from .adapter_params import classify_best_params as _classify_best_params_impl
from .adapter_results import (
    is_model_fitted,
    list_available_plots,
    render_inference_plot,
    render_plot,
    task_for_model,
)
from .adapter_schema import (
    enforce_iv_exclusivity,
    get_default_search_space,
    normalize_inner_valid,
    prepare_tune_overrides,
    strip_for_backend,
)
from .adapter_views import (
    view_fit_result,
    view_prediction_result,
    view_tune_progress,
    view_tuning_result,
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

# Re-export for tests / external imports that historically pulled these
# symbols from ``lizyml_widget.adapter``.
__all__ = [
    "LIZYML_MAX_VERSION",
    "LIZYML_MIN_VERSION",
    "BackendAdapter",
    "LizyMLAdapter",
    "_check_lizyml_version",
    "_parse_lizyml_version",
    "_serialize_boundary_report",
    "_serialize_rounds",
    "_serialize_trials",
]

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

    # P-037: tune state cross-process persistence (#152). The subprocess
    # writes ``model._tuning_result`` (and best-effort ``_study``) to a
    # path; the parent reads the path back onto a freshly-created model so
    # ``optimization-history`` plot renders without re-fitting.
    def export_tune_state(self, model: Any, path: str) -> None: ...

    def restore_tune_state(self, model: Any, path: str) -> None: ...

    def classify_best_params(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: ...

    def plot_inference(self, predictions: pd.DataFrame, plot_type: str) -> PlotData: ...


class LizyMLAdapter:
    """Adapter for the LizyML backend library."""

    def __init__(self) -> None:
        _check_lizyml_version()
        self._last_worker_thread: threading.Thread | None = None
        # #116: Adapter-side config registry keyed by ``id(model)``. Replaces
        # the previous ``model._widget_config`` private write. Used as the
        # fallback path when reading task off lizyml's internal ``_cfg``
        # also fails (e.g. an older / mocked model that lacks ``_cfg``).
        self._model_configs: dict[int, dict[str, Any]] = {}

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

    # Backward-compatible static aliases for helpers now living in
    # ``adapter_internals`` (#137). Tests and external callers historically
    # accessed these via ``LizyMLAdapter._<name>``; the aliases keep that
    # interface stable.
    _enforce_auto_num_leaves = staticmethod(enforce_auto_num_leaves)
    _deep_merge = staticmethod(deep_merge)
    _get_nested = staticmethod(get_nested)
    _set_nested = staticmethod(set_nested)
    _unset_nested = staticmethod(unset_nested)
    _extract_defaults = staticmethod(extract_defaults)
    _convert_metric_entries = staticmethod(convert_metric_entries)

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
        config = extract_defaults(schema)
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
                set_nested(result, parts, op.value)
            elif op.op == "unset":
                unset_nested(result, parts)
            elif op.op == "merge":
                existing = get_nested(result, parts)
                if isinstance(existing, dict) and isinstance(op.value, dict):
                    set_nested(result, parts, {**existing, **op.value})
                else:
                    set_nested(result, parts, op.value)

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
        result["model"] = enforce_auto_num_leaves(cur_model)

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

    # ── Canonicalize config ───────────────────────────────────

    # Keys managed by Service (df_info / build_config), not the widget config traitlet
    _SERVICE_MANAGED_KEYS: frozenset[str] = frozenset({"data", "features", "split", "task"})

    def canonicalize_config(
        self, config: dict[str, Any], *, task: str | None = None
    ) -> dict[str, Any]:
        """Canonicalize a partial/full config by merging with backend defaults."""
        defaults = self.initialize_config(task=task)
        result = deep_merge(defaults, copy.deepcopy(config))

        # Strip Service-managed keys (data/features/split/task)
        for key in self._SERVICE_MANAGED_KEYS:
            result.pop(key, None)

        # Enforce auto_num_leaves exclusivity
        result["model"] = enforce_auto_num_leaves(result.get("model", {}))

        result = normalize_inner_valid(result)
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
        result = {**result, "model": enforce_auto_num_leaves(model)}

        if job_type == "tune":
            result = prepare_tune_overrides(result)

        result = normalize_inner_valid(result)
        result = enforce_iv_exclusivity(result)
        result = convert_metric_entries(result)
        return strip_for_backend(result)

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
        # #116: register the config in the adapter rather than writing a
        # private attr onto the lizyml object. The fallback in
        # ``_task_for_model`` reads from this registry when ``model._cfg``
        # is unavailable.
        self._model_configs[id(model)] = copy.deepcopy(config)
        return model

    def _task_for_model(self, model: Any) -> str:
        """Return the task for *model* (delegates to ``adapter_results``)."""
        return task_for_model(model, self._model_configs)

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
        view = view_fit_result(result)
        return FitSummary(
            metrics=view.metrics,
            fold_count=view.fold_count,
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
                view = view_tune_progress(info)
                msg = f"Trial {view.current_trial}/{view.total_trials}"
                if view.best_score is not None:
                    msg += f" (best: {view.best_score:.4f})"
                on_progress(
                    view.current_trial,
                    view.total_trials,
                    msg,
                    round=view.round,
                    cumulative_trials=view.cumulative_trials,
                    expanded_dims=list(view.expanded_dims),
                    latest_score=view.latest_score,
                    latest_state=view.latest_state,
                    best_score=view.best_score,
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
        view = view_tuning_result(result)

        return TuningSummary(
            best_params=view.best_params,
            best_score=view.best_score,
            trials=_serialize_trials(view.raw_trials),
            metric_name=view.metric_name,
            direction=view.direction,
            rounds=_serialize_rounds(view.rounds),
            boundary_report=_serialize_boundary_report(view.boundary_report),
        )

    def predict(
        self,
        model: Any,
        data: pd.DataFrame,
        *,
        return_shap: bool = False,
    ) -> PredictionSummary:
        import numpy as np

        # P-030: lizyml>=0.10 returns ``result.pred`` already decoded back to
        # the original target dtype (e.g. "Adelie" rather than int code 2)
        # via ``FitResult.target_encoder``. Pandas preserves that dtype when
        # we wrap it directly, so no extra conversion is needed here.
        result = model.predict(data, return_shap=return_shap)
        view = view_prediction_result(result)
        df = pd.DataFrame({"pred": view.pred})

        # Proba: expand 2D (multiclass) into per-class columns
        if view.proba is not None:
            proba = np.asarray(view.proba)
            if proba.ndim == 2:
                for i in range(proba.shape[1]):
                    df[f"proba_{i}"] = proba[:, i]
            else:
                df["proba"] = proba

        # SHAP values: include if available
        if view.shap_values is not None:
            shap_arr = np.asarray(view.shap_values)
            if shap_arr.ndim == 2:
                feature_names = list(data.columns)
                for i, name in enumerate(feature_names):
                    if i < shap_arr.shape[1]:
                        df[f"shap_{name}"] = shap_arr[:, i]

        return PredictionSummary(predictions=df, warnings=view.warnings)

    def evaluate_table(self, model: Any) -> list[dict[str, Any]]:
        # Unfit models raise ``LizyMLError(MODEL_NOT_FIT)`` from
        # ``Model.evaluate_table`` (lizyml >= 0.10). The widget's tune
        # path ends with an unfit model when the user did not run fit
        # first, so callers across both ThreadJobRunner and the
        # subprocess entry uniformly want an empty list rather than a
        # surfaced backend error. Centralising the guard here keeps the
        # rule in one place (#147 / P-036).
        if not is_model_fitted(model):
            return []
        df: pd.DataFrame = model.evaluate_table()
        return list(df.reset_index().to_dict(orient="records"))  # type: ignore[arg-type]

    def split_summary(self, model: Any) -> list[dict[str, Any]]:
        if not is_model_fitted(model):
            return []
        df: pd.DataFrame = model.split_summary()
        return list(df.to_dict(orient="records"))  # type: ignore[arg-type]

    def importance(self, model: Any, kind: str) -> dict[str, float]:
        return model.importance(kind=kind)  # type: ignore[no-any-return]

    def plot(self, model: Any, plot_type: str, **kwargs: Any) -> PlotData:
        return render_plot(model, plot_type, **kwargs)

    def available_plots(self, model: Any) -> list[str]:
        # #116: task lookup centralised in _task_for_model (registry-backed
        # fallback replaces the legacy ``_widget_config`` private read).
        task = self._task_for_model(model)
        return list_available_plots(model, task)

    def plot_inference(self, predictions: pd.DataFrame, plot_type: str) -> PlotData:
        """Generate Plotly plots from inference results (not part of Protocol)."""
        return render_inference_plot(predictions, plot_type)

    def export_model(self, model: Any, path: str) -> str:
        model.export(path)
        return path

    def export_tune_state(self, model: Any, path: str) -> None:
        """Persist tune state for IPC across the subprocess boundary (P-037).

        Writes a pickle blob containing ``_tuning_result`` (always) and
        ``_study`` (best-effort — silently dropped when the study object
        cannot be pickled, e.g., RDB-backed Optuna storage).

        Raises:
            ValueError: when *model* has no tune state to export. Callers
                MUST gate this on tune completion; calling on a fresh
                model is a programming error.
        """
        tuning_result = getattr(model, "_tuning_result", None)
        if tuning_result is None:
            msg = "Model has no tune state to export (run tune() first)"
            raise ValueError(msg)

        blob: dict[str, Any] = {"tuning_result": tuning_result, "study": None}

        study = getattr(model, "_study", None)
        if study is not None:
            try:
                pickle.dumps(study)
            except Exception as study_err:  # noqa: BLE001 — study is opaque
                _log.warning(
                    "tune-state export: dropping non-pickleable study (%s)",
                    study_err,
                )
            else:
                blob["study"] = study

        with open(path, "wb") as f:  # noqa: PTH123
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore_tune_state(self, model: Any, path: str) -> None:
        """Reattach tune state onto *model* from an IPC blob (P-037).

        Reads what :meth:`export_tune_state` wrote and assigns
        ``model._tuning_result`` (and ``model._study`` when present).
        The model remains unfit — INV-2 keeps ``is_model_fitted`` False so
        downstream guards (``available_plots``, ``evaluate_table``)
        continue to behave correctly.
        """
        with open(path, "rb") as f:  # noqa: PTH123
            blob = pickle.load(f)  # noqa: S301 — trusted, written by us

        # Private slot writes are allowed inside the adapter only.
        model._tuning_result = blob.get("tuning_result")  # noqa: SLF001
        study = blob.get("study")
        if study is not None:
            model._study = study  # noqa: SLF001

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
        import contextlib

        with contextlib.suppress(Exception):
            params_df = model.params_table()
            if params_df is not None:
                info["params"] = params_df.reset_index().to_dict(orient="records")

        # #116: task lookup centralised in _task_for_model.
        task = self._task_for_model(model)
        if task:
            info["task"] = task

        return info
