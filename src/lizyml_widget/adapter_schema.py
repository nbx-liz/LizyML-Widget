"""Schema-based config utilities for LizyML adapter.

Provides thread-safe schema caching, recursive field stripping,
default search space extraction, and Tune config preparation.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .adapter_params import MODEL_METRIC_TO_EVAL, resolve_direction

_log = logging.getLogger(__name__)

# ── Schema cache (thread-safe) ────────────────────────────

_schema_cache: dict[str, Any] | None = None
_schema_lock = threading.Lock()


def get_schema() -> dict[str, Any]:
    """Lazily load and cache the LizyML JSON schema (thread-safe)."""
    global _schema_cache  # noqa: PLW0603
    if _schema_cache is not None:
        return _schema_cache
    with _schema_lock:
        if _schema_cache is None:
            from lizyml.config.schema import LizyMLConfig

            _schema_cache = LizyMLConfig.model_json_schema()
    return _schema_cache


def reset_schema_cache() -> None:
    """Reset the cached schema (for testing)."""
    global _schema_cache  # noqa: PLW0603
    _schema_cache = None


# ── Schema resolution helpers ─────────────────────────────


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref pointer like '#/$defs/FooConfig'."""
    name = ref.rsplit("/", 1)[-1]
    result: dict[str, Any] = defs.get(name, {})
    return result


def _resolve_sub_schema(
    prop_schema: dict[str, Any], defs: dict[str, Any], value: Any
) -> dict[str, Any] | None:
    """Resolve a property schema to a concrete object schema, handling
    $ref, oneOf, anyOf, and discriminated unions."""
    if "$ref" in prop_schema:
        return _resolve_ref(prop_schema["$ref"], defs)

    # oneOf with discriminator (e.g., model, split, inner_valid)
    if "oneOf" in prop_schema and isinstance(value, dict):
        discriminator = prop_schema.get("discriminator", {})
        prop_name = discriminator.get("propertyName")
        mapping = discriminator.get("mapping", {})
        if prop_name and prop_name in value:
            ref = mapping.get(value[prop_name])
            if ref:
                return _resolve_ref(ref, defs)
        # Fallback: try each oneOf variant
        for variant in prop_schema["oneOf"]:
            resolved = _resolve_sub_schema(variant, defs, value)
            if resolved and "properties" in resolved:
                return resolved

    # anyOf (nullable types, union types)
    if "anyOf" in prop_schema and isinstance(value, dict):
        for variant in prop_schema["anyOf"]:
            resolved = _resolve_sub_schema(variant, defs, value)
            if resolved and "properties" in resolved:
                return resolved

    # Direct object with properties
    if prop_schema.get("type") == "object" and "properties" in prop_schema:
        return prop_schema

    return None


def _strip_to_schema(
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
            sub = _resolve_sub_schema(props[key], defs, value)
            if sub and "properties" in sub:
                value = _strip_to_schema(value, sub, defs)

        result[key] = value
    return result


# ── inner_valid normalization ─────────────────────────────

_INNER_VALID_ALIASES: set[str] = {"holdout", "group_holdout", "time_holdout"}

# Allowed fields per inner_valid method (LizyML Pydantic schema, extra='forbid')
_INNER_VALID_FIELDS: dict[str, set[str]] = {
    "holdout": {"method", "ratio", "random_state", "stratify"},
    "group_holdout": {"method", "ratio", "random_state"},
    "time_holdout": {"method", "ratio"},
}


def _normalize_iv_value(iv: Any) -> Any:
    """Return a normalized inner_valid value (pure, no mutation)."""
    if iv is None:
        return None
    if isinstance(iv, str):
        return {"method": iv} if iv in _INNER_VALID_ALIASES else None
    if isinstance(iv, dict):
        method = iv.get("method")
        allowed = _INNER_VALID_FIELDS.get(method or "")
        if allowed:
            return {k: v for k, v in iv.items() if k in allowed}
        if method is not None:
            return None
    return None


def normalize_inner_valid(config: dict[str, Any]) -> dict[str, Any]:
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
    new_iv = _normalize_iv_value(iv)
    if new_iv is iv:
        return config
    return {
        **config,
        "training": {
            **training,
            "early_stopping": {**es, "inner_valid": new_iv},
        },
    }


def enforce_iv_exclusivity(config: dict[str, Any]) -> dict[str, Any]:
    """Remove conflicting inner_valid/validation_ratio keys for Pydantic.

    Pydantic's ``model_fields_set`` treats an explicit ``None`` as "set",
    so having both ``validation_ratio`` and ``inner_valid: None`` in the dict
    triggers ``Specify either 'validation_ratio' or 'inner_valid', not both``.

    This function removes the losing key entirely:
    - validation_ratio is set + inner_valid is None → remove inner_valid key
    - inner_valid is set (non-None) + validation_ratio present → remove validation_ratio
    """
    training = config.get("training")
    if not isinstance(training, dict):
        return config
    es = training.get("early_stopping")
    if not isinstance(es, dict):
        return config

    iv = es.get("inner_valid")
    vr = es.get("validation_ratio")
    has_iv_key = "inner_valid" in es
    has_vr = vr is not None

    if has_vr and has_iv_key and iv is None:
        # validation_ratio wins → remove inner_valid key entirely
        new_es = {k: v for k, v in es.items() if k != "inner_valid"}
        return {**config, "training": {**training, "early_stopping": new_es}}
    if iv is not None and has_vr:
        # inner_valid wins → remove validation_ratio
        new_es = {k: v for k, v in es.items() if k != "validation_ratio"}
        return {**config, "training": {**training, "early_stopping": new_es}}
    return config


# Widget-only fields inside the tuning section (not recognized by LizyML)
_WIDGET_ONLY_TUNING_KEYS: frozenset[str] = frozenset(
    {
        "model_params",
        "training",
        "evaluation",
    }
)


def strip_for_backend(config: dict[str, Any]) -> dict[str, Any]:
    """Strip fields not recognized by LizyML schema.

    Uses the Pydantic JSON schema to whitelist allowed fields at each
    nesting level. Prevents 'Extra inputs are not permitted' errors.
    Also removes Widget-only tuning fields (P-014).
    """
    schema = get_schema()
    defs = schema.get("$defs", {})
    result = _strip_to_schema(config, schema, defs)

    # Remove Widget-only tuning fields before sending to backend
    tuning = result.get("tuning")
    if isinstance(tuning, dict):
        stripped_tuning = {k: v for k, v in tuning.items() if k not in _WIDGET_ONLY_TUNING_KEYS}
        result = {**result, "tuning": stripped_tuning}

    return result


# ── Default search space ──────────────────────────────────


def get_default_search_space(task: str) -> dict[str, Any]:
    """Return LizyML's default search space for *task* as a widget-format dict.

    Each entry uses the format expected by the SearchSpace UI component:
    ``{"type": "int"|"float"|"categorical", ...}``.
    Returns empty dict for unknown tasks.
    """
    _default_space_fn = None
    try:
        from lizyml.estimators.lgbm.defaults import default_space as _ds

        _default_space_fn = _ds
    except ImportError:
        try:
            from lizyml.tuning import search_space as _ss  # v0.1.x fallback

            _default_space_fn = getattr(_ss, "default_space", None)
        except ImportError:
            pass
    if _default_space_fn is None:
        return {}
    default_space = _default_space_fn

    try:
        dims = default_space(task)
    except Exception:
        _log.warning(
            "Failed to load default search space for task=%s",
            task,
            exc_info=True,
        )
        return {}

    space: dict[str, Any] = {}
    for d in dims:
        name: str = d.name
        if hasattr(d, "low") and hasattr(d, "high"):
            space[name] = {
                "type": "int" if isinstance(getattr(d, "low", None), int) else "float",
                "low": d.low,
                "high": d.high,
                "log": getattr(d, "log", False),
            }
        elif hasattr(d, "choices"):
            space[name] = {
                "type": "categorical",
                "choices": list(d.choices),
            }
    return space


# ── Tune config preparation (extracted from adapter.py) ────


def prepare_tune_overrides(result: dict[str, Any]) -> dict[str, Any]:
    """Apply Tune-specific overrides: defaults, P-014 fields, direction."""
    if not result.get("tuning"):
        result = {**result, "tuning": {"optuna": {"params": {"n_trials": 50}, "space": {}}}}
    else:
        existing_tuning = result["tuning"]
        existing_optuna = existing_tuning.get("optuna") or {}
        existing_params = existing_optuna.get("params") or {}
        merged_optuna = {
            **existing_optuna,
            "params": {"n_trials": 50, **existing_params},
            "space": existing_optuna.get("space") or {},
        }
        result = {**result, "tuning": {**existing_tuning, "optuna": merged_optuna}}

    tune_section = result.get("tuning", {})
    tune_model_params = tune_section.get("model_params")
    tune_training = tune_section.get("training")
    tune_evaluation = tune_section.get("evaluation")

    if isinstance(tune_model_params, dict) and tune_model_params:
        existing_params = result.get("model", {}).get("params", {})
        result = {
            **result,
            "model": {
                **result.get("model", {}),
                "params": {**existing_params, **tune_model_params},
            },
        }
    if isinstance(tune_training, dict) and tune_training:
        existing_training = dict(result.get("training", {}))
        result = {**result, "training": {**existing_training, **tune_training}}
    if isinstance(tune_evaluation, dict) and tune_evaluation:
        result = {**result, "evaluation": dict(tune_evaluation)}

    # Remove calibration (LizyML tune does not reference it).
    # Smart params are kept — LizyML backend supports them during tuning
    # (resolve_smart_params is called per trial, search space can include
    # category='smart' dimensions).
    result = {k: v for k, v in result.items() if k != "calibration"}

    return _resolve_tune_direction(result, tune_evaluation)


def _resolve_tune_direction(
    result: dict[str, Any], tune_evaluation: dict[str, Any] | None
) -> dict[str, Any]:
    """Set tuning direction from evaluation metrics or legacy widget metric."""
    eval_metrics = (result.get("evaluation") or {}).get("metrics", [])
    cur_optuna = result["tuning"]["optuna"]
    raw_params = cur_optuna.get("params", {})
    widget_metric: str | None = raw_params.get("metric")
    cur_params = {k: v for k, v in raw_params.items() if k != "metric"}

    if tune_evaluation is None and widget_metric is not None and widget_metric:
        eval_metric_name: str = MODEL_METRIC_TO_EVAL.get(widget_metric, widget_metric)
        existing = [
            m for m in (result.get("evaluation") or {}).get("metrics", []) if m != eval_metric_name
        ]
        result = {
            **result,
            "evaluation": {
                **result.get("evaluation", {}),
                "metrics": [eval_metric_name, *existing],
            },
        }
        cur_params = {**cur_params, "direction": resolve_direction(eval_metric_name)}
    elif eval_metrics:
        cur_params = {**cur_params, "direction": resolve_direction(eval_metrics[0])}
    else:
        # Fallback: derive direction from model.params.metric when eval_metrics
        # is empty (e.g. tune_evaluation exists but has metrics=[]).
        model_metric = (result.get("model") or {}).get("params", {}).get("metric")
        raw = model_metric[0] if isinstance(model_metric, list) else model_metric
        if isinstance(raw, str) and raw:
            fallback = MODEL_METRIC_TO_EVAL.get(raw, raw)
            cur_params = {**cur_params, "direction": resolve_direction(fallback)}

    return {
        **result,
        "tuning": {
            **result["tuning"],
            "optuna": {**cur_optuna, "params": cur_params},
        },
    }
