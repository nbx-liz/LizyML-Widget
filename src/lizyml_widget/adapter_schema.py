"""Schema-based config utilities for LizyML adapter.

Provides thread-safe schema caching, recursive field stripping,
and default search space extraction from LizyML's Pydantic schema.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

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


def strip_for_backend(config: dict[str, Any]) -> dict[str, Any]:
    """Strip fields not recognized by LizyML schema.

    Uses the Pydantic JSON schema to whitelist allowed fields at each
    nesting level. Prevents 'Extra inputs are not permitted' errors.
    """
    schema = get_schema()
    defs = schema.get("$defs", {})
    return _strip_to_schema(config, schema, defs)


# ── Default search space ──────────────────────────────────


def get_default_search_space(task: str) -> dict[str, Any]:
    """Return LizyML's default search space for *task* as a widget-format dict.

    Each entry uses the format expected by the SearchSpace UI component:
    ``{"type": "int"|"float"|"categorical", ...}``.
    Returns empty dict for unknown tasks.
    """
    try:
        from lizyml.tuning.search_space import default_space
    except ImportError:
        return {}

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
                "type": "int" if type(d).__name__ == "IntDim" else "float",
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
