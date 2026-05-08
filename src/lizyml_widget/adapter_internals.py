"""Internal helpers for LizyMLAdapter.

Pure module-level helpers extracted from adapter.py (#137) so the adapter
module stays under the 800-line ceiling. These are version-check guards,
serializers for tuning artifacts, dict-traversal helpers, and config
canonicalization helpers — all stateless and reusable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .adapter_views import LMBoundaryReportView, LMRoundView

#: Minimum supported lizyml version (inclusive). P-030: bumped to 0.10.0
#: because the Adapter assumes target_encoder-driven label dtype preservation
#: (FitResult.target_encoder, FORMAT_VERSION=2 — added in lizyml 0.10).
LIZYML_MIN_VERSION = (0, 10, 0)
#: Maximum supported lizyml version (exclusive). P-030: covers 0.10 / 0.11 / 0.12.
LIZYML_MAX_VERSION = (0, 13, 0)


def _parse_lizyml_version(raw: str) -> tuple[int, ...]:
    """Parse a lizyml version string into a comparable tuple of ints.

    Pre-release / dev suffixes ("0.9.0rc1", "0.9.0.dev3") are stripped so
    they compare equal to their base release. This keeps the version guard
    permissive for local dev installs.
    """
    core = re.split(r"[^0-9.]", raw, maxsplit=1)[0]
    parts = core.split(".")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            break
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def _check_lizyml_version() -> None:
    """Validate the installed lizyml version against the widget's contract.

    Raises
    ------
    ImportError
        When ``lizyml`` is missing or outside ``[LIZYML_MIN_VERSION,
        LIZYML_MAX_VERSION)``. The error message points the user at the
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
    # real string and the guard cannot reliably parse it. Skip silently
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
    out: list[dict[str, Any]] = []
    for t in trials or ():
        d = asdict(t)
        d.setdefault("round", 1)
        out.append(d)
    return out


def _serialize_rounds(rounds: Sequence[LMRoundView]) -> list[dict[str, Any]]:
    """Convert a tuple of typed round views into JSON-serialisable dicts.

    ``space_snapshot`` is stripped because each entry is a dataclass
    (``FloatDim`` / ``IntDim`` / ``CategoricalDim``) that anywidget cannot
    serialize directly. The UI only needs the round-level metadata
    (scores, expanded dim names); full space snapshots can be retrieved
    from ``boundary_report`` if needed in a future iteration.
    """
    return [
        {
            "round": r.round,
            "n_trials": r.n_trials,
            "best_score_before": r.best_score_before,
            "best_score_after": r.best_score_after,
            "expanded_dims": list(r.expanded_dims),
        }
        for r in rounds
    ]


def _serialize_boundary_report(
    report: LMBoundaryReportView | None,
) -> dict[str, Any] | None:
    """Convert a typed boundary-report view into a JSON-friendly dict (or None)."""
    if report is None:
        return None
    return {
        "dims": [
            {
                "name": d.name,
                "best_value": d.best_value,
                "low": d.low,
                "high": d.high,
                "position_pct": d.position_pct,
                "edge": d.edge,
                "expanded": d.expanded,
                "new_low": d.new_low,
                "new_high": d.new_high,
            }
            for d in report.dims
        ],
        "expanded_names": list(report.expanded_names),
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values take precedence."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_nested(obj: dict[str, Any], parts: list[str]) -> Any:
    """Get a value at a dot-path inside a nested dict."""
    current: Any = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def set_nested(obj: dict[str, Any], parts: list[str], value: Any) -> None:
    """Set a value at a dot-path inside a nested dict, creating intermediates."""
    current = obj
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def unset_nested(obj: dict[str, Any], parts: list[str]) -> None:
    """Remove a key at a dot-path inside a nested dict."""
    current = obj
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            return
        current = current[part]
    current.pop(parts[-1], None)


def extract_defaults(schema: dict[str, Any]) -> dict[str, Any]:
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


def enforce_auto_num_leaves(model: dict[str, Any]) -> dict[str, Any]:
    """Return a new model dict with auto_num_leaves exclusivity enforced."""
    auto_nl = model.get("auto_num_leaves", True)
    params = dict(model.get("params", {}))
    if auto_nl:
        params.pop("num_leaves", None)
    elif "num_leaves" not in params:
        params["num_leaves"] = 256
    return {**model, "params": params}


def convert_metric_entries(config: dict[str, Any]) -> dict[str, Any]:
    """Convert widget-only ``_precision_at_k_k`` to MetricEntry dict form.

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
            {"precision_at_k": {"k": int(pak_k)}} if m == "precision_at_k" else m for m in metric
        ]
        new_params = {**new_params, "metric": new_metric}

    return {**config, "model": {**model, "params": new_params}}
