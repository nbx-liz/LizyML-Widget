#!/usr/bin/env python3
"""ML library import lint (P-039 Phase 4 / change-gate enforcement).

Fails CI if any production source file in ``src/lizyml_widget/`` imports
an ML library outside the allowlisted modules. The allowlisted modules
are the legitimate boundary owners of ML library calls:

- ``adapter.py`` and any ``adapter_*.py`` helper — the canonical
  Adapter layer that translates LizyML / LightGBM / etc. into the
  widget's common types.
- ``backend_executor.py`` — the single caller-thread chokepoint
  introduced by P-039 Phase 3.
- ``_subprocess_entry.py`` — the subprocess body, which runs in a
  different process and therefore cannot bind libgomp on the parent
  thread.
- ``openmp_detect.py`` — imports ``lightgbm`` only to populate
  ``/proc/self/maps`` for the libgomp detection routine; never
  enters a parallel region itself.

Any other module that needs to import ``lightgbm`` / ``shap`` /
``xgboost`` / ``lizyml`` must add a ``# noqa: ML-CALL`` comment on
the offending line and document the reason in the surrounding
docstring or PR body. This is the structural enforcement for INV-H
declared in P-039: "lightgbm / shap / xgboost / sklearn API への
直接呼出は BackendExecutor モジュール外部から発生しない".

Detection patterns:

- ``import lightgbm`` / ``import lightgbm as ...``
- ``from lightgbm ...``
- ``import shap`` / ``from shap ...``
- ``import xgboost`` / ``from xgboost ...``
- ``import lizyml`` / ``from lizyml ...``

Direct method patterns (``model.predict``, etc.) are intentionally NOT
matched — they produce too many false positives on attribute access.
The import gate alone is sufficient: a file cannot use these libraries
without importing them.

Exit codes:
- 0 — clean
- 1 — violations found (printed to stderr)

Run locally:
    uv run python scripts/lint_ml_imports.py

Run as part of CI:
    .github/workflows/ci.yml -> ml-call lint job
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Files allowed to import ML libraries directly. All other source
# files in ``src/lizyml_widget/`` must funnel through one of these.
ALLOWLISTED_FILES: frozenset[str] = frozenset(
    {
        "adapter.py",
        "adapter_internals.py",
        "adapter_params.py",
        "adapter_results.py",
        "adapter_schema.py",
        "adapter_views.py",
        "backend_executor.py",
        "_subprocess_entry.py",
        "openmp_detect.py",
    }
)

# ML library names whose direct import is gate-required outside the
# allowlist. Add new ML libraries here as they enter the project.
ML_LIBRARIES: frozenset[str] = frozenset(
    {
        "lightgbm",
        "shap",
        "xgboost",
        "lizyml",
    }
)

# Per-line escape hatch comment. A line ending with ``# noqa: ML-CALL``
# is exempt from the gate (the surrounding docstring / PR body must
# explain why).
NOQA_TAG = "# noqa: ML-CALL"

# Build a regex that matches:
#   import <lib>
#   import <lib> as ...
#   from <lib> ...
#   from <lib>.something ...
_LIB_ALTERNATION = "|".join(re.escape(name) for name in sorted(ML_LIBRARIES))
_IMPORT_PATTERN = re.compile(
    rf"^\s*(?:import\s+(?:{_LIB_ALTERNATION})(?:\s|$|\.|,)"
    rf"|from\s+(?:{_LIB_ALTERNATION})(?:\s|\.|$))"
)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of (line_number, line_text) for offending imports."""
    violations: list[tuple[int, str]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"WARN: cannot read {path}: {exc}", file=sys.stderr)
        return violations
    for line_no, line in enumerate(source.splitlines(), start=1):
        if NOQA_TAG in line:
            continue
        if _IMPORT_PATTERN.match(line):
            violations.append((line_no, line.rstrip()))
    return violations


def _iter_source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.name not in ALLOWLISTED_FILES)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src" / "lizyml_widget"
    if not src_root.is_dir():
        print(f"ERROR: source directory not found: {src_root}", file=sys.stderr)
        return 1

    total_violations = 0
    for path in _iter_source_files(src_root):
        violations = _scan_file(path)
        if not violations:
            continue
        rel = path.relative_to(repo_root)
        for line_no, line in violations:
            print(
                f"{rel}:{line_no}: ML-CALL violation — {line.strip()}",
                file=sys.stderr,
            )
        total_violations += len(violations)

    if total_violations:
        print(
            f"\n{total_violations} ML-CALL violation(s) found. "
            f"Direct ML library imports are only allowed in: "
            f"{', '.join(sorted(ALLOWLISTED_FILES))}. "
            f"Add a ``{NOQA_TAG}`` comment with a documented reason if "
            f"you need an exception, or route the call through "
            f"``BackendExecutor`` / the Adapter layer. See HISTORY.md "
            f"P-039 Phase 4 and BLUEPRINT.md §3.2.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
