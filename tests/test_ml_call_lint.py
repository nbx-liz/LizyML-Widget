"""Self-tests for ``scripts/lint_ml_imports.py`` (P-039 Phase 4).

The lint script is the structural enforcement of INV-H: ML library
imports outside the allowlist must fail CI. These tests pin the
script's positive and negative paths so a future refactor cannot
silently break the gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "lint_ml_imports.py"


@pytest.fixture(scope="module")
def lint_module() -> object:
    """Load the lint script as a module so we can call ``main`` and
    helpers directly without spawning a subprocess."""
    spec = importlib.util.spec_from_file_location("lint_ml_imports", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lint_ml_imports"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Detection regex — verifies the patterns the spec calls out.
# ---------------------------------------------------------------------------


class TestImportPattern:
    @pytest.mark.parametrize(
        "line",
        [
            "import lightgbm",
            "import lightgbm as lgb",
            "from lightgbm import Booster",
            "from lightgbm.sklearn import LGBMClassifier",
            "import shap",
            "from shap import TreeExplainer",
            "import xgboost",
            "from xgboost import Booster",
            "import lizyml",
            "from lizyml import Model",
            "from lizyml.core import something",
            "    import lightgbm  # nested under indent",
        ],
    )
    def test_matches_disallowed_imports(self, lint_module: object, line: str) -> None:
        assert lint_module._IMPORT_PATTERN.match(line), (  # type: ignore[attr-defined]
            f"Pattern should match: {line!r}"
        )

    @pytest.mark.parametrize(
        "line",
        [
            "import os",
            "from typing import Any",
            "import lightgbm_helper",  # similar prefix, NOT lightgbm itself
            "from lightgbmlike import x",  # similar prefix, NOT lightgbm itself
            "x = 'import lightgbm'",  # string content, not actual import
            "# import lightgbm  # comment-only is fine via NOQA path",
        ],
    )
    def test_does_not_match_unrelated_lines(self, lint_module: object, line: str) -> None:
        # NOTE: a leading ``#`` comment line is technically not an import,
        # so the regex does not match it; the NOQA tag is for actual
        # imports that need the escape hatch.
        if line.lstrip().startswith("#"):
            assert not lint_module._IMPORT_PATTERN.match(line)  # type: ignore[attr-defined]
            return
        assert not lint_module._IMPORT_PATTERN.match(line), (  # type: ignore[attr-defined]
            f"Pattern should NOT match: {line!r}"
        )


# ---------------------------------------------------------------------------
# _scan_file — the per-file driver
# ---------------------------------------------------------------------------


class TestScanFile:
    def test_clean_file_yields_no_violations(self, tmp_path: Path, lint_module: object) -> None:
        f = tmp_path / "clean.py"
        f.write_text("import os\nfrom typing import Any\n")
        assert lint_module._scan_file(f) == []  # type: ignore[attr-defined]

    def test_dirty_file_yields_violations(self, tmp_path: Path, lint_module: object) -> None:
        f = tmp_path / "dirty.py"
        f.write_text("import os\nimport lightgbm\nfrom shap import TreeExplainer\n")
        violations = lint_module._scan_file(f)  # type: ignore[attr-defined]
        assert len(violations) == 2
        assert violations[0][0] == 2
        assert "lightgbm" in violations[0][1]
        assert violations[1][0] == 3
        assert "shap" in violations[1][1]

    def test_noqa_tag_skips_violation(self, tmp_path: Path, lint_module: object) -> None:
        f = tmp_path / "noqa.py"
        f.write_text("import os\nimport lightgbm  # noqa: ML-CALL — needed for env detection\n")
        assert lint_module._scan_file(f) == []  # type: ignore[attr-defined]

    def test_noqa_only_applies_to_tagged_line(self, tmp_path: Path, lint_module: object) -> None:
        """A ``# noqa: ML-CALL`` on one line must NOT cover an
        un-tagged import on a different line."""
        f = tmp_path / "mixed.py"
        f.write_text("import lightgbm  # noqa: ML-CALL — env detection only\nimport shap\n")
        violations = lint_module._scan_file(f)  # type: ignore[attr-defined]
        assert len(violations) == 1
        assert "shap" in violations[0][1]


# ---------------------------------------------------------------------------
# main — repo-level integration
# ---------------------------------------------------------------------------


class TestMain:
    def test_repo_passes_lint_today(self, lint_module: object) -> None:
        """The repo at HEAD must be clean — every legitimate ML-using
        module is in the allowlist."""
        rc = lint_module.main()  # type: ignore[attr-defined]
        assert rc == 0, "P-039 Phase 4 lint must pass on a clean develop"


# ---------------------------------------------------------------------------
# Allowlist sanity — every allowlisted file actually exists.
# ---------------------------------------------------------------------------


class TestAllowlistSanity:
    def test_every_allowlisted_file_exists(self, lint_module: object) -> None:
        src = REPO_ROOT / "src" / "lizyml_widget"
        for fname in lint_module.ALLOWLISTED_FILES:  # type: ignore[attr-defined]
            assert (src / fname).is_file(), (
                f"Allowlisted file {fname!r} does not exist in {src}; "
                f"a refactor renamed/removed it. Update the lint allowlist."
            )

    def test_allowlist_does_not_include_widget_or_service(self, lint_module: object) -> None:
        """Defensive assertion: a future refactor must not allow
        ``widget.py`` / ``service.py`` to import ML libraries — that
        would defeat the executor funnel."""
        assert "widget.py" not in lint_module.ALLOWLISTED_FILES  # type: ignore[attr-defined]
        assert "service.py" not in lint_module.ALLOWLISTED_FILES  # type: ignore[attr-defined]
        assert "widget_actions.py" not in lint_module.ALLOWLISTED_FILES  # type: ignore[attr-defined]
        assert "job_runner.py" not in lint_module.ALLOWLISTED_FILES  # type: ignore[attr-defined]
