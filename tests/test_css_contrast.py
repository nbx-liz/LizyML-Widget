"""Tests for CSS contrast ratio validation and hardcoded color detection.

Phase 28-5: WCAG contrast ratio auto-verification.
Phase 28-6: Hardcoded color detection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.css_contrast.contrast import (
    AA_LARGE_TEXT_AND_UI,
    AA_NORMAL_TEXT,
    contrast_ratio,
    hex_to_rgb,
    relative_luminance,
)
from tests.css_contrast.parser import extract_css_variables, find_hardcoded_colors

# Path to the built CSS (source of truth for the widget)
_WIDGET_CSS_PATH = Path(__file__).parent.parent / "js" / "src" / "widget.css"


# ── hex_to_rgb ─────────────────────────────────────────────


class TestHexToRgb:
    """Test hex color string parsing."""

    def test_six_digit_with_hash(self) -> None:
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_six_digit_without_hash(self) -> None:
        assert hex_to_rgb("00ff00") == (0, 255, 0)

    def test_three_digit_with_hash(self) -> None:
        # #abc -> #aabbcc
        assert hex_to_rgb("#abc") == (170, 187, 204)

    def test_three_digit_without_hash(self) -> None:
        assert hex_to_rgb("fff") == (255, 255, 255)

    def test_black(self) -> None:
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self) -> None:
        assert hex_to_rgb("#ffffff") == (255, 255, 255)

    def test_case_insensitive(self) -> None:
        assert hex_to_rgb("#FF00ff") == (255, 0, 255)

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError):
            hex_to_rgb("#12345")

    def test_invalid_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            hex_to_rgb("#gghhii")


# ── relative_luminance ─────────────────────────────────────


class TestRelativeLuminance:
    """Test WCAG 2.1 relative luminance calculation."""

    def test_black(self) -> None:
        assert relative_luminance(0, 0, 0) == pytest.approx(0.0)

    def test_white(self) -> None:
        assert relative_luminance(255, 255, 255) == pytest.approx(1.0)

    def test_red(self) -> None:
        # sRGB red: 0.2126 * linearize(255) = 0.2126
        assert relative_luminance(255, 0, 0) == pytest.approx(0.2126, abs=0.001)

    def test_green(self) -> None:
        # sRGB green: 0.7152 * linearize(255) = 0.7152
        assert relative_luminance(0, 255, 0) == pytest.approx(0.7152, abs=0.001)

    def test_blue(self) -> None:
        # sRGB blue: 0.0722 * linearize(255) = 0.0722
        assert relative_luminance(0, 0, 255) == pytest.approx(0.0722, abs=0.001)

    def test_mid_gray(self) -> None:
        # #808080 -> linearize(128) for each channel
        lum = relative_luminance(128, 128, 128)
        assert 0.2 < lum < 0.25  # approximately 0.2159


# ── contrast_ratio ─────────────────────────────────────────


class TestContrastRatio:
    """Test WCAG 2.1 contrast ratio calculation."""

    def test_black_on_white(self) -> None:
        ratio = contrast_ratio("#000000", "#ffffff")
        assert ratio == pytest.approx(21.0, abs=0.1)

    def test_white_on_black(self) -> None:
        # Order should not matter — always returns >= 1
        ratio = contrast_ratio("#ffffff", "#000000")
        assert ratio == pytest.approx(21.0, abs=0.1)

    def test_same_color(self) -> None:
        assert contrast_ratio("#336699", "#336699") == pytest.approx(1.0)

    def test_known_pair(self) -> None:
        # Dark gray (#333) on white (#fff): well-known ~12.6:1
        ratio = contrast_ratio("#333333", "#ffffff")
        assert 12.0 < ratio < 13.0

    def test_wcag_aa_pass(self) -> None:
        # #595959 on #fff -> ~7.0:1, passes AA normal text
        ratio = contrast_ratio("#595959", "#ffffff")
        assert ratio >= AA_NORMAL_TEXT

    def test_wcag_aa_fail(self) -> None:
        # #999 on #fff -> ~2.85:1, fails AA
        ratio = contrast_ratio("#999999", "#ffffff")
        assert ratio < AA_NORMAL_TEXT

    def test_shorthand_hex(self) -> None:
        ratio = contrast_ratio("#000", "#fff")
        assert ratio == pytest.approx(21.0, abs=0.1)


# ── extract_css_variables ──────────────────────────────────


SAMPLE_CSS = """\
.lzw-root {
  --lzw-bg-0: #ffffff;
  --lzw-bg-1: #f5f5f5;
  --lzw-fg-0: #333333;
  --lzw-fg-1: #666666;
  --lzw-accent: #1a73e8;
  --lzw-error: #d32f2f;
  font-size: 13px;
}

.lzw-header {
  background: var(--lzw-bg-1);
}

@media (prefers-color-scheme: dark) {
  .lzw-root {
    --lzw-bg-0: #1a1a1a;
    --lzw-bg-1: #2d2d2d;
    --lzw-fg-0: #e0e0e0;
    --lzw-fg-1: #aaaaaa;
    --lzw-accent: #8ab4f8;
    --lzw-error: #f28b82;
  }
}
"""


class TestExtractCssVariables:
    """Test CSS variable extraction from CSS text."""

    def test_light_mode(self) -> None:
        result = extract_css_variables(SAMPLE_CSS, scope="light")
        assert result["--lzw-bg-0"] == "#ffffff"
        assert result["--lzw-fg-0"] == "#333333"
        assert result["--lzw-accent"] == "#1a73e8"

    def test_light_mode_count(self) -> None:
        result = extract_css_variables(SAMPLE_CSS, scope="light")
        assert len(result) == 6

    def test_dark_mode(self) -> None:
        result = extract_css_variables(SAMPLE_CSS, scope="dark")
        assert result["--lzw-bg-0"] == "#1a1a1a"
        assert result["--lzw-fg-0"] == "#e0e0e0"
        assert result["--lzw-accent"] == "#8ab4f8"

    def test_dark_mode_count(self) -> None:
        result = extract_css_variables(SAMPLE_CSS, scope="dark")
        assert len(result) == 6

    def test_ignores_non_lzw_variables(self) -> None:
        css = ".lzw-root {\n  --jp-border-color: #ccc;\n  --lzw-border: #ddd;\n}"
        result = extract_css_variables(css, scope="light")
        assert "--jp-border-color" not in result
        assert result["--lzw-border"] == "#ddd"

    def test_empty_css(self) -> None:
        assert extract_css_variables("", scope="light") == {}

    def test_no_dark_block(self) -> None:
        css = ".lzw-root { --lzw-bg-0: #fff; }"
        assert extract_css_variables(css, scope="dark") == {}


# ── find_hardcoded_colors ──────────────────────────────────


class TestFindHardcodedColors:
    """Test detection of hardcoded color values in CSS."""

    def test_detects_hex_color(self) -> None:
        css = ".foo { color: #ff0000; }"
        results = find_hardcoded_colors(css)
        assert len(results) == 1
        assert results[0]["color"] == "#ff0000"

    def test_detects_short_hex(self) -> None:
        css = ".foo { color: #f00; }"
        results = find_hardcoded_colors(css)
        assert len(results) == 1
        assert results[0]["color"] == "#f00"

    def test_detects_rgb_function(self) -> None:
        css = ".foo { background: rgb(255, 0, 0); }"
        results = find_hardcoded_colors(css)
        assert len(results) == 1

    def test_detects_rgba_function(self) -> None:
        css = ".foo { background: rgba(0, 0, 0, 0.5); }"
        results = find_hardcoded_colors(css)
        assert len(results) == 1

    def test_skips_variable_definition_lines(self) -> None:
        css = "  --lzw-bg-0: #ffffff;"
        results = find_hardcoded_colors(css)
        assert len(results) == 0

    def test_skips_single_line_comment(self) -> None:
        css = "/* color: #ff0000; */"
        results = find_hardcoded_colors(css)
        assert len(results) == 0

    def test_skips_multi_line_comment(self) -> None:
        css = "/* start\ncolor: #ff0000;\n*/"
        results = find_hardcoded_colors(css)
        assert len(results) == 0

    def test_code_after_multi_line_comment(self) -> None:
        css = "/* start\ncomment\n*/\n.foo { color: #f00; }"
        results = find_hardcoded_colors(css)
        assert len(results) == 1
        assert results[0]["color"] == "#f00"

    def test_skips_var_references(self) -> None:
        css = ".foo { color: var(--lzw-fg-0, #333); }"
        # Fallback in var() is acceptable — it's the variable system working
        results = find_hardcoded_colors(css)
        assert len(results) == 0

    def test_reports_line_number(self) -> None:
        css = "line1\n.foo { color: #f00; }\nline3"
        results = find_hardcoded_colors(css)
        assert results[0]["line_number"] == 2

    def test_multiple_colors_same_line(self) -> None:
        css = ".foo { border: 1px solid #ccc; color: #333; }"
        results = find_hardcoded_colors(css)
        assert len(results) >= 2

    def test_clean_css_returns_empty(self) -> None:
        css = """\
.lzw-root {
  --lzw-bg-0: #fff;
  --lzw-fg-0: #333;
}
.foo {
  color: var(--lzw-fg-0);
  background: var(--lzw-bg-0);
}
"""
        results = find_hardcoded_colors(css)
        assert len(results) == 0


# ── widget.css WCAG contrast validation ────────────────────
#
# These tests validate the actual widget.css file.
# They are activated once Phase 28 (dark mode) CSS variables are in place.
# Until then, they are skipped with a clear reason.
#
# When Phase 28 is implemented:
#   1. Remove the skipIf decorators
#   2. Update CONTRAST_PAIRS if new --lzw-* variables are added


def _has_lzw_variables() -> bool:
    """Check if widget.css contains --lzw-* variable definitions."""
    if not _WIDGET_CSS_PATH.exists():
        return False
    css = _WIDGET_CSS_PATH.read_text()
    variables = extract_css_variables(css, scope="light")
    return len(variables) >= 3


_SKIP_REASON = "Phase 28 not yet implemented: --lzw-* CSS variables not defined"

# Foreground/background pairs to validate.
# (foreground_var, background_var, wcag_threshold)
CONTRAST_PAIRS: list[tuple[str, str, float]] = [
    # Primary text on backgrounds
    ("--lzw-fg-0", "--lzw-bg-0", AA_NORMAL_TEXT),
    ("--lzw-fg-0", "--lzw-bg-1", AA_NORMAL_TEXT),
    ("--lzw-fg-0", "--lzw-bg-2", AA_NORMAL_TEXT),
    # Secondary text on backgrounds
    ("--lzw-fg-1", "--lzw-bg-0", AA_NORMAL_TEXT),
    ("--lzw-fg-1", "--lzw-bg-1", AA_NORMAL_TEXT),
    # Muted text on backgrounds (large text / UI element threshold)
    ("--lzw-fg-2", "--lzw-bg-0", AA_LARGE_TEXT_AND_UI),
    ("--lzw-fg-2", "--lzw-bg-1", AA_LARGE_TEXT_AND_UI),
    ("--lzw-fg-2", "--lzw-bg-2", AA_LARGE_TEXT_AND_UI),
    # Text on input fields
    ("--lzw-fg-0", "--lzw-input-bg", AA_NORMAL_TEXT),
    # Status colors on backgrounds
    ("--lzw-accent", "--lzw-bg-0", AA_LARGE_TEXT_AND_UI),
    ("--lzw-error", "--lzw-bg-0", AA_LARGE_TEXT_AND_UI),
    ("--lzw-success", "--lzw-bg-0", AA_LARGE_TEXT_AND_UI),
    ("--lzw-warning", "--lzw-bg-0", AA_LARGE_TEXT_AND_UI),
]


@pytest.mark.skipif(not _has_lzw_variables(), reason=_SKIP_REASON)
class TestWidgetCssContrastLight:
    """Validate WCAG AA contrast ratios for light mode in widget.css."""

    @pytest.fixture(autouse=True)
    def _load_css(self) -> None:
        css = _WIDGET_CSS_PATH.read_text()
        self.variables = extract_css_variables(css, scope="light")

    @pytest.mark.parametrize(
        ("fg_var", "bg_var", "threshold"),
        CONTRAST_PAIRS,
        ids=[f"{fg} on {bg}" for fg, bg, _ in CONTRAST_PAIRS],
    )
    def test_contrast_ratio(self, fg_var: str, bg_var: str, threshold: float) -> None:
        fg_color = self.variables.get(fg_var)
        bg_color = self.variables.get(bg_var)
        if fg_color is None or bg_color is None:
            pytest.skip(f"Variable not defined: {fg_var if fg_color is None else bg_var}")
        ratio = contrast_ratio(fg_color, bg_color)
        assert ratio >= threshold, (
            f"Light mode: {fg_var} ({fg_color}) on {bg_var} ({bg_color}) "
            f"has contrast ratio {ratio:.2f}:1, required {threshold}:1"
        )


@pytest.mark.skipif(not _has_lzw_variables(), reason=_SKIP_REASON)
class TestWidgetCssContrastDark:
    """Validate WCAG AA contrast ratios for dark mode in widget.css."""

    @pytest.fixture(autouse=True)
    def _load_css(self) -> None:
        css = _WIDGET_CSS_PATH.read_text()
        self.variables = extract_css_variables(css, scope="dark")

    @pytest.mark.parametrize(
        ("fg_var", "bg_var", "threshold"),
        CONTRAST_PAIRS,
        ids=[f"{fg} on {bg}" for fg, bg, _ in CONTRAST_PAIRS],
    )
    def test_contrast_ratio(self, fg_var: str, bg_var: str, threshold: float) -> None:
        fg_color = self.variables.get(fg_var)
        bg_color = self.variables.get(bg_var)
        if fg_color is None or bg_color is None:
            pytest.skip(f"Variable not defined: {fg_var if fg_color is None else bg_var}")
        ratio = contrast_ratio(fg_color, bg_color)
        assert ratio >= threshold, (
            f"Dark mode: {fg_var} ({fg_color}) on {bg_var} ({bg_color}) "
            f"has contrast ratio {ratio:.2f}:1, required {threshold}:1"
        )


@pytest.mark.skipif(not _has_lzw_variables(), reason=_SKIP_REASON)
class TestWidgetCssNoHardcodedColors:
    """Verify widget.css has no hardcoded colors outside variable definitions."""

    def test_no_hardcoded_colors(self) -> None:
        css = _WIDGET_CSS_PATH.read_text()
        violations = find_hardcoded_colors(css)
        if violations:
            report = "\n".join(
                f"  Line {v['line_number']}: {v['color']} in: {v['line']}"
                for v in violations[:20]  # Show first 20
            )
            total = len(violations)
            pytest.fail(
                f"Found {total} hardcoded color(s) in widget.css "
                f"(should use --lzw-* CSS variables):\n{report}"
            )
