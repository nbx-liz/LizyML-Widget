"""Parse CSS files to extract CSS custom property (variable) definitions."""

from __future__ import annotations

import re

_VAR_DEF_RE = re.compile(r"^\s*(--lzw-[\w-]+)\s*:\s*([^;]+);", re.MULTILINE)
_DARK_MEDIA_RE = re.compile(
    r"@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)\s*\{(.*?)\}[\s\n]*\}",
    re.DOTALL,
)
_DARK_ATTR_RE = re.compile(
    r'\.lzw-root\[data-lzw-theme=["\']dark["\']\]\s*\{([^}]+)\}',
    re.DOTALL,
)
_HEX_COLOR_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?(?![0-9a-fA-F])")
_RGB_FUNC_RE = re.compile(r"rgba?\s*\([^)]+\)")
_VAR_FUNC_RE = re.compile(r"var\s*\([^)]*\)")


_VAR_FALLBACK_RE = re.compile(r"var\([^,]+,\s*([^)]+)\)")


def _resolve_value(value: str) -> str:
    """Resolve a CSS value to a concrete color.

    If the value is a var() reference with a fallback (e.g. 'var(--jp-x, #fff)'),
    return the fallback value. Otherwise return the value as-is.
    """
    match = _VAR_FALLBACK_RE.match(value)
    if match:
        return match.group(1).strip()
    return value


_LZW_VAR_REF_RE = re.compile(r"var\((--lzw-[\w-]+)\)")


def _extract_vars_from_block(block: str) -> dict[str, str]:
    """Extract --lzw-* variable definitions from a CSS block."""
    result: dict[str, str] = {}
    for match in _VAR_DEF_RE.finditer(block):
        name = match.group(1)
        value = _resolve_value(match.group(2).strip())
        result[name] = value
    # Resolve internal --lzw-* references (e.g. --lzw-input-bg: var(--lzw-bg-0))
    for name in list(result):
        val = result[name]
        ref = _LZW_VAR_REF_RE.match(val)
        if ref and ref.group(1) in result:
            result[name] = result[ref.group(1)]
    return result


def extract_css_variables(
    css_text: str,
    *,
    scope: str = "light",
) -> dict[str, str]:
    """Extract --lzw-* CSS variable definitions from CSS text.

    Args:
        css_text: Raw CSS file content.
        scope: 'light' for root/.lzw-root definitions,
               'dark' for @media (prefers-color-scheme: dark) definitions.

    Returns:
        Dict mapping variable name (e.g. '--lzw-bg-0') to its value (e.g. '#fff').
    """
    if scope == "dark":
        result: dict[str, str] = {}
        # Support both @media (prefers-color-scheme: dark) and [data-lzw-theme="dark"]
        media_match = _DARK_MEDIA_RE.search(css_text)
        if media_match:
            result.update(_extract_vars_from_block(media_match.group(1)))
        attr_match = _DARK_ATTR_RE.search(css_text)
        if attr_match:
            result.update(_extract_vars_from_block(attr_match.group(1)))
        return result

    # Light mode: extract from top-level blocks (outside @media dark / [data-lzw-theme="dark"])
    # Remove dark mode blocks first to avoid double-counting
    text_without_dark = _DARK_MEDIA_RE.sub("", css_text)
    text_without_dark = _DARK_ATTR_RE.sub("", text_without_dark)
    return _extract_vars_from_block(text_without_dark)


def find_hardcoded_colors(css_text: str) -> list[dict[str, object]]:
    """Find hardcoded color values in CSS that should use CSS variables.

    Skips CSS variable definition lines (lines containing '--lzw-')
    and comment lines.

    Returns:
        List of dicts with 'line_number', 'line', and 'color' keys.
    """
    results: list[dict[str, object]] = []
    in_comment = False

    for line_num, line in enumerate(css_text.splitlines(), start=1):
        stripped = line.strip()

        # Track block comments
        if "/*" in stripped and "*/" in stripped and stripped.index("/*") < stripped.index("*/"):
            # Single-line comment — skip entire line
            continue
        if "/*" in stripped:
            in_comment = True
            continue
        if "*/" in stripped:
            in_comment = False
            continue
        if in_comment:
            continue

        # Skip CSS variable definition lines
        if "--lzw-" in stripped:
            continue

        # Remove var(...) references (fallback colors inside var() are acceptable)
        check_line = _VAR_FUNC_RE.sub("", stripped)

        # Find hex colors
        for match in _HEX_COLOR_RE.finditer(check_line):
            results.append(
                {
                    "line_number": line_num,
                    "line": stripped,
                    "color": match.group(0),
                }
            )

        # Find rgb/rgba functions
        for match in _RGB_FUNC_RE.finditer(check_line):
            results.append(
                {
                    "line_number": line_num,
                    "line": stripped,
                    "color": match.group(0),
                }
            )

    return results
