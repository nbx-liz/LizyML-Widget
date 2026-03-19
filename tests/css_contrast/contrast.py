"""WCAG 2.1 contrast ratio calculation.

Implements the relative luminance and contrast ratio formulas
from https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio.
"""

from __future__ import annotations

# WCAG AA thresholds
AA_NORMAL_TEXT = 4.5
AA_LARGE_TEXT_AND_UI = 3.0


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to (R, G, B) tuple.

    Accepts '#rgb', '#rrggbb', 'rgb', 'rrggbb' formats.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6:
        msg = f"Invalid hex color: {hex_color!r} (expected 3 or 6 hex digits)"
        raise ValueError(msg)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        msg = f"Invalid hex color: {hex_color!r} (non-hex characters)"
        raise ValueError(msg) from None


def _linearize(channel: int) -> float:
    """Linearize an sRGB channel value (0-255) to linear light."""
    s = channel / 255.0
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4


def relative_luminance(r: int, g: int, b: int) -> float:
    """Calculate WCAG 2.1 relative luminance for an sRGB color.

    Each channel value is 0-255.
    Returns a value between 0.0 (darkest) and 1.0 (lightest).
    """
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def contrast_ratio(color1: str, color2: str) -> float:
    """Calculate WCAG 2.1 contrast ratio between two hex colors.

    Returns a value >= 1.0 (1:1 means identical, 21:1 is max).
    """
    l1 = relative_luminance(*hex_to_rgb(color1))
    l2 = relative_luminance(*hex_to_rgb(color2))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)
