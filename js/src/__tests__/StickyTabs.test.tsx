/**
 * Tests for header/tabs positioning — fixed via flex-shrink:0 + position:relative,
 * with internal scroll handled by .lzw-content.
 */
import { describe, it, expect } from "vitest";

describe("Header and tabs CSS positioning", () => {
  let styleSheet: string;

  beforeAll(async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const cssPath = path.resolve(__dirname, "../widget.css");
    styleSheet = fs.readFileSync(cssPath, "utf-8");
  });

  it(".lzw-header has position: relative and z-index >= 3", () => {
    const headerBlock = extractRuleBlock(styleSheet, ".lzw-header");
    expect(headerBlock).toContain("position: relative");
    const zMatch = headerBlock.match(/z-index:\s*(\d+)/);
    expect(zMatch).not.toBeNull();
    expect(Number(zMatch![1])).toBeGreaterThanOrEqual(3);
  });

  it(".lzw-header has flex-shrink: 0", () => {
    const headerBlock = extractRuleBlock(styleSheet, ".lzw-header");
    expect(headerBlock).toContain("flex-shrink: 0");
  });

  it(".lzw-tabs has position: relative and z-index >= 3", () => {
    const tabsBlock = extractRuleBlock(styleSheet, ".lzw-tabs");
    expect(tabsBlock).toContain("position: relative");
    const zMatch = tabsBlock.match(/z-index:\s*(\d+)/);
    expect(zMatch).not.toBeNull();
    expect(Number(zMatch![1])).toBeGreaterThanOrEqual(3);
  });

  it(".lzw-tabs has flex-shrink: 0", () => {
    const tabsBlock = extractRuleBlock(styleSheet, ".lzw-tabs");
    expect(tabsBlock).toContain("flex-shrink: 0");
  });

  it(".lzw-root has max-height and overflow: hidden", () => {
    // Match .lzw-root specifically (skip :root which also matches \.lzw-root)
    const allMatches = [...styleSheet.matchAll(/\.lzw-root\s*\{([^}]*)\}/g)];
    const rootBlock = allMatches.find((m) => m[1].includes("font-family"))?.[1] ?? "";
    expect(rootBlock).toContain("max-height:");
    expect(rootBlock).toContain("overflow: hidden");
  });

  it(".lzw-content has overflow-y: auto for internal scroll", () => {
    const contentBlock = extractRuleBlock(styleSheet, ".lzw-content");
    expect(contentBlock).toMatch(/overflow(-y)?:\s*auto/);
  });
});

/**
 * Extract the CSS rule block for a given exact selector.
 * Returns the content between { and } for the first match.
 */
function extractRuleBlock(css: string, selector: string): string {
  // Escape special regex chars in selector
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Match selector followed by optional whitespace/comma combinators, then {block}
  // Use a pattern that matches the exact selector (not sub-selectors like .lzw-header__title)
  const pattern = new RegExp(
    `(?:^|[},\\n])\\s*${escaped}\\s*\\{([^}]*)\\}`,
    "gm",
  );
  const match = pattern.exec(css);
  return match ? match[1] : "";
}
