/**
 * Tests for sticky tab bar — header and tabs stay visible when widget scrolls.
 */
import { describe, it, expect } from "vitest";

describe("Sticky tab bar CSS classes", () => {
  let styleSheet: string;

  beforeAll(async () => {
    // Read the compiled CSS to verify sticky positioning rules
    const fs = await import("node:fs");
    const path = await import("node:path");
    const cssPath = path.resolve(__dirname, "../widget.css");
    styleSheet = fs.readFileSync(cssPath, "utf-8");
  });

  it(".lzw-header has position: sticky", () => {
    // Match the .lzw-header rule block and check for sticky
    const headerBlock = extractRuleBlock(styleSheet, ".lzw-header");
    expect(headerBlock).toContain("position: sticky");
  });

  it(".lzw-header has top: 0", () => {
    const headerBlock = extractRuleBlock(styleSheet, ".lzw-header");
    expect(headerBlock).toContain("top: 0");
  });

  it(".lzw-header has z-index >= 3", () => {
    const headerBlock = extractRuleBlock(styleSheet, ".lzw-header");
    const zMatch = headerBlock.match(/z-index:\s*(\d+)/);
    expect(zMatch).not.toBeNull();
    expect(Number(zMatch![1])).toBeGreaterThanOrEqual(3);
  });

  it(".lzw-tabs has position: sticky", () => {
    const tabsBlock = extractRuleBlock(styleSheet, ".lzw-tabs");
    expect(tabsBlock).toContain("position: sticky");
  });

  it(".lzw-tabs has top offset matching header height (40px)", () => {
    const tabsBlock = extractRuleBlock(styleSheet, ".lzw-tabs");
    expect(tabsBlock).toContain("top: 40px");
  });

  it(".lzw-tabs has z-index >= 3", () => {
    const tabsBlock = extractRuleBlock(styleSheet, ".lzw-tabs");
    const zMatch = tabsBlock.match(/z-index:\s*(\d+)/);
    expect(zMatch).not.toBeNull();
    expect(Number(zMatch![1])).toBeGreaterThanOrEqual(3);
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
