/**
 * Tests for BoundaryExpansionPanel (P-027 re-tune monitoring).
 *
 * Covers:
 *   - empty / null report → renders nothing
 *   - all-dim-unchanged report → collapses to "No expansion needed" copy
 *   - expanded dims → renders table rows with correct before/after ranges
 *   - unchanged footer counter
 *   - showUnchanged flag forces the full table
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/preact";
import {
  BoundaryExpansionPanel,
  type BoundaryReport,
} from "../components/BoundaryExpansionPanel";

function makeReport(): BoundaryReport {
  return {
    dims: [
      {
        name: "learning_rate",
        best_value: 0.0001,
        low: 0.0001,
        high: 0.1,
        position_pct: 0.0,
        edge: "lower",
        expanded: true,
        new_low: 0.00001,
        new_high: 0.1,
      },
      {
        name: "num_leaves",
        best_value: 256,
        low: 16,
        high: 256,
        position_pct: 1.0,
        edge: "upper",
        expanded: true,
        new_low: 16,
        new_high: 512,
      },
      {
        name: "max_depth",
        best_value: 5,
        low: 3,
        high: 10,
        position_pct: 0.28,
        edge: "mid",
        expanded: false,
        new_low: null,
        new_high: null,
      },
    ],
    expanded_names: ["learning_rate", "num_leaves"],
  };
}

describe("BoundaryExpansionPanel — rendering", () => {
  it("returns nothing when the report is null", () => {
    const { container } = render(<BoundaryExpansionPanel report={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("returns nothing when the report has an empty dims list", () => {
    const { container } = render(
      <BoundaryExpansionPanel report={{ dims: [], expanded_names: [] }} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("shows 'No expansion needed' when every dim is unchanged", () => {
    render(
      <BoundaryExpansionPanel
        report={{
          dims: [
            {
              name: "lr",
              best_value: 0.01,
              low: 0.001,
              high: 0.1,
              position_pct: 0.5,
              edge: "mid",
              expanded: false,
              new_low: null,
              new_high: null,
            },
          ],
          expanded_names: [],
        }}
      />,
    );
    expect(screen.getByText(/No expansion needed/)).toBeDefined();
  });

  it("renders a row per expanded dim with before/after ranges", () => {
    render(<BoundaryExpansionPanel report={makeReport()} />);
    expect(screen.getByText("learning_rate")).toBeDefined();
    expect(screen.getByText("num_leaves")).toBeDefined();
    // unchanged dim is NOT in the default view
    expect(screen.queryByText("max_depth")).toBeNull();
    // shows "1 dim unchanged" footer counter
    expect(screen.getByText(/1 dim unchanged/)).toBeDefined();
    // header counter: 2/3 expanded
    expect(screen.getByText(/2\/3 dims expanded/)).toBeDefined();
  });

  it("renders all dims when showUnchanged is true", () => {
    render(
      <BoundaryExpansionPanel report={makeReport()} showUnchanged={true} />,
    );
    expect(screen.getByText("learning_rate")).toBeDefined();
    expect(screen.getByText("num_leaves")).toBeDefined();
    expect(screen.getByText("max_depth")).toBeDefined();
  });

  // Edge: a dim with edge="mid" that was still expanded (e.g. the
  // backend expanded both sides equally) must fall through to the
  // generic "↔" glyph in edgeIcon() instead of defaulting to the
  // unchanged bullet.  Exercises the final return branch of edgeIcon.
  it("renders the ↔ glyph for a mid-edge expanded dim", () => {
    const midExpandedReport: BoundaryReport = {
      dims: [
        {
          name: "feature_fraction",
          best_value: 0.6,
          low: 0.4,
          high: 0.8,
          position_pct: 0.5,
          edge: "mid",
          expanded: true,
          new_low: 0.2,
          new_high: 1.0,
        },
      ],
      expanded_names: ["feature_fraction"],
    };
    const { container } = render(
      <BoundaryExpansionPanel report={midExpandedReport} />,
    );
    // The ↔ glyph (U+2194) must appear in the DOM.
    expect(container.textContent).toContain("\u2194");
  });
});

describe("BoundaryExpansionPanel — formatNumber / formatRange edge cases", () => {
  /** Helper that renders a single-dim panel with the given numeric fields. */
  function renderOneDim(overrides: Record<string, unknown>) {
    const report: BoundaryReport = {
      dims: [
        {
          name: "p",
          best_value: null,
          low: null,
          high: null,
          position_pct: 0.5,
          edge: "upper",
          expanded: true,
          new_low: null,
          new_high: null,
          ...overrides,
        } as unknown as BoundaryReport["dims"][number],
      ],
      expanded_names: ["p"],
    };
    return render(<BoundaryExpansionPanel report={report} />);
  }

  it("renders '?' placeholder when low is null", () => {
    const { container } = renderOneDim({ low: null, high: 0.5 });
    expect(container.textContent).toContain("[?, 0.5]");
  });

  it("renders '?' placeholder when high is null", () => {
    const { container } = renderOneDim({ low: 0.01, high: null });
    expect(container.textContent).toContain("[0.01, ?]");
  });

  it("uses exponential notation for very small numbers (< 0.001)", () => {
    const { container } = renderOneDim({ low: 0.00001, high: 0.1 });
    // 0.00001 should become "1.00e-5".
    expect(container.textContent).toMatch(/1\.00e-5/);
  });

  it("uses exponential notation for very large numbers (>= 10000)", () => {
    const { container } = renderOneDim({ low: 0, high: 50000 });
    // 50000 should become "5.00e+4".
    expect(container.textContent).toMatch(/5\.00e\+4/);
  });

  it("renders integer values without decimals", () => {
    const { container } = renderOneDim({ low: 16, high: 256 });
    expect(container.textContent).toContain("[16, 256]");
  });

  it("trims trailing zeros from fractional values", () => {
    const { container } = renderOneDim({ low: 0.5, high: 0.8 });
    // Not "0.5000, 0.8000" — trailing zeros stripped.
    expect(container.textContent).toContain("[0.5, 0.8]");
  });

  it("renders literal Infinity for non-finite values", () => {
    const { container } = renderOneDim({
      low: Number.NEGATIVE_INFINITY,
      high: Number.POSITIVE_INFINITY,
    });
    expect(container.textContent).toMatch(/-Infinity/);
    expect(container.textContent).toMatch(/Infinity/);
  });

  // Coverage hole: new_low is set but new_high is null → ``d.new_high ?? d.high``
  // fallback chain on line 118, plus the "unchanged" dim counter in the
  // footer uses singular/plural ("dim" vs "dims" on line 141), and
  // edge="" triggers the `d.edge || "\u2014"` fallback on line 129.
  it("covers partial new_low expansion + singular unchanged footer + empty edge", () => {
    const report: BoundaryReport = {
      dims: [
        {
          name: "partial",
          best_value: 0.0001,
          low: 0.0001,
          high: 0.5,
          position_pct: 0.0,
          edge: "",  // ← triggers d.edge || "—" fallback on L129
          expanded: true,
          new_low: 0.00001,  // only new_low set, new_high falls back to high
          new_high: null,
        },
        {
          // Single unchanged dim → footer says "1 dim unchanged" (singular).
          name: "static",
          best_value: 0.5,
          low: 0.0,
          high: 1.0,
          position_pct: 0.5,
          edge: "mid",
          expanded: false,
          new_low: null,
          new_high: null,
        },
      ],
      expanded_names: ["partial"],
    };
    const { container } = render(<BoundaryExpansionPanel report={report} />);
    // L118: new_high falls back to high=0.5 in the "after" range.
    // 0.00001 renders in scientific notation so the range string has the
    // shape [1.00e-5, 0.5].  The key assertion is that the fallback to
    // ``d.high`` (0.5) happened — otherwise the "0.5" would be missing.
    expect(container.textContent).toMatch(/\[1\.00e-5, 0\.5\]/);
    // L129: empty edge falls back to em-dash.
    expect(container.textContent).toContain("\u2014");
    // L141: singular "1 dim unchanged" (no trailing "s").
    expect(container.textContent).toMatch(/1 dim unchanged/);
    expect(container.textContent).not.toMatch(/1 dims unchanged/);
  });
});
