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
});
