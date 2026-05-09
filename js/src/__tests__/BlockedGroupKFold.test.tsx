/**
 * Tests for BlockedGroupKFold — 2-axis CV configuration UI.
 *
 * #114 Phase A: was at 0.57% — this single render covers the bulk of the
 * component since most code paths are conditional on cv state.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/preact";
import { BlockedGroupKFold } from "../components/BlockedGroupKFold";

const baseProps = {
  cv: {
    blocks: { col: "", cutoffs: [], mode: "expanding", train_window: 1 },
    groups: { col: "", n_splits: 3, stratify: "auto", shuffle: true },
    min_train_rows: 10,
    min_valid_rows: 5,
  },
  allColumns: ["x1", "x2", "date"],
  columnStats: null,
  splitPreview: null,
  sendAction: vi.fn(),
  sendCv: vi.fn(),
};

describe("BlockedGroupKFold — initial render (no columns selected)", () => {
  it("renders without crashing for an empty cv state", () => {
    const { container } = render(<BlockedGroupKFold {...baseProps} />);
    expect(container.querySelector(".lzw-form-row")).not.toBeNull();
  });

  it("renders the period column selector", () => {
    render(<BlockedGroupKFold {...baseProps} />);
    expect(screen.getAllByRole("combobox").length).toBeGreaterThanOrEqual(1);
  });
});

describe("BlockedGroupKFold — with column stats", () => {
  it("renders period cutoff toggles when blocks.col is set and stats are loaded", () => {
    const { container } = render(
      <BlockedGroupKFold
        {...baseProps}
        cv={{
          ...baseProps.cv,
          blocks: { col: "date", cutoffs: ["2025-Q2"], mode: "expanding", train_window: 1 },
        }}
        columnStats={{
          column: "date",
          unique_count: 4,
          values: [
            { value: "2025-Q1", count: 100 },
            { value: "2025-Q2", count: 110 },
            { value: "2025-Q3", count: 105 },
            { value: "2025-Q4", count: 120 },
          ],
        }}
      />,
    );
    // Period values should be rendered as toggleable controls
    expect(container.textContent).toContain("2025-Q1");
    expect(container.textContent).toContain("2025-Q4");
  });

  it("renders the group column form when groups.col is set", () => {
    const { container } = render(
      <BlockedGroupKFold
        {...baseProps}
        cv={{
          ...baseProps.cv,
          groups: { col: "x1", n_splits: 3, stratify: "auto", shuffle: true },
        }}
      />,
    );
    // The selected group column should be reflected in the form select.
    const selects = container.querySelectorAll("select");
    const values = Array.from(selects).map((s) => (s as HTMLSelectElement).value);
    expect(values).toContain("x1");
  });
});

describe("BlockedGroupKFold — with split preview", () => {
  it("renders FoldPreview when a splitPreview is supplied", () => {
    render(
      <BlockedGroupKFold
        {...baseProps}
        cv={{
          ...baseProps.cv,
          blocks: { col: "date", cutoffs: ["2025-Q2"], mode: "expanding", train_window: 1 },
          groups: { col: "x1", n_splits: 2, stratify: "auto", shuffle: true },
        }}
        splitPreview={{
          total_folds: 4,
          time_folds: 2,
          group_folds: 2,
          periods: ["P0", "P1", "P2"],
          folds: [
            {
              fold: 1,
              period_label: "P0 -> P1",
              group_label: "G0",
              train_size: 100,
              valid_size: 25,
            },
          ],
          mode: "expanding",
        }}
      />,
    );
    expect(screen.getByText(/Total:/)).toBeDefined();
  });
});
