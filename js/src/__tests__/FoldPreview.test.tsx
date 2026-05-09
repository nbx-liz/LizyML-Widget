/**
 * Tests for FoldPreview — fold preview visualization.
 *
 * #114 Phase A: was at 1.98% — fold rendering logic is pure.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/preact";
import { FoldPreview } from "../components/FoldPreview";

describe("FoldPreview — empty state", () => {
  it("shows placeholder text when no folds are present", () => {
    render(
      <FoldPreview
        totalFolds={0}
        timeFolds={0}
        groupFolds={0}
        periods={[]}
        folds={[]}
        mode="time"
      />,
    );
    expect(screen.getByText(/Configure blocks and groups/)).toBeDefined();
  });

  it("renders the summary badge with totals", () => {
    render(
      <FoldPreview
        totalFolds={6}
        timeFolds={3}
        groupFolds={2}
        periods={[]}
        folds={[]}
        mode="time"
      />,
    );
    expect(screen.getByText(/Total:/)).toBeDefined();
    expect(screen.getByText(/3 time x 2 groups/)).toBeDefined();
  });
});

describe("FoldPreview — populated state", () => {
  const sampleFolds = [
    {
      fold: 1,
      period_label: "P0 -> P1",
      group_label: "G0",
      train_size: 100,
      valid_size: 25,
    },
    {
      fold: 2,
      period_label: "P0+P1 -> P2",
      group_label: "G1",
      train_size: 200,
      valid_size: 30,
    },
  ];

  it("renders one row per fold", () => {
    const { container } = render(
      <FoldPreview
        totalFolds={2}
        timeFolds={2}
        groupFolds={1}
        periods={["P0", "P1", "P2"]}
        folds={sampleFolds}
        mode="time"
      />,
    );
    const dataRows = container.querySelectorAll(".lzw-fold-line");
    // Header row + 2 data rows
    expect(dataRows.length).toBe(3);
  });

  it("renders period blocks for each row", () => {
    const { container } = render(
      <FoldPreview
        totalFolds={2}
        timeFolds={2}
        groupFolds={1}
        periods={["P0", "P1", "P2"]}
        folds={sampleFolds}
        mode="time"
      />,
    );
    const blocks = container.querySelectorAll(".lzw-period-block");
    expect(blocks.length).toBeGreaterThan(0);
    const trainBlocks = container.querySelectorAll(".lzw-period-block--train");
    const validBlocks = container.querySelectorAll(".lzw-period-block--valid");
    expect(trainBlocks.length).toBeGreaterThan(0);
    expect(validBlocks.length).toBeGreaterThan(0);
  });

  it("formats large train/valid sizes with locale separators", () => {
    render(
      <FoldPreview
        totalFolds={1}
        timeFolds={1}
        groupFolds={1}
        periods={["P0", "P1"]}
        folds={[
          {
            fold: 1,
            period_label: "P0 -> P1",
            group_label: "",
            train_size: 12345,
            valid_size: 6789,
          },
        ]}
        mode="time"
      />,
    );
    expect(screen.getByText(/12[,. ]?345/)).toBeDefined();
    expect(screen.getByText(/6[,. ]?789/)).toBeDefined();
  });
});
