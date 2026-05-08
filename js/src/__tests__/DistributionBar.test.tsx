/**
 * Tests for DistributionBar — horizontal bar chart for value distribution.
 *
 * #114 Phase A: small pure component, was at 10% statement coverage.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/preact";
import { DistributionBar } from "../components/DistributionBar";

describe("DistributionBar", () => {
  it("returns null when there are no values", () => {
    const { container } = render(<DistributionBar values={[]} />);
    expect(container.textContent).toBe("");
  });

  it("renders one row per value", () => {
    const { container } = render(
      <DistributionBar
        values={[
          { value: "A", count: 30 },
          { value: "B", count: 70 },
          { value: "C", count: 100 },
        ]}
      />,
    );
    const rows = container.querySelectorAll(".lzw-dist-row");
    expect(rows.length).toBe(3);
  });

  it("scales bar widths against the maximum count", () => {
    const { container } = render(
      <DistributionBar
        values={[
          { value: "A", count: 50 },
          { value: "B", count: 100 },
        ]}
      />,
    );
    const bars = container.querySelectorAll(".lzw-dist-bar");
    expect((bars[0] as HTMLElement).style.width).toBe("50%");
    expect((bars[1] as HTMLElement).style.width).toBe("100%");
  });

  it("renders counts using locale formatting", () => {
    render(
      <DistributionBar
        values={[
          { value: "Big", count: 1234567 },
        ]}
      />,
    );
    // "1,234,567" (en-US) — accept any thousands separator via flexible regex.
    expect(screen.getByText(/1[,. ]?234[,. ]?567/)).toBeDefined();
  });

  it("guards against zero-only counts (max defaults to 1)", () => {
    const { container } = render(
      <DistributionBar values={[{ value: "Z", count: 0 }]} />,
    );
    // Width should be 0% — but the row still renders.
    expect(container.querySelectorAll(".lzw-dist-row").length).toBe(1);
    const bar = container.querySelector(".lzw-dist-bar") as HTMLElement;
    expect(bar.style.width).toBe("0%");
  });
});
