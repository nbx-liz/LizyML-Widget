/**
 * Tests for ConvergenceSignal (P-027 re-tune monitoring).
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { ConvergenceSignal } from "../components/ConvergenceSignal";

describe("ConvergenceSignal", () => {
  it("renders nothing for the first round", () => {
    const { container } = render(
      <ConvergenceSignal round={1} checkedDims={5} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders the convergence banner for round 2+", () => {
    render(<ConvergenceSignal round={2} checkedDims={5} />);
    expect(screen.getByText(/Search space converged/)).toBeDefined();
    expect(screen.getByText(/Round 2 finished without expanding any boundary/)).toBeDefined();
    expect(screen.getByText(/5 dims checked/)).toBeDefined();
  });

  it("omits the dims count copy when checkedDims is 0", () => {
    render(<ConvergenceSignal round={3} checkedDims={0} />);
    expect(screen.queryByText(/dims checked/)).toBeNull();
  });

  it("invokes onProceedToFit when the button is clicked", () => {
    const onProceedToFit = vi.fn();
    render(
      <ConvergenceSignal
        round={2}
        checkedDims={4}
        onProceedToFit={onProceedToFit}
      />,
    );
    fireEvent.click(screen.getByText(/Proceed to Fit/));
    expect(onProceedToFit).toHaveBeenCalledOnce();
  });

  it("omits the button when no onProceedToFit handler is provided", () => {
    render(<ConvergenceSignal round={2} checkedDims={4} />);
    expect(screen.queryByText(/Proceed to Fit/)).toBeNull();
  });
});
