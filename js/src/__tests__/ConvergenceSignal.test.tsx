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

  // Regression test for a bug where the icon slot contained the 6-character
  // string "\u2713" instead of the actual check-mark glyph because JSX text
  // nodes do not interpret JavaScript escape sequences.  The raw literal
  // must NOT appear in the DOM; only the resolved glyph should.
  it("renders a real checkmark glyph, not the literal \\u2713", () => {
    const { container } = render(<ConvergenceSignal round={2} checkedDims={1} />);
    const icon = container.querySelector(".lzw-convergence-signal__icon");
    expect(icon).not.toBeNull();
    // The DOM text must be the glyph, not the escape sequence.
    expect(icon!.textContent).toBe("\u2713");
    expect(icon!.textContent).not.toContain("\\u");
    // Container must not contain the literal escape anywhere.
    expect(container.innerHTML).not.toContain("\\u2713");
  });
});
