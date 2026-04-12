/**
 * Tests for RetuneControls (P-028 re-tune launcher UI).
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { RetuneControls } from "../components/RetuneControls";

describe("RetuneControls", () => {
  it("renders n_trials / expand_boundary / boundary_threshold inputs", () => {
    render(<RetuneControls onRetune={vi.fn()} disabled={false} />);
    // Use role query for button since the word "Re-tune" appears in both
    // the panel title and the button label.
    expect(
      screen.getByRole("button", { name: /Re-tune \(resume\)/i }),
    ).toBeDefined();
    expect(screen.getByLabelText(/n_trials/i)).toBeDefined();
    expect(screen.getByLabelText(/expand_boundary/i)).toBeDefined();
    expect(screen.getByLabelText(/boundary_threshold/i)).toBeDefined();
  });

  it("calls onRetune with the entered values when the button is clicked", () => {
    const onRetune = vi.fn();
    render(<RetuneControls onRetune={onRetune} disabled={false} />);

    // Modify n_trials.  NumericStepper's inner <input> uses onChange, and
    // getByLabelText resolves the aria-label / htmlFor binding.
    const n_trials = screen.getByLabelText("n_trials") as HTMLInputElement;
    fireEvent.change(n_trials, { target: { value: "25" } });

    // Click the launch button.
    const button = screen.getByRole("button", { name: /Re-tune \(resume\)/i });
    fireEvent.click(button);

    expect(onRetune).toHaveBeenCalledOnce();
    const payload = onRetune.mock.calls[0][0];
    expect(payload.n_trials).toBe(25);
    // default expand_boundary is true
    expect(payload.expand_boundary).toBe(true);
    // default boundary_threshold is 0.05
    expect(payload.boundary_threshold).toBeCloseTo(0.05);
  });

  it("disables the button when `disabled` prop is true", () => {
    render(<RetuneControls onRetune={vi.fn()} disabled={true} />);
    const button = screen.getByRole("button", { name: /Re-tune \(resume\)/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });

  it("toggles expand_boundary via checkbox", () => {
    const onRetune = vi.fn();
    render(<RetuneControls onRetune={onRetune} disabled={false} />);
    const expand = screen.getByLabelText(/expand_boundary/i) as HTMLInputElement;
    fireEvent.click(expand);  // flip to false
    fireEvent.click(screen.getByRole("button", { name: /Re-tune \(resume\)/i }));
    expect(onRetune.mock.calls[0][0].expand_boundary).toBe(false);
  });

  it("honors a custom boundary_threshold input", () => {
    const onRetune = vi.fn();
    render(<RetuneControls onRetune={onRetune} disabled={false} />);
    const thr = screen.getByLabelText("boundary_threshold") as HTMLInputElement;
    fireEvent.change(thr, { target: { value: "0.2" } });
    fireEvent.click(screen.getByRole("button", { name: /Re-tune \(resume\)/i }));
    expect(onRetune.mock.calls[0][0].boundary_threshold).toBeCloseTo(0.2);
  });
});
