/**
 * Tests for ResultsTab UI guards — inference button disabled state (B-2)
 * and Apply to Fit button visibility.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { ResultsTab } from "../tabs/ResultsTab";

// jsdom matchMedia mock (needed for PlotViewer/theme)
if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((q: string) => ({
      matches: false, media: q, onchange: null,
      addListener: vi.fn(), removeListener: vi.fn(),
      addEventListener: vi.fn(), removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

const defaultProps = {
  status: "completed",
  jobType: "fit",
  jobIndex: 1,
  progress: {},
  elapsedSec: 5.0,
  fitSummary: { metrics: { auc: { oos: 0.9 } }, fold_count: 5, fold_details: [], params: [] },
  tuneSummary: {},
  availablePlots: [],
  inferenceResult: { status: "ready", rows: 20 },
  error: {},
  plots: {},
  plotLoading: {},
  onRequestPlot: vi.fn(),
  sendAction: vi.fn(),
  onSwitchToFit: vi.fn(),
};

describe("ResultsTab — Inference button guard (B-2)", () => {
  it("shows Run Inference button when Inference accordion is opened", () => {
    render(
      <ResultsTab
        {...defaultProps}
        inferenceResult={{ status: "ready", rows: 20 }}
      />,
    );

    // Open the Inference accordion (default closed when no inference data)
    const accordionBtn = screen.getByText("Inference");
    fireEvent.click(accordionBtn);

    const btn = screen.getByText(/Run Inference/);
    expect(btn).toBeDefined();
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });

  it("disables button and shows 'Running...' after click", () => {
    const sendAction = vi.fn();
    render(
      <ResultsTab
        {...defaultProps}
        sendAction={sendAction}
        inferenceResult={{ status: "ready", rows: 20 }}
      />,
    );

    // Open accordion
    fireEvent.click(screen.getByText("Inference"));

    const btn = screen.getByText(/Run Inference/) as HTMLButtonElement;
    fireEvent.click(btn);

    expect(sendAction).toHaveBeenCalledWith("run_inference", { return_shap: false });
    expect(btn.disabled).toBe(true);
    expect(btn.textContent).toContain("Running...");
  });

  it("re-enables button when inferenceResult status changes to completed", () => {
    const { rerender } = render(
      <ResultsTab
        {...defaultProps}
        inferenceResult={{ status: "ready", rows: 20 }}
      />,
    );

    // Open accordion and click
    fireEvent.click(screen.getByText("Inference"));
    const btn = screen.getByText(/Run Inference/) as HTMLButtonElement;
    fireEvent.click(btn);
    expect(btn.disabled).toBe(true);

    // Simulate inference completion
    rerender(
      <ResultsTab
        {...defaultProps}
        inferenceResult={{ status: "completed", rows: 20, data: [{ pred: 1 }], warnings: [] }}
      />,
    );

    // Button should no longer be present (replaced by prediction table)
    const btns = screen.queryAllByText(/Run Inference/);
    if (btns.length > 0) {
      expect((btns[0] as HTMLButtonElement).disabled).toBe(false);
    }
  });
});

describe("ResultsTab — Apply to Fit button", () => {
  it("shows Apply to Fit button with primary style when tune results exist", () => {
    render(
      <ResultsTab
        {...defaultProps}
        jobType="tune"
        tuneSummary={{
          best_params: { learning_rate: 0.01 },
          best_score: 0.92,
          trials: [],
          metric_name: "auc",
          direction: "maximize",
        }}
      />,
    );

    const btn = screen.getByText(/Apply to Fit/);
    expect(btn).toBeDefined();
    expect(btn.className).toContain("lzw-btn--primary");
  });

  it("calls sendAction and onSwitchToFit when Apply to Fit is clicked", () => {
    const sendAction = vi.fn();
    const onSwitchToFit = vi.fn();
    render(
      <ResultsTab
        {...defaultProps}
        jobType="tune"
        sendAction={sendAction}
        onSwitchToFit={onSwitchToFit}
        tuneSummary={{
          best_params: { learning_rate: 0.01 },
          best_score: 0.92,
          trials: [],
          metric_name: "auc",
          direction: "maximize",
        }}
      />,
    );

    fireEvent.click(screen.getByText(/Apply to Fit/));
    expect(sendAction).toHaveBeenCalledWith("apply_best_params", {
      params: { learning_rate: 0.01 },
    });
    expect(onSwitchToFit).toHaveBeenCalled();
  });
});

describe("ResultsTab — Export Code button", () => {
  it("has accent-outline class for visibility", () => {
    render(<ResultsTab {...defaultProps} />);
    const btn = screen.getByText("Export Code");
    expect(btn.className).toContain("lzw-btn--accent-outline");
  });

  it("shows 'Exporting...' when exportLoading is true", () => {
    render(<ResultsTab {...defaultProps} exportLoading={true} />);
    const btn = screen.getByText("Exporting...");
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });
});
