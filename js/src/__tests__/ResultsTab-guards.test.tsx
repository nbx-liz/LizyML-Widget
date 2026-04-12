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

describe("ResultsTab — Convergence Signal round number (P-029 bug-fix)", () => {
  // lizyml's RoundSummary.round is already 1-indexed, so the UI must not
  // add 1 to it when rendering "Round N finished ...".  Regression test
  // for the off-by-one bug reported after P-027.
  const tuneSummaryWithThreeRounds = {
    best_params: { lr: 0.01 },
    best_score: 0.9,
    trials: [],
    metric_name: "auc",
    direction: "maximize",
    rounds: [
      { round: 1, n_trials: 50, best_score_before: null, best_score_after: 0.85, expanded_dims: ["lr"] },
      { round: 2, n_trials: 30, best_score_before: 0.85, best_score_after: 0.89, expanded_dims: ["num_leaves"] },
      // Third round (2nd resume): nothing expanded → Convergence Signal
      // must label this "Round 3", not "Round 4".
      { round: 3, n_trials: 20, best_score_before: 0.89, best_score_after: 0.9, expanded_dims: [] },
    ],
    boundary_report: { dims: [{ name: "lr" }, { name: "num_leaves" }], expanded_names: [] },
  };

  it("shows the actual round number (not round+1)", () => {
    render(
      <ResultsTab
        {...defaultProps}
        status="completed"
        jobType="tune"
        fitSummary={{}}
        tuneSummary={tuneSummaryWithThreeRounds}
      />,
    );
    expect(screen.getByText(/Round 3 finished without expanding any boundary/)).toBeDefined();
    expect(screen.queryByText(/Round 4 finished/)).toBeNull();
  });

  it("does not render the Convergence Signal when the last round did expand dims", () => {
    const expandedLast = {
      ...tuneSummaryWithThreeRounds,
      rounds: [
        ...tuneSummaryWithThreeRounds.rounds.slice(0, 2),
        { round: 3, n_trials: 20, best_score_before: 0.89, best_score_after: 0.9, expanded_dims: ["feature_fraction"] },
      ],
    };
    render(
      <ResultsTab
        {...defaultProps}
        status="completed"
        jobType="tune"
        fitSummary={{}}
        tuneSummary={expandedLast}
      />,
    );
    expect(screen.queryByText(/Search space converged/)).toBeNull();
  });
});

describe("ResultsTab — Best Score null guard", () => {
  // Regression: when tune_summary is partial and best_score is missing
  // or null, the Best Score row must not render the string "undefined".
  it("renders em-dash placeholder when best_score is null", () => {
    const partialTuneSummary = {
      best_params: { lr: 0.01 },
      best_score: null,
      trials: [],
      metric_name: "auc",
      direction: "maximize",
      rounds: [],
      boundary_report: null,
    };
    const { container } = render(
      <ResultsTab
        {...defaultProps}
        status="completed"
        jobType="tune"
        fitSummary={{}}
        tuneSummary={partialTuneSummary}
      />,
    );
    expect(container.textContent).not.toContain("undefined");
    // The em-dash (U+2014) placeholder must be in the DOM.
    expect(container.textContent).toContain("\u2014");
  });

  it("renders the numeric value when best_score is a float", () => {
    const tuneSummary = {
      best_params: { lr: 0.01 },
      best_score: 0.9123,
      trials: [],
      metric_name: "auc",
      direction: "maximize",
      rounds: [],
      boundary_report: null,
    };
    render(
      <ResultsTab
        {...defaultProps}
        status="completed"
        jobType="tune"
        fitSummary={{}}
        tuneSummary={tuneSummary}
      />,
    );
    expect(screen.getByText(/auc: 0\.9123/)).toBeDefined();
  });
});
