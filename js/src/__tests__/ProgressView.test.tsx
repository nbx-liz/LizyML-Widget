/**
 * Tests for ProgressView — progress bar, elapsed time, cancel button, fold results.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { ProgressView } from "../components/ProgressView";

const defaultProps = {
  jobType: "fit",
  jobIndex: 1,
  progress: { current: 3, total: 5, message: "" },
  elapsedSec: 125,
  onCancel: vi.fn(),
};

describe("ProgressView — progress bar with current/total", () => {
  it("renders fold progress text", () => {
    render(<ProgressView {...defaultProps} />);
    expect(screen.getByText(/Fold 3 \/ 5/)).toBeDefined();
  });

  it("renders progress bar with correct width", () => {
    const { container } = render(<ProgressView {...defaultProps} />);
    const bar = container.querySelector(".lzw-progress__bar") as HTMLElement;
    expect(bar).not.toBeNull();
    // 3/5 = 60%
    expect(bar.style.width).toBe("60%");
  });

  it("shows indeterminate bar when current is 0", () => {
    const { container } = render(
      <ProgressView {...defaultProps} progress={{ current: 0, total: 5 }} />,
    );
    const bar = container.querySelector(".lzw-progress__bar--indeterminate");
    expect(bar).not.toBeNull();
  });
});

describe("ProgressView — elapsed time formatting", () => {
  it("formats 125 seconds as 02:05", () => {
    render(<ProgressView {...defaultProps} elapsedSec={125} />);
    expect(screen.getByText(/02:05/)).toBeDefined();
  });

  it("formats 0 seconds as 00:00", () => {
    render(<ProgressView {...defaultProps} elapsedSec={0} />);
    expect(screen.getByText(/00:00/)).toBeDefined();
  });
});

describe("ProgressView — cancel button", () => {
  it("fires onCancel when clicked", () => {
    const onCancel = vi.fn();
    render(<ProgressView {...defaultProps} onCancel={onCancel} />);
    fireEvent.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalled();
  });
});

describe("ProgressView — fold results", () => {
  it("shows fold results when available", () => {
    render(
      <ProgressView
        {...defaultProps}
        progress={{
          current: 2,
          total: 3,
          fold_results: [
            { label: "Fold 1", score: "0.85", done: true },
            { label: "Fold 2", score: "0.87", done: true },
            { label: "Fold 3", done: false },
          ],
        }}
      />,
    );
    expect(screen.getByText("Fold 1")).toBeDefined();
    expect(screen.getByText(/0\.85/)).toBeDefined();
    expect(screen.getByText("Fold 3")).toBeDefined();
  });

  it("does not render fold section when no fold_results", () => {
    const { container } = render(
      <ProgressView {...defaultProps} progress={{ current: 1, total: 5 }} />,
    );
    expect(container.querySelector(".lzw-progress__folds")).toBeNull();
  });
});

describe("ProgressView — job type and index", () => {
  it("renders job type title capitalized with index", () => {
    render(<ProgressView {...defaultProps} jobType="fit" jobIndex={3} />);
    expect(screen.getByText("Fit #3")).toBeDefined();
  });

  it("renders tune type with indeterminate bar when current is 0", () => {
    const { container } = render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{ current: 0, total: 50 }}
      />,
    );
    // P-027: with current=0 the bar is indeterminate but the label already
    // shows "Trial 0/50" so the user sees the total trial count.
    expect(screen.getByText(/Trial 0\/50/)).toBeDefined();
    const bar = container.querySelector(".lzw-progress__bar--indeterminate");
    expect(bar).not.toBeNull();
  });

  it("shows 'Tuning...' fallback when total is 0", () => {
    render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{ current: 0, total: 0, message: "Starting tune..." }}
      />,
    );
    expect(screen.getByText(/Tuning\.\.\./)).toBeDefined();
  });

  it("renders tune trial progress with determinate bar", () => {
    const { container } = render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{ current: 10, total: 50 }}
      />,
    );
    // P-027: in-progress tune shows "Trial N/M" with a real bar.
    expect(screen.getByText(/Trial 10\/50/)).toBeDefined();
    const bar = container.querySelector(".lzw-progress__bar");
    expect(bar).not.toBeNull();
  });
});

describe("ProgressView — P-027 round-aware tune", () => {
  it("shows Round badge and cumulative trials on resume rounds", () => {
    render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{
          current: 17,
          total: 30,
          round: 2,
          total_rounds: 2,
          cumulative_trials: 67,
          best_score: 0.279,
        }}
      />,
    );
    expect(screen.getByText(/Round 2\/2/)).toBeDefined();
    expect(screen.getByText(/Trial 17\/30/)).toBeDefined();
    expect(screen.getByText(/cumulative: 67/)).toBeDefined();
    expect(screen.getByText(/0\.2790/)).toBeDefined();
  });

  it("renders expanded_dims chips when boundary expansion happened", () => {
    render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{
          current: 5,
          total: 30,
          round: 2,
          expanded_dims: ["learning_rate", "num_leaves"],
        }}
      />,
    );
    expect(screen.getByText("learning_rate")).toBeDefined();
    expect(screen.getByText("num_leaves")).toBeDefined();
    expect(screen.getByText(/Boundary expansion this round/)).toBeDefined();
  });

  it("hides round badge for a single-round tune", () => {
    render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{ current: 5, total: 30, round: 1 }}
      />,
    );
    expect(screen.queryByText(/^Round /)).toBeNull();
  });
});
