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

  it("renders tune type with indeterminate bar", () => {
    const { container } = render(
      <ProgressView
        {...defaultProps}
        jobType="tune"
        progress={{ current: 0, total: 50, message: "Trial 10/50" }}
      />,
    );
    expect(screen.getByText(/Tuning 50 trials/)).toBeDefined();
    const bar = container.querySelector(".lzw-progress__bar--indeterminate");
    expect(bar).not.toBeNull();
  });
});
