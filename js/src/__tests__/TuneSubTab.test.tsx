/**
 * Tests for TuneSubTab — Search Space + Tuning Settings + Evaluation.
 *
 * #114 Phase A: was at 1.38% — small file, mostly composition over SearchSpace.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { TuneSubTab } from "../tabs/TuneSubTab";

const baseUiSchema = {
  option_sets: {
    objective: { binary: ["binary"] },
    metric: { binary: ["auc", "binary_logloss", "binary_error"] },
    model_metric: { binary: ["auc", "binary_logloss"] },
  },
  step_map: {},
  search_space_catalog: [
    {
      key: "learning_rate",
      title: "Learning Rate",
      paramType: "number",
      modes: ["fixed", "range"],
      group: "model_params",
    },
  ],
  additional_params: [],
  conditional_visibility: {},
};

const baseProps = {
  localConfig: {
    model: { name: "lgbm", params: {} },
    training: {},
    tuning: { optuna: { space: {}, params: { n_trials: 10 } } },
  },
  uiSchema: baseUiSchema,
  task: "binary",
  dfInfo: { columns: [] },
  handleChange: vi.fn(),
  sendAction: vi.fn(),
  rawYaml: null,
  setRawYaml: vi.fn(),
};

describe("TuneSubTab — section structure", () => {
  it("renders the three accordion sections", () => {
    render(<TuneSubTab {...baseProps} />);
    expect(screen.getByText("Tuning Settings")).toBeDefined();
    expect(screen.getByText("Search Space")).toBeDefined();
    expect(screen.getByText("Evaluation")).toBeDefined();
  });

  it("shows the n_trials stepper", () => {
    render(<TuneSubTab {...baseProps} />);
    expect(screen.getByText("n_trials")).toBeDefined();
  });
});

describe("TuneSubTab — Optimization Metric segment", () => {
  it("renders the segment button row with all metrics", () => {
    const { container } = render(<TuneSubTab {...baseProps} />);
    const segmentBtns = Array.from(
      container.querySelectorAll(".lzw-segment .lzw-segment__btn"),
    ).map((b) => b.textContent);
    expect(segmentBtns).toEqual(
      expect.arrayContaining(["auc", "binary_logloss", "binary_error"]),
    );
  });

  it("marks the first configured metric as the optimization metric", () => {
    const { container } = render(
      <TuneSubTab
        {...baseProps}
        localConfig={{
          ...baseProps.localConfig,
          tuning: {
            ...baseProps.localConfig.tuning,
            evaluation: { metrics: ["binary_logloss"] },
          },
        }}
      />,
    );
    // Scope to the Optimization Metric row (avoids SearchSpace's Fixed/Range
    // segment which is also rendered above).
    const activeBtns = Array.from(
      container.querySelectorAll(".lzw-segment__btn--active"),
    ).map((b) => b.textContent);
    expect(activeBtns).toContain("binary_logloss");
  });

  it("fires handleChange when a different optimization metric is selected", () => {
    const handleChange = vi.fn();
    const { container } = render(
      <TuneSubTab
        {...baseProps}
        handleChange={handleChange}
        localConfig={{
          ...baseProps.localConfig,
          tuning: {
            ...baseProps.localConfig.tuning,
            evaluation: { metrics: ["auc"] },
          },
        }}
      />,
    );
    const segmentBtn = Array.from(
      container.querySelectorAll(".lzw-segment .lzw-segment__btn"),
    ).find((b) => b.textContent === "binary_logloss") as HTMLButtonElement;
    fireEvent.click(segmentBtn);
    expect(handleChange).toHaveBeenCalled();
    const updated = handleChange.mock.calls.at(-1)![0];
    // First metric in the array is the optimization metric.
    expect(updated.tuning.evaluation.metrics[0]).toBe("binary_logloss");
  });
});

describe("TuneSubTab — Additional Metrics chips", () => {
  it("excludes the active optimization metric from the additional chip list", () => {
    const { container } = render(
      <TuneSubTab
        {...baseProps}
        localConfig={{
          ...baseProps.localConfig,
          tuning: {
            ...baseProps.localConfig.tuning,
            evaluation: { metrics: ["auc"] },
          },
        }}
      />,
    );
    // The Additional Metrics chip group should NOT contain the active "auc" chip.
    const chips = container.querySelectorAll(".lzw-chip-group .lzw-chip");
    const labels = Array.from(chips).map((c) => c.textContent);
    expect(labels).not.toContain("auc");
    expect(labels).toContain("binary_logloss");
  });

  it("toggles an additional metric on click", () => {
    const handleChange = vi.fn();
    const { container } = render(
      <TuneSubTab
        {...baseProps}
        handleChange={handleChange}
        localConfig={{
          ...baseProps.localConfig,
          tuning: {
            ...baseProps.localConfig.tuning,
            evaluation: { metrics: ["auc"] },
          },
        }}
      />,
    );
    // Click the chip-group instance of binary_logloss (additional metric).
    const chipBtn = Array.from(
      container.querySelectorAll(".lzw-chip-group .lzw-chip"),
    ).find((c) => c.textContent === "binary_logloss") as HTMLButtonElement;
    fireEvent.click(chipBtn);
    expect(handleChange).toHaveBeenCalled();
  });
});

describe("TuneSubTab — precision_at_k k stepper", () => {
  it("hides the k stepper when precision_at_k is not in the metrics list", () => {
    const { queryByText } = render(
      <TuneSubTab
        {...baseProps}
        uiSchema={{
          ...baseUiSchema,
          option_sets: {
            ...baseUiSchema.option_sets,
            metric: { binary: ["auc", "precision_at_k"] },
          },
        }}
        localConfig={{
          ...baseProps.localConfig,
          tuning: {
            ...baseProps.localConfig.tuning,
            evaluation: { metrics: ["auc"], params: {} },
          },
        }}
      />,
    );
    expect(queryByText("precision_at_k: k")).toBeNull();
  });

  it("shows the k stepper when precision_at_k is selected and updates evaluation.params on change", () => {
    const handleChange = vi.fn();
    const { container } = render(
      <TuneSubTab
        {...baseProps}
        handleChange={handleChange}
        uiSchema={{
          ...baseUiSchema,
          option_sets: {
            ...baseUiSchema.option_sets,
            metric: { binary: ["auc", "precision_at_k"] },
          },
        }}
        localConfig={{
          ...baseProps.localConfig,
          tuning: {
            ...baseProps.localConfig.tuning,
            evaluation: {
              metrics: ["precision_at_k"],
              params: { precision_at_k_k: 10 },
            },
          },
        }}
      />,
    );
    expect(screen.getByText("precision_at_k: k")).toBeDefined();
    // Use the stepper's "+" button to bump k from 10 → 11.
    const plusBtns = Array.from(container.querySelectorAll("button")).filter(
      (b) => b.textContent === "+" || b.textContent?.trim() === "+",
    );
    // The precision_at_k stepper is the last one rendered (after n_trials).
    const plus = plusBtns.at(-1)!;
    fireEvent.click(plus);
    const updated = handleChange.mock.calls.at(-1)![0];
    expect(updated.tuning.evaluation.params.precision_at_k_k).toBe(11);
  });
});
