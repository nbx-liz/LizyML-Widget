/**
 * Tests for SearchSpace — per-parameter mode selection for tuning (Fixed/Range/Choice).
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { SearchSpace } from "../components/SearchSpace";

const minimalUiSchema = {
  option_sets: { objective: { binary: ["binary", "cross_entropy"] } },
  step_map: { learning_rate: 0.01 },
  search_space_catalog: [
    { key: "learning_rate", title: "Learning Rate", paramType: "number", modes: ["fixed", "range"], group: "model_params" },
    { key: "num_leaves", title: "Num Leaves", paramType: "integer", modes: ["fixed", "range", "choice"], group: "model_params" },
    { key: "objective", title: "Objective", paramType: "string", modes: ["fixed"], group: "model_params" },
  ],
  additional_params: ["extra_param"],
  conditional_visibility: {},
};

const defaultProps = {
  schema: { type: "object", properties: {} },
  spaceValue: {},
  fixedModelParams: { learning_rate: 0.1 },
  fixedTraining: {},
  modelConfig: { params: { learning_rate: 0.1 } },
  trainingConfig: {},
  task: "binary",
  uiSchema: minimalUiSchema,
  onChange: vi.fn(),
};

describe("SearchSpace — initial rendering", () => {
  it("renders parameter names from catalog", () => {
    render(<SearchSpace {...defaultProps} />);
    expect(screen.getByText("Learning Rate")).toBeDefined();
    expect(screen.getByText("Num Leaves")).toBeDefined();
    expect(screen.getByText("Objective")).toBeDefined();
  });

  it("renders group header for Model Params", () => {
    render(<SearchSpace {...defaultProps} />);
    expect(screen.getByText("Model Params")).toBeDefined();
  });

  it("renders grid header with Parameter/Mode/Configuration columns", () => {
    render(<SearchSpace {...defaultProps} />);
    expect(screen.getByText("Parameter")).toBeDefined();
    expect(screen.getByText("Mode")).toBeDefined();
    expect(screen.getByText("Configuration")).toBeDefined();
  });
});

describe("SearchSpace — mode switching", () => {
  it("shows Fixed/Range buttons for params with those modes", () => {
    const { container } = render(<SearchSpace {...defaultProps} />);
    // learning_rate has modes: ["fixed", "range"]
    const fixedBtns = container.querySelectorAll('.lzw-segment__btn');
    const fixedLabels = Array.from(fixedBtns).map((b) => b.textContent);
    expect(fixedLabels).toContain("Fixed");
    expect(fixedLabels).toContain("Range");
  });

  it("shows Fixed/Range/Choice buttons for params with all three modes", () => {
    const { container } = render(<SearchSpace {...defaultProps} />);
    const allBtns = Array.from(container.querySelectorAll('.lzw-segment__btn')).map((b) => b.textContent);
    expect(allBtns).toContain("Choice");
  });

  it("fires onChange when switching to Range mode", () => {
    const onChange = vi.fn();
    render(<SearchSpace {...defaultProps} onChange={onChange} />);
    // Find the Range buttons; click the first one (for learning_rate)
    const rangeBtns = screen.getAllByText("Range");
    fireEvent.click(rangeBtns[0]);
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        space: expect.objectContaining({
          learning_rate: expect.objectContaining({ type: "float", low: expect.any(Number), high: expect.any(Number) }),
        }),
      }),
    );
  });
});

describe("SearchSpace — Fixed mode display", () => {
  it("shows Fixed tag for single-mode params", () => {
    render(<SearchSpace {...defaultProps} />);
    // Objective has modes: ["fixed"] → renders as a tag, not button
    const tags = screen.getAllByText("Fixed");
    // At least one should be a tag (for objective which has only fixed mode)
    expect(tags.length).toBeGreaterThan(0);
  });
});

describe("SearchSpace — Range mode display", () => {
  it("shows low/high inputs and Log checkbox when in range mode", () => {
    const { container } = render(
      <SearchSpace
        {...defaultProps}
        spaceValue={{ learning_rate: { type: "float", low: 0.001, high: 0.1, log: false } }}
        fixedModelParams={{}}
      />,
    );
    // Should have low/high stepper inputs
    const steppers = container.querySelectorAll(".lzw-search-space__range .lzw-stepper__input");
    expect(steppers.length).toBeGreaterThanOrEqual(2);
    // Log checkbox
    expect(screen.getByText("Log")).toBeDefined();
  });
});

describe("SearchSpace — empty catalog", () => {
  it("renders grid header even with no catalog entries", () => {
    render(
      <SearchSpace
        {...defaultProps}
        uiSchema={{ ...minimalUiSchema, search_space_catalog: [] }}
      />,
    );
    expect(screen.getByText("Parameter")).toBeDefined();
  });
});

// ── Bug 7: metric Fixed→Choice should not nest array in choices ──

describe("SearchSpace — Bug 7: metric Fixed→Choice mode switch", () => {
  const metricUiSchema = {
    option_sets: {
      objective: { binary: ["binary"] },
      model_metric: { binary: ["auc", "binary_logloss", "binary_error"] },
      metric: { binary: ["auc", "logloss", "accuracy"] },
    },
    step_map: {},
    search_space_catalog: [
      { key: "metric", title: "Metric", paramType: "string", modes: ["fixed", "choice"], group: "model_params" },
    ],
    special_search_space_fields: { metric: "model_metric" },
    additional_params: [],
    conditional_visibility: {},
  };

  it("initializes choices as flat array of strings when switching from Fixed", () => {
    const onChange = vi.fn();
    render(
      <SearchSpace
        {...defaultProps}
        uiSchema={metricUiSchema}
        fixedModelParams={{ metric: ["auc", "binary_logloss"] }}
        modelConfig={{ params: { metric: ["auc", "binary_logloss"] } }}
        onChange={onChange}
      />,
    );
    // Switch from Fixed to Choice
    const choiceBtn = screen.getByText("Choice");
    fireEvent.click(choiceBtn);
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[0][0];
    const choices = update.space.metric?.choices ?? [];
    // Every element must be a string — no nested arrays
    for (const c of choices) {
      expect(typeof c).toBe("string");
    }
  });
});

// ── #114 Phase A: regression metric option set must include smape / wape (P-030) ──

describe("SearchSpace — P-030 regression metric chips", () => {
  const regressionUiSchema = {
    option_sets: {
      objective: { regression: ["huber", "mse"] },
      // P-030: smape / wape are first-class regression metrics in lizyml 0.11.
      // The widget must surface them through the contract; SearchSpace renders
      // them as choice chips when the user picks the metric row.
      model_metric: { regression: ["rmse", "mae", "smape", "wape"] },
    },
    step_map: {},
    search_space_catalog: [
      {
        key: "metric",
        title: "Metric",
        paramType: "string",
        modes: ["fixed", "choice"],
        group: "model_params",
      },
    ],
    special_search_space_fields: { metric: "model_metric" },
    additional_params: [],
    conditional_visibility: {},
  };

  it("renders smape and wape chips for the regression metric option set", () => {
    render(
      <SearchSpace
        {...defaultProps}
        task="regression"
        uiSchema={regressionUiSchema}
        fixedModelParams={{ metric: ["rmse"] }}
        modelConfig={{ params: { metric: ["rmse"] } }}
      />,
    );
    expect(screen.getByText("smape")).toBeDefined();
    expect(screen.getByText("wape")).toBeDefined();
  });

  it("toggles smape into the fixed metric list when the chip is clicked", () => {
    const onChange = vi.fn();
    render(
      <SearchSpace
        {...defaultProps}
        task="regression"
        uiSchema={regressionUiSchema}
        fixedModelParams={{ metric: ["rmse"] }}
        modelConfig={{ params: { metric: ["rmse"] } }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByText("smape"));
    expect(onChange).toHaveBeenCalled();
    // The most-recent invocation should now contain smape in the fixed metric.
    const calls = onChange.mock.calls;
    const last = calls[calls.length - 1][0];
    const fixedMetric = last.fixedModelParams?.metric;
    expect(Array.isArray(fixedMetric)).toBe(true);
    expect(fixedMetric).toContain("smape");
  });
});

// ── Bug 8: boolean Fixed→Choice should include both true and false ──

describe("SearchSpace — Bug 8: boolean Fixed→Choice mode switch", () => {
  const boolUiSchema = {
    option_sets: { objective: { binary: ["binary"] } },
    step_map: {},
    search_space_catalog: [
      { key: "auto_num_leaves", title: "Auto Num Leaves", paramType: "boolean", modes: ["fixed", "choice"], group: "smart_params", default: true },
    ],
    additional_params: [],
    conditional_visibility: {},
  };

  it("initializes choices with both true and false", () => {
    const onChange = vi.fn();
    render(
      <SearchSpace
        {...defaultProps}
        uiSchema={boolUiSchema}
        fixedModelParams={{ auto_num_leaves: true }}
        modelConfig={{ auto_num_leaves: true }}
        onChange={onChange}
      />,
    );
    const choiceBtn = screen.getByText("Choice");
    fireEvent.click(choiceBtn);
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[0][0];
    const choices = update.space.auto_num_leaves?.choices ?? [];
    expect(choices).toContain(true);
    expect(choices).toContain(false);
  });
});
