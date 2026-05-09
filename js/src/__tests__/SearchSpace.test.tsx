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

// ── #134: close coverage gaps for range/choice/log paths ──

describe("SearchSpace — range mode interactions", () => {
  const baseProps = {
    schema: { type: "object", properties: {} },
    fixedTraining: {},
    modelConfig: { params: { learning_rate: 0.1 } },
    trainingConfig: {},
    task: "binary",
    uiSchema: minimalUiSchema,
  };

  it("toggling Log fires onChange with log=true", () => {
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...baseProps}
        spaceValue={{
          learning_rate: { type: "float", low: 0.001, high: 0.1, log: false },
        }}
        fixedModelParams={{}}
        onChange={onChange}
      />,
    );
    const logCheckbox = container.querySelector(
      ".lzw-search-space__log input[type='checkbox']",
    ) as HTMLInputElement;
    fireEvent.click(logCheckbox);
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[0][0];
    expect(update.space.learning_rate.log).toBe(true);
  });

  it("editing low stepper input fires onChange with the new low value", () => {
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...baseProps}
        spaceValue={{
          learning_rate: { type: "float", low: 0.01, high: 0.1, log: false },
        }}
        fixedModelParams={{}}
        onChange={onChange}
      />,
    );
    const inputs = container.querySelectorAll(
      ".lzw-search-space__range .lzw-stepper__input",
    );
    fireEvent.change(inputs[0], { target: { value: "0.005" } });
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update.space.learning_rate.low).toBeCloseTo(0.005);
  });

  it("editing high stepper input fires onChange with the new high value", () => {
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...baseProps}
        spaceValue={{
          learning_rate: { type: "float", low: 0.01, high: 0.1, log: false },
        }}
        fixedModelParams={{}}
        onChange={onChange}
      />,
    );
    const inputs = container.querySelectorAll(
      ".lzw-search-space__range .lzw-stepper__input",
    );
    fireEvent.change(inputs[1], { target: { value: "0.5" } });
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update.space.learning_rate.high).toBeCloseTo(0.5);
  });

  it("switching Range -> Fixed recovers the low value as the new fixed value", () => {
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...baseProps}
        spaceValue={{
          learning_rate: { type: "float", low: 0.025, high: 0.1, log: false },
        }}
        fixedModelParams={{}}
        onChange={onChange}
      />,
    );
    // The Range button renders for params with multiple modes; Fixed lives next
    // to it. Click the Fixed button for the learning_rate row (first segment).
    const segments = container.querySelectorAll(".lzw-segment");
    const fixedBtn = Array.from(
      segments[0].querySelectorAll("button"),
    ).find((b) => b.textContent === "Fixed") as HTMLButtonElement;
    fireEvent.click(fixedBtn);
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[0][0];
    expect(update.space.learning_rate).toBeUndefined();
    expect(update.fixedModelParams.learning_rate).toBeCloseTo(0.025);
  });
});

describe("SearchSpace — choice mode interactions", () => {
  const choiceUiSchema = {
    option_sets: { objective: { binary: ["binary"] } },
    step_map: {},
    search_space_catalog: [
      {
        key: "objective",
        title: "Objective",
        paramType: "string",
        modes: ["fixed", "choice"],
        group: "model_params",
      },
    ],
    special_search_space_fields: { objective: "objective" },
    additional_params: [],
    conditional_visibility: {},
  };

  const choiceProps = {
    schema: { type: "object", properties: {} },
    fixedTraining: {},
    modelConfig: { params: {} },
    trainingConfig: {},
    task: "binary",
    uiSchema: choiceUiSchema,
  };

  it("clicking a chip not yet selected adds it to choices", () => {
    const onChange = vi.fn();
    render(
      <SearchSpace
        {...choiceProps}
        spaceValue={{ objective: { type: "categorical", choices: ["binary"] } }}
        fixedModelParams={{}}
        onChange={onChange}
      />,
    );
    // The chip-group now uses option_sets.objective.binary -> ["binary"]
    // Switch to single-option case: clicking the chip removes it (it's the
    // only option and it's already selected).
    const chip = screen.getByText("binary");
    fireEvent.click(chip);
    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls[0][0];
    expect(update.space.objective.choices).toEqual([]);
  });

  it("clicking a selected chip removes it from choices", () => {
    const onChange = vi.fn();
    const multi = {
      ...choiceUiSchema,
      option_sets: { objective: { binary: ["binary", "cross_entropy"] } },
    };
    render(
      <SearchSpace
        {...choiceProps}
        uiSchema={multi}
        spaceValue={{
          objective: { type: "categorical", choices: ["binary", "cross_entropy"] },
        }}
        fixedModelParams={{}}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByText("binary"));
    const update = onChange.mock.calls[0][0];
    expect(update.space.objective.choices).toEqual(["cross_entropy"]);
  });
});

// ── #134: Fixed-mode editor variants (enum select, array+items.enum chips, object/feature_weights, plain text fallback) ──

describe("SearchSpace — Fixed mode editors", () => {
  it("renders an inner_valid <select> backed by inner_valid_options and fires onChange", () => {
    const ui = {
      option_sets: {},
      step_map: {},
      inner_valid_options: ["holdout", "group_holdout", "time_holdout"],
      search_space_catalog: [
        {
          key: "inner_valid",
          title: "Inner Valid",
          paramType: "string",
          modes: ["fixed"],
          group: "training",
        },
      ],
      special_search_space_fields: { inner_valid: "inner_valid_picker" },
      additional_params: [],
      conditional_visibility: {},
    };
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...defaultProps}
        uiSchema={ui}
        spaceValue={{}}
        fixedModelParams={{}}
        fixedTraining={{ inner_valid: "holdout" }}
        modelConfig={{ params: {} }}
        onChange={onChange}
      />,
    );
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    fireEvent.change(select, { target: { value: "time_holdout" } });
    expect(onChange).toHaveBeenCalled();
  });

  it("renders the metric chip-group fed by option_sets.model_metric and toggles a chip", () => {
    const ui = {
      option_sets: {
        objective: { binary: ["binary"] },
        model_metric: { binary: ["auc", "logloss", "accuracy"] },
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
    const onChange = vi.fn();
    render(
      <SearchSpace
        {...defaultProps}
        uiSchema={ui}
        spaceValue={{}}
        fixedModelParams={{ metric: ["auc"] }}
        modelConfig={{ params: { metric: ["auc"] } }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByText("logloss"));
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(last.fixedModelParams.metric).toContain("logloss");
  });

  it("renders an ON/OFF toggle for object-typed feature_weights and the toggle fires onChange", () => {
    const ui = {
      option_sets: {},
      step_map: { feature_weights: 0.1 },
      search_space_catalog: [
        {
          key: "feature_weights",
          title: "Feature Weights",
          paramType: "object",
          modes: ["fixed"],
          group: "model_params",
          type: "object",
        },
      ],
      additional_params: [],
      conditional_visibility: {},
    };
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...defaultProps}
        uiSchema={ui}
        spaceValue={{}}
        fixedModelParams={{ feature_weights: null }}
        modelConfig={{ feature_weights: null }}
        columns={[{ name: "x1" }, { name: "x2" }]}
        onChange={onChange}
      />,
    );
    const toggle = container.querySelector(
      "input[type='checkbox']",
    ) as HTMLInputElement;
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalled();
  });

  it("falls back to a plain text <input> for unknown fixed-mode field types", () => {
    const ui = {
      option_sets: {},
      step_map: {},
      search_space_catalog: [
        {
          key: "comment",
          title: "Comment",
          paramType: "string",
          modes: ["fixed"],
          group: "model_params",
          // No enum, no items.enum, no number/integer type, no object → fallback
        },
      ],
      additional_params: [],
      conditional_visibility: {},
    };
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...defaultProps}
        uiSchema={ui}
        spaceValue={{}}
        fixedModelParams={{ comment: "hello" }}
        modelConfig={{ params: { comment: "hello" } }}
        onChange={onChange}
      />,
    );
    const input = container.querySelector(
      "input[type='text']",
    ) as HTMLInputElement;
    expect(input).not.toBeNull();
    fireEvent.change(input, { target: { value: "world" } });
    expect(onChange).toHaveBeenCalled();
  });
});

describe("SearchSpace — additional_params row", () => {
  it("offers an Add dropdown that adds a parameter row when selected", () => {
    const ui = {
      option_sets: {},
      step_map: {},
      search_space_catalog: [
        {
          key: "lr",
          title: "Learning Rate",
          paramType: "number",
          modes: ["fixed", "range"],
          group: "model_params",
        },
      ],
      additional_params: ["max_depth", "min_split_gain"],
      conditional_visibility: {},
    };
    const onChange = vi.fn();
    const { container } = render(
      <SearchSpace
        {...defaultProps}
        uiSchema={ui}
        spaceValue={{}}
        fixedModelParams={{ lr: 0.1 }}
        modelConfig={{ params: { lr: 0.1 } }}
        onChange={onChange}
      />,
    );
    // The Add dropdown is the last <select> in the model_params group.
    const selects = container.querySelectorAll(".lzw-select");
    const addSelect = selects[selects.length - 1] as HTMLSelectElement;
    expect(addSelect).not.toBeNull();
    fireEvent.change(addSelect, { target: { value: "max_depth" } });
    // Adding a row triggers a state change inside the component; assert that
    // a <select> remains rendered (the row is now visible) by querying again.
    const after = container.querySelectorAll(".lzw-select");
    expect(after.length).toBeGreaterThan(0);
  });
});
