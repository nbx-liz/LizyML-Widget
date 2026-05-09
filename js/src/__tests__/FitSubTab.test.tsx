/**
 * Tests for FitSubTab — Model / Evaluation / Calibration / Training sections.
 *
 * #114 Phase A: was at 43% — smoke + behaviour tests pull it well above target.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { FitSubTab } from "../tabs/FitSubTab";

beforeEach(() => {
  if (!window.matchMedia) {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((q: string) => ({
        matches: false,
        media: q,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  }
});

const baseConfigSchema = {
  type: "object",
  properties: {
    model: {
      type: "object",
      properties: {
        name: { type: "string" },
        params: { type: "object" },
      },
    },
    training: {
      type: "object",
      properties: {
        seed: { type: "integer" },
      },
    },
    calibration: { type: "object" },
    evaluation: { type: "object" },
  },
};

const baseUiSchema = {
  sections: [
    { key: "model", title: "Model" },
    { key: "training", title: "Training" },
    { key: "calibration", title: "Calibration" },
    { key: "evaluation", title: "Evaluation" },
  ],
  option_sets: {
    objective: { binary: ["binary"] },
    model_metric: { binary: ["auc", "binary_logloss"] },
    metric: { binary: ["auc", "binary_logloss", "binary_error"] },
  },
  parameter_hints: [
    { key: "objective", label: "Objective", kind: "objective" },
    { key: "metric", label: "Metric", kind: "model_metric" },
    { key: "n_estimators", label: "N Estimators", kind: "integer", step: 100 },
  ],
  step_map: {},
  conditional_visibility: { calibration: { task: ["binary"] } },
  defaults: { calibration: { method: "platt", params: {} } },
  inner_valid_options: ["holdout", "group_holdout", "time_holdout"],
  search_space_catalog: [
    { key: "auto_num_leaves", title: "Auto Num Leaves", paramType: "boolean", group: "smart_params", default: true },
    { key: "balanced", title: "Balanced", paramType: "boolean", group: "smart_params", default: true },
  ],
  additional_params: ["min_child_samples"],
};

const baseProps = {
  localConfig: {
    model: { name: "lgbm", params: { objective: "binary", metric: ["auc"] }, balanced: true },
    training: { seed: 1 },
    evaluation: { metrics: ["auc"], params: {} },
  },
  configSchema: baseConfigSchema,
  uiSchema: baseUiSchema,
  capabilities: {
    cv_strategy_fields: {
      kfold: ["n_splits", "shuffle", "random_state"],
      group_kfold: ["n_splits", "group_col"],
    },
    additional_params_hidden_keys: ["verbose", "num_threads"],
    cv_default_strategy: { binary: "kfold", regression: "kfold" },
    cv_strategies: ["kfold", "group_kfold"],
  },
  task: "binary",
  dfInfo: {
    target: "y",
    columns: [{ name: "x1" }, { name: "x2" }],
    cv: { strategy: "kfold", n_splits: 5 },
  },
  handleChange: vi.fn(),
  handleSectionChange: vi.fn(),
  sendAction: vi.fn(),
  rawYaml: null,
  setRawYaml: vi.fn(),
};

describe("FitSubTab — section structure", () => {
  it("renders the four contract-driven sections", () => {
    render(<FitSubTab {...baseProps} />);
    expect(screen.getByText("Model")).toBeDefined();
    expect(screen.getAllByText("Training").length).toBeGreaterThan(0);
    expect(screen.getByText("Calibration")).toBeDefined();
    expect(screen.getAllByText("Evaluation").length).toBeGreaterThan(0);
  });

  it("renders the Calibration accordion only for tasks listed in conditional_visibility", () => {
    const { rerender } = render(<FitSubTab {...baseProps} task="binary" />);
    expect(screen.getByText("Calibration")).toBeDefined();
    rerender(<FitSubTab {...baseProps} task="regression" />);
    expect(screen.queryByText("Calibration")).toBeNull();
  });
});

describe("FitSubTab — Calibration toggle", () => {
  it("calls handleChange to set calibration to its default when toggled on", () => {
    const handleChange = vi.fn();
    render(
      <FitSubTab
        {...baseProps}
        handleChange={handleChange}
        localConfig={{ ...baseProps.localConfig, calibration: null }}
      />,
    );
    const checkbox = screen.getByLabelText("Enable calibration") as HTMLInputElement;
    fireEvent.click(checkbox);
    expect(handleChange).toHaveBeenCalled();
    const updated = handleChange.mock.calls.at(-1)![0];
    expect(updated.calibration).not.toBeNull();
    expect(updated.calibration.method).toBe("platt");
  });

  it("calls handleChange to clear calibration when toggled off", () => {
    const handleChange = vi.fn();
    render(
      <FitSubTab
        {...baseProps}
        handleChange={handleChange}
        localConfig={{
          ...baseProps.localConfig,
          calibration: { method: "isotonic", params: {} },
        }}
      />,
    );
    const checkbox = screen.getByLabelText("Enable calibration") as HTMLInputElement;
    fireEvent.click(checkbox);
    expect(handleChange).toHaveBeenCalled();
    const updated = handleChange.mock.calls.at(-1)![0];
    expect(updated.calibration).toBeNull();
  });
});

describe("FitSubTab — Evaluation metrics", () => {
  it("renders metric chips from the option set for the current task", () => {
    const { container } = render(<FitSubTab {...baseProps} />);
    const evalAccordion = Array.from(container.querySelectorAll(".lzw-chip-group"));
    const allChips = evalAccordion.flatMap((g) => Array.from(g.children).map((c) => c.textContent));
    expect(allChips).toEqual(expect.arrayContaining(["auc", "binary_logloss", "binary_error"]));
  });

  it("toggling a metric chip emits a section change with the new metrics list", () => {
    const handleSectionChange = vi.fn();
    const { container } = render(
      <FitSubTab {...baseProps} handleSectionChange={handleSectionChange} />,
    );
    // The evaluation chip-group contains binary_error (model_metric does
    // not). Identify it by content to avoid matching model.params.metric
    // chip groups.
    const allChipGroups = Array.from(
      container.querySelectorAll(".lzw-chip-group"),
    ) as HTMLElement[];
    const evalGroup = allChipGroups.find((g) =>
      Array.from(g.children).some((c) => c.textContent === "binary_error"),
    )!;
    const evalChips = Array.from(evalGroup.children) as HTMLElement[];
    const loglossChip = evalChips.find((b) => b.textContent === "binary_logloss")!;
    fireEvent.click(loglossChip);
    expect(handleSectionChange).toHaveBeenCalledWith(
      "evaluation",
      expect.objectContaining({ metrics: expect.arrayContaining(["auc", "binary_logloss"]) }),
    );

    handleSectionChange.mockClear();
    const aucChip = evalChips.find((b) => b.textContent === "auc")!;
    fireEvent.click(aucChip);
    const last = handleSectionChange.mock.calls.at(-1)!;
    expect(last[0]).toBe("evaluation");
    expect(last[1].metrics).not.toContain("auc");
  });

  it("renders the precision_at_k k stepper only when the metric is selected", () => {
    const propsWithPrecision = {
      ...baseProps,
      uiSchema: {
        ...baseUiSchema,
        option_sets: {
          ...baseUiSchema.option_sets,
          metric: { binary: ["auc", "precision_at_k"] },
        },
      },
      localConfig: {
        ...baseProps.localConfig,
        evaluation: { metrics: ["auc"], params: {} },
      },
    };
    const { rerender, queryByLabelText } = render(<FitSubTab {...propsWithPrecision} />);
    expect(queryByLabelText("precision_at_k: k")).toBeNull();

    rerender(
      <FitSubTab
        {...propsWithPrecision}
        localConfig={{
          ...propsWithPrecision.localConfig,
          evaluation: { metrics: ["auc", "precision_at_k"], params: {} },
        }}
      />,
    );
    // Stepper renders a numeric input — its label is "precision_at_k: k".
    expect(screen.getByText("precision_at_k: k")).toBeDefined();
  });
});

describe("FitSubTab — Calibration enabled state", () => {
  it("changing the method emits a section change with the new method", () => {
    const handleSectionChange = vi.fn();
    render(
      <FitSubTab
        {...baseProps}
        handleSectionChange={handleSectionChange}
        localConfig={{
          ...baseProps.localConfig,
          calibration: { method: "platt", params: {} },
        }}
        uiSchema={{
          ...baseUiSchema,
          calibration_methods: ["platt", "isotonic", "beta"],
        }}
      />,
    );
    const select = screen
      .getAllByRole("combobox")
      .find((s) => (s as HTMLSelectElement).value === "platt")!;
    fireEvent.change(select, { target: { value: "isotonic" } });
    expect(handleSectionChange).toHaveBeenCalledWith(
      "calibration",
      expect.objectContaining({ method: "isotonic" }),
    );
  });

  it("removing a calibration param strips it from params", () => {
    const handleSectionChange = vi.fn();
    render(
      <FitSubTab
        {...baseProps}
        handleSectionChange={handleSectionChange}
        localConfig={{
          ...baseProps.localConfig,
          calibration: { method: "platt", params: { c: 1.0 } },
        }}
      />,
    );
    const remove = screen.getByLabelText("Remove c");
    fireEvent.click(remove);
    const last = handleSectionChange.mock.calls.at(-1)!;
    expect(last[0]).toBe("calibration");
    expect(last[1].params).not.toHaveProperty("c");
  });

  it("adding an available param via + Add select writes it to params", () => {
    const handleSectionChange = vi.fn();
    render(
      <FitSubTab
        {...baseProps}
        handleSectionChange={handleSectionChange}
        localConfig={{
          ...baseProps.localConfig,
          calibration: { method: "platt", params: {} },
        }}
        uiSchema={{
          ...baseUiSchema,
          calibration_methods: ["platt"],
          calibration_params: { platt: ["c"] },
        }}
      />,
    );
    const addSelect = screen
      .getAllByRole("combobox")
      .find((s) => (s as HTMLSelectElement).value === "")!;
    fireEvent.change(addSelect, { target: { value: "c" } });
    const last = handleSectionChange.mock.calls.at(-1)!;
    expect(last[0]).toBe("calibration");
    expect(last[1].params).toHaveProperty("c");
  });
});

describe("FitSubTab — Training section", () => {
  it("toggling early stopping emits a section change with enabled flag flipped", () => {
    const handleSectionChange = vi.fn();
    render(
      <FitSubTab
        {...baseProps}
        handleSectionChange={handleSectionChange}
        localConfig={{
          ...baseProps.localConfig,
          training: {
            seed: 1,
            early_stopping: { enabled: true, rounds: 150, validation_ratio: 0.1 },
          },
        }}
      />,
    );
    const toggle = screen.getByLabelText("Enable early stopping") as HTMLInputElement;
    fireEvent.click(toggle);
    const last = handleSectionChange.mock.calls.at(-1)!;
    expect(last[0]).toBe("training");
    expect(last[1].early_stopping.enabled).toBe(false);
  });

  it("hides rounds / validation_ratio / inner_valid when early stopping is off", () => {
    const { queryByText } = render(
      <FitSubTab
        {...baseProps}
        localConfig={{
          ...baseProps.localConfig,
          training: { seed: 1, early_stopping: { enabled: false } },
        }}
      />,
    );
    expect(queryByText("Rounds")).toBeNull();
    expect(queryByText("Validation Ratio")).toBeNull();
    expect(queryByText("Inner Validation")).toBeNull();
  });

  it("changing inner_valid select sends a section change with the new method object", () => {
    const handleSectionChange = vi.fn();
    // Use a CV strategy that exposes group fields so group_holdout is
    // allowed by ``filterInnerValidOptions``.
    render(
      <FitSubTab
        {...baseProps}
        handleSectionChange={handleSectionChange}
        dfInfo={{
          ...baseProps.dfInfo,
          cv: { strategy: "group_kfold", n_splits: 5, group_col: "g" },
        }}
        localConfig={{
          ...baseProps.localConfig,
          training: {
            seed: 1,
            early_stopping: {
              enabled: true,
              rounds: 150,
              validation_ratio: 0.1,
              inner_valid: { method: "holdout" },
            },
          },
        }}
      />,
    );
    // The inner_valid select is the one whose options include "group_holdout".
    const select = (screen.getAllByRole("combobox") as HTMLSelectElement[]).find((s) =>
      Array.from(s.options).some((o) => o.value === "group_holdout"),
    )!;
    fireEvent.change(select, { target: { value: "group_holdout" } });
    const last = handleSectionChange.mock.calls.at(-1)!;
    expect(last[0]).toBe("training");
    expect(last[1].early_stopping.inner_valid).toEqual({ method: "group_holdout" });
  });
});
