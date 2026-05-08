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
});
