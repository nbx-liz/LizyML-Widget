/**
 * Tests for ModelEditors — Smart Params, typed params, additional params.
 *
 * #114 Phase A: ModelEditors was at 0.69% coverage; smoke + behaviour tests
 * pull it well over the 75% target since the file is mostly render code.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { ModelSection } from "../components/ModelEditors";
import type { TypedParamMeta } from "../components/ModelEditors";

const parameterHints: TypedParamMeta[] = [
  { key: "objective", label: "Objective", kind: "objective" },
  { key: "metric", label: "Metric", kind: "model_metric" },
  { key: "n_estimators", label: "N Estimators", kind: "integer", step: 100 },
  { key: "learning_rate", label: "Learning Rate", kind: "number", step: 0.001 },
  { key: "first_metric_only", label: "First Metric Only", kind: "boolean" },
];

const optionSets = {
  objective: { binary: ["binary", "cross_entropy"] },
  model_metric: { binary: ["auc", "binary_logloss"] },
};

const searchSpaceCatalog = [
  { key: "auto_num_leaves", title: "Auto Num Leaves", paramType: "boolean", group: "smart_params", default: true },
  { key: "num_leaves_ratio", title: "Num Leaves Ratio", paramType: "number", group: "smart_params", default: 1.0 },
  { key: "num_leaves", title: "Num Leaves", paramType: "integer", group: "smart_params", default: 256 },
  { key: "min_data_in_leaf_ratio", title: "Min Data In Leaf Ratio", paramType: "number", group: "smart_params", default: 0.01 },
  { key: "min_data_in_bin_ratio", title: "Min Data In Bin Ratio", paramType: "number", group: "smart_params", default: 0.01 },
  { key: "feature_weights", title: "Feature Weights", paramType: "object", group: "smart_params", default: null },
  { key: "balanced", title: "Balanced", paramType: "boolean", group: "smart_params", default: true },
];

const baseSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    auto_num_leaves: { type: "boolean" },
    num_leaves_ratio: { type: "number" },
    feature_weights: { type: "object" },
    balanced: { type: "boolean" },
    params: { type: "object" },
    min_data_in_leaf_ratio: { type: "number" },
    min_data_in_bin_ratio: { type: "number" },
  },
};

const baseValue = {
  name: "lgbm",
  auto_num_leaves: true,
  num_leaves_ratio: 1.0,
  min_data_in_leaf_ratio: 0.01,
  min_data_in_bin_ratio: 0.01,
  balanced: true,
  feature_weights: null,
  params: {
    objective: "binary",
    metric: ["auc"],
    n_estimators: 1500,
    learning_rate: 0.001,
    first_metric_only: false,
  },
};

const baseProps = {
  schema: baseSchema,
  rootSchema: baseSchema,
  value: baseValue,
  onChange: vi.fn(),
  task: "binary",
  parameterHints,
  optionSets,
  stepMap: {},
  columns: [{ name: "x1" }, { name: "x2" }, { name: "x3" }],
  additionalParams: ["min_child_samples", "subsample"],
  searchSpaceCatalog,
  additionalParamsHiddenKeys: ["verbose", "num_threads"],
};

describe("ModelEditors.ModelSection — smoke", () => {
  it("renders Model Type tag with the model name", () => {
    render(<ModelSection {...baseProps} />);
    expect(screen.getByText("lgbm")).toBeDefined();
  });

  it("renders the Smart Params section title", () => {
    render(<ModelSection {...baseProps} />);
    expect(screen.getByText("Smart Params")).toBeDefined();
  });

  it("renders the Model Params section title", () => {
    render(<ModelSection {...baseProps} />);
    expect(screen.getByText("Model Params")).toBeDefined();
  });

  it("renders the Additional Params section title", () => {
    render(<ModelSection {...baseProps} />);
    expect(screen.getByText("Additional Params")).toBeDefined();
  });

  it("renders typed parameter rows from parameterHints", () => {
    render(<ModelSection {...baseProps} />);
    expect(screen.getByText("Objective")).toBeDefined();
    expect(screen.getByText("Metric")).toBeDefined();
    expect(screen.getByText("Learning Rate")).toBeDefined();
  });

  it("renders the warning tag when model.name is missing", () => {
    render(<ModelSection {...baseProps} value={{ ...baseValue, name: "" }} />);
    expect(screen.getByText(/model.name missing/)).toBeDefined();
  });
});

describe("ModelEditors.ModelSection — Smart Params toggle", () => {
  it("hides the manual Num Leaves stepper when auto is on", () => {
    const { container } = render(<ModelSection {...baseProps} />);
    expect(container.textContent).toContain("Num Leaves Ratio");
    // The bespoke "Num Leaves" stepper appears only when auto is off.
    // When auto is on, we still see the smart-params keys in additional
    // exclusion lists, but the manual stepper row is not rendered.
    const numLeavesRow = Array.from(container.querySelectorAll(".lzw-form-row")).find(
      (row) => row.textContent?.startsWith("Num Leaves") && !row.textContent?.includes("Ratio"),
    );
    expect(numLeavesRow).toBeUndefined();
  });

  it("shows the manual Num Leaves stepper when auto is off", () => {
    const onChange = vi.fn();
    render(
      <ModelSection
        {...baseProps}
        value={{
          ...baseValue,
          auto_num_leaves: false,
          params: { ...baseValue.params, num_leaves: 128 },
        }}
        onChange={onChange}
      />,
    );
    expect(screen.getByText("Num Leaves")).toBeDefined();
  });

  it("flips auto_num_leaves and sets/removes params.num_leaves on toggle", () => {
    const onChange = vi.fn();
    const { container } = render(
      <ModelSection {...baseProps} onChange={onChange} />,
    );
    // First toggle: find by parent row with "Auto Num Leaves" label.
    const autoRow = Array.from(container.querySelectorAll(".lzw-form-row")).find(
      (r) => r.textContent?.startsWith("Auto Num Leaves"),
    ) as HTMLElement;
    const checkbox = autoRow.querySelector("input[type='checkbox']") as HTMLInputElement;
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)![0];
    expect(last.auto_num_leaves).toBe(false);
    // num_leaves should now be present in params (default flowed from catalog).
    expect(last.params.num_leaves).toBeDefined();
  });
});

describe("ModelEditors.ModelSection — Feature Weights toggle", () => {
  it("renders the toggle in disabled state when feature_weights is null", () => {
    const { container } = render(<ModelSection {...baseProps} />);
    const fwRow = Array.from(container.querySelectorAll(".lzw-form-row")).find((r) =>
      r.textContent?.startsWith("Feature Weights"),
    ) as HTMLElement;
    const checkbox = fwRow.querySelector("input[type='checkbox']") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
  });

  it("turns the toggle on when feature_weights is an object", () => {
    const { container } = render(
      <ModelSection
        {...baseProps}
        value={{ ...baseValue, feature_weights: { x1: 1.5 } }}
      />,
    );
    const fwRow = Array.from(container.querySelectorAll(".lzw-form-row")).find((r) =>
      r.textContent?.startsWith("Feature Weights"),
    ) as HTMLElement;
    const checkbox = fwRow.querySelector("input[type='checkbox']") as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
  });
});

describe("ModelEditors.ModelSection — Additional Params hidden keys", () => {
  it("does not render verbose / num_threads in Additional Params even when present in params", () => {
    const { container } = render(
      <ModelSection
        {...baseProps}
        value={{
          ...baseValue,
          params: {
            ...baseValue.params,
            verbose: -1,
            num_threads: 4,
          },
        }}
      />,
    );
    // These should NOT appear as Additional Params rows since they're in
    // additionalParamsHiddenKeys.
    const additionalSection = Array.from(container.querySelectorAll(".lzw-dynform__section-title"))
      .find((el) => el.textContent === "Additional Params")?.parentElement;
    expect(additionalSection?.textContent).not.toMatch(/verbose/);
    expect(additionalSection?.textContent).not.toMatch(/num_threads/);
  });
});

describe("ModelEditors.ModelSection — backend-driven smart_params", () => {
  it("falls back to the legacy smart_params set when searchSpaceCatalog is empty", () => {
    // Even without a catalog, we must still render the Smart Params section
    // (test fixtures sometimes omit the catalog).
    const { container } = render(
      <ModelSection {...baseProps} searchSpaceCatalog={[]} />,
    );
    expect(container.textContent).toContain("Smart Params");
    expect(container.textContent).toContain("Auto Num Leaves");
  });
});
