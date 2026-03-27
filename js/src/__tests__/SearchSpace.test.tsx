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
