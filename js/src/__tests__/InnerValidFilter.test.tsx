/**
 * Tests for inner validation options filtering based on CV strategy.
 * - group_holdout: only visible when CV strategy uses group column
 * - time_holdout: only visible when CV strategy uses time column
 * - holdout: always visible
 */
import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/preact";
import { FitSubTab } from "../tabs/FitSubTab";

const baseUiSchema = {
  sections: [
    { key: "training", title: "Training" },
  ],
  option_sets: {},
  parameter_hints: [],
  step_map: {},
  conditional_visibility: {},
  defaults: {},
  inner_valid_options: ["holdout", "group_holdout", "time_holdout"],
};

const baseConfigSchema = {
  type: "object",
  properties: {
    training: {
      type: "object",
      properties: {
        seed: { type: "integer" },
        early_stopping: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            rounds: { type: "integer" },
            validation_ratio: { type: "number" },
            inner_valid: { type: "object" },
          },
        },
      },
    },
  },
};

const baseConfig = {
  training: {
    seed: 42,
    early_stopping: {
      enabled: true,
      rounds: 150,
      validation_ratio: 0.1,
      inner_valid: { method: "holdout" },
    },
  },
};

const baseProps = {
  localConfig: baseConfig,
  configSchema: baseConfigSchema,
  uiSchema: baseUiSchema,
  task: "binary",
  dfInfo: { target: "y", task: "binary", cv: { strategy: "kfold", n_splits: 5 } },
  handleChange: vi.fn(),
  handleSectionChange: vi.fn(),
  sendAction: vi.fn(),
  rawYaml: null,
  setRawYaml: vi.fn(),
};

function getInnerValidOptions(container: HTMLElement): string[] {
  const selects = container.querySelectorAll("select.lzw-select");
  for (const sel of selects) {
    const options = Array.from(sel.querySelectorAll("option"));
    const values = options.map((o) => o.value);
    if (values.includes("holdout")) {
      return values;
    }
  }
  return [];
}

describe("FitSubTab — Inner Validation filtering by CV strategy", () => {
  it("shows only 'holdout' for non-group/non-time strategy (kfold)", () => {
    const { container } = render(<FitSubTab {...baseProps} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).not.toContain("group_holdout");
    expect(options).not.toContain("time_holdout");
  });

  it("shows 'group_holdout' for group_kfold strategy", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "group_kfold", n_splits: 5 },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).toContain("group_holdout");
    expect(options).not.toContain("time_holdout");
  });

  it("shows 'time_holdout' for time_series strategy", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "time_series", n_splits: 5 },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).not.toContain("group_holdout");
    expect(options).toContain("time_holdout");
  });

  it("shows both group_holdout and time_holdout for group_time_series", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "group_time_series", n_splits: 5 },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).toContain("group_holdout");
    expect(options).toContain("time_holdout");
  });

  it("shows 'group_holdout' for stratified_group_kfold", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "stratified_group_kfold", n_splits: 5 },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("group_holdout");
    expect(options).not.toContain("time_holdout");
  });

  it("shows 'time_holdout' for purged_time_series", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "purged_time_series", n_splits: 5 },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).not.toContain("group_holdout");
    expect(options).toContain("time_holdout");
  });

  it("resets inner_valid to 'holdout' when strategy changes to non-group", () => {
    const handleSectionChange = vi.fn();
    const props = {
      ...baseProps,
      localConfig: {
        ...baseConfig,
        training: {
          ...baseConfig.training,
          early_stopping: {
            ...baseConfig.training.early_stopping,
            inner_valid: { method: "group_holdout" },
          },
        },
      },
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "kfold", n_splits: 5 },  // non-group strategy
      },
      handleSectionChange,
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).not.toContain("group_holdout");
    // useEffect should auto-reset config to holdout
    expect(handleSectionChange).toHaveBeenCalledWith(
      "training",
      expect.objectContaining({
        early_stopping: expect.objectContaining({
          inner_valid: { method: "holdout" },
        }),
      }),
    );
  });
});
