/**
 * Tests for inner validation options filtering based on group_column / time_column availability.
 * - group_holdout: only visible when dfInfo.cv.group_column is set
 * - time_holdout: only visible when dfInfo.cv.time_column is set
 * - holdout: always visible
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/preact";
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
  // Find the Inner Validation select element and extract its option values
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

describe("FitSubTab — Inner Validation filtering", () => {
  it("shows only 'holdout' when neither group_column nor time_column is set", () => {
    const { container } = render(<FitSubTab {...baseProps} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).not.toContain("group_holdout");
    expect(options).not.toContain("time_holdout");
  });

  it("shows 'holdout' and 'group_holdout' when group_column is set", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "group_kfold", n_splits: 5, group_column: "user_id" },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).toContain("group_holdout");
    expect(options).not.toContain("time_holdout");
  });

  it("shows 'holdout' and 'time_holdout' when time_column is set", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: { strategy: "time_series", n_splits: 5, time_column: "date" },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).not.toContain("group_holdout");
    expect(options).toContain("time_holdout");
  });

  it("shows all options when both group_column and time_column are set", () => {
    const props = {
      ...baseProps,
      dfInfo: {
        ...baseProps.dfInfo,
        cv: {
          strategy: "group_time_series",
          n_splits: 5,
          group_column: "user_id",
          time_column: "date",
        },
      },
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    expect(options).toContain("holdout");
    expect(options).toContain("group_holdout");
    expect(options).toContain("time_holdout");
  });

  it("resets inner_valid to 'holdout' if current method becomes unavailable", () => {
    // Scenario: inner_valid is set to group_holdout but group_column is removed
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
        cv: { strategy: "kfold", n_splits: 5 },  // no group_column
      },
      handleSectionChange,
    };
    const { container } = render(<FitSubTab {...props} />);
    const options = getInnerValidOptions(container);
    // group_holdout should not be in the options list
    expect(options).not.toContain("group_holdout");
    // The select value should fall back to holdout
    const selects = container.querySelectorAll("select.lzw-select");
    for (const sel of selects) {
      const opts = Array.from(sel.querySelectorAll("option")).map((o) => o.value);
      if (opts.includes("holdout")) {
        expect((sel as HTMLSelectElement).value).toBe("holdout");
      }
    }
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
