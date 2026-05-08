/**
 * Tests for DataTab — Target/Task selection, Column settings, CV settings, Feature summary.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/preact";
import { DataTab } from "../tabs/DataTab";

const baseDfInfo = {
  shape: [100, 5] as [number, number],
  target: "y",
  task: "binary",
  auto_task: "binary",
  columns: [
    { name: "x1", unique_count: 10, excluded: false, col_type: "numeric" },
    { name: "x2", unique_count: 5, excluded: false, col_type: "categorical" },
    { name: "x3", unique_count: 2, excluded: true, col_type: "numeric", exclude_reason: "id" },
  ],
  cv: { strategy: "kfold", n_splits: 5 },
  feature_summary: {
    total: 3,
    numeric: 1,
    categorical: 1,
    excluded: 1,
    excluded_id: 1,
    excluded_const: 0,
    excluded_manual: 0,
  },
};

/** Mirrors the live backend contract for cv strategies — see adapter_contract.build_capabilities. */
const baseBackendContract = {
  capabilities: {
    cv_strategies: [
      "kfold",
      "stratified_kfold",
      "group_kfold",
      "stratified_group_kfold",
      "time_series",
      "purged_time_series",
      "group_time_series",
      "blocked_group_kfold",
    ],
    cv_strategy_labels: {
      kfold: "KFold",
      stratified_kfold: "StratifiedKFold",
      group_kfold: "GroupKFold",
      stratified_group_kfold: "StratifiedGroup",
      time_series: "TimeSeriesSplit",
      purged_time_series: "PurgedTimeSeriesSplit",
      group_time_series: "GroupTimeSeriesSplit",
      blocked_group_kfold: "BlockedGroup",
    },
    cv_strategy_fields: {
      kfold: ["n_splits", "shuffle", "random_state"],
      stratified_kfold: ["n_splits", "shuffle", "random_state"],
      group_kfold: ["n_splits", "group_col"],
      stratified_group_kfold: ["n_splits", "group_col"],
      time_series: ["n_splits", "time_col", "gap", "max_train_size", "max_test_size"],
      purged_time_series: ["n_splits", "time_col", "purge_gap", "embargo"],
      group_time_series: ["n_splits", "group_col", "time_col", "gap"],
      blocked_group_kfold: ["blocks_col", "groups_col"],
    },
  },
};

const defaultProps = {
  dfInfo: baseDfInfo,
  allColumns: ["x1", "x2", "x3", "y"],
  columnStats: null,
  splitPreview: null,
  sendAction: vi.fn(),
  backendContract: baseBackendContract,
};

describe("DataTab — Target dropdown", () => {
  it("renders all columns as options", () => {
    const { container } = render(<DataTab {...defaultProps} />);
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    expect(select).not.toBeNull();
    // "-- Select --" + 4 columns
    const options = select.querySelectorAll("option");
    expect(options.length).toBe(5);
  });

  it("shows current target as selected value", () => {
    const { container } = render(<DataTab {...defaultProps} />);
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    expect(select.value).toBe("y");
  });

  it("fires sendAction('set_target') on selection change", () => {
    const sendAction = vi.fn();
    const { container } = render(<DataTab {...defaultProps} sendAction={sendAction} />);
    const select = container.querySelector(".lzw-select") as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "x1" } });
    expect(sendAction).toHaveBeenCalledWith("set_target", { target: "x1" });
  });
});

describe("DataTab — Task toggle", () => {
  it("renders binary/multiclass/regression buttons", () => {
    render(<DataTab {...defaultProps} />);
    // Active task with auto_task match shows "⚡binary"
    expect(screen.getByText(/binary/)).toBeDefined();
    expect(screen.getByText("multiclass")).toBeDefined();
    expect(screen.getByText("regression")).toBeDefined();
  });

  it("marks current task as active with aria-pressed", () => {
    const { container } = render(<DataTab {...defaultProps} />);
    const activeBtn = container.querySelector(
      '.lzw-segment__btn--active[aria-pressed="true"]',
    );
    // The binary button should contain the lightning emoji prefix for auto_task match
    expect(activeBtn).not.toBeNull();
    expect(activeBtn!.textContent).toContain("binary");
  });

  it("fires sendAction('set_task') on click", () => {
    const sendAction = vi.fn();
    render(<DataTab {...defaultProps} sendAction={sendAction} />);
    fireEvent.click(screen.getByText("regression"));
    expect(sendAction).toHaveBeenCalledWith("set_task", { task: "regression" });
  });

  it("disables task buttons when no target is set", () => {
    const { container } = render(
      <DataTab {...defaultProps} dfInfo={{ ...baseDfInfo, target: null }} />,
    );
    const taskButtons = container.querySelectorAll(".lzw-segment__btn");
    // Task segment buttons (binary/multiclass/regression) should be disabled
    const taskBtns = Array.from(taskButtons).filter((b) =>
      ["binary", "multiclass", "regression"].includes(b.textContent ?? ""),
    );
    for (const btn of taskBtns) {
      expect((btn as HTMLButtonElement).disabled).toBe(true);
    }
  });
});

describe("DataTab — CV strategy segment", () => {
  it("renders all CV strategy options", () => {
    render(<DataTab {...defaultProps} />);
    expect(screen.getByText("KFold")).toBeDefined();
    expect(screen.getByText("StratifiedKFold")).toBeDefined();
    expect(screen.getByText("GroupKFold")).toBeDefined();
    expect(screen.getByText("TimeSeriesSplit")).toBeDefined();
  });

  it("fires sendAction('update_cv') when strategy changes", () => {
    const sendAction = vi.fn();
    render(<DataTab {...defaultProps} sendAction={sendAction} />);
    fireEvent.click(screen.getByText("StratifiedKFold"));
    expect(sendAction).toHaveBeenCalledWith(
      "update_cv",
      expect.objectContaining({ strategy: "stratified_kfold" }),
    );
  });

  it("renders an unknown CV strategy reported by backend without a JS edit (#119)", () => {
    // Acceptance criterion: adding a new strategy in adapter_contract.py should
    // surface in the UI without touching DataTab.tsx.
    render(
      <DataTab
        {...defaultProps}
        backendContract={{
          capabilities: {
            cv_strategies: ["future_strategy"],
            cv_strategy_labels: { future_strategy: "Future Strategy" },
            cv_strategy_fields: { future_strategy: ["n_splits"] },
          },
        }}
      />,
    );
    expect(screen.getByText("Future Strategy")).toBeDefined();
  });

  it("falls back to humanised label when cv_strategy_labels lacks an entry", () => {
    render(
      <DataTab
        {...defaultProps}
        backendContract={{
          capabilities: {
            cv_strategies: ["weird_one"],
            cv_strategy_fields: { weird_one: ["n_splits"] },
            // cv_strategy_labels deliberately omitted
          },
        }}
      />,
    );
    expect(screen.getByText("Weird One")).toBeDefined();
  });
});

describe("DataTab — Feature summary display", () => {
  it("renders feature counts correctly", () => {
    render(<DataTab {...defaultProps} />);
    // "Features: 3 cols (Numeric: 1, Categ.: 1)"
    expect(screen.getByText(/Features: 3 cols/)).toBeDefined();
    expect(screen.getByText(/Numeric: 1/)).toBeDefined();
    expect(screen.getByText(/Excluded: 1/)).toBeDefined();
  });

  it("does not render feature summary when absent", () => {
    const { container } = render(
      <DataTab
        {...defaultProps}
        dfInfo={{ ...baseDfInfo, feature_summary: undefined }}
      />,
    );
    const summary = container.querySelector(".lzw-feature-summary");
    expect(summary).toBeNull();
  });
});

describe("DataTab — DataFrame shape display", () => {
  it("renders row and column counts", () => {
    render(<DataTab {...defaultProps} />);
    const shape = screen.getByText(/100 rows/);
    expect(shape).toBeDefined();
    expect(shape.textContent).toContain("5 cols");
  });
});

describe("DataTab — Group/Time column conditional display", () => {
  it("hides group column selector for non-group strategy (kfold)", () => {
    const { container } = render(<DataTab {...defaultProps} />);
    expect(container.textContent).not.toContain("Group column");
  });

  it("hides time column selector for non-time strategy (kfold)", () => {
    const { container } = render(<DataTab {...defaultProps} />);
    expect(container.textContent).not.toContain("Time column");
  });

  it("shows group column selector for group_kfold", () => {
    render(
      <DataTab
        {...defaultProps}
        dfInfo={{ ...baseDfInfo, cv: { strategy: "group_kfold", n_splits: 5 } }}
      />,
    );
    expect(screen.getByText("Group column")).toBeDefined();
  });

  it("shows time column selector for time_series", () => {
    render(
      <DataTab
        {...defaultProps}
        dfInfo={{ ...baseDfInfo, cv: { strategy: "time_series", n_splits: 5 } }}
        backendContract={{
          capabilities: {
            cv_strategy_fields: {
              time_series: ["n_splits", "time_col", "gap", "max_train_size", "max_test_size"],
            },
          },
        }}
      />,
    );
    expect(screen.getByText("Time column")).toBeDefined();
  });

  it("shows both for group_time_series", () => {
    render(
      <DataTab
        {...defaultProps}
        dfInfo={{ ...baseDfInfo, cv: { strategy: "group_time_series", n_splits: 5 } }}
        backendContract={{
          capabilities: {
            cv_strategy_fields: {
              group_time_series: ["n_splits", "group_col", "time_col", "gap"],
            },
          },
        }}
      />,
    );
    expect(screen.getByText("Group column")).toBeDefined();
    expect(screen.getByText("Time column")).toBeDefined();
  });
});
