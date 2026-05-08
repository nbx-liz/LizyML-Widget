/**
 * Tests for ConfigTab debounce/traitlet race (#136).
 *
 * If a user is editing the config (debounce timer pending) when Python
 * pushes a new ``config`` prop (e.g. after ``apply_best_params`` completes),
 * the pending timer must be cancelled.  Otherwise it fires later and
 * computes a patch against a stale baseline, silently emitting an
 * incorrect ``patch_config``.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render } from "@testing-library/preact";
import type { ComponentChildren } from "preact";
import { ConfigTab } from "../tabs/ConfigTab";
import { createMockModel } from "./mock-model";

// Capture handleChange from FitSubTab so the test can drive an edit.
let capturedHandleChange: ((newConfig: Record<string, any>) => void) | null = null;

vi.mock("../tabs/FitSubTab", () => ({
  FitSubTab: (props: { handleChange: (cfg: Record<string, any>) => void; children?: ComponentChildren }) => {
    capturedHandleChange = props.handleChange;
    return null;
  },
}));

vi.mock("../tabs/TuneSubTab", () => ({
  TuneSubTab: () => null,
}));

const minimalContract = {
  config_schema: { type: "object", properties: {} },
  ui_schema: {
    sections: [],
    option_sets: {},
    parameter_hints: [],
    step_map: {},
    conditional_visibility: {},
  },
  capabilities: {},
};

const baseConfig = { model: { name: "lgbm", params: { learning_rate: 0.1 } } };

beforeEach(() => {
  vi.useFakeTimers();
  capturedHandleChange = null;
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("ConfigTab — debounce/traitlet race (#136)", () => {
  it("cancels a pending debounce when an external config update arrives", () => {
    const sendAction = vi.fn();
    const { rerender } = render(
      <ConfigTab
        backendContract={minimalContract}
        config={baseConfig}
        dfInfo={{ target: "y", task: "binary", shape: [100, 5] }}
        status="data_loaded"
        sendAction={sendAction}
        model={createMockModel()}
      />,
    );

    expect(capturedHandleChange).not.toBeNull();
    sendAction.mockClear();

    // 1. User edits a field at t=0 — debounce starts
    capturedHandleChange!({
      ...baseConfig,
      model: { ...baseConfig.model, params: { learning_rate: 0.5 } },
    });

    // 2. Before debounce fires, Python pushes a new config (e.g. after
    //    apply_best_params completes).
    vi.advanceTimersByTime(100); // halfway through the 300ms debounce
    const externalConfig = {
      model: { name: "lgbm", params: { learning_rate: 0.05, num_leaves: 64 } },
    };
    rerender(
      <ConfigTab
        backendContract={minimalContract}
        config={externalConfig}
        dfInfo={{ target: "y", task: "binary", shape: [100, 5] }}
        status="data_loaded"
        sendAction={sendAction}
        model={createMockModel()}
      />,
    );

    // 3. Drain all pending timers — the cancelled one MUST NOT fire.
    vi.runAllTimers();

    // Without the fix, the pending debounce fires after the rerender and
    // computes a patch from ``lastSentRef.current`` (now the *external*
    // config) to ``newConfig`` captured in the closure (the user's edit),
    // spuriously overwriting Python's push. With the fix we cancel the
    // timer in the ``[config]`` useEffect so no patch_config call escapes.
    expect(sendAction).not.toHaveBeenCalled();
  });

  it("clears the timer on unmount (regression — pre-existing cleanup still works)", () => {
    const sendAction = vi.fn();
    const { unmount } = render(
      <ConfigTab
        backendContract={minimalContract}
        config={baseConfig}
        dfInfo={{ target: "y", task: "binary", shape: [100, 5] }}
        status="data_loaded"
        sendAction={sendAction}
        model={createMockModel()}
      />,
    );

    capturedHandleChange!({
      ...baseConfig,
      model: { ...baseConfig.model, params: { learning_rate: 0.7 } },
    });
    sendAction.mockClear();
    unmount();
    vi.runAllTimers();

    expect(sendAction).not.toHaveBeenCalled();
  });
});
