/**
 * Tests for useModel hooks — useTraitlet, useSendAction, useCustomMsg.
 * These are the fundamental building blocks for Python↔JS communication.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/preact";
import { createMockModel, MockModel } from "./mock-model";
import { useTraitlet, useSendAction, useCustomMsg } from "../hooks/useModel";

describe("useTraitlet", () => {
  let model: MockModel;

  beforeEach(() => {
    model = createMockModel({ status: "idle", job_index: 0 });
  });

  it("returns initial value from model.get", () => {
    const { result } = renderHook(() => useTraitlet<string>(model, "status"));
    expect(result.current).toBe("idle");
  });

  it("updates when traitlet changes", () => {
    const { result } = renderHook(() => useTraitlet<string>(model, "status"));
    expect(result.current).toBe("idle");

    act(() => { model.simulateTraitletChange("status", "running"); });
    expect(result.current).toBe("running");
  });

  it("updates on multiple changes", () => {
    const { result } = renderHook(() => useTraitlet<number>(model, "job_index"));
    expect(result.current).toBe(0);

    act(() => { model.simulateTraitletChange("job_index", 1); });
    expect(result.current).toBe(1);

    act(() => { model.simulateTraitletChange("job_index", 2); });
    expect(result.current).toBe(2);
  });

  it("does not react to changes on other traitlets", () => {
    const { result } = renderHook(() => useTraitlet<string>(model, "status"));

    act(() => { model.simulateTraitletChange("job_index", 99); });
    expect(result.current).toBe("idle"); // unchanged
  });

  it("cleans up listener on unmount", () => {
    const { unmount } = renderHook(() => useTraitlet<string>(model, "status"));
    expect(model.listenerCount("change:status")).toBe(1);

    unmount();
    expect(model.listenerCount("change:status")).toBe(0);
  });

  it("handles Dict traitlet (object values)", () => {
    const { result } = renderHook(() =>
      useTraitlet<Record<string, any>>(model, "progress"),
    );
    expect(result.current).toEqual({});

    act(() => {
      model.simulateTraitletChange("progress", {
        current: 3,
        total: 5,
        message: "Fold 3/5",
      });
    });
    expect(result.current.current).toBe(3);
  });
});

describe("useSendAction", () => {
  let model: MockModel;

  beforeEach(() => {
    model = createMockModel();
  });

  it("sends msg:custom with correct format", () => {
    const { result } = renderHook(() => useSendAction(model));

    act(() => { result.current("fit"); });

    expect(model.sentMessages).toHaveLength(1);
    expect(model.sentMessages[0]).toEqual({
      type: "action",
      action_type: "fit",
      payload: {},
    });
  });

  it("includes payload when provided", () => {
    const { result } = renderHook(() => useSendAction(model));

    act(() => {
      result.current("patch_config", {
        ops: [{ op: "set", path: "model.params.lr", value: 0.05 }],
      });
    });

    const msg = model.sentMessages[0];
    expect(msg.action_type).toBe("patch_config");
    expect(msg.payload.ops[0].value).toBe(0.05);
  });

  it("returns stable callback reference", () => {
    const { result, rerender } = renderHook(() => useSendAction(model));
    const first = result.current;

    rerender({});
    expect(result.current).toBe(first);
  });
});

describe("useCustomMsg", () => {
  let model: MockModel;

  beforeEach(() => {
    model = createMockModel();
  });

  it("receives msg:custom messages", () => {
    const received: any[] = [];
    renderHook(() =>
      useCustomMsg(model, (msg: any) => received.push(msg)),
    );

    act(() => {
      model.simulateCustomMessage({ type: "plot_data", plot_type: "roc" });
    });

    expect(received).toHaveLength(1);
    expect(received[0].type).toBe("plot_data");
  });

  it("receives multiple messages", () => {
    const received: any[] = [];
    renderHook(() =>
      useCustomMsg(model, (msg: any) => received.push(msg)),
    );

    act(() => {
      model.simulateCustomMessage({ type: "plot_data" });
      model.simulateCustomMessage({ type: "yaml_export" });
    });

    expect(received).toHaveLength(2);
  });

  it("uses latest handler without re-subscribing", () => {
    let count = 0;
    const { rerender } = renderHook(
      ({ cb }) => useCustomMsg(model, cb),
      { initialProps: { cb: () => { count += 1; } } },
    );

    // Re-render with a new handler
    rerender({ cb: () => { count += 10; } });

    // Listener count should still be 1 (no double-registration)
    expect(model.listenerCount("msg:custom")).toBe(1);

    act(() => {
      model.simulateCustomMessage({ type: "test" });
    });

    // Should use the LATEST handler (count += 10, not += 1)
    expect(count).toBe(10);
  });

  it("cleans up listener on unmount", () => {
    const { unmount } = renderHook(() =>
      useCustomMsg(model, () => {}),
    );
    expect(model.listenerCount("msg:custom")).toBe(1);

    unmount();
    expect(model.listenerCount("msg:custom")).toBe(0);
  });
});
