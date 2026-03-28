/**
 * Tests for usePlot metrics options (P-026).
 * Verifies that requestPlot forwards options to Python via msg:custom.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/preact";
import { createMockModel, MockModel } from "./mock-model";
import { usePlot } from "../hooks/usePlot";

describe("usePlot metrics options (P-026)", () => {
  let model: MockModel;

  beforeEach(() => {
    model = createMockModel();
  });

  it("sends options.metrics in payload when provided", () => {
    const { result } = renderHook(() => usePlot(model));

    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["auc"] });
    });

    expect(model.sentMessages).toHaveLength(1);
    const msg = model.sentMessages[0];
    expect(msg.payload.plot_type).toBe("learning-curve");
    expect(msg.payload.options).toEqual({ metrics: ["auc"] });
    expect(msg.payload.request_id).toBeDefined();
  });

  it("sends no options field when not provided", () => {
    const { result } = renderHook(() => usePlot(model));

    act(() => {
      result.current.requestPlot("roc-curve");
    });

    expect(model.sentMessages).toHaveLength(1);
    const msg = model.sentMessages[0];
    expect(msg.payload.options).toBeUndefined();
  });

  it("invalidates cache and re-requests when options change", () => {
    const { result } = renderHook(() => usePlot(model));

    // First request
    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["auc"] });
    });
    const rid1 = model.sentMessages[0].payload.request_id;

    // Simulate response
    act(() => {
      model.simulateCustomMessage({
        type: "plot_data",
        plot_type: "learning-curve",
        plotly_json: '{"data": [{"name": "auc"}]}',
        request_id: rid1,
      });
    });
    expect(result.current.plots["learning-curve"]).toBeDefined();

    model.clearSentMessages();

    // Request with different options — should bypass cache
    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["binary_logloss"] });
    });

    expect(model.sentMessages).toHaveLength(1);
    expect(model.sentMessages[0].payload.options).toEqual({ metrics: ["binary_logloss"] });
  });

  it("uses cache when same options are requested again", () => {
    const { result } = renderHook(() => usePlot(model));

    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["auc"] });
    });
    const rid = model.sentMessages[0].payload.request_id;

    act(() => {
      model.simulateCustomMessage({
        type: "plot_data",
        plot_type: "learning-curve",
        plotly_json: '{"data": []}',
        request_id: rid,
      });
    });

    model.clearSentMessages();

    // Same options — should use cache (no new request)
    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["auc"] });
    });

    expect(model.sentMessages).toHaveLength(0);
  });

  it("re-requests when options change while loading", () => {
    const { result } = renderHook(() => usePlot(model));

    // First request
    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["auc"] });
    });
    expect(result.current.loading["learning-curve"]).toBe(true);

    model.clearSentMessages();

    // Change options while still loading — should send new request
    act(() => {
      result.current.requestPlot("learning-curve", { metrics: ["binary_logloss"] });
    });

    expect(model.sentMessages).toHaveLength(1);
    expect(model.sentMessages[0].payload.options).toEqual({ metrics: ["binary_logloss"] });
  });
});
