/**
 * Tests for usePlot hook — request_id stale-response filtering (B-1)
 * and cache management.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/preact";
import { createMockModel, MockModel } from "./mock-model";
import { usePlot } from "../hooks/usePlot";

describe("usePlot", () => {
  let model: MockModel;

  beforeEach(() => {
    model = createMockModel();
  });

  // ── Basic request/response ──

  describe("basic plot request", () => {
    it("sends action with request_id and caches response", () => {
      const { result } = renderHook(() => usePlot(model));

      // Request a plot
      act(() => { result.current.requestPlot("roc-curve"); });

      // Should have sent an action
      expect(model.sentMessages).toHaveLength(1);
      const msg = model.sentMessages[0];
      expect(msg.type).toBe("action");
      expect(msg.action_type).toBe("request_plot");
      expect(msg.payload.plot_type).toBe("roc-curve");
      expect(msg.payload.request_id).toBeDefined();

      // Capture the request_id
      const rid = msg.payload.request_id;

      // Simulate response with matching request_id
      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "roc-curve",
          plotly_json: '{"data": [{"x": [1]}]}',
          request_id: rid,
        });
      });

      // Plot should be cached
      expect(result.current.plots["roc-curve"]).toBeDefined();
      expect(result.current.plots["roc-curve"].data[0].x[0]).toBe(1);
      expect(result.current.loading["roc-curve"]).toBe(false);
    });

    it("does not re-request already cached plot", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("roc-curve"); });
      const rid = model.sentMessages[0].payload.request_id;

      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "roc-curve",
          plotly_json: '{"data": []}',
          request_id: rid,
        });
      });

      model.clearSentMessages();

      // Request same plot again — should be no-op (cached)
      act(() => { result.current.requestPlot("roc-curve"); });
      expect(model.sentMessages).toHaveLength(0);
    });

    it("does not re-request while loading", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("roc-curve"); });
      expect(result.current.loading["roc-curve"]).toBe(true);

      model.clearSentMessages();

      // Request same plot again while loading — should be no-op
      act(() => { result.current.requestPlot("roc-curve"); });
      expect(model.sentMessages).toHaveLength(0);
    });
  });

  // ── B-1: Stale response filtering ──

  describe("stale response filtering (B-1)", () => {
    it("rejects response with mismatched request_id", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("roc-curve"); });

      // Simulate response with WRONG request_id (stale from previous request)
      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "roc-curve",
          plotly_json: '{"data": "stale"}',
          request_id: "old-stale-id",
        });
      });

      // Should NOT be cached
      expect(result.current.plots["roc-curve"]).toBeUndefined();
      // Loading should still be true (waiting for correct response)
      expect(result.current.loading["roc-curve"]).toBe(true);
    });

    it("rejects response without request_id when pending exists", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("roc-curve"); });

      // Simulate response WITHOUT request_id (legacy or stripped response)
      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "roc-curve",
          plotly_json: '{"data": "legacy"}',
          // no request_id
        });
      });

      // Should NOT be cached because we have a pending request_id
      expect(result.current.plots["roc-curve"]).toBeUndefined();
      expect(result.current.loading["roc-curve"]).toBe(true);
    });

    it("accepts response without request_id when no pending (backward compat)", () => {
      const { result } = renderHook(() => usePlot(model));

      // Simulate unsolicited response (e.g., from old Python code)
      // without any prior requestPlot call
      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "feature-importance",
          plotly_json: '{"data": "unsolicited"}',
        });
      });

      // Should be accepted (no pending request_id to check against)
      expect(result.current.plots["feature-importance"]).toBeDefined();
    });
  });

  // ── Error responses ──

  describe("error responses", () => {
    it("clears loading on plot_error with matching request_id", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("shap-summary"); });
      const rid = model.sentMessages[0].payload.request_id;

      act(() => {
        model.simulateCustomMessage({
          type: "plot_error",
          plot_type: "shap-summary",
          message: "No model available",
          request_id: rid,
        });
      });

      expect(result.current.loading["shap-summary"]).toBe(false);
      expect(result.current.plots["shap-summary"]).toBeUndefined();
    });

    it("ignores plot_error with mismatched request_id", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("shap-summary"); });

      act(() => {
        model.simulateCustomMessage({
          type: "plot_error",
          plot_type: "shap-summary",
          message: "stale error",
          request_id: "wrong-id",
        });
      });

      // Loading should still be true (error was for wrong request)
      expect(result.current.loading["shap-summary"]).toBe(true);
    });
  });

  // ── Cache clearing ──

  describe("clearCache", () => {
    it("clears all plots and loading states", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("roc-curve"); });
      const rid = model.sentMessages[0].payload.request_id;

      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "roc-curve",
          plotly_json: '{"data": []}',
          request_id: rid,
        });
      });

      expect(result.current.plots["roc-curve"]).toBeDefined();

      act(() => { result.current.clearCache(); });

      expect(result.current.plots).toEqual({});
      expect(result.current.loading).toEqual({});
    });

    it("accepts delayed response after clearCache (no pending)", () => {
      const { result } = renderHook(() => usePlot(model));

      act(() => { result.current.requestPlot("roc-curve"); });
      const rid = model.sentMessages[0].payload.request_id;

      // Clear cache before response arrives
      act(() => { result.current.clearCache(); });

      // Delayed response arrives — no pending, so accepted (backward compat)
      act(() => {
        model.simulateCustomMessage({
          type: "plot_data",
          plot_type: "roc-curve",
          plotly_json: '{"data": "delayed"}',
          request_id: rid,
        });
      });

      // Accepted because clearCache removes pending tracking
      expect(result.current.plots["roc-curve"]).toBeDefined();
    });
  });
});
