/**
 * Tests for useJobPolling hook — the polling fallback for Colab/stalled
 * traitlet sync, and the state lifecycle that caused the "frozen UI" bug.
 *
 * Key regression: polled state must be cleared on completed/failed transitions,
 * otherwise effectiveStatus = polled?.status ?? traitletStatus stays "running".
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/preact";
import { createMockModel, MockModel } from "./mock-model";
import { useJobPolling } from "../hooks/useJobPolling";

// Non-Colab environment (jsdom has no window.google.colab),
// so stall detection activates after STALL_DETECT_MS (2000ms).
// Use fake timers to control timing precisely.

describe("useJobPolling", () => {
  let model: MockModel;

  beforeEach(() => {
    vi.useFakeTimers();
    model = createMockModel();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ── Regression test: the bug that caused "frozen UI after Tune" ──

  describe("polled state lifecycle (regression)", () => {
    it("clears polled when traitletStatus transitions to 'completed'", () => {
      // Start with running status — polling will activate after stall detection
      const { result, rerender } = renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      // Advance past stall detection (2000ms) to start polling
      act(() => { vi.advanceTimersByTime(2100); });

      // Simulate a poll response with "running" state
      act(() => {
        model.simulateCustomMessage({
          type: "job_state",
          status: "running",
          progress: { current: 1, total: 5, message: "Fold 1/5" },
          elapsed_sec: 1.5,
          job_type: "tune",
          job_index: 1,
          error: {},
        });
      });

      // polled should now be set
      expect(result.current).not.toBeNull();
      expect(result.current!.status).toBe("running");

      // *** THE BUG: traitletStatus changes to "completed" ***
      // polled MUST be cleared so effectiveStatus falls back to traitletStatus
      rerender({ status: "completed", jobIndex: 1 });

      expect(result.current).toBeNull();
    });

    it("clears polled when traitletStatus transitions to 'failed'", () => {
      const { result, rerender } = renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      act(() => { vi.advanceTimersByTime(2100); });

      act(() => {
        model.simulateCustomMessage({
          type: "job_state",
          status: "running",
          progress: {},
          elapsed_sec: 0.5,
          job_type: "fit",
          job_index: 1,
          error: {},
        });
      });

      expect(result.current).not.toBeNull();

      rerender({ status: "failed", jobIndex: 1 });

      expect(result.current).toBeNull();
    });
  });

  // ── A-1: Consecutive jobs restart polling ──

  describe("consecutive jobs (A-1)", () => {
    it("restarts polling when jobIndex changes while status stays 'running'", () => {
      const { result, rerender } = renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      // Start polling via stall detection
      act(() => { vi.advanceTimersByTime(2100); });

      // Simulate first job completing via poll
      act(() => {
        model.simulateCustomMessage({
          type: "job_state",
          status: "completed",
          progress: {},
          elapsed_sec: 5.0,
          job_type: "tune",
          job_index: 1,
          error: {},
          tune_summary: { best_params: { lr: 0.01 } },
        });
      });

      expect(result.current).not.toBeNull();
      expect(result.current!.status).toBe("completed");

      // *** Colab scenario: traitletStatus stays "running" but jobIndex changes ***
      // (On Colab, BG thread traitlet writes don't reach JS)
      model.clearSentMessages();
      rerender({ status: "running", jobIndex: 2 });

      // Polling should restart — advance past stall detection again
      act(() => { vi.advanceTimersByTime(2100); });

      // polled should be reset (null or re-initialized)
      // and new poll messages should be sent
      expect(model.sentMessages.some(m => m.type === "poll")).toBe(true);
    });
  });

  // ── Stall detection ──

  describe("stall detection (non-Colab)", () => {
    it("does not start polling immediately", () => {
      renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      // Before stall detection fires, no poll messages should be sent
      expect(model.sentMessages).toHaveLength(0);
    });

    it("starts polling after STALL_DETECT_MS", () => {
      renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      act(() => { vi.advanceTimersByTime(2100); });

      // Poll message should have been sent
      expect(model.sentMessages.some(m => m.type === "poll")).toBe(true);
    });

    it("does not start polling when status is not 'running'", () => {
      renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "data_loaded", jobIndex: 0 } },
      );

      act(() => { vi.advanceTimersByTime(3000); });

      expect(model.sentMessages).toHaveLength(0);
    });
  });

  // ── Duplicate registration guard ──

  describe("duplicate registration guard", () => {
    it("does not register duplicate msg:custom listeners", () => {
      renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      // Start polling
      act(() => { vi.advanceTimersByTime(2100); });

      const listenerCount = model.listenerCount("msg:custom");

      // Advance more time — polling interval fires again but should NOT add more listeners
      act(() => { vi.advanceTimersByTime(2000); });

      expect(model.listenerCount("msg:custom")).toBe(listenerCount);
    });
  });

  // ── Poll response handling ──

  describe("poll response handling", () => {
    it("updates polled state from job_state message", () => {
      const { result } = renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      act(() => { vi.advanceTimersByTime(2100); });

      act(() => {
        model.simulateCustomMessage({
          type: "job_state",
          status: "running",
          progress: { current: 3, total: 5, message: "Fold 3/5" },
          elapsed_sec: 2.5,
          job_type: "fit",
          job_index: 1,
          error: {},
        });
      });

      expect(result.current).not.toBeNull();
      expect(result.current!.status).toBe("running");
      expect(result.current!.progress.current).toBe(3);
      expect(result.current!.elapsed_sec).toBe(2.5);
      expect(result.current!.job_type).toBe("fit");
    });

    it("ignores non-job_state messages", () => {
      const { result } = renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      act(() => { vi.advanceTimersByTime(2100); });

      act(() => {
        model.simulateCustomMessage({ type: "plot_data", plot_type: "roc" });
      });

      // polled should still be null (no job_state received)
      expect(result.current).toBeNull();
    });

    it("includes fit_summary and tune_summary on completed poll", () => {
      const { result } = renderHook(
        ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
        { initialProps: { status: "running", jobIndex: 1 } },
      );

      act(() => { vi.advanceTimersByTime(2100); });

      act(() => {
        model.simulateCustomMessage({
          type: "job_state",
          status: "completed",
          progress: {},
          elapsed_sec: 10.0,
          job_type: "tune",
          job_index: 1,
          error: {},
          fit_summary: { metrics: {} },
          tune_summary: { best_params: { lr: 0.01 }, best_score: 0.95 },
          available_plots: ["optimization-history"],
        });
      });

      expect(result.current).not.toBeNull();
      expect(result.current!.tune_summary?.best_score).toBe(0.95);
      expect(result.current!.available_plots).toContain("optimization-history");
    });
  });
});
