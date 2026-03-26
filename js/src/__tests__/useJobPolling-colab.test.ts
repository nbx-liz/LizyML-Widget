/**
 * Tests for useJobPolling Colab path — window.google.colab is set.
 *
 * IN_COLAB is evaluated at module load time, so we set window.google.colab
 * BEFORE importing the hook via dynamic import + vi.resetModules.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/preact";
import { createMockModel, MockModel } from "./mock-model";

// Set up Colab environment before importing the hook
(window as any).google = { colab: {} };

// Dynamic import AFTER setting window.google.colab
const { useJobPolling } = await import("../hooks/useJobPolling");

describe("useJobPolling (Colab path)", () => {
  let model: MockModel;

  beforeEach(() => {
    vi.useFakeTimers();
    model = createMockModel();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts polling immediately on Colab (no stall detection delay)", () => {
    renderHook(
      ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
      { initialProps: { status: "running", jobIndex: 1 } },
    );

    // On Colab, polling should start immediately — no need to wait 2000ms
    // Just advance a tiny bit for microtask processing
    act(() => { vi.advanceTimersByTime(50); });

    expect(model.sentMessages.some(m => m.type === "poll")).toBe(true);
  });

  it("sends periodic poll messages at 1s intervals", () => {
    renderHook(
      ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
      { initialProps: { status: "running", jobIndex: 1 } },
    );

    act(() => { vi.advanceTimersByTime(50); });
    const initialCount = model.sentMessages.filter(m => m.type === "poll").length;

    act(() => { vi.advanceTimersByTime(1000); });
    const afterOneInterval = model.sentMessages.filter(m => m.type === "poll").length;

    expect(afterOneInterval).toBeGreaterThan(initialCount);
  });

  it("stops polling when status leaves 'running'", () => {
    const { rerender } = renderHook(
      ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
      { initialProps: { status: "running", jobIndex: 1 } },
    );

    act(() => { vi.advanceTimersByTime(50); });
    model.clearSentMessages();

    // Status changes to completed
    rerender({ status: "completed", jobIndex: 1 });
    act(() => { vi.advanceTimersByTime(2000); });

    // No more poll messages should be sent
    expect(model.sentMessages.filter(m => m.type === "poll")).toHaveLength(0);
  });

  it("clears polled state on completed (regression — Colab path)", () => {
    const { result, rerender } = renderHook(
      ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
      { initialProps: { status: "running", jobIndex: 1 } },
    );

    act(() => { vi.advanceTimersByTime(50); });

    // Simulate poll response
    act(() => {
      model.simulateCustomMessage({
        type: "job_state",
        status: "running",
        progress: {},
        elapsed_sec: 1.0,
        job_type: "tune",
        job_index: 1,
        error: {},
      });
    });

    expect(result.current).not.toBeNull();

    // Complete
    rerender({ status: "completed", jobIndex: 1 });
    expect(result.current).toBeNull();
  });

  it("restarts polling on jobIndex change (A-1 Colab scenario)", () => {
    const { rerender } = renderHook(
      ({ status, jobIndex }) => useJobPolling(model, status, 0, jobIndex),
      { initialProps: { status: "running", jobIndex: 1 } },
    );

    act(() => { vi.advanceTimersByTime(50); });

    // Simulate completion via poll
    act(() => {
      model.simulateCustomMessage({
        type: "job_state",
        status: "completed",
        progress: {},
        elapsed_sec: 5.0,
        job_type: "tune",
        job_index: 1,
        error: {},
        tune_summary: { best_params: {} },
      });
    });

    model.clearSentMessages();

    // New job starts — same status but different jobIndex
    rerender({ status: "running", jobIndex: 2 });
    act(() => { vi.advanceTimersByTime(50); });

    // New poll messages should be sent
    expect(model.sentMessages.some(m => m.type === "poll")).toBe(true);
  });
});
